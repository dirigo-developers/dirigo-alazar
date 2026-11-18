"""Alazar connections and preparation of finite externally triggered records."""

from dataclasses import fields
from importlib import import_module
from importlib import metadata as importlib_metadata
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
from dirigo import units
from dirigo.hardware.capture import (
    CaptureInfo,
    CaptureLayout,
    CaptureSettings,
    SampleArray,
)
from dirigo.hardware.capture_session import DigitizerCaptureSession
from dirigo.hardware.digitizer import (
    ChannelCoupling,
    Digitizer,
    DigitizerProfile,
    ExternalTriggerCoupling,
)
from dirigo.hardware.digitizer import (
    ExternalTriggerRange as ProfileTriggerRange,
)
from dirigo.hardware.metadata import DeviceIdentity, DriverIdentity
from dirigo.hardware.profiles import capture_settings_from_profile
from dirigo.hardware.timing import SignalEdge

from dirigo_alazar.dma import _DMAPool
from dirigo_alazar.settings import (
    AlazarCaptureSettings,
    AlazarDigitizerConfig,
    AuxOutputMode,
    Coupling,
    ExternalTriggerRange,
)

if TYPE_CHECKING:
    from atsbindings import Board


class AlazarDigitizer(Digitizer):
    """Describe one Alazar connection and discover its board during preparation.

    Construction and create_capture() do not load the SDK or access hardware.
    prepare() loads the SDK, discovers the board, and applies run settings.
    SDK initialization may reset boards, so discovery is deferred deliberately.
    Use one logical digitizer per board. Reservations do not coordinate other
    objects or applications accessing the same physical board.

    Samples are signed, right-justified ADC codes stored in int8 or int16.
    """

    config_type = AlazarDigitizerConfig

    def __init__(self, config: AlazarDigitizerConfig) -> None:
        """Store connection settings without loading the native library."""
        if not isinstance(config, AlazarDigitizerConfig):
            raise TypeError("config must be an AlazarDigitizerConfig.")
        super().__init__()
        self.config = config
        self._connection: tuple[ModuleType, Board] | None = None
        self._channel_count: int | None = None
        self._bit_depth: int | None = None
        self._dma_pool: _DMAPool | None = None
        self._identity = DeviceIdentity()

    def _connect(self) -> tuple[ModuleType, Board]:
        """Discover and cache a board while a capture owns the reservation."""
        if self._connection is not None:
            return self._connection

        sdk = import_module("atsbindings")
        board = sdk.Board(self.config.system_id, self.config.board_id)
        count = int(board.bsi.channels)
        _, bits = board.get_channel_info()
        bits = int(bits)

        if count <= 0 or not 2 <= bits <= 16:
            raise NotImplementedError(
                "Unsupported Alazar input count or ADC resolution."
            )

        identity = DeviceIdentity(
            manufacturer="AlazarTech",
            model=board.get_board_kind().name,
            serial_number=str(
                board.query_capability(sdk.Ats.Capabilities.GET_SERIAL_NUMBER)
            ),
        )

        self._channel_count = count
        self._bit_depth = bits
        self._identity = identity
        self._connection = (sdk, board)
        return self._connection

    @property
    def channel_count(self) -> int:
        """Physical input count, available after the first successful discovery."""
        if self._channel_count is None:
            raise RuntimeError("Channel count is unavailable before board discovery.")
        return self._channel_count

    @property
    def bit_depth(self) -> int:
        """ADC resolution in bits, available after board discovery."""
        if self._bit_depth is None:
            raise RuntimeError("ADC resolution is unavailable before board discovery.")
        return self._bit_depth

    @property
    def data_range(self) -> units.IntRange:
        """Inclusive bounds of the planned signed, right-justified sample codes."""
        half = 2 ** (self.bit_depth - 1)
        return units.IntRange(-half, half - 1)

    @property
    def reported_identity(self) -> DeviceIdentity:
        """Return cached board identity without making SDK calls."""
        return self._identity

    @property
    def driver_identity(self) -> DriverIdentity:
        """Identify this implementation and its installed distribution."""
        try:
            version = importlib_metadata.version("dirigo-alazar")
        except importlib_metadata.PackageNotFoundError:
            version = None

        return DriverIdentity(
            implementation=f"{type(self).__module__}.{type(self).__qualname__}",
            distribution="dirigo-alazar",
            version=version,
        )

    def _snapshot_settings(self) -> dict[str, object]:
        """Describe the connection and cached input geometry."""
        return {
            "system_id": self.config.system_id,
            "board_id": self.config.board_id,
            "channel_count": self._channel_count,
            "bit_depth": self._bit_depth,
        }

    def _validate_channel_indices(self, indices: tuple[int, ...]) -> None:
        """Check known bounds, deferring physical discovery until prepare()."""
        if self._channel_count is not None:
            super()._validate_channel_indices(indices)

    def _create_capture(
        self,
        settings: CaptureSettings,
    ) -> DigitizerCaptureSession:
        """Validate complete settings and construct an unprepared session.

        Construction does not load the SDK, access hardware, or reserve
        the digitizer.
        """
        if self._dma_pool is not None and self._capture_owner is None:
            raise RuntimeError("Previous DMA cleanup failed; close this digitizer.")

        if not isinstance(settings, AlazarCaptureSettings):
            raise TypeError("Alazar capture requires AlazarCaptureSettings.")

        self._validate_capture_settings(settings)
        return _AlazarCaptureSession(self, settings)

    def _create_profile_capture(
        self,
        profile: DigitizerProfile,
    ) -> DigitizerCaptureSession:
        """Construct Alazar capture with deferred geometry."""
        if self._dma_pool is not None and self._capture_owner is None:
            raise RuntimeError("Previous DMA cleanup failed; close this digitizer.")

        return _AlazarCaptureSession(self, profile=profile)

    def _settings_from_profile(
        self,
        profile: DigitizerProfile,
        *,
        record_length: int,
        records_per_batch: int,
        batch_count: int | None,
        timestamps_enabled: bool,
        trigger_offset_samples: int,
    ) -> AlazarCaptureSettings:
        """Resolve profile settings supported by the present Alazar backend."""
        selected = tuple(channel for channel in profile.channels if channel.enabled)
        first = selected[0]

        if first.input_range is None:
            raise ValueError(
                "Alazar profiles must specify enabled-channel input ranges."
            )

        if not isinstance(first.impedance, units.Resistance):
            raise ValueError("Alazar profiles must specify a numeric input impedance.")

        for channel in selected:
            if (
                channel.input_range != first.input_range
                or channel.impedance != first.impedance
                or channel.coupling != first.coupling
            ):
                raise NotImplementedError(
                    "Alazar capture currently requires shared range, "
                    "impedance, and coupling across enabled channels."
                )

            if channel.inverted or channel.offset is not None:
                raise NotImplementedError(
                    "The current Alazar backend does not implement "
                    "profile inversion or offset."
                )

        if first.coupling not in (
            None,
            ChannelCoupling.AC,
            ChannelCoupling.DC,
        ):
            raise NotImplementedError("Unsupported Alazar input coupling.")

        trigger = profile.trigger

        if trigger.level is not None:
            raise NotImplementedError(
                "The current Alazar backend supports TTL triggering "
                "without an adjustable trigger level."
            )

        if trigger.external_coupling not in (
            None,
            ExternalTriggerCoupling.DC,
        ):
            raise NotImplementedError(
                "The current Alazar backend requires DC-coupled triggers."
            )

        if trigger.external_range not in (
            None,
            ProfileTriggerRange.TTL,
        ):
            raise NotImplementedError(
                "The current Alazar backend requires the named TTL trigger mode."
            )

        if trigger.external_impedance is not None:
            raise NotImplementedError(
                "The current Alazar backend does not implement independent "
                "external-trigger impedance selection."
            )

        common = capture_settings_from_profile(
            profile,
            record_length=record_length,
            records_per_batch=records_per_batch,
            batch_count=batch_count,
            timestamps_enabled=timestamps_enabled,
            trigger_offset_samples=trigger_offset_samples,
        )

        settings = AlazarCaptureSettings(
            **{
                item.name: getattr(common, item.name)
                for item in fields(CaptureSettings)
            },
            input_voltage_range=first.input_range,
            input_impedance=first.impedance,
            input_coupling=(
                Coupling.AC if first.coupling is ChannelCoupling.AC else Coupling.DC
            ),
        )
        self._validate_capture_settings(settings)
        return settings

    @staticmethod
    def _validate_capture_settings(
        settings: AlazarCaptureSettings,
    ) -> None:
        """Reject unsupported acquisition modes without accessing hardware.

        Board-specific checks remain deferred until preparation. This
        validation also applies to settings translated from a stored profile.
        """
        trigger = settings.record_trigger

        if (
            settings.batch_count is None
            or settings.external_clock is not None
            or settings.start_trigger is not None
            or settings.timestamps_enabled
            or settings.trigger_offset_samples != 0
            or trigger is None
            or trigger.source != "external"
            or settings.external_trigger_range is not ExternalTriggerRange.TTL
            or settings.external_trigger_coupling is not Coupling.DC
        ):
            raise NotImplementedError(
                "This backend currently prepares finite, internally clocked "
                "records with external DC-coupled TTL triggers, zero offset, "
                "and no timestamps."
            )

    def _close(self) -> None:
        """Retry retained DMA cleanup; native board handles remain SDK-owned."""
        if self._dma_pool is not None:
            self._dma_pool.close()
            self._dma_pool = None


class _AlazarCaptureSession(DigitizerCaptureSession):
    """Reserve and configure one run; DMA execution follows in a later increment."""

    def __init__(
        self,
        digitizer: AlazarDigitizer,
        settings: AlazarCaptureSettings | None = None,
        *,
        profile: DigitizerProfile | None = None,
    ) -> None:
        """Accept complete settings or an immutable profile snapshot."""
        if (settings is None) == (profile is None):
            raise ValueError("Supply exactly one of settings or profile.")

        super().__init__(digitizer, profile=profile)

        if settings is not None:
            self._set_capture_settings(settings)

        self._device = digitizer
        self._pool: _DMAPool | None = None

    @property
    def settings(self) -> AlazarCaptureSettings:
        """Complete Alazar request after configuration."""
        settings = super().settings

        if not isinstance(settings, AlazarCaptureSettings):
            raise TypeError("Alazar capture requires AlazarCaptureSettings.")

        return settings

    def _prepare_capture(self) -> CaptureLayout:
        """Configure the board, allocate and post buffers, without arming capture.

        SDK configuration failures can leave partially applied settings. A
        later preparation reapplies them. Failed native cleanup retains its
        buffers and prevents reuse until device closure retries cleanup.
        """
        sdk, board = self._device._connect()
        ats = sdk.Ats
        info = board.bsi
        settings = self.settings
        self._device._validate_channel_indices(settings.channel_indices)

        if (
            settings.record_length < info.min_record_size
            or settings.record_length % info.record_resolution
        ):
            raise ValueError("record_length does not meet this board's requirements.")

        rate = next(
            (r for r in info.sample_rates if r.to_hertz == float(settings.sample_rate)),
            None,
        )
        if rate is None or ats.ClockSources.INTERNAL_CLOCK not in info.supported_clocks:
            raise ValueError(
                "sample_rate is not supported by this board's internal clock."
            )

        coupling = (
            ats.Couplings.AC_COUPLING
            if settings.input_coupling is Coupling.AC
            else ats.Couplings.DC_COUPLING
        )
        if coupling not in info.input_couplings:
            raise ValueError("input_coupling is not supported by this board.")

        impedance_name = {
            50.0: "IMPEDANCE_50_OHM",
            75.0: "IMPEDANCE_75_OHM",
            100.0: "IMPEDANCE_100_OHM",
            300.0: "IMPEDANCE_300_OHM",
            1_000_000.0: "IMPEDANCE_1M_OHM",
        }.get(float(settings.input_impedance))
        impedance = (
            getattr(ats.Impedances, impedance_name)
            if impedance_name is not None
            else None
        )
        if impedance is None or impedance not in info.input_impedances:
            raise ValueError("input_impedance is not supported by this board.")

        input_range = next(
            (
                r
                for r in info.input_ranges(impedance)
                if r.to_volts == float(settings.input_voltage_range.max)
            ),
            None,
        )
        if input_range is None:
            raise ValueError("input_voltage_range is not supported at this impedance.")

        if (
            ats.ExternalTriggerRanges.ETR_TTL not in info.external_trigger_ranges
            or ats.TriggerSources.TRIG_EXTERNAL not in info.supported_trigger_sources
        ):
            raise ValueError("External TTL triggering is not supported by this board.")

        layout = CaptureLayout(
            shape=(
                settings.records_per_batch,
                settings.record_length,
                len(settings.channel_indices),
            ),
            channel_indices=settings.channel_indices,
            dtype=np.dtype(np.int8 if self._device.bit_depth <= 8 else np.int16),
        )

        board.set_capture_clock(
            ats.ClockSources.INTERNAL_CLOCK,
            rate,
            ats.ClockEdges.CLOCK_EDGE_RISING,
        )
        for index in settings.channel_indices:
            board.input_control_ex(
                channel=ats.Channels.from_int(index),
                coupling=coupling,
                input_range=input_range,
                impedance=impedance,
            )

        board.set_external_trigger(
            coupling=ats.Couplings.DC_COUPLING,
            range=ats.ExternalTriggerRanges.ETR_TTL,
        )
        trigger = settings.record_trigger
        assert trigger is not None
        slope = (
            ats.TriggerSlopes.TRIGGER_SLOPE_POSITIVE
            if trigger.active_edge is SignalEdge.RISING
            else ats.TriggerSlopes.TRIGGER_SLOPE_NEGATIVE
        )
        board.set_trigger_operation(
            operation=ats.TriggerOperations.TRIG_ENGINE_OP_J,
            engine1=ats.TriggerEngines.TRIG_ENGINE_J,
            source1=ats.TriggerSources.TRIG_EXTERNAL,
            slope1=slope,
            level1=128,
            engine2=ats.TriggerEngines.TRIG_ENGINE_K,
            source2=ats.TriggerSources.TRIG_DISABLE,
            slope2=ats.TriggerSlopes.TRIGGER_SLOPE_POSITIVE,
            level2=128,
        )
        board.set_trigger_time_out(0)
        board.set_trigger_delay(0)
        board.set_record_size(
            pre_trigger_samples=0,
            post_trigger_samples=settings.record_length,
        )

        aux = settings.aux_output_mode
        if aux is not None:
            mode = (
                ats.AuxIOModes.AUX_OUT_TRIGGER
                if aux is AuxOutputMode.ACQUISITION_ACTIVE
                else ats.AuxIOModes.AUX_OUT_SERIAL_DATA
            )
            board.configure_aux_io(mode=mode, parameter=0)

        pool = _DMAPool(sdk, board, settings)
        self._pool = pool
        self._device._dma_pool = pool
        pool.prepare()
        return layout

    def _start_capture(self) -> None:
        """Arm the prepared DMA run without waiting for incoming triggers."""
        assert self._pool is not None
        self._pool.start()

    def _read_into(
        self,
        destination: SampleArray,
        *,
        timeout: float | None,
    ) -> CaptureInfo:
        """Deliver one complete batch using the prepared native buffer pool."""
        assert self._pool is not None
        self._pool.read_into(destination, timeout=timeout)
        return CaptureInfo()

    def _stop_capture(self) -> None:
        """Abort DMA while retaining buffers and the input reservation."""
        if self._pool is not None:
            self._pool.stop()

    def _close_capture(self) -> None:
        """Release the pool only after its DMA activity has been aborted."""
        if self._pool is not None:
            self._pool.close()
            self._device._dma_pool = None
            self._pool = None
