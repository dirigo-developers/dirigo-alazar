"""Exercise Alazar preparation without loading the native SDK."""

from collections.abc import Iterator
from dataclasses import dataclass, replace
from enum import Enum
from types import ModuleType, SimpleNamespace
from typing import cast
from unittest.mock import Mock, call

import numpy as np
import pytest
from dirigo import units
from dirigo.hardware.capture import CaptureSettings
from dirigo.hardware.capture_session import CaptureState, DigitizerCaptureSession
from dirigo.hardware.timing import (
    DigitalRecordTrigger,
    DigitalStartTrigger,
    ExternalSampleClock,
    SignalEdge,
)

from dirigo_alazar import digitizer as digitizer_module
from dirigo_alazar.digitizer import AlazarDigitizer
from dirigo_alazar.settings import (
    AlazarCaptureSettings,
    AlazarDigitizerConfig,
    Coupling,
    ExternalTriggerRange,
)


@dataclass
class FakeSDK:
    """Keep connection, allocation, and native-call observations together."""

    board: Mock
    factory: Mock
    ats: SimpleNamespace
    load: Mock
    buffer_factory: Mock
    buffers: list[Mock]
    events: list[str]


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> FakeSDK:
    """Substitute the SDK loader without importing the native library.

    Board methods record calls. The capability table describes two 14-bit
    inputs, a 100 MS/s internal clock, and external TTL triggering. Symbols
    are local enums: these tests check adapter routing, not vendor constants.
    """
    ats = SimpleNamespace()
    for name, members in {
        "Couplings": "AC_COUPLING DC_COUPLING",
        "Impedances": "IMPEDANCE_50_OHM IMPEDANCE_1M_OHM",
        "ClockSources": "INTERNAL_CLOCK",
        "ClockEdges": "CLOCK_EDGE_RISING",
        "ExternalTriggerRanges": "ETR_TTL",
        "TriggerOperations": "TRIG_ENGINE_OP_J",
        "TriggerEngines": "TRIG_ENGINE_J TRIG_ENGINE_K",
        "TriggerSources": "TRIG_EXTERNAL TRIG_DISABLE",
        "TriggerSlopes": "TRIGGER_SLOPE_POSITIVE TRIGGER_SLOPE_NEGATIVE",
        "Capabilities": "GET_SERIAL_NUMBER CAP_SUPPORTS_NPT_AUTODMA",
        "Parameters": "PACK_MODE SET_DATA_FORMAT",
    }.items():
        setattr(ats, name, Enum(name, members))

    ats.Channels = SimpleNamespace(CHANNEL_ALL=0, from_int=lambda index: 1 << index)
    ats.PackModes = SimpleNamespace(PACK_DEFAULT=0)
    ats.ADMAModes = SimpleNamespace(ADMA_NPT=0x200)
    ats.ADMAFlags = SimpleNamespace(ADMA_EXTERNAL_STARTCAPTURE=1)
    rate = SimpleNamespace(to_hertz=100_000_000)
    voltage_range = SimpleNamespace(to_volts=1.0)

    board = Mock()
    board.bsi = SimpleNamespace(
        channels=2,
        channel_configs=[1, 2, 3],
        min_record_size=256,
        record_resolution=128,
        sample_rates=[rate],
        supported_clocks=[ats.ClockSources.INTERNAL_CLOCK],
        input_couplings=[ats.Couplings.DC_COUPLING],
        input_impedances=[ats.Impedances.IMPEDANCE_50_OHM],
        input_ranges=lambda impedance: [voltage_range],
        external_trigger_ranges=[ats.ExternalTriggerRanges.ETR_TTL],
        supported_trigger_sources=[ats.TriggerSources.TRIG_EXTERNAL],
    )
    board.get_channel_info.return_value = (2**20, 14)
    board.get_board_kind.return_value = SimpleNamespace(name="ATS9440")
    board.query_capability.side_effect = lambda cap: (
        123456 if cap is ats.Capabilities.GET_SERIAL_NUMBER else 1
    )
    events: list[str] = []
    buffers: list[Mock] = []

    def allocate(**kwargs: object) -> Mock:
        """Record allocation and provide a uniquely identified transfer buffer."""
        buffer = Mock(spec=["address", "size", "close"])
        buffer.address = len(buffers) + 1
        buffer.size = 32768
        buffer.close.side_effect = lambda: events.append("free")
        buffers.append(buffer)
        events.append("allocate")
        return buffer

    buffer_factory = Mock(side_effect=allocate)
    board.before_async_read.side_effect = lambda **kwargs: events.append("before")
    board.post_async_buffer.side_effect = lambda *args: events.append("post")
    board.start_capture.side_effect = lambda: events.append("start")
    board.abort_async_read.side_effect = lambda: events.append("abort")

    factory = Mock(return_value=board)
    module = ModuleType("atsbindings")
    module.__dict__.update(Board=factory, Ats=ats, Buffer=buffer_factory)
    loader = Mock(return_value=module)
    monkeypatch.setattr(digitizer_module, "import_module", loader)
    return FakeSDK(board, factory, ats, loader, buffer_factory, buffers, events)


@pytest.fixture
def settings() -> AlazarCaptureSettings:
    """Request two records per batch with reversed physical-channel delivery."""
    return AlazarCaptureSettings(
        sample_rate=units.SampleRate("100 MS/s"),
        record_length=4096,
        channel_indices=(1, 0),
        records_per_batch=2,
        batch_count=3,
        record_trigger=DigitalRecordTrigger(source="external"),
        input_voltage_range=units.VoltageRange("-1 V", "1 V"),
        input_impedance=units.Resistance("50 ohm"),
    )


@pytest.fixture
def digitizer(sdk: FakeSDK) -> Iterator[AlazarDigitizer]:
    """Supply a passive device and close it after all borrowing sessions."""
    device = AlazarDigitizer(AlazarDigitizerConfig())
    sdk.board.reset_mock()
    try:
        yield device
    finally:
        device.close()


@pytest.fixture
def capture(
    digitizer: AlazarDigitizer,
    settings: AlazarCaptureSettings,
) -> Iterator[DigitizerCaptureSession]:
    """Ensure every borrowing session closes before its owning device."""
    session = digitizer.create_capture(settings)
    try:
        yield session
    finally:
        session.close()


def test_construction_does_not_load_or_query_sdk(sdk: FakeSDK) -> None:
    """Keep construction passive even though first SDK loading may reset boards."""
    device = AlazarDigitizer(AlazarDigitizerConfig(system_id=2, board_id=3))
    try:
        assert device.config.system_id == 2
        assert device.config.board_id == 3
        sdk.factory.assert_not_called()
        sdk.load.assert_not_called()
        assert sdk.board.mock_calls == []

        with pytest.raises(RuntimeError, match="before board discovery"):
            _ = device.channel_count
        with pytest.raises(RuntimeError, match="before board discovery"):
            _ = device.data_range

        snapshot = device.snapshot()
        assert snapshot["settings"]["channel_count"] is None
        assert snapshot["settings"]["bit_depth"] is None
    finally:
        device.close()


def test_create_capture_is_passive(
    capture: DigitizerCaptureSession,
    sdk: FakeSDK,
    digitizer: AlazarDigitizer,
) -> None:
    """Construct a NEW session without SDK calls or an input reservation."""
    assert capture.state is CaptureState.NEW
    sdk.factory.assert_not_called()
    sdk.load.assert_not_called()
    assert sdk.board.mock_calls == []
    digitizer.close()


@pytest.mark.parametrize("edge", [SignalEdge.RISING, SignalEdge.FALLING])
def test_prepare_applies_requested_settings_without_starting(
    digitizer: AlazarDigitizer,
    settings: AlazarCaptureSettings,
    sdk: FakeSDK,
    edge: SignalEdge,
) -> None:
    """Configure selected inputs, clock, trigger edge, and record geometry.

    Both trigger engines are explicitly configured and automatic trigger
    timeout is disabled. Preparation reserves the logical device and posts
    buffers without arming.
    """
    session = digitizer.create_capture(
        replace(
            settings,
            record_trigger=DigitalRecordTrigger(source="external", active_edge=edge),
        )
    )
    try:
        session.prepare()
        sdk.factory.assert_called_once_with(1, 1)
        assert digitizer.channel_count == 2
        assert digitizer.bit_depth == 14
        assert digitizer.data_range == units.IntRange(-8192, 8191)
        assert session.state is CaptureState.PREPARED
        assert session.layout.shape == (2, 4096, 2)
        assert session.layout.channel_indices == (1, 0)
        assert session.layout.dtype == np.dtype(np.int16)

        sdk.board.set_capture_clock.assert_called_once_with(
            sdk.ats.ClockSources.INTERNAL_CLOCK,
            sdk.board.bsi.sample_rates[0],
            sdk.ats.ClockEdges.CLOCK_EDGE_RISING,
        )
        impedance = sdk.ats.Impedances.IMPEDANCE_50_OHM
        assert sdk.board.input_control_ex.call_args_list == [
            call(
                channel=1 << index,
                coupling=sdk.ats.Couplings.DC_COUPLING,
                input_range=sdk.board.bsi.input_ranges(impedance)[0],
                impedance=impedance,
            )
            for index in (1, 0)
        ]
        sdk.board.set_external_trigger.assert_called_once_with(
            coupling=sdk.ats.Couplings.DC_COUPLING,
            range=sdk.ats.ExternalTriggerRanges.ETR_TTL,
        )

        trigger_call = sdk.board.set_trigger_operation.call_args
        assert trigger_call is not None
        trigger_args = trigger_call.kwargs
        assert trigger_args["operation"] is sdk.ats.TriggerOperations.TRIG_ENGINE_OP_J
        assert trigger_args["source1"] is sdk.ats.TriggerSources.TRIG_EXTERNAL
        assert trigger_args["source2"] is sdk.ats.TriggerSources.TRIG_DISABLE
        expected_slope = (
            sdk.ats.TriggerSlopes.TRIGGER_SLOPE_POSITIVE
            if edge is SignalEdge.RISING
            else sdk.ats.TriggerSlopes.TRIGGER_SLOPE_NEGATIVE
        )
        assert trigger_args["slope1"] is expected_slope

        sdk.board.set_trigger_time_out.assert_called_once_with(0)
        sdk.board.set_trigger_delay.assert_called_once_with(0)
        sdk.board.set_record_size.assert_called_once_with(
            pre_trigger_samples=0,
            post_trigger_samples=4096,
        )
        sdk.board.start_capture.assert_not_called()
        sdk.board.before_async_read.assert_called_once()

        with pytest.raises(RuntimeError, match="reserved"):
            digitizer.close()
    finally:
        session.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"record_length": 128},
        {"record_length": 4097},
        {"sample_rate": units.SampleRate("101 MS/s")},
        {"input_coupling": Coupling.AC},
        {"input_impedance": units.Resistance("1 Mohm")},
        {"input_voltage_range": units.VoltageRange("-2 V", "2 V")},
    ],
)
def test_board_constraints_are_checked_before_configuration(
    digitizer: AlazarDigitizer,
    settings: AlazarCaptureSettings,
    sdk: FakeSDK,
    changes: dict[str, object],
) -> None:
    """Reject unsupported board settings before the first configuration call.

    The session closes and releases its reservation on failure, allowing a
    subsequent valid session to prepare using the same logical device.
    """
    session = digitizer.create_capture(replace(settings, **changes))
    with pytest.raises(ValueError, match=r"record_length|sample_rate|input_"):
        session.prepare()

    assert session.state is CaptureState.CLOSED
    assert sdk.board.mock_calls == [
        call.get_channel_info(),
        call.get_board_kind(),
        call.query_capability(sdk.ats.Capabilities.GET_SERIAL_NUMBER),
    ]

    retry = digitizer.create_capture(settings)
    try:
        retry.prepare()
    finally:
        retry.close()


@pytest.mark.parametrize(
    "operation",
    ["set_capture_clock", "input_control_ex", "set_record_size"],
)
def test_sdk_failure_releases_reservation(
    digitizer: AlazarDigitizer,
    settings: AlazarCaptureSettings,
    sdk: FakeSDK,
    operation: str,
) -> None:
    """Preserve a configuration error and permit a fresh preparation afterward."""
    failure = RuntimeError("SDK configuration failed")
    method = getattr(sdk.board, operation)
    method.side_effect = failure
    session = digitizer.create_capture(settings)

    with pytest.raises(RuntimeError, match="SDK configuration failed") as caught:
        session.prepare()

    assert caught.value is failure
    assert session.state is CaptureState.CLOSED
    sdk.board.abort_async_read.assert_not_called()

    method.side_effect = None
    retry = digitizer.create_capture(settings)
    try:
        retry.prepare()
    finally:
        retry.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"batch_count": None},
        {"record_trigger": None},
        {"timestamps_enabled": True},
        {"trigger_offset_samples": -128},
        {"start_trigger": DigitalStartTrigger(source="run_gate")},
        {
            "external_clock": ExternalSampleClock(
                source="clock",
                max_rate=units.SampleRate("100 MS/s"),
            )
        },
        {"external_trigger_coupling": Coupling.AC},
        {
            "external_trigger_range": ExternalTriggerRange.V5_50_OHM,
            "external_trigger_level": units.Voltage("1.5 V"),
        },
    ],
)
def test_unsupported_modes_fail_without_sdk_calls(
    digitizer: AlazarDigitizer,
    settings: AlazarCaptureSettings,
    sdk: FakeSDK,
    changes: dict[str, object],
) -> None:
    """Reject unsupported requests during pure session creation."""
    with pytest.raises(NotImplementedError, match="currently prepares"):
        digitizer.create_capture(replace(settings, **changes))

    assert sdk.board.mock_calls == []
    sdk.factory.assert_not_called()
    sdk.load.assert_not_called()


def test_identity_and_snapshot_use_cached_discovery(
    digitizer: AlazarDigitizer,
    sdk: FakeSDK,
    capture: DigitizerCaptureSession,
) -> None:
    """Report physical identity and geometry without repeated board queries."""
    capture.prepare()
    sdk.board.reset_mock()

    snapshot = digitizer.snapshot()
    assert snapshot["identity"] == {
        "manufacturer": "AlazarTech",
        "model": "ATS9440",
        "serial_number": "123456",
    }
    assert snapshot["settings"]["bit_depth"] == 14
    assert snapshot["driver"]["distribution"] == "dirigo-alazar"
    assert sdk.board.mock_calls == []


def test_readout_remains_explicitly_unavailable(
    capture: DigitizerCaptureSession,
    sdk: FakeSDK,
) -> None:
    """Reject reads until batch delivery is implemented, aborting the armed run."""
    capture.prepare()
    capture.start()
    samples = np.empty(capture.layout.shape, dtype=capture.layout.dtype)

    with pytest.raises(NotImplementedError, match="DMA readout"):
        capture.read_into(samples)

    assert capture.state is CaptureState.CLOSED
    sdk.board.start_capture.assert_called_once()
    sdk.board.abort_async_read.assert_called_once()


def test_wrong_config_is_rejected_before_connecting(sdk: FakeSDK) -> None:
    """Require the declared initialization model before touching the SDK."""
    with pytest.raises(TypeError, match="AlazarDigitizerConfig"):
        AlazarDigitizer(cast(AlazarDigitizerConfig, object()))

    sdk.factory.assert_not_called()


def test_capture_requires_backend_settings(
    digitizer: AlazarDigitizer,
    sdk: FakeSDK,
) -> None:
    """Reject generic requests missing the Alazar analog-input settings."""
    with pytest.raises(TypeError, match="AlazarCaptureSettings"):
        digitizer.create_capture(
            CaptureSettings(
                sample_rate=units.SampleRate("100 MS/s"),
                record_length=4096,
            )
        )

    assert sdk.board.mock_calls == []


def test_channel_bounds_are_deferred_then_cached(
    digitizer: AlazarDigitizer,
    settings: AlazarCaptureSettings,
    sdk: FakeSDK,
) -> None:
    """Defer unknown bounds to discovery and reject known bounds during creation.

    The invalid physical input is discovered before any configuration call.
    Discovery remains cached even though that session fails preparation.
    """
    invalid = replace(settings, channel_indices=(2,))
    session = digitizer.create_capture(invalid)
    sdk.load.assert_not_called()

    with pytest.raises(ValueError, match="available inputs"):
        session.prepare()

    assert session.state is CaptureState.CLOSED
    sdk.board.set_capture_clock.assert_not_called()

    sdk.board.reset_mock()
    with pytest.raises(ValueError, match="available inputs"):
        digitizer.create_capture(invalid)

    assert sdk.board.mock_calls == []
    sdk.factory.assert_called_once_with(1, 1)


def test_prepare_posts_every_buffer_before_start(
    capture: DigitizerCaptureSession,
    sdk: FakeSDK,
) -> None:
    """Allocate before configuring DMA, then post without arming.

    Three finite batches cap the requested four-buffer pool at three buffers.
    The run requests six records. Closing an unstarted session must still
    abort its posted DMA buffers before releasing their allocations.
    """
    capture.prepare()
    assert sdk.events == ["allocate"] * 3 + ["before"] + ["post"] * 3
    assert sdk.buffer_factory.call_count == 3
    sdk.board.before_async_read.assert_called_once_with(
        channels=3,
        transfer_offset=0,
        transfer_length=4096,
        records_per_buffer=2,
        records_per_acquisition=6,
        flags=sdk.ats.ADMAModes.ADMA_NPT | sdk.ats.ADMAFlags.ADMA_EXTERNAL_STARTCAPTURE,
    )
    sdk.board.start_capture.assert_not_called()

    capture.close()
    assert sdk.events[-4:] == ["abort", "free", "free", "free"]


def test_stop_retains_buffers_and_reservation_until_close(
    capture: DigitizerCaptureSession,
    digitizer: AlazarDigitizer,
    sdk: FakeSDK,
) -> None:
    """Abort on stop, retain ownership, and free once when the session closes."""
    capture.prepare()
    capture.start()
    assert capture.state is CaptureState.STARTED
    assert sdk.events[-1] == "start"

    capture.stop()
    assert capture.state is CaptureState.STOPPED
    sdk.board.abort_async_read.assert_called_once()
    assert all(buffer.close.call_count == 0 for buffer in sdk.buffers)

    with pytest.raises(RuntimeError, match="reserved"):
        digitizer.close()

    capture.close()
    capture.close()
    sdk.board.abort_async_read.assert_called_once()
    assert all(buffer.close.call_count == 1 for buffer in sdk.buffers)


@pytest.mark.parametrize("stage", ["allocate", "before", "post", "start"])
def test_dma_failures_cleanup_partial_preparation_and_start(
    capture: DigitizerCaptureSession,
    digitizer: AlazarDigitizer,
    settings: AlazarCaptureSettings,
    sdk: FakeSDK,
    stage: str,
) -> None:
    """Preserve the operation error, clean owned resources, and permit reuse.

    Allocation fails on the second buffer. Posting fails on the second post.
    A before_async_read failure still requires abort because native setup
    may have partially succeeded. Allocation failure precedes DMA setup.
    """
    failure = RuntimeError("DMA operation failed")
    methods = {
        "allocate": sdk.buffer_factory,
        "before": sdk.board.before_async_read,
        "post": sdk.board.post_async_buffer,
        "start": sdk.board.start_capture,
    }
    method = methods[stage]
    original = method.side_effect
    assert callable(original)
    attempts = 0

    def fail(*args: object, **kwargs: object) -> object:
        """Fail after the selected operation's requested partial progress."""
        nonlocal attempts
        attempts += 1
        if attempts == (2 if stage in {"allocate", "post"} else 1):
            raise failure
        return original(*args, **kwargs)

    if stage == "start":
        capture.prepare()

    method.side_effect = fail
    operation = capture.start if stage == "start" else capture.prepare
    with pytest.raises(RuntimeError, match="DMA operation failed") as caught:
        operation()

    assert caught.value is failure
    assert capture.state is CaptureState.CLOSED
    assert all(buffer.close.call_count == 1 for buffer in sdk.buffers)

    if stage == "allocate":
        sdk.board.abort_async_read.assert_not_called()
    else:
        sdk.board.abort_async_read.assert_called_once()
        assert sdk.events.index("abort") < sdk.events.index("free")

    method.side_effect = original
    retry = digitizer.create_capture(settings)
    try:
        retry.prepare()
        retry.start()
    finally:
        retry.close()

    sdk.load.assert_called_once_with("atsbindings")
    sdk.factory.assert_called_once_with(1, 1)


def test_abort_failure_retains_memory_and_blocks_device_reuse(
    capture: DigitizerCaptureSession,
    digitizer: AlazarDigitizer,
    settings: AlazarCaptureSettings,
    sdk: FakeSDK,
) -> None:
    """Keep buffers pinned when abort cannot establish that DMA has stopped.

    Core closure releases the logical reservation despite cleanup errors.
    The device retains the failed pool and rejects new captures. Device
    close retries abort before freeing its retained allocations.
    """
    capture.prepare()
    capture.start()
    sdk.board.abort_async_read.side_effect = RuntimeError("abort failed")

    try:
        with pytest.raises(RuntimeError, match="abort failed"):
            capture.close()

        assert capture.state is CaptureState.CLOSED
        assert all(buffer.close.call_count == 0 for buffer in sdk.buffers)
        with pytest.raises(RuntimeError, match="Previous DMA cleanup failed"):
            digitizer.create_capture(settings)
    finally:
        sdk.board.abort_async_read.side_effect = None
        digitizer.close()

    assert all(buffer.close.call_count == 1 for buffer in sdk.buffers)


def test_buffer_release_failure_does_not_skip_remaining_releases(
    capture: DigitizerCaptureSession,
    digitizer: AlazarDigitizer,
    settings: AlazarCaptureSettings,
    sdk: FakeSDK,
) -> None:
    """Attempt every release, retain the failed allocation, and retry only it."""
    capture.prepare()
    failed = sdk.buffers[1]
    failed.close.side_effect = RuntimeError("free failed")

    try:
        with pytest.raises(RuntimeError, match="free failed"):
            capture.close()

        assert all(buffer.close.call_count == 1 for buffer in sdk.buffers)
        with pytest.raises(RuntimeError, match="Previous DMA cleanup failed"):
            digitizer.create_capture(settings)
    finally:
        failed.close.side_effect = None
        digitizer.close()

    assert [buffer.close.call_count for buffer in sdk.buffers] == [1, 2, 1]
    sdk.board.abort_async_read.assert_called_once()
