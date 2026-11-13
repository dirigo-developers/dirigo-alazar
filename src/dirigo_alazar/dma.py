"""Own native DMA buffers until transfers stop and cleanup succeeds."""

from types import ModuleType
from typing import TYPE_CHECKING

from dirigo_alazar.settings import AlazarCaptureSettings

if TYPE_CHECKING:
    from atsbindings import Board, Buffer


class _DMAPool:
    """Own one finite NPT run with unpacked, noninterleaved sample storage.

    An internal registry keeps allocated pools alive if cleanup fails. Their
    Buffer finalizers must not free memory that may still belong to DMA.
    All methods are called serially by the borrowing capture session.
    """

    def __init__(
        self,
        sdk: ModuleType,
        board: Board,
        settings: AlazarCaptureSettings,
    ) -> None:
        """Store a connection and request without allocating native resources."""
        self._sdk = sdk
        self._board = board
        self._settings = settings
        self.buffers: list[Buffer] = []
        self._pending = False

    def prepare(self) -> None:
        """Allocate every buffer before configuring DMA and posting the pool.

        Arm acquisition only in start(). Mark DMA pending before the native
        configuration call so a partially failed call still requires abort.
        """
        settings = self._settings
        ats = self._sdk.Ats
        count = settings.batch_count
        assert count is not None

        records = settings.records_per_batch * count
        if records >= 0x7FFFFFFF:
            raise ValueError("Finite record count must be less than 0x7FFFFFFF.")

        if not callable(getattr(self._sdk.Buffer, "close", None)):
            raise RuntimeError(
                "Install atsbindings with explicit Buffer.close() support."
            )

        if not self._board.query_capability(ats.Capabilities.CAP_SUPPORTS_NPT_AUTODMA):
            raise NotImplementedError("This board does not support NPT AutoDMA.")

        mask = sum(1 << index for index in settings.channel_indices)
        if mask not in self._board.bsi.channel_configs:
            raise ValueError("This physical-channel combination is not supported.")

        memory, bits = self._board.get_channel_info()
        if memory == 0:
            raise NotImplementedError(
                "This first DMA backend requires on-board memory."
            )

        size = (
            len(settings.channel_indices)
            * settings.record_length
            * settings.records_per_batch
            * ((bits + 7) // 8)
        )
        if size >= 64 * 1024**2:
            raise ValueError("DMA buffers must be smaller than 64 MiB.")

        self._board.set_parameter(
            ats.Channels.CHANNEL_ALL,
            ats.Parameters.PACK_MODE,
            ats.PackModes.PACK_DEFAULT,
        )
        self._board.set_parameter(
            ats.Channels.CHANNEL_ALL,
            ats.Parameters.SET_DATA_FORMAT,
            0,
        )

        _retained_pools.add(self)
        for _ in range(min(settings.dma_buffer_count, count)):
            self.buffers.append(
                self._sdk.Buffer(
                    board=self._board,
                    channels=len(settings.channel_indices),
                    records_per_buffer=settings.records_per_batch,
                    samples_per_record=settings.record_length,
                    include_header=False,
                    include_footer=False,
                    interleave_samples=False,
                    data_packing=ats.PackModes.PACK_DEFAULT,
                )
            )

        self._pending = True
        self._board.before_async_read(
            channels=mask,
            transfer_offset=0,
            transfer_length=settings.record_length,
            records_per_buffer=settings.records_per_batch,
            records_per_acquisition=records,
            flags=ats.ADMAModes.ADMA_NPT | ats.ADMAFlags.ADMA_EXTERNAL_STARTCAPTURE,
        )
        for buffer in self.buffers:
            self._board.post_async_buffer(buffer.address, buffer.size)

    def start(self) -> None:
        """Arm the prepared run without waiting for a record trigger."""
        self._board.start_capture()

    def stop(self) -> None:
        """Abort outstanding transfers while retaining every allocated buffer."""
        if self._pending:
            self._board.abort_async_read()
            self._pending = False

    def close(self) -> None:
        """Abort before freeing; retain ownership if abort or any release fails."""
        self.stop()

        failure: BaseException | None = None
        remaining: list[Buffer] = []
        for buffer in reversed(self.buffers):
            try:
                buffer.close()
            except BaseException as exc:
                remaining.append(buffer)
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(f"Another DMA buffer release failed: {exc!r}")

        self.buffers = list(reversed(remaining))
        if not self.buffers:
            _retained_pools.discard(self)

        if failure is not None:
            raise failure


_retained_pools: set[_DMAPool] = set()
