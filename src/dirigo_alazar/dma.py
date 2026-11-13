"""Own native DMA buffers until transfers stop and cleanup succeeds."""

from time import monotonic
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
from dirigo.hardware.capture import SampleArray

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

        self._completed = 0
        self._posted = 0
        self._sample_bits: int | None = None
        physical_order = sorted(settings.channel_indices)
        self._channel_order = tuple(
            physical_order.index(index) for index in settings.channel_indices
        )

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
        if not isinstance(getattr(self._sdk, "AlazarError", None), type):
            raise RuntimeError("Install atsbindings with AlazarError support.")

        if not self._board.query_capability(ats.Capabilities.CAP_SUPPORTS_NPT_AUTODMA):
            raise NotImplementedError("This board does not support NPT AutoDMA.")

        mask = sum(1 << index for index in settings.channel_indices)
        if mask not in self._board.bsi.channel_configs:
            raise ValueError("This physical-channel combination is not supported.")

        memory, bits = self._board.get_channel_info()
        self._sample_bits = bits
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
            self._posted += 1

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

    def read_into(self, destination: SampleArray, *, timeout: float | None) -> None:
        """Copy one completed NPT batch, then repost if more work remains.

        The session validates destination and timeout before this hook runs.
        Native storage is channel-major, with physical channels in ascending
        order. Copy into caller storage before allowing DMA to reuse a buffer.
        """
        settings = self._settings
        buffer = self.buffers[self._completed % len(self.buffers)]
        self._wait_for_buffer(buffer, timeout=timeout)

        bits = self._sample_bits
        assert bits is not None
        unsigned = np.dtype(np.uint8 if bits <= 8 else np.uint16)
        channels = len(settings.channel_indices)
        records = settings.records_per_batch
        length = settings.record_length

        raw = buffer.buffer
        if (
            not isinstance(raw, np.ndarray)
            or raw.dtype != unsigned
            or raw.size != channels * records * length
            or not raw.flags.c_contiguous
        ):
            raise RuntimeError("Unexpected Alazar DMA sample layout.")

        native = raw.reshape(channels, records, length)
        storage_bits = unsigned.itemsize * 8
        sign_bit = 1 << (storage_bits - 1)

        for target, source in enumerate(self._channel_order):
            np.bitwise_xor(
                native[source],
                sign_bit,
                out=destination[:, :, target].view(unsigned),
            )

        shift = storage_bits - bits
        if shift:
            np.right_shift(destination, shift, out=destination)

        count = settings.batch_count
        assert count is not None
        if self._posted < count:
            self._board.post_async_buffer(buffer.address, buffer.size)
            self._posted += 1

        self._completed += 1

    def _wait_for_buffer(self, buffer: Buffer, *, timeout: float | None) -> None:
        """Wait in bounded SDK calls, preserving queue position on timeout.

        SDK waits use milliseconds, rounded down; a submillisecond remainder
        polls. None retries bounded waits indefinitely without relying on a
        special native infinity value. Only ApiWaitTimeout (579) is retryable.
        """
        deadline = None if timeout is None else monotonic() + timeout

        while True:
            remaining = (
                1.0 if deadline is None else min(1.0, max(0.0, deadline - monotonic()))
            )
            milliseconds = int(remaining * 1000)

            try:
                self._board.wait_async_buffer_complete(
                    buffer.address,
                    timeout_ms=milliseconds,
                )
            except self._sdk.AlazarError as exc:
                if exc.code != 579:
                    raise
                if deadline is not None and (
                    milliseconds == 0 or monotonic() >= deadline
                ):
                    raise TimeoutError(
                        "Timed out waiting for an Alazar DMA buffer."
                    ) from exc
            else:
                return


_retained_pools: set[_DMAPool] = set()
