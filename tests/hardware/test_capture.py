"""Smoke-test finite external-trigger capture on configured hardware."""

import math
import tomllib
from contextlib import ExitStack
from pathlib import Path
from time import monotonic

import numpy as np
import pytest
from dirigo import load_system, units
from dirigo.hardware.capture_session import CaptureState
from dirigo.hardware.timing import DigitalRecordTrigger, SignalEdge

from dirigo_alazar.digitizer import AlazarDigitizer
from dirigo_alazar.settings import AlazarCaptureSettings, Coupling


@pytest.mark.hardware
def test_finite_external_trigger_capture() -> None:
    """Capture finite batches and reprepare the same device through the real SDK.

    Read connection and run settings from hardware_local.toml. The operator
    supplies an external TTL trigger source; the test does not generate it.
    Check delivered geometry, signed integer storage, batch count, finite
    completion, and resource reuse without evaluating measured signal values.

    Apply one waiting budget across the capture. Close sessions before Hardware
    on success or failure. Reprepare a second session without starting it to
    verify that the first session released its native resources and reservation.
    """
    config_path = Path(__file__).with_name("hardware_local.toml")
    if not config_path.is_file():
        pytest.skip(
            "Copy hardware_example.toml to hardware_local.toml and configure it."
        )

    with config_path.open("rb") as stream:
        document = tomllib.load(stream)
    parameters = document.get("test", {}).get("capture")
    if parameters is None:
        pytest.skip("Configure [test.capture] in hardware_local.toml.")

    timeout = float(units.Time(parameters["timeout"]))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("The capture test timeout must be finite and positive.")

    voltage_range = parameters["input_voltage_range"]
    request = AlazarCaptureSettings(
        sample_rate=units.SampleRate(parameters["sample_rate"]),
        record_length=parameters["record_length"],
        records_per_batch=parameters["records_per_batch"],
        batch_count=parameters["batch_count"],
        channel_indices=tuple(parameters["channel_indices"]),
        record_trigger=DigitalRecordTrigger(
            source="external",
            active_edge=SignalEdge(parameters["trigger_edge"]),
        ),
        input_voltage_range=units.VoltageRange(
            voltage_range["min"],
            voltage_range["max"],
        ),
        input_impedance=units.Resistance(parameters["input_impedance"]),
        input_coupling=Coupling(parameters["input_coupling"]),
        dma_buffer_count=parameters["dma_buffer_count"],
    )
    batch_count = request.batch_count
    if batch_count is None:
        raise ValueError("The hardware test requires finite batch_count.")

    with ExitStack() as cleanup:
        system = load_system(config_path)
        cleanup.callback(system.close)
        digitizer = system.require_device(parameters["digitizer"], AlazarDigitizer)

        capture = digitizer.create_capture(request)
        cleanup.callback(capture.close)
        capture.prepare()

        layout = capture.layout
        expected_shape = (
            request.records_per_batch,
            request.record_length,
            len(request.channel_indices),
        )
        assert layout.shape == expected_shape
        assert layout.channel_indices == request.channel_indices
        assert layout.dtype.kind == "i"
        assert layout.dtype.isnative
        samples = np.empty(layout.shape, dtype=layout.dtype)

        deadline = monotonic() + timeout
        capture.start()

        for _ in range(batch_count):
            remaining = deadline - monotonic()
            assert remaining > 0, "Capture exceeded its configured waiting budget."

            try:
                info = capture.read_into(samples, timeout=remaining)
            except TimeoutError as exc:
                exc.add_note("Check that the external TTL trigger source is running.")
                raise

            assert info is not None, (
                "Capture ended before all requested batches arrived."
            )
            assert info.timestamps is None
            assert samples.shape == expected_shape
            assert samples.dtype == layout.dtype

        assert capture.batches_acquired == batch_count
        assert capture.state is CaptureState.COMPLETED
        assert capture.read_into(samples, timeout=0) is None

        capture.close()
        assert capture.state is CaptureState.CLOSED
        assert not digitizer.is_active

        retry = digitizer.create_capture(request)
        cleanup.callback(retry.close)
        retry.prepare()
        assert retry.state is CaptureState.PREPARED
        assert retry.layout == layout
