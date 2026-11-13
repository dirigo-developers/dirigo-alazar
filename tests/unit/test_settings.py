"""Validate Alazar request models independently of the SDK and hardware."""

from dataclasses import replace

import pytest
from dirigo import units
from dirigo.hardware.config import parse_config
from dirigo.hardware.timing import DigitalRecordTrigger

from dirigo_alazar.settings import (
    AlazarCaptureSettings,
    AlazarDigitizerConfig,
    Coupling,
    ExternalTriggerRange,
)


@pytest.fixture
def settings() -> AlazarCaptureSettings:
    """Request finite triggered records with explicit analog-input settings."""
    return AlazarCaptureSettings(
        sample_rate=units.SampleRate("100 MS/s"),
        record_length=4096,
        records_per_batch=2,
        batch_count=3,
        record_trigger=DigitalRecordTrigger(source="external"),
        input_voltage_range=units.VoltageRange("-1 V", "1 V"),
        input_impedance=units.Resistance("50 ohm"),
    )


def test_connection_config_uses_the_existing_parser() -> None:
    """Parse defaults and explicit board IDs through core's configuration path."""
    assert parse_config(AlazarDigitizerConfig, {}) == AlazarDigitizerConfig(
        system_id=1,
        board_id=1,
    )
    assert parse_config(
        AlazarDigitizerConfig,
        {"system_id": 2, "board_id": 3},
    ) == AlazarDigitizerConfig(system_id=2, board_id=3)


@pytest.mark.parametrize("name", ["system_id", "board_id"])
@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_connection_ids_require_integers(name: str, value: object) -> None:
    """Reject implicit conversions, including bool's integer subclass behavior."""
    with pytest.raises(TypeError, match=name):
        replace(AlazarDigitizerConfig(), **{name: value})


@pytest.mark.parametrize("name", ["system_id", "board_id"])
@pytest.mark.parametrize("value", [0, -1])
def test_connection_ids_are_one_based(name: str, value: int) -> None:
    """Reject zero and negative connection identifiers."""
    with pytest.raises(ValueError, match=name):
        replace(AlazarDigitizerConfig(), **{name: value})


def test_capture_defaults_and_inherited_validation(
    settings: AlazarCaptureSettings,
) -> None:
    """Retain generic record geometry and run its validation in the subclass."""
    assert settings.input_coupling is Coupling.DC
    assert settings.external_trigger_range is ExternalTriggerRange.TTL
    assert settings.external_trigger_level is None
    assert settings.dma_buffer_count == 4
    assert settings.records_per_batch == 2
    assert settings.batch_count == 3

    with pytest.raises(ValueError, match="record_length"):
        replace(settings, record_length=0)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("input_voltage_range", object()),
        ("input_impedance", 50.0),
        ("input_coupling", "DC"),
        ("external_trigger_coupling", "DC"),
        ("external_trigger_range", "TTL"),
        ("external_trigger_level", 0.5),
        ("dma_buffer_count", True),
        ("dma_buffer_count", 1.5),
    ],
)
def test_capture_fields_require_explicit_types(
    settings: AlazarCaptureSettings,
    name: str,
    value: object,
) -> None:
    """Reject untyped substitutes before an SDK adapter interprets the request."""
    with pytest.raises(TypeError, match=name):
        replace(settings, **{name: value})


@pytest.mark.parametrize(
    "bounds",
    [(0.0, 1.0), (-1.0, 2.0), (-float("inf"), 1.0)],
)
def test_input_range_requires_finite_symmetric_bounds(
    settings: AlazarCaptureSettings,
    bounds: tuple[float, float],
) -> None:
    """Reject unipolar, asymmetric, and unbounded acquisition voltage ranges."""
    with pytest.raises(ValueError, match="input_voltage_range"):
        replace(settings, input_voltage_range=units.VoltageRange(*bounds))


@pytest.mark.parametrize(
    "impedance",
    [0.0, -50.0, float("inf"), float("nan")],
)
def test_input_impedance_requires_finite_positive_resistance(
    settings: AlazarCaptureSettings,
    impedance: float,
) -> None:
    """Validate physical bounds while leaving supported values to the board."""
    with pytest.raises(ValueError, match="input_impedance"):
        replace(settings, input_impedance=units.Resistance(impedance))


def test_ttl_does_not_accept_a_requested_voltage_threshold(
    settings: AlazarCaptureSettings,
) -> None:
    """Keep TTL requests distinct from adjustable analog trigger thresholds."""
    with pytest.raises(ValueError, match="TTL"):
        replace(
            settings,
            external_trigger_level=units.Voltage("1.5 V"),
        )


@pytest.mark.parametrize(
    ("mode", "full_scale"),
    [
        (ExternalTriggerRange.V1_50_OHM, 1.0),
        (ExternalTriggerRange.V2_5_50_OHM, 2.5),
        (ExternalTriggerRange.V5_50_OHM, 5.0),
        (ExternalTriggerRange.V5_300_OHM, 5.0),
    ],
)
def test_analog_trigger_requires_an_in_range_threshold(
    settings: AlazarCaptureSettings,
    mode: ExternalTriggerRange,
    full_scale: float,
) -> None:
    """Require an explicit threshold and retain accepted voltages unchanged.

    Exercise both limits and zero for every declared analog connector mode.
    Conversion to a discrete SDK trigger code belongs to the later adapter.
    """
    with pytest.raises(ValueError, match="require external_trigger_level"):
        replace(settings, external_trigger_range=mode)

    for voltage in (-full_scale, 0.0, full_scale):
        requested = replace(
            settings,
            external_trigger_range=mode,
            external_trigger_level=units.Voltage(voltage),
        )
        assert requested.external_trigger_level == units.Voltage(voltage)

    for voltage in (-full_scale - 0.1, full_scale + 0.1, float("nan")):
        with pytest.raises(ValueError, match="finite and in range"):
            replace(
                settings,
                external_trigger_range=mode,
                external_trigger_level=units.Voltage(voltage),
            )


@pytest.mark.parametrize("count", [0, -1])
def test_dma_buffer_count_is_positive(
    settings: AlazarCaptureSettings,
    count: int,
) -> None:
    """Reject a DMA buffer pool that cannot supply a transfer buffer."""
    with pytest.raises(ValueError, match="dma_buffer_count"):
        replace(settings, dma_buffer_count=count)


def test_aux_output_defaults_to_unmanaged(settings: AlazarCaptureSettings) -> None:
    """Keep AUX configuration opt-in rather than assigning a pin mode implicitly."""
    assert settings.aux_output_mode is None


@pytest.mark.parametrize("value", ["acquisition_active", True, 0])
def test_aux_output_requires_a_typed_mode(
    settings: AlazarCaptureSettings,
    value: object,
) -> None:
    """Reject strings, booleans, and raw SDK integers as AUX mode substitutes."""
    with pytest.raises(TypeError, match="aux_output_mode"):
        replace(settings, aux_output_mode=value)
