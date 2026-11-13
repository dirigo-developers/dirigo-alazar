"""Describe Alazar connections and capture requests without loading the SDK."""

import math
from dataclasses import dataclass, field
from enum import StrEnum

from dirigo import units
from dirigo.hardware.capture import CaptureSettings


class Coupling(StrEnum):
    """Select AC or DC coupling for an input supporting the requested mode."""

    AC = "AC"
    DC = "DC"


class ExternalTriggerRange(StrEnum):
    """Select the external-trigger voltage range and termination together.

    Voltage ranges are bipolar. Our TTL request uses the board's TTL mode
    without a user-selected voltage threshold. Availability is board-specific.
    """

    TTL = "TTL"
    V1_50_OHM = "1 V, 50 ohm"
    V2_5_50_OHM = "2.5 V, 50 ohm"
    V5_50_OHM = "5 V, 50 ohm"
    V5_300_OHM = "5 V, 300 ohm"


_TRIGGER_FULL_SCALE = {
    ExternalTriggerRange.V1_50_OHM: 1.0,
    ExternalTriggerRange.V2_5_50_OHM: 2.5,
    ExternalTriggerRange.V5_50_OHM: 5.0,
    ExternalTriggerRange.V5_300_OHM: 5.0,
}


@dataclass(frozen=True, slots=True, kw_only=True)
class AlazarDigitizerConfig:
    """Identify one board using one-based system and board IDs.

    Construction validates identifiers without loading the SDK or accessing
    hardware. IDs select a connection; they do not assert its serial number.
    Configured manufacturer, model, and serial number remain DeviceIdentity.
    """

    system_id: int = field(
        default=1,
        metadata={"description": "One-based Alazar system ID."},
    )
    board_id: int = field(
        default=1,
        metadata={"description": "One-based board ID within the system."},
    )

    def __post_init__(self) -> None:
        """Reject noninteger or nonpositive connection identifiers."""
        for name in ("system_id", "board_id"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")


@dataclass(frozen=True, slots=True, kw_only=True)
class AlazarCaptureSettings(CaptureSettings):
    """Add Alazar analog-input, external-trigger, and DMA settings to a run.

    All selected acquisition channels share input_voltage_range,
    input_impedance, and input_coupling. Input voltage ranges must be finite
    and symmetric about zero. Backend preparation validates board support.

    record_trigger identifies the trigger source and edge. When its source
    is external, external_trigger_range and external_trigger_coupling select
    that connector's electrical mode. TTL requires external_trigger_level=None;
    analog ranges require a finite voltage threshold within their bounds.
    The backend translates requested thresholds to the SDK's discrete codes.

    dma_buffer_count controls the maximum number of reusable transfer buffers, not
    the total number of batches. Generic sample timing, record windows, and
    batch geometry retain their CaptureSettings meanings. This model does
    not establish that a board supports a requested acquisition mode.

    aux_output_mode=None makes no AUX configuration call; the mode established
    by SDK initialization or a previous run remains in effect. An explicit mode
    is applied in prepare() and remains configured after close(); previous pin
    modes are not restored. Use LOW in a later request to select a static low.
    ACQUISITION_ACTIVE selects the SDK's AUX_OUT_TRIGGER signal, not an echo
    of every incoming record trigger or a source of software timestamps.
    """

    input_voltage_range: units.VoltageRange
    input_impedance: units.Resistance
    input_coupling: Coupling = Coupling.DC
    external_trigger_range: ExternalTriggerRange = ExternalTriggerRange.TTL
    external_trigger_coupling: Coupling = Coupling.DC
    external_trigger_level: units.Voltage | None = None
    dma_buffer_count: int = 4
    aux_output_mode: AuxOutputMode | None = None

    def __post_init__(self) -> None:
        """Validate inherited geometry and board-independent electrical bounds."""
        CaptureSettings.__post_init__(self)

        if self.aux_output_mode is not None and not isinstance(
            self.aux_output_mode, AuxOutputMode
        ):
            raise TypeError("aux_output_mode must be an AuxOutputMode or None.")

        if not isinstance(self.input_voltage_range, units.VoltageRange):
            raise TypeError("input_voltage_range must be a VoltageRange.")

        low = float(self.input_voltage_range.min)
        high = float(self.input_voltage_range.max)
        if (
            not math.isfinite(low)
            or not math.isfinite(high)
            or not low < 0 < high
            or low != -high
        ):
            raise ValueError("input_voltage_range must be finite and symmetric.")

        if not isinstance(self.input_impedance, units.Resistance):
            raise TypeError("input_impedance must be a Resistance.")
        if (
            not math.isfinite(float(self.input_impedance))
            or float(self.input_impedance) <= 0
        ):
            raise ValueError("input_impedance must be finite and positive.")

        for name in ("input_coupling", "external_trigger_coupling"):
            if not isinstance(getattr(self, name), Coupling):
                raise TypeError(f"{name} must be a Coupling.")

        if not isinstance(self.external_trigger_range, ExternalTriggerRange):
            raise TypeError("external_trigger_range must be an ExternalTriggerRange.")

        level = self.external_trigger_level
        if level is not None and not isinstance(level, units.Voltage):
            raise TypeError("external_trigger_level must be a Voltage or None.")

        if self.external_trigger_range is ExternalTriggerRange.TTL:
            if level is not None:
                raise ValueError("TTL requires external_trigger_level=None.")
        else:
            if level is None:
                raise ValueError(
                    "Analog trigger ranges require external_trigger_level."
                )

            full_scale = _TRIGGER_FULL_SCALE[self.external_trigger_range]
            voltage = float(level)
            if not math.isfinite(voltage) or not -full_scale <= voltage <= full_scale:
                raise ValueError("external_trigger_level must be finite and in range.")

        if not isinstance(self.dma_buffer_count, int) or isinstance(
            self.dma_buffer_count, bool
        ):
            raise TypeError("dma_buffer_count must be an integer.")
        if self.dma_buffer_count <= 0:
            raise ValueError("dma_buffer_count must be positive.")


class AuxOutputMode(StrEnum):
    """Select the software-configurable AUX output for a capture.

    ACQUISITION_ACTIVE is high while samples are acquired to onboard memory.
    LOW selects a static low output. Connector labels and support depend on
    the board. These modes do not configure the record-trigger input.
    """

    ACQUISITION_ACTIVE = "acquisition_active"
    LOW = "low"
