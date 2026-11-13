"""AlazarTech hardware support for Dirigo."""

from dirigo_alazar.digitizer import AlazarDigitizer
from dirigo_alazar.settings import (
    AlazarCaptureSettings,
    AlazarDigitizerConfig,
    AuxOutputMode,
    Coupling,
    ExternalTriggerRange,
)

__all__ = [
    "AlazarCaptureSettings",
    "AlazarDigitizer",
    "AlazarDigitizerConfig",
    "AuxOutputMode",
    "Coupling",
    "ExternalTriggerRange",
]
