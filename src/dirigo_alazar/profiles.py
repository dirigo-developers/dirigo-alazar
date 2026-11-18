"""Read common digitizer settings and Alazar-specific profile options."""

import tomllib
from os import PathLike
from pathlib import Path

from dirigo.hardware.profiles import parse_digitizer_profile

from dirigo_alazar.settings import (
    AlazarDigitizerProfile,
    AuxOutputMode,
)


def load_alazar_digitizer_profile(
    path: str | PathLike[str],
) -> AlazarDigitizerProfile:
    """Read an Alazar profile without importing or accessing the native SDK.

    Common settings use the core profile format. The optional [driver]
    section accepts dma_buffer_count and aux_output_mode.
    """
    with Path(path).open("rb") as stream:
        values = tomllib.load(stream)

    options = values.pop("driver", {})

    if not isinstance(options, dict):
        raise TypeError("The profile driver section must be a table.")

    unknown = options.keys() - {
        "dma_buffer_count",
        "aux_output_mode",
    }
    if unknown:
        raise ValueError(
            f"Unknown Alazar profile options: {', '.join(sorted(unknown))}."
        )

    buffer_count = options.get("dma_buffer_count", 4)
    if type(buffer_count) is not int:
        raise TypeError("dma_buffer_count must be an integer.")

    raw_mode = options.get("aux_output_mode")
    mode: AuxOutputMode | None = None

    if raw_mode is not None:
        if not isinstance(raw_mode, str):
            raise TypeError("aux_output_mode must be a string.")

        mode = AuxOutputMode(raw_mode.strip().casefold())

    common = parse_digitizer_profile(values)

    return AlazarDigitizerProfile(
        sample_clock=common.sample_clock,
        channels=common.channels,
        trigger=common.trigger,
        dma_buffer_count=buffer_count,
        aux_output_mode=mode,
    )
