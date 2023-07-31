"""Main package for the neurotechdevkit."""
from __future__ import annotations

import os
from enum import Enum

from . import materials, scenarios, sources
from .results import load_result_from_disk

__all__ = [
    "results",
    "scenarios",
    "materials",
    "sources",
    "load_result_from_disk",
    "BUILTIN_SCENARIOS",
]

if "DEVITO_ARCH" not in os.environ:
    print(
        "WARNING: DEVITO_ARCH environment variable not set "
        "and might cause compilation errors. See NDK documentation for help."
    )


class BUILTIN_SCENARIOS(Enum):
    """Enum of built-in scenarios."""

    SCENARIO_0 = scenarios.Scenario0
    SCENARIO_1_2D = scenarios.Scenario1_2D
    SCENARIO_1_3D = scenarios.Scenario1_3D
    SCENARIO_2_2D = scenarios.Scenario2_2D
    SCENARIO_2_3D = scenarios.Scenario2_3D
