"""Main package for the neurotechdevkit."""
from __future__ import annotations

import os

from . import materials, scenarios, sources
from .results import load_result_from_disk
from .scenarios import built_in

__all__ = [
    "results",
    "scenarios",
    "materials",
    "sources",
    "built_in",
    "load_result_from_disk",
]

if "DEVITO_ARCH" not in os.environ:
    print(
        "WARNING: DEVITO_ARCH environment variable not set "
        "and might cause compilation errors. See NDK documentation for help."
    )
