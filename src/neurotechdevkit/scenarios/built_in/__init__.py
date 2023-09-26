"""Contain built-in scenarios for NDK."""
from ._scenario_simple import ScenarioSimple
from ._scenario_flat_skull import ScenarioFlatSkull_2D, ScenarioFlatSkull_3D
from ._scenario_realistic_skull import (
    ScenarioRealisticSkull_2D,
    ScenarioRealisticSkull_2D_Benchmark7,
    ScenarioRealisticSkull_3D,
    ScenarioRealisticSkull_3D_Benchmark7,
)
from ._scenario_ultrasound_phantom import ScenarioUltrasoundPhantom

__all__ = [
    "ScenarioSimple",
    "ScenarioFlatSkull_2D",
    "ScenarioFlatSkull_3D",
    "ScenarioRealisticSkull_2D",
    "ScenarioRealisticSkull_3D",
    "ScenarioRealisticSkull_2D_Benchmark7",
    "ScenarioRealisticSkull_3D_Benchmark7",
    "ScenarioUltrasoundPhantom",
]

BUILT_IN_SCENARIOS = {
    "ScenarioSimple": ScenarioSimple,
    "ScenarioFlatSkull_2D": ScenarioFlatSkull_2D,
    "ScenarioFlatSkull_3D": ScenarioFlatSkull_3D,
    "ScenarioRealisticSkull_2D": ScenarioRealisticSkull_2D,
    "ScenarioRealisticSkull_3D": ScenarioRealisticSkull_3D,
    "ScenarioRealisticSkull_2D_Benchmark7": ScenarioRealisticSkull_2D_Benchmark7,
    "ScenarioRealisticSkull_3D_Benchmark7": ScenarioRealisticSkull_3D_Benchmark7,
    "ScenarioUltrasoundPhantom": ScenarioUltrasoundPhantom,
}
