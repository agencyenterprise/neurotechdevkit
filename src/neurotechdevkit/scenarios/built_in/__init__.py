"""Contain built-in scenarios for NDK."""
from ._scenario_simple import ScenarioSimple
from ._scenario_flat_skull import ScenarioFlatSkull_2D, ScenarioFlatSkull_3D
from ._scenario_realistic_skull import (
    Scenario2_2D,
    Scenario2_2D_Benchmark7,
    Scenario2_3D,
    Scenario2_3D_Benchmark7,
)
from ._scenario_ultrasound_phantom import Scenario3

__all__ = [
    "ScenarioSimple",
    "ScenarioFlatSkull_2D",
    "ScenarioFlatSkull_3D",
    "Scenario2_2D",
    "Scenario2_3D",
    "Scenario2_2D_Benchmark7",
    "Scenario2_3D_Benchmark7",
    "Scenario3",
]

BUILT_IN_SCENARIOS = {
    "ScenarioSimple": ScenarioSimple,
    "ScenarioFlatSkull_2D": ScenarioFlatSkull_2D,
    "ScenarioFlatSkull_3D": ScenarioFlatSkull_3D,
    "Scenario2_2D": Scenario2_2D,
    "Scenario2_3D": Scenario2_3D,
    "Scenario2_2D_Benchmark7": Scenario2_2D_Benchmark7,
    "Scenario2_3D_Benchmark7": Scenario2_3D_Benchmark7,
    "Scenario3": Scenario3,
}
