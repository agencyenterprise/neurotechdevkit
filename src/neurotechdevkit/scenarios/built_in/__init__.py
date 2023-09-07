"""Contain built-in scenarios for NDK."""
from ._scenario_0 import Scenario0
from ._scenario_1 import Scenario1_2D, Scenario1_3D
from ._scenario_2 import Scenario2_2D, Scenario2_3D
from ._scenario_3 import Scenario3

__all__ = [
    "Scenario0",
    "Scenario1_2D",
    "Scenario1_3D",
    "Scenario2_2D",
    "Scenario2_3D",
    "Scenario3",
]

BUILT_IN_SCENARIOS = {
    "Scenario0": Scenario0,
    "Scenario1_2D": Scenario1_2D,
    "Scenario1_3D": Scenario1_3D,
    "Scenario2_2D": Scenario2_2D,
    "Scenario2_3D": Scenario2_3D,
    "Scenario3": Scenario3,
}
