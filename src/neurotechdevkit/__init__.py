from __future__ import annotations

from . import scenarios
from .scenarios import load_result_from_disk

__all__ = [
    "scenarios",
    "make",
    "ScenarioNotFoundError",
    "load_result_from_disk",
]


class ScenarioNotFoundError(Exception):
    pass


def make(scenario_id, complexity="fast"):
    """
    Initialize a scenario. This will return an object which represents
    the simulation.

    Args:
        scenario_id (str): the id of the scenario to load. Supported
            scenarios are:

            - [Scenario 0][neurotechdevkit.scenarios.Scenario0._SCENARIO_ID]
            - [Scenario 1 2D][neurotechdevkit.scenarios.Scenario1_2D._SCENARIO_ID]
            - [Scenario 1 3D][neurotechdevkit.scenarios.Scenario1_3D._SCENARIO_ID]
            - [Scenario 2 2D][neurotechdevkit.scenarios.Scenario2_2D._SCENARIO_ID]
            - [Scenario 2 3D][neurotechdevkit.scenarios.Scenario2_3D._SCENARIO_ID]

        complexity (str, optional): allow the user to choose from a few
            pre-selected options for parameters such as PPP and PPW that
            determine the compute, memory, and time requirements of a
            simulation as well as the accuracy of the simulation results.
            Defaults to "fast".

    Raises:
        ScenarioNotFoundError: Raised when the scenario id is not found.

    Returns:
        scenarios.Scenario: an object representing the simulation.
    """

    if scenario_id not in _scenario_map:
        raise ScenarioNotFoundError(
            f"Scenario '{scenario_id}' does not exist. Please refer to documentation"
            " for the list of provided scenarios."
        )
    return _scenario_map[scenario_id](complexity=complexity)


_scenario_map = {
    scenarios.Scenario0._SCENARIO_ID: scenarios.Scenario0,
    scenarios.Scenario1_2D._SCENARIO_ID: scenarios.Scenario1_2D,
    scenarios.Scenario1_3D._SCENARIO_ID: scenarios.Scenario1_3D,
    scenarios.Scenario2_2D._SCENARIO_ID: scenarios.Scenario2_2D,
    scenarios.Scenario2_3D._SCENARIO_ID: scenarios.Scenario2_3D,
}
