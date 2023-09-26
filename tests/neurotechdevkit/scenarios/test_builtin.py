"""Test built-in scenario 2."""

import pytest

from neurotechdevkit.scenarios.built_in import (
    ScenarioSimple,
    ScenarioFlatSkull_2D,
    ScenarioFlatSkull_3D,
    Scenario2_2D,
    Scenario2_2D_Benchmark7,
    Scenario2_3D,
    Scenario2_3D_Benchmark7,
    Scenario3,
)


@pytest.fixture(
    params=[
        ScenarioSimple,
        ScenarioFlatSkull_2D,
        ScenarioFlatSkull_3D,
        Scenario2_2D,
        Scenario2_3D,
        Scenario2_2D_Benchmark7,
        Scenario2_3D_Benchmark7,
        Scenario3,
    ]
)
def scenario_cls(request):
    """Iterate over all scenarios"""
    return request.param


@pytest.fixture(params=[1e5, 5e5])
def center_frequency(request):
    """Iterate over different center frequencies"""
    return request.param


@pytest.mark.integration
def test_compile_problem(scenario_cls, center_frequency):
    """Test compiling each built-in scenario's default problem."""
    scenario = scenario_cls()
    scenario.center_frequency = center_frequency
    scenario.make_grid()
    scenario.compile_problem()
