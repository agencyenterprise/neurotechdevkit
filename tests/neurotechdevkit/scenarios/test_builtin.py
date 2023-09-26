"""Test built-in scenarios."""

import pytest

from neurotechdevkit.scenarios.built_in import (
    ScenarioSimple,
    ScenarioFlatSkull_2D,
    ScenarioFlatSkull_3D,
    ScenarioRealisticSkull_2D,
    ScenarioRealisticSkullCropped_2D,
    ScenarioRealisticSkull_3D,
    ScenarioRealisticSkullCropped_3D,
    ScenarioUltrasoundPhantom,
)


@pytest.fixture(
    params=[
        ScenarioSimple,
        ScenarioFlatSkull_2D,
        ScenarioFlatSkull_3D,
        ScenarioRealisticSkull_2D,
        ScenarioRealisticSkull_3D,
        ScenarioRealisticSkullCropped_2D,
        ScenarioRealisticSkullCropped_3D,
        ScenarioUltrasoundPhantom,
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
