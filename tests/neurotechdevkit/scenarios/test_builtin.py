"""Test built-in scenario 2."""

import pytest

from neurotechdevkit.scenarios import (
    Scenario0,
    Scenario1_2D,
    Scenario1_3D,
    Scenario2_2D,
    Scenario2_3D,
)


@pytest.fixture(
    params=[
        Scenario0,
        Scenario1_2D,
        Scenario1_3D,
        Scenario2_2D,
        Scenario2_3D,
    ]
)
def scenario_cls(request):
    """Iterate over all scenarios"""
    return request.param


@pytest.fixture(params=[1e5, 5e5])
def center_frequency(request):
    """Iterate over different center frequencies"""
    return request.param


def test_compile_problem(scenario_cls, center_frequency):
    """Test compiling each built-in scenario's default problem."""
    scenario = scenario_cls()
    scenario.center_frequency = center_frequency
    scenario.make_grid()
    problem = scenario.compile_problem()
    assert problem == scenario.problem
