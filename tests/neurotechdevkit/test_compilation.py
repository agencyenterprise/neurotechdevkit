import pytest

import neurotechdevkit as ndk


@pytest.mark.integration
def test_compilation():
    """Will run a simulation requiring compilation."""
    scenario = ndk.scenarios.built_in.Scenario0()
    scenario.make_grid()
    scenario.compile_problem()
    result = scenario.simulate_steady_state()
    assert result.wavefield.shape == (101, 81, 59)
