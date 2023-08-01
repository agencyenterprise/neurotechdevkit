import pytest

import neurotechdevkit as ndk


@pytest.mark.integration
def test_compilation():
    """Will run a simulation requiring compilation."""
    scenario = ndk.BUILTIN_SCENARIOS.SCENARIO_0.value()
    scenario.make_grid(center_frequency=5e5)
    scenario.compile_problem(center_frequency=5e5)
    result = scenario.simulate_steady_state()
    assert result.wavefield.shape == (101, 81, 59)
