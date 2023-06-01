import pytest

import neurotechdevkit as ndk


@pytest.mark.integration
def test_compilation():
    """Will run a simulation requiring compilation."""
    scenario_id = "scenario-0-v0"
    scenario = ndk.make(scenario_id)
    result = scenario.simulate_steady_state()
    assert result.wavefield.shape == (101, 81, 59)
