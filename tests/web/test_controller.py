from web import controller

from neurotechdevkit.scenarios.built_in import BUILT_IN_SCENARIOS


def test_get_builtin_scenarios():
    """Test that the built-in scenarios can be parsed into IndexBuiltInScenario."""
    built_in_scenarios = controller.get_built_in_scenarios()
    assert len(built_in_scenarios) == len(BUILT_IN_SCENARIOS)
