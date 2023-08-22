"""Tests for the web.messages module."""
from web.messages.requests import (
    IndexBuiltInScenario,
    RenderLayoutRequest,
    SimulateRequest,
)

from neurotechdevkit.scenarios.built_in import BUILT_IN_SCENARIOS


def test_load_render_layout_request(request_payload_example):
    """Test that the RenderLayoutRequest can be loaded from JSON."""
    settings = RenderLayoutRequest.parse_obj(request_payload_example)
    assert settings.scenarioSettings.scenarioId == "Scenario0"


def test_load_simulate_request(request_payload_example):
    """Test that the SimulateRequest can be loaded from JSON."""
    settings = SimulateRequest.parse_obj(request_payload_example)
    assert settings.scenarioSettings.scenarioId == "Scenario0"


def test_parse_builtin_scenarios():
    """Test that the built-in scenarios can be parsed into IndexBuiltInScenario."""
    for scenario_id, scenario in BUILT_IN_SCENARIOS.items():
        settings = IndexBuiltInScenario.from_scenario(scenario_id, scenario)
        assert settings.scenarioSettings.scenarioId == scenario_id
