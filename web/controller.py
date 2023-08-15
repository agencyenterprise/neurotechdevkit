"""Controller for the web app."""
import base64
import io
import tempfile
from typing import Dict, Tuple

from neurotechdevkit.scenarios import Scenario2D, Scenario3D
from neurotechdevkit.scenarios.built_in import BUILT_IN_SCENARIOS
from web.built_in_scenario_info import BuiltInScenarioInfo
from web.messages import RenderLayoutRequest, SimulateRequest


def get_scenarios() -> Dict[str, BuiltInScenarioInfo]:
    """
    Return a dictionary containing information about all the built-in scenarios.

    Returns:
        The dictionary of built-in scenario infos.
    """
    scenarios: Dict[str, BuiltInScenarioInfo] = {}
    for scenario_id, scenario in BUILT_IN_SCENARIOS.items():
        assert issubclass(scenario, (Scenario2D, Scenario3D))

        scenario_info = BuiltInScenarioInfo.from_scenario(scenario_id, scenario)
        scenarios[scenario_id] = scenario_info
    return scenarios


def get_scenario_layout(config: RenderLayoutRequest) -> str:
    """
    Render the layout of a scenario and return the result as a base64 png.

    Args:
        config (RenderLayoutRequest): The configuration for the scenario.

    Returns:
        The base64 encoded png image.
    """
    if config.is2d:
        if config.scenarioSettings.isPreBuilt:
            print("prebuilt scenario")
            scenario = BUILT_IN_SCENARIOS[config.scenarioSettings.scenario_id]()
        else:
            print("custom scenario")
            scenario = Scenario2D()

        _configure_scenario(scenario, config)
        scenario.make_grid()
        fig = scenario.render_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return data
    raise NotImplementedError


def get_simulation_image(config: SimulateRequest) -> Tuple[str, str]:
    """
    Simulate a scenario and return the result as a base64 gif or png.

    Args:
        config (SimulateRequest): The configuration for the scenario.

    Returns:
        Tuple[str, str]: The base64 encoded image and the image format.
    """
    if config.is2d:
        if config.scenarioSettings.isPreBuilt:
            scenario = BUILT_IN_SCENARIOS[config.scenarioSettings.scenario_id]()
        else:
            scenario = Scenario2D()

        _configure_scenario(scenario, config)
        scenario.make_grid()
        scenario.compile_problem()
        if config.simulationSettings.isSteadySimulation:
            result = scenario.simulate_steady_state()
            fig = result.render_steady_state_amplitudes(show_material_outlines=False)
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            data = base64.b64encode(buf.getbuffer()).decode("ascii")
            return data, "png"
        else:
            result = result = scenario.simulate_pulse()
            animation = result.render_pulsed_simulation_animation()
            with tempfile.NamedTemporaryFile(suffix=".gif") as tmpfile:
                animation.save(tmpfile.name, writer="imagemagick", fps=30)
                tmpfile.seek(0)
                buf = io.BytesIO(tmpfile.read())
                data = base64.b64encode(buf.getbuffer()).decode("ascii")
                return data, "gif"
    raise NotImplementedError


def _configure_scenario(scenario, config):
    # TODO: configure all parameters
    pass
