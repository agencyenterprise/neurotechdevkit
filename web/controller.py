"""Controller for the web app."""
import base64
import io
import tempfile
from typing import Dict, Optional, Tuple, Union

from neurotechdevkit.results import PulsedResult2D, SteadyStateResult2D
from neurotechdevkit.scenarios import Scenario2D, Scenario3D, Target
from neurotechdevkit.scenarios.built_in import BUILT_IN_SCENARIOS
from web.messages.requests import (
    IndexBuiltInScenario,
    RenderLayoutRequest,
    SimulateRequest,
)


class BuiltInScenariosShelf(object):
    """Singleton class for storing the already instantiated built-in scenarios."""

    scenarios: Dict[str, Scenario2D] = {}

    def __new__(cls):
        """Create a new instance of the BuiltInScenariosShelf class."""
        if not hasattr(cls, "instance"):
            cls.instance = super(BuiltInScenariosShelf, cls).__new__(cls)
        return cls.instance

    def get(self, scenario_id: str) -> Scenario2D:
        """
        Return the instantiated built-in scenario with the given id.

        Args:
            scenario_id (str): The id of the scenario.

        Returns:
            The built-in scenario.
        """
        if scenario_id not in self.scenarios:
            builtin_scenario = BUILT_IN_SCENARIOS[scenario_id]()
            builtin_scenario.make_grid()
            builtin_scenario.compile_problem()
            self.scenarios[scenario_id] = builtin_scenario
        return self.scenarios[scenario_id]


def get_built_in_scenarios() -> Dict[str, Dict]:
    """
    Return a dictionary containing information about all the built-in scenarios.

    Returns:
        The dictionary of built-in scenario infos.
    """
    scenarios = {}
    for scenario_id, scenario in BUILT_IN_SCENARIOS.items():
        assert issubclass(scenario, (Scenario2D, Scenario3D))
        settings = IndexBuiltInScenario.from_scenario(scenario_id, scenario)
        assert isinstance(settings, IndexBuiltInScenario)
        settings.title = scenario_id.replace("_", " ").title()
        scenarios[scenario_id] = settings.dict()
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
        scenario = _instantiate_scenario(
            config.scenarioSettings.isPreBuilt, config.scenarioSettings.scenario_id
        )
        _configure_scenario(scenario, config)
        fig = scenario.render_layout(show_sources=len(scenario.sources) > 0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return data
    raise NotImplementedError


def _instantiate_scenario(is_prebuilt: bool, scenario_id: Optional[str]) -> Scenario2D:
    """Instantiate the scenario for the web app.

    Args:
        is_prebuilt (bool): Whether the scenario is prebuilt or not.
        scenario_id (Optional[str]): The id of the scenario.

    Returns:
        The instantiated scenario.
    """
    if is_prebuilt:
        assert scenario_id is not None
        builtin_scenario = BuiltInScenariosShelf().get(scenario_id)
        material_properties = {
            key: value for key, value in builtin_scenario.material_properties.items()
        }
        scenario = Scenario2D(
            center_frequency=builtin_scenario.center_frequency,
            material_properties=material_properties,
            material_masks=builtin_scenario.material_masks,
            origin=builtin_scenario.origin,
            sources=[source for source in builtin_scenario.sources],
            target=builtin_scenario.target,
            problem=builtin_scenario.problem,
            grid=builtin_scenario.grid,
        )
    else:
        raise NotImplementedError
    return scenario


def get_simulation_image(config: SimulateRequest) -> Tuple[str, str]:
    """
    Simulate a scenario and return the result as a base64 gif or png.

    Args:
        config (SimulateRequest): The configuration for the scenario.

    Returns:
        Tuple[str, str]: The base64 encoded image and the image format.
    """
    if config.is2d:
        scenario = _instantiate_scenario(
            config.scenarioSettings.isPreBuilt, config.scenarioSettings.scenario_id
        )
        _configure_scenario(scenario, config)
        scenario.compile_problem()
        if config.simulationSettings.isSteadySimulation:
            result = scenario.simulate_steady_state()
            assert isinstance(result, SteadyStateResult2D)
            fig = result.render_steady_state_amplitudes(show_material_outlines=False)
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            data = base64.b64encode(buf.getbuffer()).decode("ascii")
            return data, "png"
        else:
            pulse_result = scenario.simulate_pulse()
            assert isinstance(pulse_result, PulsedResult2D)
            animation = pulse_result.render_pulsed_simulation_animation()
            with tempfile.NamedTemporaryFile(suffix=".gif") as tmpfile:
                animation.save(tmpfile.name, writer="imagemagick", fps=30)
                tmpfile.seek(0)
                buf = io.BytesIO(tmpfile.read())
                data = base64.b64encode(buf.getbuffer()).decode("ascii")
                return data, "gif"
    raise NotImplementedError


def _configure_scenario(
    scenario: Scenario2D, config: Union[RenderLayoutRequest, SimulateRequest]
):
    """Configure a scenario based on the given configuration."""
    if config_target := config.target:
        scenario.target = Target(
            target_id="target_1",
            center=[config_target.centerY, config_target.centerX],
            radius=config_target.radius,
            description="",
        )

    scenario.center_frequency = config.simulationSettings.centerFrequency

    config_material_properties = config.simulationSettings.materialProperties
    if config_material_properties:
        scenario.material_properties = (
            config_material_properties.to_ndk_material_properties()
        )

    config_transducers = config.transducers
    scenario.sources.clear()
    for configured_transducer in config_transducers:
        source = configured_transducer.to_ndk_source()
        scenario.sources.append(source)
