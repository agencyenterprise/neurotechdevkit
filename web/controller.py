"""Controller for the web app."""
import base64
import io
import tempfile
from typing import Dict, Optional, Tuple, Union

from neurotechdevkit.results import SteadyStateResult2D, SteadyStateResult3D
from neurotechdevkit.scenarios import Scenario2D, Scenario3D
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
    scenario = _instantiate_scenario(
        config.scenarioSettings.isPreBuilt,
        config.is2d,
        config.scenarioSettings.scenario_id,
    )
    _configure_scenario(scenario, config)
    fig = scenario.render_layout(show_sources=len(scenario.sources) > 0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    return data


def _instantiate_scenario(
    is_prebuilt: bool, is_2d: bool, scenario_id: Optional[str]
) -> Union[Scenario2D, Scenario3D]:
    """Instantiate the scenario for the web app.

    Args:
        is_prebuilt (bool): Whether the scenario is prebuilt or not.
        is_2d (bool): Whether the scenario is 2D or not.
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
        scenario_class = Scenario2D if is_2d else Scenario3D
        scenario = scenario_class(
            center_frequency=builtin_scenario.center_frequency,
            material_properties=material_properties,
            material_masks=builtin_scenario.material_masks,
            material_outline_upsample_factor=16,  # TODO: make this configurable
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
    scenario = _instantiate_scenario(
        config.scenarioSettings.isPreBuilt,
        config.is2d,
        config.scenarioSettings.scenario_id,
    )
    _configure_scenario(scenario, config)
    scenario.compile_problem()

    if config.simulationSettings.isSteadySimulation:
        result = scenario.simulate_steady_state()
        assert isinstance(result, (SteadyStateResult2D, SteadyStateResult3D))
        fig = result.render_steady_state_amplitudes(show_material_outlines=False)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return data, "png"
    else:
        pulse_result = scenario.simulate_pulse()
        animation = pulse_result.render_pulsed_simulation_animation()
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmpfile:
            animation.save(tmpfile.name, writer="imagemagick", fps=30)
            tmpfile.seek(0)
            buf = io.BytesIO(tmpfile.read())
            data = base64.b64encode(buf.getbuffer()).decode("ascii")
            return data, "gif"


def _configure_scenario(
    scenario: Union[Scenario2D, Scenario3D],
    config: Union[RenderLayoutRequest, SimulateRequest],
):
    """Configure a scenario based on the given configuration."""
    if config_target := config.target:
        scenario.target = config_target.to_ndk_target()

    scenario.center_frequency = config.simulationSettings.centerFrequency
    if isinstance(scenario, Scenario3D) and config.scenarioSettings.sliceAxis:
        scenario.slice_axis = config.scenarioSettings.sliceAxis.to_ndk_axis()
        assert config.scenarioSettings.slicePosition is not None
        scenario.slice_position = config.scenarioSettings.slicePosition

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
