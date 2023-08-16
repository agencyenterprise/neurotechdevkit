"""Controller for the web app."""
import base64
import io
import tempfile
from typing import Dict, List, Tuple, Union

from neurotechdevkit.results import PulsedResult2D, SteadyStateResult2D
from neurotechdevkit.scenarios import Scenario2D, Scenario3D, Target
from neurotechdevkit.scenarios.built_in import BUILT_IN_SCENARIOS
from web.messages import IndexBuiltInScenario, RenderLayoutRequest, SimulateRequest


def get_supported_materials() -> List[Tuple[str, str]]:
    """
    Return a list of supported materials and their descriptions.

    Returns:
        The list of supported materials.
    """
    return [
        ("water", "Water"),
        ("brain", "Brain"),
        ("trabecularBone", "Trabecular Bone"),
        ("corticalBone", "Cortical Bone"),
        ("skin", "Skin"),
        ("tumor", "Tumor"),
    ]


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
        scenario = _instantiate_scenario(config)
        _configure_scenario(scenario, config)
        fig = scenario.render_layout(show_sources=len(scenario.sources) > 0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return data
    raise NotImplementedError


def _instantiate_scenario(
    config: Union[RenderLayoutRequest, SimulateRequest]
) -> Scenario2D:
    if config.scenarioSettings.isPreBuilt:
        builtin_scenario = BUILT_IN_SCENARIOS[config.scenarioSettings.scenario_id]()
        builtin_scenario.make_grid()
        builtin_scenario.compile_problem()
        scenario = Scenario2D(
            center_frequency=builtin_scenario.center_frequency,
            material_properties=builtin_scenario.material_properties,
            material_masks=builtin_scenario.material_masks,
            origin=builtin_scenario.origin,
            sources=[source for source in builtin_scenario.sources],
            target=builtin_scenario.target,
            problem=builtin_scenario.problem,
            grid=builtin_scenario.grid,
        )
    else:
        scenario = Scenario2D()
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
        scenario = _instantiate_scenario(config)
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
    config_target = config.target
    if config_target:
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
