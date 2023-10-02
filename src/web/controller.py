"""Controller for the web app."""
import base64
import io
import tempfile
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
from flask import current_app
from web.computed_tomography import get_ct_image
from web.messages.material_properties import Material, MaterialName, MaterialProperties
from web.messages.requests import (
    IndexBuiltInScenario,
    RenderLayoutRequest,
    SimulateRequest,
)
from web.messages.settings import Axis

from neurotechdevkit.grid import Grid
from neurotechdevkit.materials import DEFAULT_MATERIALS, get_material
from neurotechdevkit.results import SteadyStateResult2D, SteadyStateResult3D
from neurotechdevkit.scenarios import Scenario2D, Scenario3D
from neurotechdevkit.scenarios.built_in import BUILT_IN_SCENARIOS


class BuiltInScenariosShelf(object):
    """Singleton class for storing the already instantiated built-in scenarios."""

    scenarios: Dict[str, Dict[float, Union[Scenario2D, Scenario3D]]] = {}

    def __new__(cls):
        """Create a new instance of the BuiltInScenariosShelf class."""
        if not hasattr(cls, "instance"):
            cls.instance = super(BuiltInScenariosShelf, cls).__new__(cls)
        return cls.instance

    def get(
        self, scenario_id: str, center_frequency: float
    ) -> Union[Scenario2D, Scenario3D]:
        """
        Return the instantiated built-in scenario with the given id.

        Args:
            scenario_id: The id of the scenario.

        Returns:
            The built-in scenario.
        """
        if scenario_id not in self.scenarios:
            self.scenarios[scenario_id] = {}

        if center_frequency not in self.scenarios[scenario_id]:
            builtin_scenario = BUILT_IN_SCENARIOS[scenario_id]()
            builtin_scenario.center_frequency = center_frequency
            builtin_scenario.make_grid()
            builtin_scenario.compile_problem()
            self.scenarios[scenario_id][center_frequency] = builtin_scenario
        return self.scenarios[scenario_id][center_frequency]


def get_built_in_scenarios() -> Dict[str, Dict]:
    """
    Return a dictionary containing information about all the built-in scenarios.

    Returns:
        The dictionary of built-in scenario info.
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
        config.simulationSettings.centerFrequency,
    )
    if not config.scenarioSettings.isPreBuilt:
        assert config.scenarioSettings.ctFile is not None
        ct_image = get_ct_image(
            ct_path=current_app.config["CT_FOLDER"] / config.scenarioSettings.ctFile,
            slice_axis=config.scenarioSettings.ctSliceAxis,
            slice_position=config.scenarioSettings.ctSlicePosition,
        )
        spacing = ct_image.spacing_in_meters
        if config.is2d:
            assert config.scenarioSettings.ctSliceAxis is not None
            if config.scenarioSettings.ctSliceAxis == Axis.x:
                spacing = (spacing[0], spacing[2])
            elif config.scenarioSettings.ctSliceAxis == Axis.y:
                spacing = (spacing[1], spacing[2])
            else:
                spacing = spacing = (spacing[0], spacing[1])
        scenario.grid = Grid.make_shaped_grid(
            shape=ct_image.data.shape, spacing=spacing
        )
        scenario.material_masks = ct_image.material_masks
    _configure_scenario(scenario, config)
    fig = scenario.render_layout(show_sources=len(scenario.sources) > 0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    plt.close(fig)
    return data


def _instantiate_scenario(
    is_prebuilt: bool,
    is_2d: bool,
    scenario_id: Optional[str],
    center_frequency: float,
) -> Union[Scenario2D, Scenario3D]:
    """Instantiate the scenario for the web app.

    Args:
        is_prebuilt: Whether the scenario is pre-built or not.
        is_2d: Whether the scenario is 2D or not.
        scenario_id: The id of the scenario.
        center_frequency: The center frequency of the scenario.

    Returns:
        The instantiated scenario.
    """
    if is_prebuilt:
        assert scenario_id is not None
        builtin_scenario = BuiltInScenariosShelf().get(scenario_id, center_frequency)
        material_properties = {
            key: value for key, value in builtin_scenario.material_properties.items()
        }
        sources = []
        if (
            hasattr(builtin_scenario, "sources")
            and builtin_scenario.sources is not None
        ):
            for source in builtin_scenario.sources:
                sources.append(source)

        scenario_class = Scenario2D if is_2d else Scenario3D
        scenario = scenario_class(
            center_frequency=builtin_scenario.center_frequency,
            material_properties=material_properties,
            material_masks=builtin_scenario.material_masks,
            material_outline_upsample_factor=16,  # TODO: make this configurable
            origin=builtin_scenario.origin,
            sources=sources,
            target=builtin_scenario.target,
            problem=builtin_scenario.problem,
            grid=builtin_scenario.grid,
        )
    else:
        scenario_class = Scenario2D if is_2d else Scenario3D
        scenario = scenario_class(
            sources=[],
            origin=[0, 0] if is_2d else [0, 0, 0],
            material_outline_upsample_factor=16,  # TODO: make this configurable
        )

    return scenario


async def get_simulation_image(
    config: SimulateRequest, app_config: dict
) -> Tuple[str, str]:
    """
    Simulate a scenario and return the result as a base64 GIF or PNG.

    Args:
        config (SimulateRequest): The configuration for the scenario.
        app_config(dict): The configuration of the app.

    Returns:
        Tuple[str, str]: The base64 encoded image and the image format.
    """
    scenario = _instantiate_scenario(
        config.scenarioSettings.isPreBuilt,
        config.is2d,
        config.scenarioSettings.scenario_id,
        config.simulationSettings.centerFrequency,
    )
    if not config.scenarioSettings.isPreBuilt:
        assert config.scenarioSettings.ctFile is not None
        ct_image = get_ct_image(
            ct_path=app_config["CT_FOLDER"] / config.scenarioSettings.ctFile,
            slice_axis=config.scenarioSettings.ctSliceAxis,
            slice_position=config.scenarioSettings.ctSlicePosition,
        )
        scenario.grid = Grid.make_shaped_grid(
            shape=ct_image.data.shape, spacing=ct_image.spacing[0] / 1000
        )
        scenario.material_masks = ct_image.material_masks
    _configure_scenario(scenario, config)
    scenario.compile_problem()

    if config.simulationSettings.isSteadySimulation:
        result = scenario.simulate_steady_state()
        assert isinstance(result, (SteadyStateResult2D, SteadyStateResult3D))
        fig = result.render_steady_state_amplitudes(show_material_outlines=False)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        plt.close(fig)
        return data, "png"
    else:
        pulse_result = scenario.simulate_pulse()
        animation = pulse_result.render_pulsed_simulation_animation()
        assert hasattr(animation, "_fig")
        with tempfile.NamedTemporaryFile(suffix=".gif") as tmpfile:
            animation.save(tmpfile.name, writer="imagemagick", fps=30)
            tmpfile.seek(0)
            buf = io.BytesIO(tmpfile.read())
            data = base64.b64encode(buf.getbuffer()).decode("ascii")
            plt.close(animation._fig)
            return data, "gif"


def get_default_material_properties(center_frequency: float) -> MaterialProperties:
    """Get the default material properties with center frequency.

    Args:
        center_frequency: The center frequency of the scenario.

    Returns:
        The default material properties.
    """
    material_properties = {}
    for ndk_material_name in DEFAULT_MATERIALS:
        ndk_material = get_material(ndk_material_name, center_frequency)
        material = Material.from_ndk_material(ndk_material)
        material_name = MaterialName.get_material_name(ndk_material_name)
        material_properties[material_name] = material

    return MaterialProperties(**material_properties)


def _configure_scenario(
    scenario: Union[Scenario2D, Scenario3D],
    config: Union[RenderLayoutRequest, SimulateRequest],
):
    """Configure a scenario based on the given configuration.

    Args:
        scenario: The scenario to configure.
        config: The configuration.
    """
    scenario.center_frequency = config.simulationSettings.centerFrequency
    if config_target := config.target:
        scenario.target = config_target.to_ndk_target()

    if isinstance(scenario, Scenario3D) and config.displaySettings:
        scenario.slice_axis = config.displaySettings.sliceAxis.to_ndk_axis()
        scenario.slice_position = config.displaySettings.slicePosition

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
