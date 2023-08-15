"""Pydantic models for the messages exchanged between server and client."""
from enum import Enum
from typing import Dict, List, Optional, Type, Union

from pydantic import BaseModel, validator

from neurotechdevkit.materials import Material as NDKMaterial
from neurotechdevkit.materials import get_material
from neurotechdevkit.scenarios import Scenario2D, Scenario3D
from neurotechdevkit.scenarios import Target as NDKTarget
from neurotechdevkit.sources import (
    FocusedSource2D,
    FocusedSource3D,
    PhasedArraySource2D,
    PlanarSource2D,
    PointSource2D,
    Source,
)


class Axis(str, Enum):
    """Axis enum for the axis of CT scans."""

    x = "x"
    y = "y"
    z = "z"


class TransducerType(str, Enum):
    """Transducer type enum for the type of transducer."""

    pointSource = "pointSource"
    phasedArray = "phasedArray"
    focusedSource = "focusedSource"
    planarSource = "planarSource"

    @classmethod
    def from_source(
        cls,
        source: Union[
            PointSource2D,
            PhasedArraySource2D,
            FocusedSource2D,
            PlanarSource2D,
        ],
    ):
        """Instantiate the transducer type from a source."""
        if isinstance(source, PointSource2D):
            return cls.pointSource
        elif isinstance(source, PhasedArraySource2D):
            return cls.phasedArray
        elif isinstance(source, (FocusedSource2D, FocusedSource3D)):
            return cls.focusedSource
        elif isinstance(source, PlanarSource2D):
            return cls.planarSource


class _BaseSourceSettings(BaseModel):
    @classmethod
    def from_source(cls, source) -> BaseModel:
        """Instantiate the source settings from a source."""
        raise NotImplementedError


class PointSourceSettings(_BaseSourceSettings):
    """Settings for a point source transducer."""

    delay: float
    position: List[float]

    @classmethod
    def from_source(cls, source: PointSource2D) -> "PointSourceSettings":
        """Instantiate the source settings from a source."""
        return cls(position=source._position, delay=source._delay)


class PhasedArraySettings(_BaseSourceSettings):
    """Settings for a phased array transducer."""

    position: List[float]
    direction: List[float]
    numPoints: int
    numElements: int
    pitch: float
    elementWidth: float
    tiltAngle: float
    focalLength: float
    delay: float
    elementDelays: List[float]

    @classmethod
    def from_source(cls, source: PhasedArraySource2D) -> "PhasedArraySettings":
        """Instantiate the source settings from a source."""
        return cls(
            position=source._position,
            direction=source._direction,
            num_points=source._num_points,
            num_elements=source._num_elements,
            pitch=source._pitch,
            element_width=source._element_width,
            tilt_angle=source._tilt_angle,
            focal_length=source._focal_length,
            delay=source._delay,
            element_delays=source._element_delays,
        )


class FocusedSourceSettings(_BaseSourceSettings):
    """Settings for a focused source transducer."""

    position: List[float]
    aperture: float
    direction: List[float]
    focalLength: float
    numPoints: int
    delay: float

    @classmethod
    def from_source(cls, source: FocusedSource2D) -> "FocusedSourceSettings":
        """Instantiate the source settings from a source."""
        return cls(
            position=source._position,
            direction=source._direction,
            aperture=source._aperture,
            focalLength=source._focal_length,
            numPoints=source._num_points,
            delay=source._delay,
        )


class PlanarSourceSettings(_BaseSourceSettings):
    """Settings for a planar source transducer."""

    aperture: float
    delay: float
    direction: List[float]
    focalLength: float
    numPoints: int
    position: List[float]

    @classmethod
    def from_source(cls, source: PlanarSource2D) -> "PlanarSourceSettings":
        """Instantiate the source settings from a source."""
        return cls(
            aperture=source._aperture,
            delay=source._delay,
            direction=source._direction,
            focalLength=source._focal_length,
            numPoints=source._num_points,
            position=source._position,
        )


TRANSDUCER_SETTINGS: Dict[Type[Source], Type[_BaseSourceSettings]] = {
    PointSource2D: PointSourceSettings,
    PhasedArraySource2D: PhasedArraySettings,
    FocusedSource2D: FocusedSourceSettings,
    FocusedSource3D: FocusedSourceSettings,
    PlanarSource2D: PlanarSourceSettings,
}


class Transducer(BaseModel):
    """Transducer model for the transducer settings."""

    transducerType: TransducerType
    transducerSettings: Union[
        PointSourceSettings,
        PhasedArraySettings,
        FocusedSourceSettings,
        PlanarSourceSettings,
    ]

    @classmethod
    def from_source(
        cls,
        source: Union[
            PointSource2D,
            PhasedArraySource2D,
            FocusedSource2D,
            PlanarSource2D,
        ],
    ) -> "Transducer":
        """Instantiate the transducer settings from a source."""
        transducer_type = TransducerType.from_source(source)
        settings_model = TRANSDUCER_SETTINGS[type(source)]
        transducer_settings = settings_model.from_source(source)
        return cls(
            transducerType=transducer_type,
            transducerSettings=transducer_settings,
        )


class Target(BaseModel):
    """Target model for the target settings."""

    centerX: float
    centerY: float
    radius: float


class ScenarioSettings(BaseModel):
    """Scenario settings model for the scenario settings."""

    axis: Optional[Axis]
    distanceFromOrigin: Optional[List[float]]
    isPreBuilt: bool
    scenarioId: Optional[str]

    @validator("scenarioId", always=True)
    def _validate(cls, value, values):
        if values.get("isPreBuilt") and not value:
            raise ValueError("scenarioId is required for prebuilt scenarios")
        return value

    @property
    def scenario_id(self) -> str:
        """Get the scenario id."""
        assert self.scenarioId
        return self.scenarioId


class Material(BaseModel):
    """Material model for the material properties."""

    vp: float
    rho: float
    alpha: float
    renderColor: str

    @classmethod
    def from_ndk_material(cls, material: NDKMaterial) -> "Material":
        """Instantiate the web material from an NDKMaterial."""
        return cls(
            vp=material.vp,
            rho=material.rho,
            alpha=material.alpha,
            renderColor=material.render_color,
        )


class MaterialProperties(BaseModel):
    """Material properties model for the material properties."""

    water: Material
    brain: Material
    trabecularBone: Material


class SimulationSettings(BaseModel):
    """Simulation settings model for the simulation settings."""

    simulationPrecision: int
    centerFrequency: float
    isSteadySimulation: bool
    materialProperties: MaterialProperties


class _DefaultSettings(BaseModel):
    """Default request model."""

    is2d: bool
    scenarioSettings: ScenarioSettings
    transducers: list[Transducer]
    target: Target
    simulationSettings: SimulationSettings

    @classmethod
    def from_scenario(
        cls, scenario_id: str, scenario: Union[Type[Scenario2D], Type[Scenario3D]]
    ) -> "_DefaultSettings":
        """Instantiate the settings from a built in scenario."""
        is_2d = issubclass(scenario, Scenario2D)
        center_frequency = scenario.center_frequency
        assert isinstance(center_frequency, float)
        material_properties = MaterialProperties(
            water=Material.from_ndk_material(get_material("water", center_frequency)),
            brain=Material.from_ndk_material(get_material("brain", center_frequency)),
            trabecularBone=Material.from_ndk_material(
                get_material("trabecular_bone", center_frequency)
            ),
        )
        simulation_settings = SimulationSettings(
            simulationPrecision=5,  # TODO: make this a parameter
            centerFrequency=center_frequency,
            isSteadySimulation=True,  # default value
            materialProperties=material_properties,
        )
        scenario_settings = ScenarioSettings(scenarioId=scenario_id, isPreBuilt=True)
        scenario_target = scenario.target
        assert isinstance(scenario_target, NDKTarget)
        target = Target(
            centerX=scenario_target.center[1],
            centerY=scenario_target.center[0],
            radius=scenario_target.radius,
        )
        transducers = [Transducer.from_source(source) for source in scenario.sources]
        return cls(
            is2d=is_2d,
            scenarioSettings=scenario_settings,
            transducers=transducers,
            target=target,
            simulationSettings=simulation_settings,
        )


class RenderLayoutRequest(_DefaultSettings):
    """Render layout request model for the render layout request."""

    pass


class SimulateRequest(_DefaultSettings):
    """Simulate request model for the simulate request."""

    pass


class IndexBuiltInScenario(_DefaultSettings):
    """The Scenario settings for the index page."""

    scenarioName: str

    @classmethod
    def from_scenario(
        cls, scenario_id: str, scenario: Union[Type[Scenario2D], Type[Scenario3D]]
    ) -> "IndexBuiltInScenario":
        """Instantiate the settings from a built in scenario."""
        super_settings = _DefaultSettings.from_scenario(scenario_id, scenario)
        return cls(scenarioName=scenario_id, **super_settings.dict())
