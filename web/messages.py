"""Pydantic models for the messages exchanged between server and client."""
from enum import Enum
from typing import Dict, List, Literal, Optional, Type, Union

import numpy as np
from pydantic import BaseModel, Field, validator
from typing_extensions import Annotated

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
    phasedArraySource = "phasedArraySource"
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
            return cls.phasedArraySource
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

    transducerType: Literal["pointSource"]
    delay: float
    position: List[float]

    @classmethod
    def from_source(cls, source: PointSource2D) -> "PointSourceSettings":
        """Instantiate the source settings from a source."""
        return cls(
            transducerType="pointSource", position=source._position, delay=source._delay
        )

    def to_ndk_source(self) -> PointSource2D:
        """Instantiate the NDK source from transducer settings."""
        return PointSource2D(
            position=self.position,
            delay=self.delay,
        )


class PhasedArraySettings(_BaseSourceSettings):
    """Settings for a phased array transducer."""

    transducerType: Literal["phasedArraySource"]
    position: List[float]
    direction: List[float]
    numPoints: int
    numElements: int
    pitch: float
    elementWidth: float
    tiltAngle: float
    focalLength: float
    delay: float
    elementDelays: Optional[List[float]]

    @classmethod
    def from_source(cls, source: PhasedArraySource2D) -> "PhasedArraySettings":
        """Instantiate the source settings from a source."""
        return cls(
            transducerType="phasedArraySource",
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

    def to_ndk_source(self) -> PhasedArraySource2D:
        """Instantiate the NDK source from transducer settings."""
        return PhasedArraySource2D(
            position=self.position,
            direction=self.direction,
            num_points=self.numPoints,
            num_elements=self.numElements,
            pitch=self.pitch,
            element_width=self.elementWidth,
            tilt_angle=self.tiltAngle,
            focal_length=self.focalLength,
            delay=self.delay,
            element_delays=np.array(self.elementDelays) if self.elementDelays else None,
        )


class FocusedSourceSettings(_BaseSourceSettings):
    """Settings for a focused source transducer."""

    transducerType: Literal["focusedSource"]
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
            transducerType="focusedSource",
            position=source._position,
            direction=source._direction,
            aperture=source._aperture,
            focalLength=source._focal_length,
            numPoints=source._num_points,
            delay=source._delay,
        )

    def to_ndk_source(self) -> FocusedSource2D:
        """Instantiate the NDK source from transducer settings."""
        return FocusedSource2D(
            position=self.position,
            direction=self.direction,
            aperture=self.aperture,
            focal_length=self.focalLength,
            num_points=self.numPoints,
            delay=self.delay,
        )


class PlanarSourceSettings(_BaseSourceSettings):
    """Settings for a planar source transducer."""

    transducerType: Literal["planarSource"]
    aperture: float
    delay: float
    direction: List[float]
    numPoints: int
    position: List[float]

    @classmethod
    def from_source(cls, source: PlanarSource2D) -> "PlanarSourceSettings":
        """Instantiate the source settings from a source."""
        return cls(
            transducerType="planarSource",
            aperture=source._aperture,
            delay=source._delay,
            direction=source._direction,
            numPoints=source._num_points,
            position=source._position,
        )

    def to_ndk_source(self) -> PlanarSource2D:
        """Instantiate the NDK source from transducer settings."""
        return PlanarSource2D(
            aperture=self.aperture,
            delay=self.delay,
            direction=self.direction,
            num_points=self.numPoints,
            position=self.position,
        )


TRANSDUCER_SETTINGS: Dict[Type[Source], Type[_BaseSourceSettings]] = {
    PointSource2D: PointSourceSettings,
    PhasedArraySource2D: PhasedArraySettings,
    FocusedSource2D: FocusedSourceSettings,
    FocusedSource3D: FocusedSourceSettings,
    PlanarSource2D: PlanarSourceSettings,
}


Transducer = Annotated[
    Union[
        PointSourceSettings,
        PhasedArraySettings,
        FocusedSourceSettings,
        PlanarSourceSettings,
    ],
    Field(discriminator="transducerType"),
]


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

    def to_ndk_material(self) -> NDKMaterial:
        """Instantiate the NDKMaterial from the web material."""
        return NDKMaterial(
            vp=self.vp,
            rho=self.rho,
            alpha=self.alpha,
            render_color=self.renderColor,
        )


class MaterialProperties(BaseModel):
    """Material properties model for the material properties."""

    water: Material
    brain: Material
    trabecularBone: Material
    corticalBone: Material
    skin: Material
    tumor: Material

    def to_ndk_material_properties(self) -> Dict[str, NDKMaterial]:
        """Instantiate the NDKMaterialProperties from the web material properties."""
        return {
            "water": self.water.to_ndk_material(),
            "brain": self.brain.to_ndk_material(),
            "trabecular_bone": self.trabecularBone.to_ndk_material(),
            "cortical_bone": self.corticalBone.to_ndk_material(),
            "skin": self.skin.to_ndk_material(),
            "tumor": self.tumor.to_ndk_material(),
        }


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
    target: Optional[Target]
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
            corticalBone=Material.from_ndk_material(
                get_material("cortical_bone", center_frequency)
            ),
            skin=Material.from_ndk_material(get_material("skin", center_frequency)),
            tumor=Material.from_ndk_material(get_material("tumor", center_frequency)),
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

        transducers = []
        for source in scenario.sources:
            settings_model = TRANSDUCER_SETTINGS[type(source)]
            transducer_settings = settings_model.from_source(source)
            transducers.append(transducer_settings)

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

    pass
