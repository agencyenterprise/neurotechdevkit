"""Pydantic models for the messages sent to the server."""
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, validator


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


class PointSourceSettings(BaseModel):
    """Settings for a point source transducer."""

    aperture: float
    delay: float
    direction: List[float]
    focalLength: float
    numPoints: int
    position: List[float]


class PhasedArraySettings(BaseModel):
    """Settings for a phased array transducer."""

    aperture: float
    delay: float
    direction: List[float]
    focalLength: float
    numPoints: int
    position: List[float]


class FocusedSourceSettings(BaseModel):
    """Settings for a focused source transducer."""

    aperture: float
    delay: float
    direction: List[float]
    focalLength: float
    numPoints: int
    position: List[float]


class PlanarSourceSettings(BaseModel):
    """Settings for a planar source transducer."""

    aperture: float
    delay: float
    direction: List[float]
    focalLength: float
    numPoints: int
    position: List[float]


class Transducer(BaseModel):
    """Transducer model for the transducer settings."""

    transducerType: TransducerType
    transducerSettings: Union[
        PointSourceSettings,
        PhasedArraySettings,
        FocusedSourceSettings,
        PlanarSourceSettings,
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


class MaterialProperties(BaseModel):
    """Material properties model for the material properties."""

    vp: float
    rho: float
    alpha: float
    renderColor: str


class SimulationSettings(BaseModel):
    """Simulation settings model for the simulation settings."""

    simulationPrecision: int
    centerFrequency: float
    isSteadySimulation: bool
    materialProperties: dict[str, MaterialProperties]


class DefaultRequest(BaseModel):
    """Default request model."""

    is2d: bool
    scenarioSettings: ScenarioSettings
    transducers: list[Transducer]
    target: Target
    simulationSettings: SimulationSettings


class RenderLayoutRequest(DefaultRequest):
    """Render layout request model for the render layout request."""

    pass


class SimulateRequest(DefaultRequest):
    """Simulate request model for the simulate request."""

    pass
