"""Pydantic models for the scenario settings."""
from enum import Enum
from typing import Optional, Type, Union

from pydantic import BaseModel, validator

from neurotechdevkit.scenarios import Scenario2D, Scenario3D, SliceAxis
from neurotechdevkit.scenarios import Target as NDKTarget
from web.messages.material_properties import MaterialProperties
from web.messages.transducers import TRANSDUCER_SETTINGS, Transducer


class Axis(str, Enum):
    """Axis enum for the axis of CT scans."""

    x = "x"
    y = "y"
    z = "z"

    @classmethod
    def from_ndk_axis(cls, axis: Optional[SliceAxis]) -> Optional["Axis"]:
        """Get the axis from the NDK axis."""
        if axis == SliceAxis.X:
            return cls.x
        elif axis == SliceAxis.Y:
            return cls.y
        elif axis == SliceAxis.Z:
            return cls.z
        return None

    def to_ndk_axis(self) -> SliceAxis:
        """Get the NDK axis from the axis."""
        if self == self.x:
            return SliceAxis.X
        elif self == self.y:
            return SliceAxis.Y
        elif self == self.z:
            return SliceAxis.Z
        raise ValueError(f"Invalid axis: {self}")


class Target(BaseModel):
    """Target model for the target settings."""

    centerX: float
    centerY: float
    centerZ: Optional[float]
    radius: float

    def to_ndk_target(self) -> NDKTarget:
        """Get the NDK target from the target."""
        center = [self.centerY, self.centerX]
        if self.centerZ is not None:
            center.append(self.centerZ)
        return NDKTarget(
            target_id="target_1",
            center=center,
            radius=self.radius,
            description="",
        )

    @classmethod
    def from_ndk_target(cls, target: NDKTarget) -> "Target":
        """Get the target from the NDK target."""
        return cls(
            centerX=target.center[1],
            centerY=target.center[0],
            centerZ=target.center[2] if len(target.center) > 2 else None,
            radius=target.radius,
        )


class ScenarioSettings(BaseModel):
    """Scenario settings model for the scenario settings."""

    sliceAxis: Optional[Axis]
    slicePosition: Optional[float]
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


class SimulationSettings(BaseModel):
    """Simulation settings model for the simulation settings."""

    simulationPrecision: int
    centerFrequency: float
    isSteadySimulation: bool
    materialProperties: MaterialProperties


class DefaultSettings(BaseModel):
    """Pydantic model with the default parameters to run a simulation."""

    is2d: bool
    scenarioSettings: ScenarioSettings
    transducers: list[Transducer]
    target: Optional[Target]
    simulationSettings: SimulationSettings

    @classmethod
    def from_scenario(
        cls, scenario_id: str, scenario: Union[Type[Scenario2D], Type[Scenario3D]]
    ) -> "DefaultSettings":
        """Instantiate the settings from a built in scenario."""
        is_2d = issubclass(scenario, Scenario2D)
        center_frequency = scenario.center_frequency
        assert isinstance(center_frequency, float)
        material_properties = MaterialProperties.from_scenario(scenario)
        simulation_settings = SimulationSettings(
            simulationPrecision=5,  # TODO: make this a parameter
            centerFrequency=center_frequency,
            isSteadySimulation=True,  # default value
            materialProperties=material_properties,
        )
        scenario_settings = ScenarioSettings(
            scenarioId=scenario_id,
            isPreBuilt=True,
            sliceAxis=Axis.from_ndk_axis(getattr(scenario, "slice_axis", None)),
            slicePosition=getattr(scenario, "slice_position", None),
        )
        scenario_target = scenario.target
        assert isinstance(scenario_target, NDKTarget)
        target = Target.from_ndk_target(scenario_target)

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
