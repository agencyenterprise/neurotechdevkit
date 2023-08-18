"""Pydantic models for the scenario settings."""
from enum import Enum
from typing import List, Optional, Type, Union

from pydantic import BaseModel, validator

from neurotechdevkit.scenarios import Scenario2D, Scenario3D
from neurotechdevkit.scenarios import Target as NDKTarget
from web.messages.material_properties import MaterialProperties
from web.messages.transducers import TRANSDUCER_SETTINGS, Transducer


class Axis(str, Enum):
    """Axis enum for the axis of CT scans."""

    x = "x"
    y = "y"
    z = "z"


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
