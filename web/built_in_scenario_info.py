"""Built-in scenario info for the web app."""
from dataclasses import asdict, dataclass
from typing import Dict, List, Type, Union

from neurotechdevkit.scenarios import Scenario2D, Scenario3D, Target
from neurotechdevkit.sources import (
    FocusedSource2D,
    FocusedSource3D,
    PhasedArraySource2D,
    PhasedArraySource3D,
    PlanarSource2D,
    PlanarSource3D,
    PointSource2D,
    PointSource3D,
)

TRANSDUCER_TYPES = {
    PointSource2D: "pointSource",
    PointSource3D: "pointSource",
    PhasedArraySource2D: "phasedArray",
    PhasedArraySource3D: "phasedArray",
    FocusedSource2D: "focusedSource",
    FocusedSource3D: "focusedSource",
    PlanarSource2D: "planarSource",
    PlanarSource3D: "planarSource",
}


@dataclass
class TargetInfo:
    """Target info for the web app."""

    center: list[float]
    radius: float


@dataclass
class TransducerInfo:
    """Transducer info for the web app."""

    transducer_type: str
    transducer: Dict


@dataclass
class BuiltInScenarioInfo:
    """Built-in scenario info for the web app."""

    id: str
    name: str
    is_2d: bool
    target: Dict
    origin: list[float]
    transducers: List[Dict]
    precision: float
    center_frequency: float
    material_properties: dict[str, Dict]

    @classmethod
    def from_scenario(
        cls, scenario_id: str, scenario: Union[Type[Scenario2D], Type[Scenario3D]]
    ) -> "BuiltInScenarioInfo":
        """
        Create a BuiltInScenarioInfo from a scenario.

        Args:
            scenario_id (str): the scenario id
            scenario (Union[Type[Scenario2D], Type[Scenario3D]]): the scenario

        Returns:
            BuiltInScenarioInfo: The built-in scenario info
        """
        center_frequency = scenario.center_frequency
        assert isinstance(center_frequency, float)
        return cls(
            id=scenario_id,
            name=scenario.__name__,
            is_2d=issubclass(scenario, Scenario2D),
            target=_get_target_info(scenario),
            origin=scenario.origin,
            transducers=_get_transducers(scenario),
            precision=5,  # TODO: get this from somewhere
            center_frequency=center_frequency,
            material_properties=_get_material_properties(scenario),
        )


def _get_target_info(scenario: Union[Type[Scenario2D], Type[Scenario3D]]) -> Dict:
    scenario_target = scenario.target
    assert isinstance(scenario_target, Target)
    target_info = TargetInfo(
        center=scenario_target.center, radius=scenario_target.radius
    )
    return asdict(target_info)


def _get_transducers(scenario: Union[Type[Scenario2D], Type[Scenario3D]]) -> List[Dict]:
    transducers = []
    for source in scenario.sources:
        # TODO: improve this
        dict_source = source.to_dict()
        dict_source["numPoints"] = dict_source["num_points"]
        dict_source["focalLength"] = dict_source["focal_length"]
        del dict_source["num_points"]
        del dict_source["focal_length"]
        transducer_info = TransducerInfo(
            transducer_type=TRANSDUCER_TYPES[source.__class__],
            transducer=dict_source,
        )
        transducers.append(asdict(transducer_info))
    return transducers


def _get_material_properties(
    scenario: Union[Type[Scenario2D], Type[Scenario3D]]
) -> Dict:
    material_properties = {}
    for material_id, material in scenario.material_properties.items():
        material_properties[material_id] = asdict(material)
    return material_properties
