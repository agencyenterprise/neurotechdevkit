import numpy as np
import stride
from mosaic.types import Struct

from ..sources import PhasedArraySource2D
from . import materials
from ._base import Scenario2D, Target
from ._utils import (
    add_material_fields_to_problem,
    create_grid_circular_mask,
    make_grid,
)


_PHANTOM_1_CENTER = np.array([0.05, 0.0])
_PHANTOM_2_CENTER = np.array([0.08, 0.03])


class Scenario3(Scenario2D):
    """Scenario 3: circular phantoms in water."""

    _SCENARIO_ID = "scenario-3-v0"
    _TARGET_OPTIONS = {
        "target_1": Target(
            target_id="target_1",
            center=_PHANTOM_1_CENTER,
            radius=0.005,
            description="Imaging phantom 1.",
        ),
        "target_2": Target(
            target_id="target_2",
            center=_PHANTOM_2_CENTER,
            radius=0.005,
            description="Imaging phantom 2.",
        ),
    }

    def __init__(self, complexity="fast"):
        """Create a new instance of scenario 3."""
        self._target_id = "target_1"
        self._extent = np.array([0.12, 0.1])  # m

        super().__init__(
            origin=np.array([0.0, -self._extent[1] / 2]),
            complexity=complexity,
        )

    @property
    def _material_layers(self) -> list[tuple[str, Struct]]:
        return [
            ("water", materials.water),
            ("agar_hydrogel", materials.agar_hydrogel),
        ]

    @property
    def _material_outline_upsample_factor(self) -> int:
        return 16

    def _get_material_masks(self, problem):
        return {
            name: _create_scenario_3_mask(name, problem.grid, self._origin)
            for name in self.materials.keys()
        }

    def _compile_problem(self) -> stride.Problem:

        # scenario constants
        speed_water = 1500  # m/s
        c_freq = 500e3  # hz

        # desired resolution for complexity=fast
        ppw = 6

        # compute resolution
        dx = speed_water / c_freq / ppw  # m

        grid = make_grid(extent=self._extent, dx=dx)
        problem = stride.Problem(
            name=f"{self.scenario_id}-{self.complexity}", grid=grid
        )
        problem = add_material_fields_to_problem(
            problem=problem,
            materials=self.materials,
            layer_ids=self.layer_ids,
            masks=self._get_material_masks(problem),
        )
        return problem

    def get_default_source(self):
        """Return the default source for the scenario."""
        return PhasedArraySource2D(
            position=np.array([0.0, 0.0]),
            direction=np.array([1.0, 0.0]),
            num_elements=64,
            num_points=960,
            pitch=0.5e-3,
            element_width=0.4e-3,
        )


def _create_scenario_3_mask(material, grid, origin):
    if material == "water":
        phantom_mask = _create_phantom_mask(grid, origin)
        water_mask = ~phantom_mask
        return water_mask

    elif material == "agar_hydrogel":
        phantom_mask = _create_phantom_mask(grid, origin)
        return phantom_mask

    else:
        raise ValueError(material)


def _create_phantom_mask(grid, origin):
    phantom_radius = 0.01  # m
    phantom_mask = (
        create_grid_circular_mask(grid, origin, _PHANTOM_1_CENTER, phantom_radius)
        | create_grid_circular_mask(grid, origin, _PHANTOM_2_CENTER, phantom_radius)
    )
    return phantom_mask

