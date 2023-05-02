import numpy as np
import stride
from mosaic.types import Struct

from ..sources import FocusedSource2D
from . import materials
from ._base import Scenario2D, Target
from ._utils import (
    add_material_fields_to_problem,
    create_grid_circular_mask,
    create_grid_elliptical_mask,
    make_grid,
)


class Scenario0(Scenario2D):
    _SCENARIO_ID = "scenario-0-v0"
    _TARGET_OPTIONS = {
        "target_1": Target(
            target_id="target_1",
            center=np.array([0.0285, 0.0024]),
            radius=0.0017,
            description="Represents a simulated tumor.",
        ),
    }

    def __init__(self, complexity="fast"):
        self._target_id = "target_1"

        super().__init__(
            origin=np.array([0.0, -0.02]),
            complexity=complexity,
        )

    def render_material_property(
        self, name, show_orientation=True, show_sources=True, show_target=True
    ):
        raise NotImplementedError()

    @property
    def _material_layers(self) -> list[tuple[str, Struct]]:
        return [
            ("water", materials.water),
            ("skull", materials.cortical_bone),
            ("brain", materials.brain),
            ("tumor", materials.tumor),
        ]

    @property
    def _material_outline_upsample_factor(self) -> int:
        return 16

    def _get_material_masks(self, problem):
        return {
            name: _create_scenario_0_mask(name, problem.grid, self._origin)
            for name in self.materials.keys()
        }

    def _compile_problem(self) -> stride.Problem:
        extent = np.array([0.05, 0.04])  # m
        origin = self.origin  # m

        # scenario constants
        speed_water = 1500  # m/s
        c_freq = 500e3  # hz

        # desired resolution for complexity=fast
        ppw = 6

        # compute resolution
        dx = speed_water / c_freq / ppw  # m

        grid = make_grid(extent=extent, dx=dx)
        self._origin = origin
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
        return FocusedSource2D(
            position=np.array([0.01, 0.0]),
            direction=np.array([1.0, 0.0]),
            aperture=0.01,
            focal_length=0.01,
            num_points=1000,
        )


def _create_scenario_0_mask(material, grid, origin):
    if material == "water":
        outer_skull_mask = _create_skull_interface_mask(grid, origin)
        water_mask = ~outer_skull_mask
        return water_mask

    elif material == "skull":
        outer_skull_mask = _create_skull_interface_mask(grid, origin)
        outer_brain_mask = _create_brain_interface_mask(grid, origin)
        skull_mask = outer_skull_mask & ~outer_brain_mask
        return skull_mask

    elif material == "brain":
        outer_brain_mask = _create_brain_interface_mask(grid, origin)
        tumor_mask = _create_tumor_mask(grid, origin)
        brain_mask = outer_brain_mask & ~tumor_mask
        return brain_mask

    elif material == "tumor":
        tumor_mask = _create_tumor_mask(grid, origin)
        return tumor_mask

    else:
        raise ValueError(material)


def _create_skull_interface_mask(grid, origin):
    skull_outer_radii = (0.01275, 0.01)
    skull_center = (0.025, 0.0)

    skull_a, skull_b = skull_outer_radii
    outer_skull_mask = create_grid_elliptical_mask(
        grid, origin, skull_center, skull_a, skull_b
    )
    outer_skull_mask[50, 20] = False
    outer_skull_mask[50, 60] = False
    return outer_skull_mask


def _create_brain_interface_mask(grid, origin):
    skull_outer_radii = (0.01275, 0.01)
    skull_thickness = 0.001
    skull_center = (0.025, 0.0)

    skull_a, skull_b = skull_outer_radii
    brain_center = skull_center
    brain_a = skull_a - skull_thickness
    brain_b = skull_b - skull_thickness
    outer_brain_mask = create_grid_elliptical_mask(
        grid, origin, brain_center, brain_a, brain_b
    )
    outer_brain_mask[50, 22] = False
    outer_brain_mask[50, 58] = False
    return outer_brain_mask


def _create_tumor_mask(grid, origin):
    tumor_radius = 0.0013
    tumor_center = (0.0285, 0.0025)
    tumor_mask = create_grid_circular_mask(grid, origin, tumor_center, tumor_radius)
    return tumor_mask
