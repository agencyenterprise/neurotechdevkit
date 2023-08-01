from typing import Mapping

import numpy as np
import numpy.typing as npt

from ..grid import Grid
from ..materials import Material
from ..problem import Problem
from ..sources import FocusedSource2D
from ._base import Scenario2D, Target
from ._utils import create_grid_circular_mask, create_grid_elliptical_mask


class Scenario0(Scenario2D):
    """Scenario 0."""

    target = Target(
        target_id="target_1",
        center=np.array([0.0285, 0.0024]),
        radius=0.0017,
        description="Represents a simulated tumor.",
    )
    sources = [
        FocusedSource2D(
            position=np.array([0.01, 0.0]),
            direction=np.array([1.0, 0.0]),
            aperture=0.01,
            focal_length=0.01,
            num_points=1000,
        )
    ]
    origin = np.array([0.0, -0.02])
    material_layers = [
        "water",
        "cortical_bone",
        "brain",
        "tumor",
    ]
    material_properties = {
        "water": Material(vp=1500.0, rho=1000.0, alpha=0.0, render_color="#2E86AB"),
        "cortical_bone": Material(
            vp=2800.0, rho=1850.0, alpha=4.0, render_color="#FAF0CA"
        ),
        "brain": Material(vp=1560.0, rho=1040.0, alpha=0.3, render_color="#DB504A"),
        "tumor": Material(vp=1650.0, rho=1150.0, alpha=0.8, render_color="#94332F"),
    }

    def _make_material_masks(self) -> Mapping[str, npt.NDArray[np.bool_]]:
        """Make the material masks for scenario 0."""
        assert self.origin is not None
        material_masks = {
            name: _create_scenario_0_mask(name, self.grid, self.origin)
            for name in self.material_layers
        }
        return material_masks

    def make_grid(self):
        """
        Make the grid for scenario 0.

        Args:
            center_frequency (float): the center frequency of the transducer
        """
        extent = np.array([0.05, 0.04])  # m

        # scenario constants
        speed_water = 1500  # m/s

        # desired resolution for complexity=fast
        ppw = 6
        # compute resolution
        dx = speed_water / self.center_frequency / ppw  # m

        self.grid = Grid.make_grid(extent=extent, dx=dx)
        self.material_masks = self._make_material_masks()

    def compile_problem(self) -> Problem:
        """
        Compile the problem for scenario 0.

        Returns:
            Problem: the compiled problem
        """
        assert self.grid is not None
        assert self.layer_ids is not None
        assert self.material_masks is not None

        self.problem = Problem(center_frequency=self.center_frequency, grid=self.grid)
        self.problem.add_material_fields(
            materials=self.get_materials(self.center_frequency),
            layer_ids=self.layer_ids,
            masks=self.material_masks,
        )
        return self.problem


def _create_scenario_0_mask(material, grid, origin):
    if material == "water":
        outer_skull_mask = _create_skull_interface_mask(grid, origin)
        water_mask = ~outer_skull_mask
        return water_mask

    elif material == "cortical_bone":
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
    skull_outer_radii = np.array([0.01275, 0.01])
    skull_center = np.array([0.025, 0.0])

    skull_a, skull_b = skull_outer_radii
    outer_skull_mask = create_grid_elliptical_mask(
        grid, origin, skull_center, skull_a, skull_b
    )
    outer_skull_mask[50, 20] = False
    outer_skull_mask[50, 60] = False
    return outer_skull_mask


def _create_brain_interface_mask(grid, origin):
    skull_outer_radii = np.array([0.01275, 0.01])
    skull_thickness = 0.001
    skull_center = np.array([0.025, 0.0])

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
    tumor_center = np.array([0.0285, 0.0025])
    tumor_mask = create_grid_circular_mask(grid, origin, tumor_center, tumor_radius)
    return tumor_mask
