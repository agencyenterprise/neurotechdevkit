from typing import Mapping

import numpy as np
import numpy.typing as npt

from ...grid import Grid
from ...materials import Material
from ...sources import FocusedSource2D
from .._base import Scenario2D
from .._utils import Target, create_grid_circular_mask, create_grid_elliptical_mask


class Scenario0(Scenario2D):
    """Scenario 0."""

    center_frequency = 5e5  # Hz
    target = Target(
        target_id="target_1",
        center=[0.0285, 0.0024],
        radius=0.0017,
        description="Represents a simulated tumor.",
    )
    sources = [
        FocusedSource2D(
            position=[0.01, 0.0],
            direction=[1.0, 0.0],
            aperture=0.01,
            focal_length=0.01,
            num_points=1000,
        )
    ]
    origin = [0.0, -0.02]

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
        material_layers = [
            "water",
            "cortical_bone",
            "brain",
            "tumor",
        ]
        material_masks = {
            name: _create_scenario_0_mask(
                name, self.grid, np.array(self.origin, dtype=float)
            )
            for name in material_layers
        }
        return material_masks

    def make_grid(self):
        """Make the grid for scenario 0."""
        self.grid = Grid.make_grid(
            extent=(0.05, 0.04),  # m
            speed_water=1500,  # m/s
            ppw=6,  # desired resolution for complexity=fast
            center_frequency=self.center_frequency,
        )
        self.material_masks = self._make_material_masks()


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
    return outer_brain_mask


def _create_tumor_mask(grid, origin):
    tumor_radius = 0.0013
    tumor_center = np.array([0.0285, 0.0025])
    tumor_mask = create_grid_circular_mask(grid, origin, tumor_center, tumor_radius)
    return tumor_mask
