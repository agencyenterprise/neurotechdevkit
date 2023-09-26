"""Imaging scenario: grainy phantoms in water."""

from typing import Mapping, Optional

import numpy as np
import numpy.typing as npt
import stride
from mosaic.types import Struct

from ...grid import Grid
from ...materials import Material
from ...problem import Problem
from .._base import Scenario2D, Target
from .._utils import create_grid_circular_mask


class Scenario3(Scenario2D):
    """Imaging Scenario: grainy phantoms in water."""

    _PHANTOM_RADIUS = 0.01  # m

    center_frequency = 5e5  # Hz
    material_properties = {
        "water": Material(vp=1500.0, rho=1000.0, alpha=0.0, render_color="#2E86AB"),
        "agar_hydrogel": Material(
            vp=1485.0,
            rho=1050.0,
            alpha=0.1,
            render_color="#E9E6C9",
        ),
    }
    material_speckle_scale = {
        "water": 0.00067,
        "agar_hydrogel": 0.008,
    }  # Acoustic property heterogeneity: relative scale
    PREDEFINED_TARGET_OPTIONS = {
        "agar-phantom-center": Target(
            target_id="agar-phantom-center",
            center=[0.06, 0.0],
            radius=_PHANTOM_RADIUS,
            description="Center imaging phantom.",
        ),
        "agar-phantom-right": Target(
            target_id="agar-phantom-right",
            center=[0.07, 0.03],
            radius=_PHANTOM_RADIUS,
            description="Right imaging phantom.",
        ),
    }
    target = PREDEFINED_TARGET_OPTIONS["agar-phantom-center"]

    _extent = np.array([0.1, 0.1])  # m
    origin = [0, -_extent[1] / 2]

    material_outline_upsample_factor = 4

    def make_grid(self):
        """Make the grid for scenario 2 3D."""
        self.grid = Grid.make_grid(
            extent=self._extent,
            speed_water=1500,  # m/s
            ppw=6,  # desired resolution for complexity=fast
            center_frequency=self.center_frequency,
        )
        self.material_masks = self._make_material_masks()

    def compile_problem(
        self,
        rng: Optional[np.random.Generator] = None,
    ):
        """Compiles the problem for the scenario.

        Imaging scenarios have a special requirement of needing heterogeneous
        material to image.
        """
        assert self.grid is not None
        assert self.material_masks is not None

        self.problem = Problem(grid=self.grid)
        _add_material_fields_to_problem(
            self.problem,
            materials=self.materials,
            masks=self.material_masks,
            material_speckle=self.material_speckle_scale,
            rng=rng,
        )

    def _make_material_masks(self) -> Mapping[str, npt.NDArray[np.bool_]]:
        """Make the material masks for the scenario."""
        material_layers = [
            "water",
            "agar_hydrogel",
        ]
        material_masks = {
            name: self._create_scenario_mask(name, grid=self.grid, origin=self.origin)
            for name in material_layers
        }
        return material_masks

    def _create_scenario_mask(self, material, grid, origin):
        phantom_mask = np.zeros(grid.space.shape, dtype=bool)

        for target in self.PREDEFINED_TARGET_OPTIONS.values():
            phantom_mask |= create_grid_circular_mask(
                grid, origin, target.center, self._PHANTOM_RADIUS
            )

        if material == "agar_hydrogel":
            return phantom_mask

        elif material == "water":
            return ~phantom_mask

        else:
            raise ValueError(material)


def _add_material_fields_to_problem(
    problem: Problem,
    materials: Mapping[str, Struct],
    masks: Mapping[str, npt.NDArray[np.bool_]],
    material_speckle: Mapping[str, float],
    rng: Optional[np.random.Generator] = None,
):
    """Add material fields as media to the problem.

    This is an adaptation of the default `compile_problem()` that adds
    in heterogeneous material properties.

    Included fields are:

    - the speed of sound (in m/s)
    - density (in kg/m^3)
    - absorption (in dB/cm)

    Args:
        problem: the stride Problem object to which the
            media should be added.
        materials: a mapping from material names
            to Structs containing the material properties.
        masks: a mapping from material
            names to boolean masks indicating the gridpoints.
        material_speckle: a mapping from material names to the relative scale
            of material heterogeneity
    """
    property_names = [
        "vp",  # [m/s]
        "rho",  # [kg/m^3]
        "alpha",  # [dB/cm]
    ]

    rng = np.random.default_rng(rng)
    for prop_name in property_names:
        field = stride.ScalarField(name=prop_name, grid=problem.grid)

        for name, material in materials.items():
            material_mask = masks[name]
            material_prop = getattr(material, prop_name)
            speckle = rng.normal(
                scale=material_speckle[name] * material_prop,
                size=material_mask.sum(),
            )
            # Add heterogeneity to the material property
            # Really, only vp and rho affect the wave scattering,
            # but we add it to alpha for completeness
            field.data[material_mask] = material_prop + speckle

        field.pad()
        problem.medium.add(field)
