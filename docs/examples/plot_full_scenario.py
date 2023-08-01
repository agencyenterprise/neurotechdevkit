# -*- coding: utf-8 -*-
"""
Implementing a full scenario
====================================

!!! note
    NDK and its examples are under constant development, more information
    and content will be added to this example soon!

The following code is a simplified implementation of NDK's Scenario 1.
"""
from typing import Mapping

# %%
# ## Implementing a Scenario
import numpy as np
from numpy import typing as npt

from neurotechdevkit import sources
from neurotechdevkit.problem import Problem
from neurotechdevkit.results import SteadyStateResult2D
from neurotechdevkit.scenarios import Scenario2D, Target, make_grid


class FullScenario(Scenario2D):
    """This Scenario is based on benchmark 4 of the following paper:

    Jean-Francois Aubry, Oscar Bates, Christian Boehm, et al., "Benchmark problems for
    transcranial ultrasound simulation: Intercomparison of compressional wave models",
    The Journal of the Acoustical Society of America 152, 1003 (2022);
    doi: 10.1121/10.0013426
    https://asa.scitation.org/doi/pdf/10.1121/10.0013426
    """

    """
    Target attributes:
        target_id: the string id of the target.
        center: the location of the center of the target (in meters).
        radius: the radius of the target (in meters).
        description: a text describing the target.
    """
    target = Target("target_1", np.array([0.064, 0.0]), 0.004, "")

    """
    The order of returned materials defines the layering of the scenario.
    """
    material_layers = [
        "water",
        "skin",
        "cortical_bone",
        "trabecular_bone",
        "brain",
    ]
    material_properties = {}
    origin = np.array([0.0, -0.035])
    sources = [
        sources.FocusedSource2D(
            position=np.array([0.0, 0.0]),
            direction=np.array([1.0, 0.0]),
            aperture=0.064,
            focal_length=0.064,
            num_points=1000,
        )
    ]
    material_outline_upsample_factor = 8

    def make_grid(self, center_frequency):
        """Make the grid for scenario 1 2D."""
        extent = np.array([0.12, 0.07])  # m
        # scenario constants
        speed_water = 1500  # m/s

        # desired resolution for complexity=fast
        ppw = 6

        # compute resolution
        dx = speed_water / center_frequency / ppw  # m

        self.grid = make_grid(extent=extent, dx=dx)
        self.material_masks = self._make_material_masks()

    def _make_material_masks(self) -> Mapping[str, npt.NDArray[np.bool_]]:
        """Make the material masks for scenario 1."""
        material_masks = {
            name: _create_scenario_1_mask(name, self.grid)
            for name in self.material_layers
        }
        return material_masks

    def compile_problem(self, center_frequency) -> Problem:
        """The problem definition for the scenario."""
        self.problem = Problem(center_frequency=center_frequency, grid=self.grid)
        self.problem.add_material_fields(
            materials=self.get_materials(center_frequency),
            layer_ids=self.layer_ids,
            masks=self.material_masks,
        )
        return self.problem


def _create_scenario_1_mask(material, grid):
    # layers are defined by X position
    dx = grid.space.spacing[0]

    layers_m = np.array(
        [
            0.026,  # water
            0.004,  # skin
            0.0015,  # cortical bone
            0.004,  # trabecular bone
            0.001,  # cortical bone
            0.0835,  # brain
        ]
    )
    interfaces = np.cumsum(layers_m)

    mask = np.zeros(grid.space.shape, dtype=bool)

    if material == "water":
        _fill_mask(mask, start=0, end=interfaces[0], dx=dx)

    elif material == "skin":
        _fill_mask(mask, start=interfaces[0], end=interfaces[1], dx=dx)

    elif material == "cortical_bone":
        _fill_mask(mask, start=interfaces[1], end=interfaces[2], dx=dx)
        _fill_mask(mask, start=interfaces[3], end=interfaces[4], dx=dx)

    elif material == "trabecular_bone":
        _fill_mask(mask, start=interfaces[2], end=interfaces[3], dx=dx)

    elif material == "brain":
        _fill_mask(mask, start=interfaces[4], end=None, dx=dx)

    else:
        raise ValueError(material)

    return mask


def _fill_mask(mask, start, end, dx):
    # fill linearly along the x axis
    if end is None:
        n = int(start / dx)
        mask[n:] = True
    else:
        n = int(start / dx)
        m = int(end / dx)
        mask[n:m] = True


# %%
# ## Creating the scenario
scenario = FullScenario()
scenario.make_grid(center_frequency=5e5)
scenario.render_layout()

# %%
# ## Rendering the simulation
scenario.compile_problem(center_frequency=5e5)
result = scenario.simulate_steady_state()
assert isinstance(result, SteadyStateResult2D)
result.render_steady_state_amplitudes(show_material_outlines=False)


# %%
