# -*- coding: utf-8 -*-
"""
Implementing a full scenario
====================================

!!! note
    NDK and its examples are under constant development, more information
    and content will be added to this example soon!

The following code is a simplified implementation of NDK's Scenario 1.
"""
# %%
# Implementing a Scenario
import numpy as np

from neurotechdevkit import sources
from neurotechdevkit.grid import Grid
from neurotechdevkit.problem import Problem
from neurotechdevkit.results import SteadyStateResult2D
from neurotechdevkit.scenarios import Scenario2D, Target

# %%
# Creating the scenario
scenario = Scenario2D()
scenario.center_frequency = 5e5

scenario.target = Target(
    target_id="target_1", center=np.array([0.064, 0.0]), radius=0.004, description=""
)
scenario.material_layers = [
    "water",
    "skin",
    "cortical_bone",
    "trabecular_bone",
    "brain",
]
scenario.material_properties = {}
scenario.origin = np.array([0.0, -0.035])
scenario.sources = [
    sources.FocusedSource2D(
        position=np.array([0.0, 0.0]),
        direction=np.array([1.0, 0.0]),
        aperture=0.064,
        focal_length=0.064,
        num_points=1000,
    )
]
scenario.material_outline_upsample_factor = 8

# %%
# Creating grid
extent = np.array([0.12, 0.07])  # m
speed_water = 1500  # m/s

ppw = 6  # desired resolution for complexity=fast
dx = speed_water / scenario.center_frequency / ppw  # m
grid = Grid.make_grid(extent=extent, dx=dx)

scenario.grid = grid

# %%
# Creating masks


def fill_mask(mask, start, end, dx):
    # fill linearly along the x axis
    if end is None:
        n = int(start / dx)
        mask[n:] = True
    else:
        n = int(start / dx)
        m = int(end / dx)
        mask[n:m] = True


def create_mask(material, grid):
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
        fill_mask(mask, start=0, end=interfaces[0], dx=dx)

    elif material == "skin":
        fill_mask(mask, start=interfaces[0], end=interfaces[1], dx=dx)

    elif material == "cortical_bone":
        fill_mask(mask, start=interfaces[1], end=interfaces[2], dx=dx)
        fill_mask(mask, start=interfaces[3], end=interfaces[4], dx=dx)

    elif material == "trabecular_bone":
        fill_mask(mask, start=interfaces[2], end=interfaces[3], dx=dx)

    elif material == "brain":
        fill_mask(mask, start=interfaces[4], end=None, dx=dx)

    else:
        raise ValueError(material)

    return mask


scenario.material_masks = {
    name: create_mask(name, grid) for name in scenario.material_layers
}


# %%
# Rendering the layout
scenario.render_layout()

# %%
# Creating problem

problem = Problem(center_frequency=scenario.center_frequency, grid=grid)
problem.add_material_fields(
    materials=scenario.materials,
    layer_ids=scenario.layer_ids,
    masks=scenario.material_masks,
)
scenario.problem = problem

# %%
# Rendering the simulation
result = scenario.simulate_steady_state()
assert isinstance(result, SteadyStateResult2D)
result.render_steady_state_amplitudes(show_material_outlines=False)

# %%
