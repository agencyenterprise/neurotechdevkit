# -*- coding: utf-8 -*-
"""
Create a scenario by manually adding its properties
====================================================

This example demonstrates how to create a scenario by manually adding its properties
"""
# %%
import numpy as np
import neurotechdevkit as ndk
from neurotechdevkit.sources import FocusedSource2D
from neurotechdevkit.scenarios import materials
from neurotechdevkit.scenarios import Target
from neurotechdevkit.scenarios import (
    create_grid_circular_mask,
    create_grid_elliptical_mask,
    make_grid,
)


def _create_masks(grid, origin):
    outer_skull_mask = _create_skull_interface_mask(grid, origin)
    outer_brain_mask = _create_brain_interface_mask(grid, origin)
    water_mask = ~outer_skull_mask
    tumor_mask = _create_tumor_mask(grid, origin)
    brain_mask = outer_brain_mask & ~tumor_mask
    masks = {
        "water": water_mask,
        "skull": outer_skull_mask & ~outer_brain_mask,
        "brain": brain_mask,
        "tumor": tumor_mask,
    }
    return masks


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


scenario = ndk.scenarios.ProceduralScenario()

origin = np.array([0.0, -0.02])
scenario.add_origin(origin)

grid = make_grid(extent=np.array([0.05, 0.04]), dx=0.0005)
scenario.add_grid(grid)

scenario.add_target(
    Target(
        target_id="target_1",
        center=np.array([0.0285, 0.0024]),
        radius=0.0017,
        description="Represents a simulated tumor.",
    )
)
scenario.add_source(
    FocusedSource2D(
        position=np.array([0.01, 0.0]),
        direction=np.array([1.0, 0.0]),
        aperture=0.01,
        focal_length=0.01,
        num_points=1000,
    )
)
scenario.add_material_layers(
    [
        ("water", materials.water),
        ("skull", materials.cortical_bone),
        ("brain", materials.brain),
        ("tumor", materials.tumor),
    ]
)
scenario.add_material_masks(_create_masks(grid, origin))

result = scenario.simulate_steady_state()

result.render_steady_state_amplitudes(show_material_outlines=False)

# %%
result = scenario.simulate_pulse()
result.render_pulsed_simulation_animation()
# %%
