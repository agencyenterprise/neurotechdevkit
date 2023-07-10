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
    make_grid,
)
import hdf5storage


def _create_masks(convert_2d=False):
    data_file = "/Users/newtonsander/workspace/bci/neurotechdevkit/src/neurotechdevkit/scenarios/data/skull_mask_bm8_dx_0.5mm.mat"
    mat_data = hdf5storage.loadmat(str(data_file))

    skull_mask = mat_data["skull_mask"].astype(np.bool_)
    brain_mask = mat_data["brain_mask"].astype(np.bool_)

    masks = {
        "skull": skull_mask,
        "brain": brain_mask,
        "water": ~(skull_mask | brain_mask),
    }
    if convert_2d:
        for key, mask in masks.items():
            masks[key] = mask[:, :, 185]
    return masks


scenario = ndk.scenarios.ProceduralScenario(material_outline_upsample_factor=4)

origin = np.array([0.0, -0.085])
scenario.add_origin(origin)

grid = make_grid(extent=np.array([0.225, 0.170]), dx=0.0005)
scenario.add_grid(grid)

scenario.add_target(
    Target(
        target_id="primary-visual-cortex",
        center=np.array([0.047, 0.002]),
        radius=0.010,
        description=(
            "A region of the primary visual cortex (approximate 2D location and"
            " size). Studies suggest that transcranial focused ultrasound"
            " stimulation of this brain region can lead to phosphene perception."
            " See for more details: https://doi.org/10.1038/srep34026"
        ),
    )
)
scenario.add_source(
    FocusedSource2D(
        position=np.array([0.0, 0.0]),
        direction=np.array([1.0, 0.0]),
        aperture=0.064,
        focal_length=0.064,
        num_points=1000,
    )
)
scenario.add_material_layers(
    [
        ("water", materials.water),
        ("skull", materials.cortical_bone),
        ("brain", materials.brain),
    ]
)
scenario.add_material_masks(_create_masks(convert_2d=True))

result = scenario.simulate_steady_state()

result.render_steady_state_amplitudes(show_material_outlines=False)

# %%
result = scenario.simulate_pulse()
result.render_pulsed_simulation_animation()
# %%
