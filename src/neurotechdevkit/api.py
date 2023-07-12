"""New API for the NeuroTech DevKit."""

import pathlib

import hdf5storage
import numpy as np

import neurotechdevkit as ndk
from neurotechdevkit.scenarios import Target, ct_loader, make_grid, materials
from neurotechdevkit.sources import FocusedSource2D


def _create_masks():
    cur_dir = pathlib.Path(__file__).parent
    data_file = cur_dir / "scenarios" / "data" / "skull_mask_bm8_dx_0.5mm.mat"
    mat_data = hdf5storage.loadmat(str(data_file))

    skull_mask = mat_data["skull_mask"].astype(np.bool_)
    brain_mask = mat_data["brain_mask"].astype(np.bool_)

    masks = {
        "water": ~(skull_mask | brain_mask)[:, :, 185],
        "skull": skull_mask[:, :, 185],
        "brain": brain_mask[:, :, 185],
    }
    return masks


def create_scenario(all_params):
    scenario = ndk.scenarios.ProceduralScenario()

    origin = np.array([float(all_params["origin_y"]), float(all_params["origin_x"])])
    scenario.add_origin(origin)

    grid = make_grid(
        extent=np.array([float(all_params["extent_y"]), float(all_params["extent_x"])]),
        dx=float(all_params["dx"]),
    )
    scenario.add_grid(grid)
    scenario.add_target(
        Target(
            target_id="target_1",
            center=np.array(
                [float(all_params["center_y"]), float(all_params["center_x"])]
            ),
            radius=float(all_params["radius"]),
            description="Represents a simulated tumor.",
        )
    )
    scenario.add_source(
        FocusedSource2D(
            position=np.array(
                [float(all_params["position_y"]), float(all_params["position_x"])]
            ),
            direction=np.array(
                [float(all_params["direction_y"]), float(all_params["direction_x"])]
            ),
            aperture=float(all_params["aperture"]),
            focal_length=float(all_params["focal_length"]),
            num_points=int(all_params["num_points"]),
        )
    )
    scenario.add_material_layers(
        [
            ("water", materials.water),
            ("skull", materials.cortical_bone),
            ("brain", materials.brain),
        ]
    )
    if all_params.get("ct_file"):
        brain_mask, skull_mask = ct_loader.get_masks(all_params["ct_file"])
        masks = {
            "water": ~(skull_mask | brain_mask),
            "skull": skull_mask,
            "brain": brain_mask,
        }
        scenario.add_material_masks(masks)
    else:
        scenario.add_material_masks(_create_masks())

    scenario.create_problem()
    return scenario
