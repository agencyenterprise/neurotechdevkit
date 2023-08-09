import numpy as np
import pytest

from neurotechdevkit.grid import Grid
from neurotechdevkit.materials import get_material
from neurotechdevkit.problem import Problem
from neurotechdevkit.scenarios import Scenario2D, Target
from neurotechdevkit.sources import FocusedSource2D


def _create_mask(material, grid):
    def fill_mask(mask, start, end, dx):
        # fill linearly along the x axis
        if end is None:
            n = int(start / dx)
            mask[n:] = True
        else:
            n = int(start / dx)
            m = int(end / dx)
            mask[n:m] = True

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


@pytest.mark.integration
def test_scenario_with_all_parameters():
    """Test that the scenario can be created/executed with all needed parameters."""
    center_frequency = 5e5
    target = Target(
        target_id="target_1", center=[0.064, 0.0], radius=0.004, description=""
    )
    sources = [
        FocusedSource2D(
            position=[0.0, 0.0],
            direction=[1.0, 0.0],
            aperture=0.064,
            focal_length=0.064,
            num_points=1000,
        )
    ]
    grid = Grid.make_grid(
        extent=(0.12, 0.07),
        speed_water=1500,
        center_frequency=center_frequency,
        ppw=6,
    )

    materials = {}
    material_masks = {}
    for layer in [
        "water",
        "skin",
        "cortical_bone",
        "trabecular_bone",
        "brain",
    ]:
        materials[layer] = get_material(layer, center_frequency).to_struct()
        material_masks[layer] = _create_mask(layer, grid)

    problem = Problem(grid=grid)
    problem.add_material_fields(
        materials=materials,
        masks=material_masks,
    )
    scenario = Scenario2D(
        center_frequency=center_frequency,
        material_masks=material_masks,
        material_properties={},
        material_outline_upsample_factor=8,
        origin=[0.0, -0.035],
        problem=problem,
        sources=sources,
        target=target,
        grid=grid,
    )

    result = scenario.simulate_steady_state()
    assert result.wavefield.shape == (241, 141, 59)
