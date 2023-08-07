import numpy as np

from neurotechdevkit.grid import Grid
from neurotechdevkit.materials import get_material
from neurotechdevkit.problem import Problem


def test_add_material_fields():
    """Validates we are able to add material fields to the problem"""
    grid = Grid.make_shaped_grid(shape=(2, 2), spacing=0.0005)
    problem = Problem(grid=grid)
    water = get_material("water")
    brain = get_material("brain")
    problem.add_material_fields(
        materials={
            "water": water.to_struct(),
            "brain": brain.to_struct(),
        },
        masks={
            "water": np.array([[0, 1], [1, 0]], dtype=bool),
            "brain": np.array([[1, 0], [0, 1]], dtype=bool),
        },
    )
    assert np.array_equal(
        problem.medium.vp.data,
        np.array([[brain.vp, water.vp], [water.vp, brain.vp]], dtype=np.float32),
    )
    assert np.array_equal(
        problem.medium.rho.data,
        np.array([[brain.rho, water.rho], [water.rho, brain.rho]], dtype=np.float32),
    )
    assert np.array_equal(
        problem.medium.alpha.data,
        np.array(
            [[brain.alpha, water.alpha], [water.alpha, brain.alpha]], dtype=np.float32
        ),
    )
