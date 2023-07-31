from __future__ import annotations

from typing import Mapping

import numpy as np
import numpy.typing as npt

from .. import rendering, sources
from ..materials import Material
from ..problem import Problem
from ._base import Scenario, Scenario2D, Scenario3D, Target
from ._utils import make_grid


class Scenario1(Scenario):
    """Specific implementation detail for scenario 1.

    Scenario 1 is based on benchmark 4 of the following paper:

        Jean-Francois Aubry, Oscar Bates, Christian Boehm, et al., "Benchmark problems
        for transcranial ultrasound simulation: Intercomparison of compressional wave
        models",
        The Journal of the Acoustical Society of America 152, 1003 (2022);
        doi: 10.1121/10.0013426
        https://asa.scitation.org/doi/pdf/10.1121/10.0013426
    """

    material_layers = [
        "water",
        "skin",
        "cortical_bone",
        "trabecular_bone",
        "brain",
    ]
    material_properties = {
        "water": Material(vp=1500.0, rho=1000.0, alpha=0.0, render_color="#2E86AB"),
        "skin": Material(vp=1610.0, rho=1090.0, alpha=0.2, render_color="#FA8B53"),
        "cortical_bone": Material(
            vp=2800.0, rho=1850.0, alpha=4.0, render_color="#FAF0CA"
        ),
        "trabecular_bone": Material(
            vp=2300.0, rho=1700.0, alpha=8.0, render_color="#EBD378"
        ),
        "brain": Material(vp=1560.0, rho=1040.0, alpha=0.3, render_color="#DB504A"),
    }

    def _get_material_masks(
        self, problem: Problem
    ) -> Mapping[str, npt.NDArray[np.bool_]]:
        return {
            name: _create_scenario_1_mask(name, problem.grid)
            for name in self.material_layers
        }

    def _compile_scenario_1_problem(
        self, extent: npt.NDArray[np.float_], center_frequency: float
    ) -> Problem:
        # scenario constants
        speed_water = 1500  # m/s

        # desired resolution for complexity=fast
        ppw = 6

        # compute resolution
        dx = speed_water / center_frequency / ppw  # m

        grid = make_grid(extent=extent, dx=dx)
        problem = Problem(center_frequency=center_frequency, grid=grid)
        problem.add_material_fields(
            materials=self.get_materials(center_frequency),
            layer_ids=self.layer_ids,
            masks=self._get_material_masks(problem),
        )
        return problem


class Scenario1_2D(Scenario1, Scenario2D):
    """A 2D implementation of scenario 1.

    Scenario 1 is based on benchmark 4 of the following paper:

        Jean-Francois Aubry, Oscar Bates, Christian Boehm, et al., "Benchmark problems
        for transcranial ultrasound simulation: Intercomparison of compressional wave
        models",
        The Journal of the Acoustical Society of America 152, 1003 (2022);
        doi: 10.1121/10.0013426
        https://asa.scitation.org/doi/pdf/10.1121/10.0013426
    """

    scenario_id = "scenario-1-2d-v0"
    target = Target("target_1", np.array([0.064, 0.0]), 0.004, "")
    sources = [
        sources.FocusedSource2D(
            position=np.array([0.0, 0.0]),
            direction=np.array([1.0, 0.0]),
            aperture=0.064,
            focal_length=0.064,
            num_points=1000,
        )
    ]
    origin = np.array([0.0, -0.035])
    material_outline_upsample_factor = 8

    def compile_problem(self, center_frequency: float) -> Problem:
        """
        Compile the problem for scenario 1.

        Args:
            center_frequency (float): the center frequency of the transducer

        Returns:
            Problem: the compiled problem
        """
        extent = np.array([0.12, 0.07])  # m
        self.problem = self._compile_scenario_1_problem(extent, center_frequency)
        return self.problem


class Scenario1_3D(Scenario1, Scenario3D):
    """A 3D implementation of scenario 1.

    Scenario 1 is based on benchmark 4 of the following paper:

        Jean-Francois Aubry, Oscar Bates, Christian Boehm, et al., "Benchmark problems
        for transcranial ultrasound simulation: Intercomparison of compressional wave
        models",
        The Journal of the Acoustical Society of America 152, 1003 (2022);
        doi: 10.1121/10.0013426
        https://asa.scitation.org/doi/pdf/10.1121/10.0013426
    """

    scenario_id = "scenario-1-3d-v0"
    target = Target(
        target_id="target_1",
        center=np.array([0.064, 0.0, 0.0]),
        radius=0.004,
        description=(
            "A centered location below the skull at approximately the focal point."
        ),
    )
    origin = np.array([0.0, -0.035, -0.035])
    sources = [
        sources.FocusedSource3D(
            position=np.array([0.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            aperture=0.064,
            focal_length=0.064,
            num_points=20_000,
        )
    ]
    viewer_config_3d = rendering.ViewerConfig3D(
        init_angles=(-15, 45, 160),
        init_zoom=3.0,
        colormaps={
            "water": "blue",
            "skin": "viridis",
            "cortical_bone": "magma",
            "trabecular_bone": "inferno",
            "brain": "bop orange",
        },
        opacities={
            "water": 0.8,
            "skin": 0.2,
            "cortical_bone": 0.2,
            "trabecular_bone": 0.2,
            "brain": 0.4,
        },
    )
    slice_axis = 1
    slice_position = 0.0
    material_outline_upsample_factor = 8

    def compile_problem(self, center_frequency: float) -> Problem:
        """
        Compile the problem for scenario 1.

        Args:
            center_frequency (float): the center frequency of the transducer

        Returns:
            Problem: the compiled problem
        """
        extent = np.array([0.12, 0.07, 0.07])  # m
        self.problem = self._compile_scenario_1_problem(extent, center_frequency)
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
