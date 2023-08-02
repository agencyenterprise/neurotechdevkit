from __future__ import annotations

import pathlib
from typing import Mapping

import hdf5storage
import numpy as np
import numpy.typing as npt

from .. import rendering, sources
from ..grid import Grid
from ..materials import Material
from ..problem import Problem
from ._base import Scenario, Scenario2D, Scenario3D, Target


class Scenario2(Scenario):
    """Specific implementation detail for scenario 2.

    Scenario 2 is based on benchmark 8 of the following paper:

        Jean-Francois Aubry, Oscar Bates, Christian Boehm, et al., "Benchmark problems
        for transcranial ultrasound simulation: Intercomparison of compressional wave
        models",
        The Journal of the Acoustical Society of America 152, 1003 (2022);
        doi: 10.1121/10.0013426
        https://asa.scitation.org/doi/pdf/10.1121/10.0013426
    """

    material_layers = [
        "water",
        "cortical_bone",
        "brain",
    ]
    material_properties = {
        "water": Material(vp=1500.0, rho=1000.0, alpha=0.0, render_color="#2E86AB"),
        "cortical_bone": Material(
            vp=2800.0, rho=1850.0, alpha=4.0, render_color="#FAF0CA"
        ),
        "brain": Material(vp=1560.0, rho=1040.0, alpha=0.3, render_color="#DB504A"),
    }

    def _make_grid(self, extent: npt.NDArray[np.float_]) -> Grid:
        # scenario constants
        speed_water = 1500  # m/s

        # desired resolution for complexity=fast
        ppw = 6
        # compute resolution
        dx = speed_water / self.center_frequency / ppw  # m

        grid = Grid.make_grid(extent=extent, dx=dx)
        return grid

    def _make_material_masks(
        self, convert_2d: bool
    ) -> Mapping[str, npt.NDArray[np.bool_]]:
        """Make the material masks for scenario 2."""
        material_masks = {
            name: _create_scenario_2_mask(name, convert_2d=convert_2d)
            for name in self.material_layers
        }
        return material_masks

    def compile_problem(self) -> Problem:
        """
        Compile the problem for scenario 2.

        Returns:
            Problem: the compiled problem
        """
        assert self.grid is not None
        assert self.layer_ids is not None
        assert self.material_masks is not None

        self.problem = Problem(center_frequency=self.center_frequency, grid=self.grid)
        self.problem.add_material_fields(
            materials=self.materials,
            layer_ids=self.layer_ids,
            masks=self.material_masks,
        )

        return self.problem


class Scenario2_2D(Scenario2D, Scenario2):
    """A 2D implementation of scenario 2.

    Scenario 2 is based on benchmark 8 of the following paper:

        Jean-Francois Aubry, Oscar Bates, Christian Boehm, et al., "Benchmark problems
        for transcranial ultrasound simulation: Intercomparison of compressional wave
        models",
        The Journal of the Acoustical Society of America 152, 1003 (2022);
        doi: 10.1121/10.0013426
        https://asa.scitation.org/doi/pdf/10.1121/10.0013426
    """

    target = Target(
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
    origin = np.array([0.0, -0.085])
    sources = [
        sources.FocusedSource2D(
            position=np.array([0.0, 0.0]),
            direction=np.array([1.0, 0.0]),
            aperture=0.064,
            focal_length=0.064,
            num_points=1000,
        )
    ]
    material_outline_upsample_factor = 4

    def make_grid(self):
        """Make the grid for scenario 2 2D."""
        self.grid = self._make_grid(np.array([0.225, 0.170]))
        self.material_masks = self._make_material_masks(convert_2d=True)


class Scenario2_3D(Scenario2, Scenario3D):
    """A 3D implementation of scenario 2.

    Scenario 2 is based on benchmark 8 of the following paper:

        Jean-Francois Aubry, Oscar Bates, Christian Boehm, et al., "Benchmark problems
        for transcranial ultrasound simulation: Intercomparison of compressional wave
        models",
        The Journal of the Acoustical Society of America 152, 1003 (2022);
        doi: 10.1121/10.0013426
        https://asa.scitation.org/doi/pdf/10.1121/10.0013426
    """

    target = Target(
        target_id="primary-visual-cortex",
        center=np.array([0.047, 0.002, 0.005]),
        radius=0.010,
        description=(
            "A region of the primary visual cortex (estimated location and size)."
            " Studies suggest that transcranial focused ultrasound stimulation of"
            " this brain region can lead to phosphene perception."
            " See for more details: https://doi.org/10.1038/srep34026"
        ),
    )

    origin = np.array([0.0, -0.085, -0.095])
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
        init_angles=(90, 10, -60),
        init_zoom=2.0,
        colormaps={
            "water": "blue",
            "cortical_bone": "magma",
            "brain": "bop orange",
        },
        opacities={
            "water": 0.8,
            "cortical_bone": 0.2,
            "brain": 0.2,
        },
    )
    slice_axis = 2
    slice_position = 0.0
    material_outline_upsample_factor = 4

    def make_grid(self):
        """Make the grid for scenario 2 3D."""
        self.grid = self._make_grid(np.array([0.225, 0.170, 0.190]))
        self.material_masks = self._make_material_masks(convert_2d=False)


def _create_scenario_2_mask(material, convert_2d=False) -> npt.NDArray[np.bool_]:

    cur_dir = pathlib.Path(__file__).parent
    data_file = cur_dir / "data" / "skull_mask_bm8_dx_0.5mm.mat"
    mat_data = hdf5storage.loadmat(str(data_file))

    skull_mask = mat_data["skull_mask"].astype(np.bool_)
    brain_mask = mat_data["brain_mask"].astype(np.bool_)

    if material == "cortical_bone":
        mask = skull_mask

    elif material == "brain":
        mask = brain_mask

    elif material == "water":
        mask = ~(skull_mask | brain_mask)

    else:
        raise ValueError(material)

    if convert_2d:
        mask = mask[:, :, 185]
        # slice through the center of the transducer

    return mask
