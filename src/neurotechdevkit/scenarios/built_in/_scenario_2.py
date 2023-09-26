from __future__ import annotations

import enum
import pathlib
from typing import Mapping, Tuple, Union

import hdf5storage
import numpy as np
import numpy.typing as npt
import scipy.ndimage

from ... import rendering, sources
from ...grid import Grid
from ...materials import Material
from .._base import Scenario, Scenario2D, Scenario3D
from .._utils import SliceAxis, Target


@enum.unique
class BenchmarkSkullMaskFile(enum.Enum):
    """Aliases for the skull mask filename for each Scenario2 benchmark."""

    BENCHMARK_7 = "skull_mask_bm7_dx_0.5mm.mat"
    BENCHMARK_8 = "skull_mask_bm8_dx_0.5mm.mat"


class BenchmarkExtent(enum.Enum):
    """Aliases for the extent for each Scenario2 benchmark."""

    BENCHMARK_7 = (0.120, 0.070, 0.070)  # m
    BENCHMARK_8 = (0.225, 0.170, 0.190)  # m


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

    center_frequency = 5e5  # Hz
    material_properties = {
        "water": Material(vp=1500.0, rho=1000.0, alpha=0.0, render_color="#2E86AB"),
        "cortical_bone": Material(
            vp=2800.0, rho=1850.0, alpha=4.0, render_color="#FAF0CA"
        ),
        "brain": Material(vp=1560.0, rho=1040.0, alpha=0.3, render_color="#DB504A"),
    }

    def _make_grid(
        self, extent: Union[Tuple[float, float], Tuple[float, float, float]]
    ) -> Grid:
        grid = Grid.make_grid(
            extent=extent,
            speed_water=1500,  # m/s
            ppw=6,  # desired resolution for complexity=fast
            center_frequency=self.center_frequency,
        )
        return grid

    def _make_material_masks(
        self,
        mask_file_name: str = BenchmarkSkullMaskFile.BENCHMARK_8.value,
        convert_2d: bool = False,
    ) -> Mapping[str, npt.NDArray[np.bool_]]:
        """Make the material masks for scenario 2."""
        material_layers = [
            "water",
            "cortical_bone",
            "brain",
        ]
        assert self.grid.space is not None, "Grid-space must be created first."
        material_masks = {
            name: _create_scenario_2_mask(
                name,
                output_shape=self.grid.space.shape,
                mask_file_name=mask_file_name,
                convert_2d=convert_2d,
            )
            for name in material_layers
        }
        return material_masks


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

    PREDEFINED_TARGET_OPTIONS = {
        "primary-visual-cortex": Target(
            target_id="primary-visual-cortex",
            center=[0.047, 0.002],
            radius=0.010,
            description=(
                "A region of the primary visual cortex (approximate 2D location and"
                " size). Studies suggest that transcranial focused ultrasound"
                " stimulation of this brain region can lead to phosphene perception."
                " See for more details: https://doi.org/10.1038/srep34026"
            ),
        ),
        "right-inferior-frontal-gyrus": Target(
            target_id="right-inferior-frontal-gyrus",
            center=[0.175, 0.048],
            radius=0.008,
            description=(
                "The right inferior frontal gyrus (approximate 2D location and size)."
                " Studies suggest that transcranial focused ultrasound stimulation of"
                " this region can be used to improve mood."
                " See for more details: https://doi.org/10.3389/fnhum.2020.00052"
            ),
        ),
        "posterior-cingulate-cortex": Target(
            target_id="posterior-cingulate-cortex",
            center=[0.10, 0.005],
            radius=0.01,
            description=(
                "The posterior cingulate cortex (approximate 2D location and size)."
                " Studies suggest this brain region could be linked with meditation"
                " and mindfulness."
                " See for more details: https://doi.org/10.1111/nyas.12246"
            ),
        ),
    }
    target = Target(
        target_id="primary-visual-cortex",
        center=[0.047, 0.002],
        radius=0.010,
        description=(
            "A region of the primary visual cortex (approximate 2D location and"
            " size). Studies suggest that transcranial focused ultrasound"
            " stimulation of this brain region can lead to phosphene perception."
            " See for more details: https://doi.org/10.1038/srep34026"
        ),
    )
    origin = [0.0, -0.085]
    sources = [
        sources.FocusedSource2D(
            position=[0.0, 0.0],
            direction=[1.0, 0.0],
            aperture=0.064,
            focal_length=0.064,
            num_points=1000,
        )
    ]
    material_outline_upsample_factor = 4

    def make_grid(self):
        """Make the grid for scenario 2 2D."""
        self.grid = self._make_grid(BenchmarkExtent.BENCHMARK_8.value[:2])
        self.material_masks = self._make_material_masks(
            mask_file_name=BenchmarkSkullMaskFile.BENCHMARK_8.value, convert_2d=True
        )


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

    PREDEFINED_TARGET_OPTIONS = {
        "primary-visual-cortex": Target(
            target_id="primary-visual-cortex",
            center=[0.047, 0.002, 0.005],
            radius=0.010,
            description=(
                "A region of the primary visual cortex (estimated location and size)."
                " Studies suggest that transcranial focused ultrasound stimulation of"
                " this brain region can lead to phosphene perception."
                " See for more details: https://doi.org/10.1038/srep34026"
            ),
        ),
        "right-inferior-frontal-gyrus": Target(
            target_id="right-inferior-frontal-gyrus",
            center=[0.175, 0.048, -0.010],
            radius=0.008,
            description=(
                "The right inferior frontal gyrus (estimated location and size)."
                " Studies suggest that transcranial focused ultrasound stimulation of"
                " this region can be used to improve mood."
                " See for more details: https://doi.org/10.3389/fnhum.2020.00052"
            ),
        ),
        "posterior-cingulate-cortex": Target(
            target_id="posterior-cingulate-cortex",
            center=[0.10, 0.005, 0.020],
            radius=0.01,
            description=(
                "The posterior cingulate cortex (estimated location and size). Studies"
                " suggest this brain region could be linked with meditation and"
                " mindfulness."
                " See for more details: https://doi.org/10.1111/nyas.12246"
            ),
        ),
        "ventral-intermediate-nucleus": Target(
            target_id="ventral-intermediate-nucleus",
            center=[0.12, 0.01, -0.015],
            radius=0.005,
            description=(
                "The right ventral intermediate nucleus of the thalamus (estimated"
                " location and size). Studies are being conducted on the use of"
                " transcranial ultrasound therapy to help treat essential tremor."
                " See for more details:"
                " https://clinicaltrials.gov/ct2/show/NCT04074031"
            ),
        ),
        "left-temporal-lobe": Target(
            target_id="left-temporal-lobe",
            center=[0.14, -0.03, -0.03],
            radius=0.015,
            description=(
                "The left temporal lobe (estimated location and size). Studies are"
                " being conducted on the use of low-intensity ultrasound targeting the"
                " temporal lobe to help treat drug-resistant epilepsy."
                " See for more details:"
                " https://clinicaltrials.gov/ct2/show/NCT03868293"
            ),
        ),
    }

    target = Target(
        target_id="primary-visual-cortex",
        center=[0.047, 0.002, 0.005],
        radius=0.010,
        description=(
            "A region of the primary visual cortex (estimated location and size)."
            " Studies suggest that transcranial focused ultrasound stimulation of"
            " this brain region can lead to phosphene perception."
            " See for more details: https://doi.org/10.1038/srep34026"
        ),
    )

    origin = [0.0, -0.085, -0.095]
    sources = [
        sources.FocusedSource3D(
            position=[0.0, 0.0, 0.0],
            direction=[1.0, 0.0, 0.0],
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
    slice_axis = SliceAxis.Z
    slice_position = 0.0
    material_outline_upsample_factor = 4

    def make_grid(self):
        """Make the grid for scenario 2 3D."""
        self.grid = self._make_grid(BenchmarkExtent.BENCHMARK_8.value)
        self.material_masks = self._make_material_masks(
            mask_file_name=BenchmarkSkullMaskFile.BENCHMARK_8.value, convert_2d=False
        )


class Scenario2_2D_Benchmark7(Scenario2_2D):
    """An adaptation of scenario 2 2D that uses the benchmark 7 sub-extent.

    From Aubry et al. (2022):
        > Benchmark 7... uses a subset of the skull mask and the
        > same relative transducer position as benchmark 8, with a reduced
        > comparison domain size
    """

    # origin at [top, center] of extent
    origin = [0.0, -0.035]  # m

    def make_grid(self):
        """Make the grid for 2-D version of Benchmark 7."""
        self.grid = self._make_grid(BenchmarkExtent.BENCHMARK_7.value[:2])
        self.material_masks = self._make_material_masks(
            mask_file_name=BenchmarkSkullMaskFile.BENCHMARK_7.value, convert_2d=True
        )


class Scenario2_3D_Benchmark7(Scenario2_3D):
    """An adaptation of scenario 2 3D that uses the benchmark 7 sub-extent.

    From Aubry et al. (2022):
        > Benchmark 7... uses a subset of the skull mask and the
        > same relative transducer position as benchmark 8, with a reduced
        > comparison domain size
    """

    # origin at [top, center, center] of extent
    origin = [0.0, -0.035, -0.035]  # m

    def make_grid(self):
        """Make the grid for Benchmark 7."""
        self.grid = self._make_grid(BenchmarkExtent.BENCHMARK_7.value)
        self.material_masks = self._make_material_masks(
            mask_file_name=BenchmarkSkullMaskFile.BENCHMARK_7.value, convert_2d=False
        )


def _create_scenario_2_mask(
    material: str,
    output_shape: Tuple[int],
    mask_file_name: str = BenchmarkSkullMaskFile.BENCHMARK_8.value,  # m
    convert_2d: bool = False,
) -> npt.NDArray[np.bool_]:
    """Create material mask for scenario 2.

    Args:
        material: name of the material to create the mask for
        output_shape: shape of the simulation grid
        mask_file_name: name of the mask file in the data directory
        convert_2d: whether to take a 2D slice of the 3D mask

    Returns:
        boolean mask array indicating which voxels are of the given material
    """
    cur_dir = pathlib.Path(__file__).parent
    mask_file_abs_path = cur_dir.joinpath("data", mask_file_name)
    mat_data = hdf5storage.loadmat(str(mask_file_abs_path))

    skull_mask = mat_data["skull_mask"].astype(np.bool_)
    brain_mask = mat_data["brain_mask"].astype(np.bool_)
    assert skull_mask.shape == brain_mask.shape

    if convert_2d:
        # slice through the center of the transducer
        z_center_idx = skull_mask.shape[2] // 2
        if mask_file_name == BenchmarkSkullMaskFile.BENCHMARK_7.value:
            assert z_center_idx == 70
        elif mask_file_name == BenchmarkSkullMaskFile.BENCHMARK_8.value:
            assert z_center_idx == 190
        skull_mask = skull_mask[:, :, z_center_idx]
        brain_mask = brain_mask[:, :, z_center_idx]

    if skull_mask.shape != output_shape:
        # resample the mask to match the grid
        scale_factor = np.array(output_shape) / np.array(skull_mask.shape)
        skull_mask = scipy.ndimage.zoom(skull_mask, scale_factor, order=0)
        brain_mask = scipy.ndimage.zoom(brain_mask, scale_factor, order=0)
        # The resampling could have introduced some overlap between the masks.
        # Ensure that each voxel is only assigned to one mask.
        brain_mask &= ~skull_mask

    if material == "cortical_bone":
        mask = skull_mask

    elif material == "brain":
        mask = brain_mask

    elif material == "water":
        mask = ~(skull_mask | brain_mask)

    else:
        raise ValueError(material)

    return mask
