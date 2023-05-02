from __future__ import annotations

import pathlib
from typing import Mapping, Protocol

import hdf5storage
import numpy as np
import numpy.typing as npt
import stride
from mosaic.types import Struct

from .. import rendering, sources
from . import materials
from ._base import Scenario2D, Scenario3D, Target
from ._utils import add_material_fields_to_problem, make_grid


class _Scenario2MixinProtocol(Protocol):
    """Provide type-hinting for Scenario 2 members used by mixins"""

    @property
    def scenario_id(self) -> str:
        ...

    @property
    def complexity(self) -> str:
        ...

    @property
    def materials(self) -> Mapping[str, Struct]:
        ...

    @property
    def layer_ids(self) -> Mapping[str, int]:
        ...

    def _get_material_masks(self) -> Mapping[str, npt.NDArray[np.bool_]]:
        ...


class _Scenario2Mixin:
    """A mixin providing specific implementation detail for scenario 2.

    Scenario 2 is based on benchmark 8 of the following paper:

    Jean-Francois Aubry, Oscar Bates, Christian Boehm, et al., "Benchmark problems for
    transcranial ultrasound simulation: Intercomparison of compressional wave models",
    The Journal of the Acoustical Society of America 152, 1003 (2022);
    doi: 10.1121/10.0013426
    https://asa.scitation.org/doi/pdf/10.1121/10.0013426
    """

    @property
    def _material_layers(self: _Scenario2MixinProtocol) -> list[tuple[str, Struct]]:
        return [
            ("water", materials.water),
            ("skull", materials.cortical_bone),
            ("brain", materials.brain),
        ]

    @property
    def _material_outline_upsample_factor(self) -> int:
        return 4

    def _compile_scenario_2_problem(
        self: _Scenario2MixinProtocol, extent: npt.NDArray[np.float_]
    ) -> stride.Problem:
        # scenario constants
        speed_water = 1500  # m/s
        c_freq = 500e3  # hz

        # desired resolution for complexity=fast
        ppw = 6

        # compute resolution
        dx = speed_water / c_freq / ppw  # m

        grid = make_grid(extent=extent, dx=dx)
        problem = stride.Problem(
            name=f"{self.scenario_id}-{self.complexity}", grid=grid
        )
        problem = add_material_fields_to_problem(
            problem=problem,
            materials=self.materials,
            layer_ids=self.layer_ids,
            masks=self._get_material_masks(),
        )
        return problem


class Scenario2_2D(_Scenario2Mixin, Scenario2D):
    """A 2D implementation of scenario 2.

    Scenario 2 is based on benchmark 8 of the following paper:

    Jean-Francois Aubry, Oscar Bates, Christian Boehm, et al., "Benchmark problems for
    transcranial ultrasound simulation: Intercomparison of compressional wave models",
    The Journal of the Acoustical Society of America 152, 1003 (2022);
    doi: 10.1121/10.0013426
    https://asa.scitation.org/doi/pdf/10.1121/10.0013426
    """

    _SCENARIO_ID = "scenario-2-2d-v0"
    _TARGET_OPTIONS = {
        "primary-visual-cortex": Target(
            target_id="primary-visual-cortex",
            center=np.array([0.047, 0.002]),
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
            center=np.array([0.175, 0.048]),
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
            center=np.array([0.10, 0.005]),
            radius=0.01,
            description=(
                "The posterior cingulate cortex (approximate 2D location and size)."
                " Studies suggest this brain region could be linked with meditation"
                " and mindfulness."
                " See for more details: https://doi.org/10.1111/nyas.12246"
            ),
        ),
    }

    def __init__(self, complexity="fast"):
        self._target_id = "primary-visual-cortex"

        super().__init__(
            origin=np.array([0.0, -0.085]),
            complexity=complexity,
        )

    def render_material_property(
        self, name, show_orientation=True, show_sources=True, show_target=True
    ):
        raise NotImplementedError()

    def _compile_problem(self) -> stride.Problem:
        extent = np.array([0.225, 0.170])  # m
        return self._compile_scenario_2_problem(extent)

    def _get_material_masks(self) -> Mapping[str, npt.NDArray[np.bool_]]:
        return {
            name: _create_scenario_2_mask(name, convert_2d=True)
            for name in self.materials.keys()
        }

    def get_default_source(self) -> sources.Source:
        return sources.FocusedSource2D(
            position=np.array([0.0, 0.0]),
            direction=np.array([1.0, 0.0]),
            aperture=0.064,
            focal_length=0.064,
            num_points=1000,
        )


class Scenario2_3D(_Scenario2Mixin, Scenario3D):
    """A 3D implementation of scenario 2.

    Scenario 2 is based on benchmark 8 of the following paper:

    Jean-Francois Aubry, Oscar Bates, Christian Boehm, et al., "Benchmark problems for
    transcranial ultrasound simulation: Intercomparison of compressional wave models",
    The Journal of the Acoustical Society of America 152, 1003 (2022);
    doi: 10.1121/10.0013426
    https://asa.scitation.org/doi/pdf/10.1121/10.0013426
    """

    _SCENARIO_ID = "scenario-2-3d-v0"
    _TARGET_OPTIONS = {
        "primary-visual-cortex": Target(
            target_id="primary-visual-cortex",
            center=np.array([0.047, 0.002, 0.005]),
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
            center=np.array([0.175, 0.048, -0.010]),
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
            center=np.array([0.10, 0.005, 0.020]),
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
            center=np.array([0.12, 0.01, -0.015]),
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
            center=np.array([0.14, -0.03, -0.03]),
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

    def __init__(self, complexity="fast"):
        self._target_id = "primary-visual-cortex"

        super().__init__(
            origin=np.array([0.0, -0.085, -0.095]),
            complexity=complexity,
        )

    @property
    def viewer_config_3d(self) -> rendering.ViewerConfig3D:
        return rendering.ViewerConfig3D(
            init_angles=(90, 10, -60),
            init_zoom=2.0,
            colormaps={
                "water": "blue",
                "skull": "magma",
                "brain": "bop orange",
            },
            opacities={
                "water": 0.8,
                "skull": 0.2,
                "brain": 0.2,
            },
        )

    def get_default_slice_axis(self) -> int:
        return 2

    def get_default_slice_position(self, axis: int) -> float:
        default_positions = np.array([0.1, 0.0, 0.0])
        return default_positions[axis]

    def render_material_property(
        self, name, show_orientation=True, show_sources=True, show_target=True
    ):
        raise NotImplementedError()

    def _compile_problem(self) -> stride.Problem:
        extent = np.array([0.225, 0.170, 0.190])  # m
        return self._compile_scenario_2_problem(extent)

    def _get_material_masks(self):
        return {
            name: _create_scenario_2_mask(name, convert_2d=False)
            for name in self.materials.keys()
        }

    def get_default_source(self):
        return sources.FocusedSource3D(
            position=np.array([0.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            aperture=0.064,
            focal_length=0.064,
            num_points=20_000,
        )


def _create_scenario_2_mask(material, convert_2d=False) -> npt.NDArray[np.bool_]:

    cur_dir = pathlib.Path(__file__).parent
    data_file = cur_dir / "data" / "skull_mask_bm8_dx_0.5mm.mat"
    mat_data = hdf5storage.loadmat(str(data_file))

    skull_mask = mat_data["skull_mask"].astype(np.bool_)
    brain_mask = mat_data["brain_mask"].astype(np.bool_)

    if material == "skull":
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
