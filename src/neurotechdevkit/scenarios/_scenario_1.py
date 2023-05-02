from __future__ import annotations

from typing import Mapping, Protocol

import numpy as np
import numpy.typing as npt
import stride
from mosaic.types import Struct

from .. import rendering, sources
from . import materials
from ._base import Scenario2D, Scenario3D, Target
from ._utils import add_material_fields_to_problem, make_grid


class _Scenario1MixinProtocol(Protocol):
    """Provide type-hinting for Scenario 1 members used by mixins"""

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

    def _get_material_masks(
        self, problem: stride.Problem
    ) -> Mapping[str, npt.NDArray[np.bool_]]:
        ...


class _Scenario1Mixin:
    """A mixin providing specific implementation detail for scenario 1.

    Scenario 1 is based on benchmark 4 of the following paper:

    Jean-Francois Aubry, Oscar Bates, Christian Boehm, et al., "Benchmark problems for
    transcranial ultrasound simulation: Intercomparison of compressional wave models",
    The Journal of the Acoustical Society of America 152, 1003 (2022);
    doi: 10.1121/10.0013426
    https://asa.scitation.org/doi/pdf/10.1121/10.0013426
    """

    @property
    def _material_layers(self: _Scenario1MixinProtocol) -> list[tuple[str, Struct]]:
        return [
            ("water", materials.water),
            ("skin", materials.skin),
            ("cortical bone", materials.cortical_bone),
            ("trabecular bone", materials.trabecular_bone),
            ("brain", materials.brain),
        ]

    @property
    def _material_outline_upsample_factor(self) -> int:
        return 8

    def _get_material_masks(
        self: _Scenario1MixinProtocol, problem: stride.Problem
    ) -> Mapping[str, npt.NDArray[np.bool_]]:
        return {
            name: _create_scenario_1_mask(name, problem.grid)
            for name in self.materials.keys()
        }

    def _compile_scenario_1_problem(
        self: _Scenario1MixinProtocol, extent: npt.NDArray[np.float_]
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
            masks=self._get_material_masks(problem),
        )
        return problem


class Scenario1_2D(_Scenario1Mixin, Scenario2D):
    """A 2D implementation of scenario 1.

    Scenario 1 is based on benchmark 4 of the following paper:

    Jean-Francois Aubry, Oscar Bates, Christian Boehm, et al., "Benchmark problems for
    transcranial ultrasound simulation: Intercomparison of compressional wave models",
    The Journal of the Acoustical Society of America 152, 1003 (2022);
    doi: 10.1121/10.0013426
    https://asa.scitation.org/doi/pdf/10.1121/10.0013426
    """

    _SCENARIO_ID = "scenario-1-2d-v0"
    _TARGET_OPTIONS = {
        "target_1": Target("target_1", np.array([0.064, 0.0]), 0.004, ""),
    }

    def __init__(self, complexity="fast"):
        self._target_id = "target_1"

        super().__init__(
            origin=np.array([0.0, -0.035]),
            complexity=complexity,
        )

    def render_material_property(
        self, name, show_orientation=True, show_sources=True, show_target=True
    ):
        raise NotImplementedError()

    def _compile_problem(self) -> stride.Problem:
        extent = np.array([0.12, 0.07])  # m
        return self._compile_scenario_1_problem(extent)

    def get_default_source(self) -> sources.Source:
        return sources.FocusedSource2D(
            position=np.array([0.0, 0.0]),
            direction=np.array([1.0, 0.0]),
            aperture=0.064,
            focal_length=0.064,
            num_points=1000,
        )


class Scenario1_3D(_Scenario1Mixin, Scenario3D):
    """A 3D implementation of scenario 1.

    Scenario 1 is based on benchmark 4 of the following paper:

    Jean-Francois Aubry, Oscar Bates, Christian Boehm, et al., "Benchmark problems for
    transcranial ultrasound simulation: Intercomparison of compressional wave models",
    The Journal of the Acoustical Society of America 152, 1003 (2022);
    doi: 10.1121/10.0013426
    https://asa.scitation.org/doi/pdf/10.1121/10.0013426
    """

    _SCENARIO_ID = "scenario-1-3d-v0"
    _TARGET_OPTIONS = {
        "target_1": Target(
            target_id="target_1",
            center=np.array([0.064, 0.0, 0.0]),
            radius=0.004,
            description=(
                "A centered location below the skull at approximately the focal point."
            ),
        ),
        "target_2": Target(
            target_id="target_2",
            center=np.array([0.09, 0.01, -0.01]),
            radius=0.006,
            description=("An off-center location below the skull."),
        ),
    }

    def __init__(self, complexity="fast"):
        self._target_id = "target_1"

        super().__init__(
            origin=np.array([0.0, -0.035, -0.035]),
            complexity=complexity,
        )

    @property
    def viewer_config_3d(self) -> rendering.ViewerConfig3D:
        return rendering.ViewerConfig3D(
            init_angles=(-15, 45, 160),
            init_zoom=3.0,
            colormaps={
                "water": "blue",
                "skin": "viridis",
                "cortical bone": "magma",
                "trabecular bone": "inferno",
                "brain": "bop orange",
            },
            opacities={
                "water": 0.8,
                "skin": 0.2,
                "cortical bone": 0.2,
                "trabecular bone": 0.2,
                "brain": 0.4,
            },
        )

    def get_default_slice_axis(self) -> int:
        return 1

    def get_default_slice_position(self, axis: int) -> float:
        default_positions = np.array([0.064, 0.0, 0.0])
        return default_positions[axis]

    def render_material_property(
        self, name, show_orientation=True, show_sources=True, show_target=True
    ):
        raise NotImplementedError()

    def _compile_problem(self) -> stride.Problem:
        extent = np.array([0.12, 0.07, 0.07])  # m
        return self._compile_scenario_1_problem(extent)

    def get_default_source(self) -> sources.Source:
        return sources.FocusedSource3D(
            position=np.array([0.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            aperture=0.064,
            focal_length=0.064,
            num_points=20_000,
        )


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

    elif material == "cortical bone":
        _fill_mask(mask, start=interfaces[1], end=interfaces[2], dx=dx)
        _fill_mask(mask, start=interfaces[3], end=interfaces[4], dx=dx)

    elif material == "trabecular bone":
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
