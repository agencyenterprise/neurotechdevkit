# -*- coding: utf-8 -*-
"""
Loading material mask from a CT scan
====================================
This example shows how to load the brain and skull masks of a CT scan and use them
to create a scenario.

The currently supported CT scan file formats are
[DICOM](https://www.dicomstandard.org/) and
[NIfTI](https://nifti.nimh.nih.gov/nifti-1).

You can use NDK to load the brain and skull masks of a CT scan with:
```
from neurotechdevkit.scenarios import ct_loader
skull_mask, brain_mask = ct_loader.get_masks('PathToCTScanFile.dcm_or_nii')
```
You can find the above call in the method `_create_scenario_2_mask`.
"""
# %%
# Implementing the scenario with brain and skull masks of a CT scan
from __future__ import annotations

import pathlib
from typing import Mapping

import numpy as np
import numpy.typing as npt
import stride

from neurotechdevkit import sources
from neurotechdevkit.results import SteadyStateResult2D
from neurotechdevkit.scenarios import (
    Scenario2D,
    Target,
    add_material_fields_to_problem,
    ct_loader,
    make_grid,
)


# The following code is based on NDK's Scenario 2 implementation.
class ScenarioWithMasksFromCTScan(Scenario2D):
    _SCENARIO_ID = "scenario-2-2d-custom-ct"
    _TARGET_OPTIONS = {
        "primary-visual-cortex": Target(
            target_id="primary-visual-cortex",
            center=np.array([0.047, 0.002]),
            radius=0.010,
            description="description",
        ),
    }
    material_layers = ["water", "trabecular_bone", "brain"]

    def __init__(self, complexity="fast"):
        self._target_id = "primary-visual-cortex"

        super().__init__(
            origin=np.array([0.0, -0.085]),
            complexity=complexity,
        )

    @property
    def _material_outline_upsample_factor(self) -> int:
        return 4

    def _compile_problem(self, center_frequency) -> stride.Problem:
        extent = np.array([0.225, 0.170])  # m
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
            materials=self.get_materials(center_frequency),
            layer_ids=self.layer_ids,
            masks=self._get_material_masks(),
        )
        return problem

    def _get_material_masks(self) -> Mapping[str, npt.NDArray[np.bool_]]:
        return {
            name: self._create_scenario_2_mask(name, convert_2d=True)
            for name in self.material_layers
        }

    def get_default_source(self) -> sources.Source:
        """Get the default source for the scenario."""
        return sources.FocusedSource2D(
            position=np.array([0.0, 0.0]),
            direction=np.array([1.0, 0.0]),
            aperture=0.064,
            focal_length=0.064,
            num_points=1000,
        )

    def _create_scenario_2_mask(
        self, material, convert_2d=False
    ) -> npt.NDArray[np.bool_]:
        cur_dir = pathlib.Path("__file__").parent

        # Here we load the CT scan and get the brain and skull masks:
        data_file = cur_dir / "ID_0000ca2f6.dcm"
        skull_mask, brain_mask = ct_loader.get_masks(data_file, convert_2d)

        if material == "trabecular_bone":
            mask = skull_mask

        elif material == "brain":
            mask = brain_mask

        elif material == "water":
            mask = ~(skull_mask | brain_mask)

        else:
            raise ValueError(material)

        return mask


# %%
# Running the scenario
scenario = ScenarioWithMasksFromCTScan()
result = scenario.simulate_steady_state()
assert isinstance(result, SteadyStateResult2D)
result.render_steady_state_amplitudes(show_material_outlines=True)

# %%
