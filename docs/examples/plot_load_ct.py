# -*- coding: utf-8 -*-
"""
Loading material masks from a CT scan
========================================
This example shows how to load the brain and skull masks segmented
from a CT scan to create a scenario.

The currently supported CT segmentation file formats is
[DICOM](https://www.dicomstandard.org/) exported from
[Slicer](https://www.slicer.org/)

You can use NDK to load the brain and skull masks of a CT scan with:
```
from neurotechdevkit.scenarios import ct_loader
skull_mask, brain_mask = ct_loader.get_masks(
    'PathToTheDicomFolder',
    ct_loader.MaterialMap(
        brain_mask_id=1,
        skull_mask_id=2,
    ),
    convert_2d=True
)
```
Where the `brain_mask_id` and `skull_mask_id` are the integer values
of the masks in the DICOM files.

You can find the above call in the method `_create_scenario_2_mask`.
"""
# %%
# Implementing the scenario with brain and skull masks of a CT scan
from __future__ import annotations

import pathlib
import tempfile
import zipfile
from typing import Mapping

import numpy as np
import numpy.typing as npt
import pooch
import stride

from neurotechdevkit import sources
from neurotechdevkit.results import SteadyStateResult2D
from neurotechdevkit.scenarios import (
    Scenario2D,
    Target,
    add_material_fields_to_problem,
    ct_loader,
    make_grid_with_shape,
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
        masks = self._get_material_masks()
        shape = masks[self.material_layers[0]].shape
        speed_water = 1500  # m/s
        c_freq = 500e3  # hz

        # desired resolution for complexity=fast
        ppw = 6

        # compute resolution
        dx = speed_water / c_freq / ppw  # m

        grid = make_grid_with_shape(shape=shape, dx=dx)
        problem = stride.Problem(
            name=f"{self.scenario_id}-{self.complexity}", grid=grid
        )
        problem = add_material_fields_to_problem(
            problem=problem,
            materials=self.get_materials(center_frequency),
            layer_ids=self.layer_ids,
            masks=masks,
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

        # Here we load the CT scan and get the brain and skull masks:

        URL = "https://neurotechdevkit.s3.us-west-2.amazonaws.com/ct_example.zip"
        known_hash = "370cf1b9f61247ae466230828a1764af4c1157476fe15e594726d18891ebca41"
        downloaded_file_path = pooch.retrieve(
            url=URL, known_hash=known_hash, progressbar=True
        )
        temp_directory = tempfile.TemporaryDirectory()

        with zipfile.ZipFile(downloaded_file_path, "r") as zip_ref:
            zip_ref.extractall(temp_directory.name)

        skull_mask, brain_mask = ct_loader.get_masks(
            pathlib.Path(temp_directory.name),
            ct_loader.MaterialMap(
                brain_mask_id=1,
                skull_mask_id=2,
            ),
            convert_2d=convert_2d,
        )
        temp_directory.cleanup()

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
scenario.render_layout()

# %%
result = scenario.simulate_steady_state()
assert isinstance(result, SteadyStateResult2D)
result.render_steady_state_amplitudes(show_material_outlines=True)

# %%
