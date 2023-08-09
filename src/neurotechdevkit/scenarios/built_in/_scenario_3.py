"""Helper functions to modify the scenario class to record traces at the source elements"""

from types import SimpleNamespace
from typing import Mapping, Optional

from mosaic.types import Struct
import numpy as np
import numpy.typing as npt
import stride
from stride.problem import StructuredData

from ... import results
from .._resources import budget_time_and_memory_resources
from .._utils import create_grid_circular_mask
from ...grid import Grid
from ...materials import Material
from ...problem import Problem
from ...results import PulsedResult2D
from ...scenarios import Scenario2D, Target
from ...scenarios._time import (
    create_time_grid,
    find_largest_delay_in_sources,
    select_simulation_time_for_pulsed,
)


class Scenario3(Scenario2D):
    """Imaging Scenario: grainy phantoms in water."""
    """Overwrite simulate_pulse to also record traces"""

    _PHANTOM_RADIUS = 0.01  # m

    center_frequency = 5e5  # Hz
    material_properties = {
        "water": Material(vp=1500.0, rho=1000.0, alpha=0.0, render_color="#2E86AB"),
        "agar_hydrogel": Material(
            vp=1485.0, rho=1050.0, alpha=0.1, render_color="#E9E6C9",
        ),
    }
    material_speckle_scale = {
        "water": 0.001,
        "agar_hydrogel": 0.01,
    }  # Acoustic property heterogeneity: relative scale
    PREDEFINED_TARGET_OPTIONS = {
        "agar-phantom-center": Target(
            target_id="agar-phantom-center",
            center=[0.06, 0.0],
            radius=_PHANTOM_RADIUS,
            description="Center imaging phantom.",
        ),
        "agar-phantom-right": Target(
            target_id="agar-phantom-right",
            center=[0.07, 0.03],
            radius=_PHANTOM_RADIUS,
            description="Right imaging phantom.",
        ),
    }
    target = PREDEFINED_TARGET_OPTIONS["agar-phantom-center"]

    _extent = np.array([0.1, 0.1])  # m
    origin = [0, -_extent[1] / 2]

    material_outline_upsample_factor = 4

    def make_grid(self):
        """Make the grid for scenario 2 3D."""
        self.grid = Grid.make_grid(
            extent=self._extent,
            speed_water=1500,  # m/s
            ppw=6,  # desired resolution for complexity=fast
            center_frequency=self.center_frequency,
        )
        self.material_masks = self._make_material_masks()

    def compile_problem(
        self,
        rng: Optional[np.random.Generator] = None,
    ):
        """Compiles the problem for the scenario.

        Imaging scenarios have a special requirement of needing heteroegeous
        material to image.
        """
        assert self.grid is not None
        assert self.material_masks is not None

        self.problem = Problem(grid=self.grid)
        _add_material_fields_to_problem(
            self.problem,
            materials=self.materials,
            masks=self.material_masks,
            material_speckle=self.material_speckle_scale,
            rng=rng,
        )

    def _make_material_masks(
        self, convert_2d: bool
    ) -> Mapping[str, npt.NDArray[np.bool_]]:
        """Make the material masks for the scenario."""
        material_layers = [
            "water",
            "agar_hydrogel",
        ]
        material_masks = {
            name: self._create_scenario_mask(name, grid=self.grid, origin=self.origin)
            for name in material_layers
        }
        return material_masks

    def _create_scenario_mask(self, material, grid, origin):
        phantom_mask = np.zeros(grid.space.shape, dtype=bool)

        for target in self.PREDEFINED_TARGET_OPTIONS.values():
            phantom_mask |= create_grid_circular_mask(
                grid, origin, target.center, self._PHANTOM_RADIUS
            )

        if material == "agar_hydrogel":
            return phantom_mask

        elif material == "water":
            return ~phantom_mask

        else:
            raise ValueError(material)

    def simulate_pulse(
        self,
        points_per_period: int = 24,
        simulation_time: float | None = None,
        recording_time_undersampling: int = 1,
        record_traces: bool = True,
        n_jobs: int | None = None,
    ) -> PulsedResult2D:
        """Execute a pulsed simulation.

        In this simulation, the sources will emit a pulse containing a few cycles of
        oscillation and then let the pulse propagate out to all edges of the scenario.

        Args:
            center_frequency: the center frequency (in hertz) to use for the
                continuous-wave source output.
            points_per_period: the number of points in time to simulate for each cycle
                of the wave.
            simulation_time: the amount of time (in seconds) the simulation should run.
                If the value is None, this time will automatically be set to the amount
                of time it would take to propagate from one corner to the opposite in
                the medium with the slowest speed of sound in the scenario.
            recording_time_undersampling: the undersampling factor to apply to the time
                axis when recording simulation results. One out of every this many
                consecutive time points will be recorded and all others will be dropped.
            record_traces: whether to record the traces of the wavefield at the
                source elements.
            n_jobs: the number of threads to be used for the computation. Use None to
                leverage Devito automatic tuning.

        Returns:
            An object containing the result of the pulsed simulation.
        """
        problem = self.problem

        if simulation_time is None:
            simulation_time = select_simulation_time_for_pulsed(
                grid=problem.grid,
                materials=self.materials,
                delay=find_largest_delay_in_sources(self.sources),
            )
        problem.grid.time = create_time_grid(
            freq_hz=self.center_frequency,
            ppp=points_per_period,
            sim_time=simulation_time,
        )

        budget_time_and_memory_resources(
            grid_shape=self.shape,
            recording_time_undersampling=recording_time_undersampling,
            time_steps=problem.grid.time.grid.shape[0],
            is_pulsed=True,
        )

        sub_problem = self._setup_sub_problem("pulsed")
        if record_traces:
            # Standard is for NDK to populate the geometry with many copies of the
            # same Transducer type
            transducer = sub_problem.geometry.transducers.get(0)
            
            # Add PointTrasnducers to the geometry at our element locations
            source_element_positions = np.vstack([
                source.element_positions for source in self.sources
            ]) - self.origin[np.newaxis, :]
            
            assert sub_problem.acquisitions.shots[0].num_receivers == 0
            offset = sub_problem.geometry.num_locations
            for idx, element_position in enumerate(source_element_positions):
                sub_problem.geometry.add(offset + idx, transducer, element_position)
                sub_problem.shot._receivers[idx] = sub_problem.geometry.get(offset + idx)
                # Note: future implementation may want to distribute the receivers
                # over the physical extent of the element, similar to how the
                # transmit elements are implemented.

        pde = self._create_pde()
        traces = self._execute_pde(
            pde=pde,
            sub_problem=sub_problem,
            save_bounds=self._get_pulsed_recording_time_bounds(),
            save_undersampling=recording_time_undersampling,
            wavefield_slice=None,
            n_jobs=n_jobs,
        )
        assert isinstance(pde.wavefield, (StructuredData, SimpleNamespace))
        assert sub_problem.shot is not None

        # put the time axis last and remove the empty last frame
        wavefield = np.moveaxis(pde.wavefield.data[:-1], 0, -1)

        return results.create_pulsed_result(
            scenario=self,
            center_frequency=self.center_frequency,
            effective_dt=self.dt * recording_time_undersampling,
            pde=pde,
            shot=sub_problem.shot,
            wavefield=wavefield,
            traces=traces,
            recorded_slice=None,
        )


def _add_material_fields_to_problem(
    problem: Problem,
    materials: Mapping[str, Struct],
    masks: Mapping[str, npt.NDArray[np.bool_]],
    material_speckle: Mapping[str, float],
    rng: Optional[np.random.Generator] = None,
):
    """Add material fields as media to the problem.

    This is an adaptation of the default `compile_problem()` that adds
    in heterogenous material properties.

    Included fields are:

    - the speed of sound (in m/s)
    - density (in kg/m^3)
    - absorption (in dB/cm)

    Args:
        problem: the stride Problem object to which the
            media should be added.
        materials: a mapping from material names
            to Structs containing the material properties.
        masks: a mapping from material
            names to boolean masks indicating the gridpoints.
        material_speckle: a mapping from material names to the relative scale
            of material hetereogeneity
    """
    property_names = [
        "vp",  # [m/s]
        "rho",  # [kg/m^3]
        "alpha",  # [dB/cm]
    ]

    rng = np.random.default_rng(rng)
    for prop_name in property_names:
        field = stride.ScalarField(name=prop_name, grid=problem.grid)

        for name, material in materials.items():
            material_mask = masks[name]
            material_prop = getattr(material, prop_name)
            speckle = rng.normal(
                scale=material_speckle[name] * material_prop,
                size=material_mask.sum(),
            )
            # Add hetereogeneity to the material property
            # Really, only vp and rho affect the wave scattering,
            # but we add it to alpha for completeness
            field.data[material_mask] = material_prop + speckle

        field.pad()
        problem.medium.add(field)
