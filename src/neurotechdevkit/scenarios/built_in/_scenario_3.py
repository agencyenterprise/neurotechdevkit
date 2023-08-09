"""Helper functions to modify the scenario class to record traces at the source elements"""

from types import SimpleNamespace

import numpy as np
from stride.problem import StructuredData

from ... import results
from .._resources import budget_time_and_memory_resources
from ...results import PulsedResult2D
from ...scenarios import Scenario2D, Target
from ...scenarios._time import (
    create_time_grid,
    find_largest_delay_in_sources,
    select_simulation_time_for_pulsed,
)


class Scenario3(Scenario2D):
    """Overwrite simulate_pulse to also record traces"""
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