from __future__ import annotations

import abc
import asyncio
import os
from types import SimpleNamespace
from typing import Mapping, Optional, Union

import nest_asyncio
import numpy as np
import numpy.typing as npt
import stride
from mosaic.types import Struct
from stride.problem import StructuredData

from .. import rendering, results
from ..grid import Grid
from ..materials import Material, get_material, get_render_color
from ..problem import Problem
from ..sources import Source
from ._resources import budget_time_and_memory_resources
from ._shots import create_shot
from ._time import (
    create_time_grid,
    find_largest_delay_in_sources,
    select_simulation_time_for_pulsed,
    select_simulation_time_for_steady_state,
)
from ._utils import (
    SliceAxis,
    Target,
    choose_wavelet_for_mode,
    create_grid_circular_mask,
    create_grid_spherical_mask,
    drop_element,
    slice_field,
    wavelet_helper,
)

nest_asyncio.apply()


class Scenario(abc.ABC):
    """The base scenario."""

    # The customization to the material layers.
    material_properties: dict[str, Material]

    material_masks: Mapping[str, npt.NDArray[np.bool_]]

    origin: list[float]

    # The list of sources in the scenario.
    sources: list[Source]
    # Coordinates of point receivers in the scenario
    receiver_coords: npt.NDArray[np.float_] | list[npt.NDArray[np.float_]] = []

    material_outline_upsample_factor: int = 16

    _center_frequency: float
    _problem: Problem
    _target: Target
    _grid: Grid

    def __init__(
        self,
        center_frequency: Optional[float] = None,
        material_properties: Optional[dict[str, Material]] = None,
        material_masks: Optional[Mapping[str, npt.NDArray[np.bool_]]] = None,
        origin: Optional[list[float]] = None,
        sources: Optional[list[Source]] = None,
        material_outline_upsample_factor: Optional[int] = None,
        target: Optional[Target] = None,
        problem: Optional[Problem] = None,
        grid: Optional[Grid] = None,
    ) -> None:
        """
        Initialize a scenario.

        All arguments are optional and can be set after initialization.

        Args:
            center_frequency (Optional[float], optional): The center frequency (in
                hertz) of the scenario. Defaults to None.
            material_properties (Optional[dict[str, Material]], optional): A map
                between material name and material properties. Defaults to None.
            material_masks (Optional[Mapping[str, npt.NDArray[np.bool_]]], optional):
                A map between material name and a boolean mask indicating which grid
                points are in that material. Defaults to None.
            origin (Optional[list[float]], optional): The location of the origin of the
                scenario (in meters). Defaults to None.
            sources (Optional[list[Source]], optional): The list of sources in the
                scenario. Defaults to None.
            material_outline_upsample_factor (Optional[int], optional): The factor by
                which to upsample the material outline when rendering the scenario.
                Defaults to None.
            target (Optional[Target], optional): The target in the scenario. Defaults
                to None.
            problem (Optional[Problem], optional): The problem definition for the
                scenario. Defaults to None.
            grid (Optional[Grid], optional): The grid for the scenario. Defaults to
                None.
        """
        if center_frequency is not None:
            self.center_frequency = center_frequency
        if material_properties is not None:
            self.material_properties = material_properties
        if material_masks is not None:
            self.material_masks = material_masks
        if origin is not None:
            self.origin = origin
        if sources is not None:
            self.sources = sources
        if material_outline_upsample_factor is not None:
            self.material_outline_upsample_factor = material_outline_upsample_factor
        if target is not None:
            self.target = target
        if problem is not None:
            self.problem = problem
        if grid is not None:
            self.grid = grid

    @property
    def extent(self) -> npt.NDArray[np.float_]:
        """The extent of the spatial grid (in meters)."""
        assert self.grid.space is not None
        return np.array(self.grid.space.size, dtype=float)

    @property
    def shape(self) -> npt.NDArray[np.int_]:
        """The shape of the spatial grid (in number of grid points)."""
        assert self.grid.space is not None
        return np.array(self.grid.space.shape, dtype=int)

    @property
    def dx(self) -> float:
        """The spacing (in meters) between spatial grid points.

        Spacing is the same in each spatial direction.
        """
        assert self.grid.space is not None
        return self.grid.space.spacing[0]

    @property
    def t_min(self) -> float:
        """The starting time (in seconds) of the simulation.

        Only available once a simulation has been completed.
        """
        if self.problem.time is None:
            raise ValueError(
                "t_min is not defined until the simulation frequency is defined."
            )
        return self.problem.time.start

    @property
    def t_max(self) -> float:
        """The maximum time (in seconds) of the simulation.

        Only available once a simulation has been completed.
        """
        if self.problem.time is None:
            raise ValueError(
                "t_max is not defined until the simulation frequency is defined."
            )
        return self.problem.time.stop

    @property
    def dt(self) -> float:
        """The spacing (in seconds) between consecutive timesteps of the simulation.

        Only available once a simulation has been completed.
        """
        if self.problem.time is None:
            raise ValueError(
                "dt is not defined until the simulation frequency is defined."
            )
        return self.problem.time.step

    @property
    def target_center(self) -> npt.NDArray[np.float_]:
        """The coordinates of the center of the target region (in meters)."""
        return np.array(self.target.center, dtype=float)

    @property
    def target_radius(self) -> float:
        """The radius of the target region (in meters)."""
        return self.target.radius

    @property
    def materials(self) -> Mapping[str, Struct]:
        """Return a map between material name and material properties.

        - vp: the speed of sound (in m/s).
        - rho: the mass density (in kg/m³).
        - alpha: the absorption (in dB/cm).
        - render_color: the color used when rendering this material in the
        scenario layout plot.
        """
        materials = {}
        for layer in self.material_layers:
            if layer not in self.material_properties:
                material_properties = get_material(layer, self.center_frequency)
            else:
                material_properties = self.material_properties[layer]
            materials[layer] = material_properties.to_struct()
        return materials

    @property
    def material_colors(self) -> dict[str, str]:
        """
        A map between material name and material render color.

        Returns:
            dict[str, str]: keys are material names and values are the hex color
        """
        material_colors = {}
        for material in self.material_layers:
            if material in self.material_properties:
                color = self.material_properties[material].render_color
            else:
                color = get_render_color(material_name=material)
            material_colors[material] = color
        return material_colors

    @property
    def layer_ids(self) -> Mapping[str, int]:
        """A map between material names and their layer id."""
        return {name: n for n, name in enumerate(self.material_layers)}

    @property
    def material_layers(self) -> list[str]:
        """The list of material layers in the scenario."""
        return list(self.material_masks.keys())

    @property
    def target(self) -> Target:
        """The target in the scenario."""
        assert hasattr(self, "_target")
        return self._target

    @target.setter
    def target(self, target: Target) -> None:
        """Set the target in the scenario.

        Args:
            target: the target in the scenario.
        """
        self._target = target

    @property
    def problem(self) -> Problem:
        """The problem definition for the scenario."""
        assert hasattr(self, "_problem")
        return self._problem

    @problem.setter
    def problem(self, problem: Problem) -> None:
        """Set the problem definition for the scenario.

        Args:
            problem: the problem definition for the scenario.
        """
        self._problem = problem

    @problem.deleter
    def problem(self) -> None:
        """Delete the problem definition for the scenario."""
        if hasattr(self, "_problem"):
            del self._problem

    @property
    def grid(self) -> Grid:
        """The grid for the scenario."""
        assert hasattr(self, "_grid")
        return self._grid

    @grid.setter
    def grid(self, grid: Grid) -> None:
        """Set the grid for the scenario.

        Args:
            grid: the grid for the scenario.
        """
        self._grid = grid

    @grid.deleter
    def grid(self) -> None:
        """Delete the grid for the scenario."""
        if hasattr(self, "_grid"):
            del self._grid

    @property
    def center_frequency(self) -> float:
        """The center frequency (in hertz) of the scenario."""
        assert hasattr(self, "_center_frequency")
        return self._center_frequency

    @center_frequency.setter
    def center_frequency(self, center_frequency: float) -> None:
        """Set the center frequency of the scenario.

        Args:
            center_frequency: the center frequency (in hertz) of the scenario.
        """
        if hasattr(self, "_center_frequency"):
            if self._center_frequency != center_frequency:
                # Invalidating problem and grid as they should be re-generated
                # with the new center frequency
                del self.problem
                del self.grid

        self._center_frequency = center_frequency

    def compile_problem(self):
        """Compiles the problem for the scenario."""
        assert self.grid is not None
        assert self.material_masks is not None

        self.problem = Problem(grid=self.grid)
        self.problem.add_material_fields(
            materials=self.materials,
            masks=self.material_masks,
        )

    def get_field_data(self, field: str) -> npt.NDArray[np.float_]:
        """Return the array of field values across the scenario for a particular field.

        Common fields include:

        - vp: the speed of sound (in m/s)
        - rho: the density (in kg/m³)
        - alpha: absorption (in dB/cm)

        Args:
            field: the name of the field to return.

        Returns:
            An array containing the field data.
        """
        return self.problem.medium.fields[field].data

    @property
    def material_layer_ids(self) -> npt.NDArray[np.int_]:
        """Return the layer id for each grid point in the scenario."""
        assert hasattr(self, "layer_ids")
        assert hasattr(self, "material_masks")
        assert self.grid.space is not None

        layer = np.zeros(self.grid.space.shape, dtype=int)

        for layer_name in self.material_layers:
            material_mask = self.material_masks[layer_name]
            layer_id = self.layer_ids[layer_name]

            layer[material_mask] = layer_id
        return layer

    def simulate_steady_state(
        self,
        points_per_period: int = 24,
        n_cycles_steady_state: int = 10,
        time_to_steady_state: float | None = None,
        recording_time_undersampling: int = 4,
        n_jobs: int | None = None,
    ) -> results.SteadyStateResult:
        """Execute a steady-state simulation.

        In this simulation, the sources will emit pressure waves with a continuous
        waveform until steady-state has been reached. The steady-state wave amplitude is
        found by taking the Fourier transform of the last `n_cycles_steady_state` cycles
        of data and taking the amplitude of the component at the `center_frequency`.

        !!! warning
            A poor choice of arguments to this function can lead to a failed
            simulation. Make sure you understand the impact of supplying parameter
            values other than the default if you chose to do so.

        Args:
            points_per_period: the number of points in time to simulate for each cycle
                of the wave.
            n_cycles_steady_state: the number of complete cycles to use when calculating
                the steady-state wave amplitudes.
            time_to_steady_state: the amount of time (in seconds) the simulation should
                run before measuring the steady-state amplitude. If the value is None,
                this time will automatically be set to the amount of time it would take
                to propagate from one corner to the opposite and back in the medium with
                the slowest speed of sound in the scenario.
            recording_time_undersampling: the undersampling factor to apply to the time
                axis when recording simulation results. One out of every this many
                consecutive time points will be recorded and all others will be dropped.
            n_jobs: the number of threads to be used for the computation. Use None to
                leverage Devito automatic tuning.

        Returns:
            An object containing the result of the steady-state simulation.
        """
        problem = self.problem
        sim_time = select_simulation_time_for_steady_state(
            grid=problem.grid,
            materials=self.materials,
            freq_hz=self.center_frequency,
            time_to_steady_state=time_to_steady_state,
            n_cycles_steady_state=n_cycles_steady_state,
            delay=find_largest_delay_in_sources(self.sources),
        )
        problem.grid.time = create_time_grid(
            freq_hz=self.center_frequency, ppp=points_per_period, sim_time=sim_time
        )

        budget_time_and_memory_resources(
            grid_shape=self.shape,
            recording_time_undersampling=recording_time_undersampling,
            n_cycles_steady_state=n_cycles_steady_state,
            time_steps=problem.grid.time.grid.shape[0],
            is_pulsed=False,
        )

        sub_problem = self._setup_sub_problem("steady-state")
        pde = self._create_pde()
        traces = self._execute_pde(
            pde=pde,
            sub_problem=sub_problem,
            save_bounds=self._get_steady_state_recording_time_bounds(
                points_per_period, n_cycles_steady_state
            ),
            save_undersampling=recording_time_undersampling,
            wavefield_slice=self._wavefield_slice(),
            n_jobs=n_jobs,
        )
        assert isinstance(pde.wavefield, (StructuredData, SimpleNamespace))
        assert sub_problem.shot is not None

        # put the time axis last and remove the empty last frame
        wavefield = np.moveaxis(pde.wavefield.data[:-1], 0, -1)

        return results.create_steady_state_result(
            scenario=self,  # type: ignore
            center_frequency=self.center_frequency,
            effective_dt=self.dt * recording_time_undersampling,
            pde=pde,
            shot=sub_problem.shot,
            wavefield=wavefield,
            traces=traces,
        )

    def _simulate_pulse(
        self,
        points_per_period: int = 24,
        simulation_time: float | None = None,
        recording_time_undersampling: int = 4,
        n_jobs: int | None = None,
        slice_axis: int | None = None,
        slice_position: float | None = None,
    ) -> Union[results.PulsedResult2D, results.PulsedResult3D]:
        """Execute a pulsed simulation.

        In this simulation, the sources will emit a pulse containing a few cycles of
        oscillation and then let the pulse propagate out to all edges of the scenario.

        Warning: A poor choice of arguments to this function can lead to a failed
        simulation. Make sure you understand the impact of supplying parameter values
        other than the default if you chose to do so.

        Args:
            points_per_period: the number of points in time to simulate for each cycle
                of the wave.
            simulation_time: the amount of time (in seconds) the simulation should run.
                If the value is None, this time will automatically be set to the amount
                of time it would take to propagate from one corner to the opposite in
                the medium with the slowest speed of sound in the scenario.
            recording_time_undersampling: the undersampling factor to apply to the time
                axis when recording simulation results. One out of every this many
                consecutive time points will be recorded and all others will be dropped.
            n_jobs: the number of threads to be used for the computation. Use None to
                leverage Devito automatic tuning.
            slice_axis: the axis along which to slice the 3D field to be recorded. If
                None, then the complete field will be recorded. Use 0 for X axis, 1 for
                Y axis and 2 for Z axis. Only valid if `slice_position` is not None.
            slice_position: the position (in meters) along the slice axis at
                which the slice of the 3D field should be made. Only valid if
                `slice_axis` is not None.

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

        if slice_axis is not None and slice_position is not None:
            recorded_slice = (slice_axis, slice_position)
        else:
            recorded_slice = None

        budget_time_and_memory_resources(
            grid_shape=self.shape,
            recording_time_undersampling=recording_time_undersampling,
            time_steps=problem.grid.time.grid.shape[0],
            is_pulsed=True,
        )

        sub_problem = self._setup_sub_problem("pulsed")
        pde = self._create_pde()
        traces = self._execute_pde(
            pde=pde,
            sub_problem=sub_problem,
            save_bounds=self._get_pulsed_recording_time_bounds(),
            save_undersampling=recording_time_undersampling,
            wavefield_slice=self._wavefield_slice(slice_axis, slice_position),
            n_jobs=n_jobs,
        )
        assert isinstance(pde.wavefield, (StructuredData, SimpleNamespace))
        assert sub_problem.shot is not None

        # put the time axis last and remove the empty last frame
        wavefield = np.moveaxis(pde.wavefield.data[:-1], 0, -1)

        return results.create_pulsed_result(
            scenario=self,  # type: ignore
            center_frequency=self.center_frequency,
            effective_dt=self.dt * recording_time_undersampling,
            pde=pde,
            shot=sub_problem.shot,
            wavefield=wavefield,
            traces=traces,
            recorded_slice=recorded_slice,
        )

    def _setup_sub_problem(self, simulation_mode: str) -> stride.SubProblem:
        """Set up a stride `SubProblem` for the simulation.

        A SubProblem requires at least one source transducer. If no source is defined, a
        default source is used.

        Args:
            simulation_mode: the type of simulation which will be run.

        Returns:
            The `SubProblem` to use for the simulation.
        """
        shot = self._setup_shot(
            sources=self.sources,
            receiver_coords=self.receiver_coords,
            freq_hz=self.center_frequency,
            simulation_mode=simulation_mode,
        )
        return self.problem.sub_problem(shot.id)

    def _setup_shot(
        self,
        sources: list[Source],
        receiver_coords: npt.NDArray[np.float_] | list[npt.NDArray[np.float_]],
        freq_hz: float,
        simulation_mode: str,
    ) -> stride.Shot:
        """Create the stride `Shot` for the simulation.

        Args:
            sources: the source transducers to use within the shot.
            freq_hz: the center frequency (in hertz) to use for the source wavelet.
            simulation_mode: the type of simulation which will be run.

        Returns:
            The `Shot` to use for the simulation.
        """
        problem = self.problem
        assert problem.grid.time is not None

        wavelet_name = choose_wavelet_for_mode(simulation_mode)
        wavelet = wavelet_helper(
            name=wavelet_name, freq_hz=freq_hz, time=problem.grid.time
        )
        return create_shot(
            problem=problem,
            sources=sources,
            receiver_coords=receiver_coords,
            origin=np.array(np.array(self.origin, dtype=float), dtype=float),
            wavelet=wavelet,
            dx=self.dx,
        )

    def _create_pde(self) -> stride.IsoAcousticDevito:
        """Instantiate the stride `Operator` representing the PDE for the scenario.

        All existing scenarios use the `IsoAcousticDevito` operator.

        Returns:
            The PDE `Operator` ready for simulation.
        """
        problem = self.problem
        return stride.IsoAcousticDevito(
            space=problem.space,
            time=problem.time,
            devito_config={"autotuning": "off"},
        )

    def _wavefield_slice(
        self, slice_axis: int | None = None, slice_position: float | None = None
    ) -> tuple[slice, ...]:
        """Define the region of of the grid that should be recorded.

        The first element of the tuple is for time, while all remaining elements are for
        space and should match the dimensionality of space.

        The returned time slice selects all time points.

        The returned space slices select only the data inside the defined extent of the
        scenario and drop data from the boundary layers.

        Args:
            slice_axis: the axis along which to slice the 3D field to be recorded. If
                None, then the complete field will be recorded. Use 0 for X axis, 1 for
                Y axis and 2 for Z axis.
            slice_position: the position (in meters) along the slice axis at
                which the slice of the 3D field should be made.

        Returns:
            A tuple of slices defining the region of the grid to record.
        """
        space = self.problem.space
        assert space is not None
        standard_slice = tuple(
            [
                # save all time points
                slice(0, None),
                *[
                    # we want to ignore extra plus an additional 10
                    # gridpoints that stride adds to each side
                    slice(extra + 10, extra + 10 + shape)
                    for shape, extra in zip(space.shape, space.extra)
                ],
            ]
        )
        if slice_axis is None and slice_position is None:
            return standard_slice

        self._validate_slice_args(slice_axis, slice_position)

        assert space.dim == 3
        assert slice_position is not None
        assert slice_axis is not None

        offset_distance = (
            slice_position - np.array(self.origin, dtype=float)[slice_axis]
        )
        slice_idx = np.clip(
            int(offset_distance / self.dx), 0, self.shape[slice_axis] - 1
        )
        standard_slice_updated = list(standard_slice)
        # first element of standard_slice is the time component
        current_slice = standard_slice[slice_axis + 1]
        start = current_slice.start
        standard_slice_updated[slice_axis + 1] = slice(
            start + slice_idx, start + slice_idx + 1, None
        )

        return tuple(standard_slice_updated)

    def _validate_slice_args(
        self, slice_axis: int | None, slice_position: float | None
    ) -> None:
        """Validate that slicing axis and position are within scenario range.

        `slice_axis` should be either 0, 1, or 2 (for X, Y, Z).
        `slice_position` must be within boundaries for `slice_axis` extent.

        Args:
            slice_axis: the axis along which to slice the 3D field to be recorded. If
                None, then the complete field will be recorded. Use 0 for X axis, 1
                for Y axis and 2 for Z axis.
            slice_position: the position (in meters) along the slice axis at
                which the slice of the 3D field should be made.

        Raises:
            ValueError if axis is not 0, 1, 2.
            ValueError if `slice_position` falls outside the current range of
                `slice_axis`.
            ValueError if  `slice_axis` is None or `slice_position` is None.
        """
        if slice_axis is None or slice_position is None:
            raise ValueError(
                "Both `slice_axis` and `slice_position` must be passed together "
                "to correctly define how to slice the field. "
            )
        if slice_axis not in (0, 1, 2):
            raise ValueError(
                "Unexpected value received for `slice_axis`. ",
                "Expected axis are 0 (X), 1 (Y) and/or 2 (Z).",
            )

        origin = np.array(self.origin, dtype=float)
        extent = self.extent

        current_range = (origin[slice_axis], origin[slice_axis] + extent[slice_axis])
        if (slice_position < current_range[0]) or (slice_position > current_range[1]):
            raise ValueError(
                "`slice_position` is out of range for `slice_axis`. ",
                f"Received value {slice_position} and "
                f"current range is {current_range}.",
            )

    def _get_steady_state_recording_time_bounds(
        self, ppp: int, n_cycles: int
    ) -> tuple[int, int]:
        """Define the indices bounding the period of time to be recorded.

        For steady-state simulations, we only want to keep the last few cycles of the
        simulation.

        Args:
            ppp: the number of points in time per phase to simulate for each cycle of
                the wave.
            n_cycles: the number of complete cycles to record at the end of the
                simulation.

        Returns:
            A tuple containing the time indices bounding the period of time which should
                be recorded for a steady-state simulation.
        """
        n_frames = ppp * n_cycles
        time = self.problem.time
        assert time is not None
        return (time.num - n_frames, time.num - 1)

    def _get_pulsed_recording_time_bounds(self) -> tuple[int, int]:
        """Define the indices bounding the period of time to be recorded.

        For pulsed simulations, we want to keep the data from all timesteps.

        Returns:
            A tuple containing the time indices bounding the period of time which should
                be recorded for a pulsed simulation.
        """
        time = self.problem.time
        assert time is not None
        return (0, time.num - 1)

    def _execute_pde(
        self,
        pde: stride.Operator,
        sub_problem: stride.SubProblem,
        save_bounds: tuple[int, int],
        save_undersampling: int,
        wavefield_slice: tuple[slice, ...],
        n_jobs: int | None = None,
    ) -> stride.Traces:
        """Execute the PDE for the simulation.

        Args:
            pde: the `Operator` containing the PDE to execute.
            sub_problem: the `SubProblem` containing details of the source and waveform
                for the simulation.
            save_bounds: the time indices bounding the period of time to be recorded.
            save_undersampling: the undersampling factor to apply to the time axis when
                recording simulation results.
            wavefield_slice: A tuple of slices defining the region of the grid to
                record.
            n_jobs: the number of threads to be used for the computation. Use None to
                leverage Devito automatic tuning.
        Returns:
            The `Traces` which are produced by the simulation.
        """
        problem = self.problem

        devito_args = {}
        if n_jobs is not None:
            devito_args = dict(nthreads=n_jobs)

        assert sub_problem.shot is not None
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            pde(
                wavelets=sub_problem.shot.wavelets,
                vp=problem.medium.fields["vp"],
                rho=problem.medium.fields["rho"],
                alpha=problem.medium.fields["alpha"],
                problem=sub_problem,
                boundary_type="complex_frequency_shift_PML_2",
                diff_source=True,
                save_wavefield=True,
                platform=os.environ.get("PLATFORM"),
                save_bounds=save_bounds,
                save_undersampling=save_undersampling,
                wavefield_slice=wavefield_slice,
                devito_args=devito_args,
            )
        )


class Scenario2D(Scenario):
    """A 2D scenario."""

    def get_target_mask(self) -> npt.NDArray[np.bool_]:
        """Return the mask for the target region."""
        target_mask = create_grid_circular_mask(
            grid=self.problem.grid,
            origin=np.array(self.origin, dtype=float),
            center=self.target_center,
            radius=self.target_radius,
        )
        return target_mask

    def simulate_pulse(
        self,
        points_per_period: int = 24,
        simulation_time: float | None = None,
        recording_time_undersampling: int = 4,
        n_jobs: int | None = None,
    ) -> results.PulsedResult2D:
        """Execute a pulsed simulation in 2D.

        In this simulation, the sources will emit a pulse containing a few cycles of
        oscillation and then let the pulse propagate out to all edges of the scenario.

        !!! warning
            A poor choice of arguments to this function can lead to a failed
            simulation. Make sure you understand the impact of supplying parameter
            values other than the default if you chose to do so.

        Args:
            points_per_period: the number of points in time to simulate for each cycle
                of the wave.
            simulation_time: the amount of time (in seconds) the simulation should run.
                If the value is None, this time will automatically be set to the amount
                of time it would take to propagate from one corner to the opposite in
                the medium with the slowest speed of sound in the scenario.
            recording_time_undersampling: the undersampling factor to apply to the time
                axis when recording simulation results. One out of every this many
                consecutive time points will be recorded and all others will be dropped.
            n_jobs: the number of threads to be used for the computation. Use None to
                leverage Devito automatic tuning.

        Returns:
            An object containing the result of the 2D pulsed simulation.
        """
        result = self._simulate_pulse(
            points_per_period=points_per_period,
            simulation_time=simulation_time,
            recording_time_undersampling=recording_time_undersampling,
            n_jobs=n_jobs,
            slice_axis=None,
            slice_position=None,
        )
        assert isinstance(result, results.PulsedResult2D)
        return result

    def render_layout(
        self,
        show_sources: bool = True,
        show_target: bool = True,
        show_material_outlines: bool = False,
    ) -> None:
        """Create a matplotlib figure showing the 2D scenario layout.

        The grid can be turned on via: `plt.grid(True)`

        Args:
            show_sources: whether or not to show the source transducer layer.
            show_target: whether or not to show the target layer.
            show_material_outlines: whether or not to display a thin white outline of
                the transition between different materials.
        """
        color_sequence = list(self.material_colors.values())
        fig, ax = rendering.create_layout_fig(
            self.extent,
            np.array(self.origin, dtype=float),
            color_sequence,
            self.material_layer_ids,
        )

        # add layers
        if show_material_outlines:
            rendering.draw_material_outlines(
                ax=ax,
                material_field=self.material_layer_ids,
                dx=self.dx,
                origin=np.array(self.origin, dtype=float),
                upsample_factor=self.material_outline_upsample_factor,
            )
        if show_target:
            rendering.draw_target(ax, self.target_center, self.target_radius)
        if show_sources:
            assert self.sources
            for source in self.sources:
                drawing_params = rendering.SourceDrawingParams.from_source(source)
                rendering.draw_source(ax, drawing_params)

        rendering.configure_layout_plot(
            fig=fig,
            ax=ax,
            color_sequence=color_sequence,
            layer_labels=self.material_layers,
            show_sources=show_sources,
            show_target=show_target,
            extent=self.extent,
            origin=np.array(self.origin, dtype=float),
        )


class Scenario3D(Scenario):
    """A 3D scenario."""

    slice_axis: SliceAxis
    slice_position: float
    viewer_config_3d: rendering.ViewerConfig3D

    def __init__(
        self,
        center_frequency: Optional[float] = None,
        material_properties: Optional[dict[str, Material]] = None,
        material_masks: Optional[Mapping[str, npt.NDArray[np.bool_]]] = None,
        origin: Optional[list[float]] = None,
        sources: Optional[list[Source]] = None,
        material_outline_upsample_factor: Optional[int] = None,
        target: Optional[Target] = None,
        problem: Optional[Problem] = None,
        grid: Optional[Grid] = None,
        slice_axis: Optional[SliceAxis] = None,
        slice_position: Optional[float] = None,
        viewer_config_3d: Optional[rendering.ViewerConfig3D] = None,
    ):
        """Initialize a 3D Scenario.

        All arguments are optional and can be set after initialization.

        Args:
            center_frequency (Optional[float], optional): The center frequency (in
                hertz) of the scenario. Defaults to None.
            material_properties (Optional[dict[str, Material]], optional): A map
                between material name and material properties. Defaults to None.
            material_masks (Optional[Mapping[str, npt.NDArray[np.bool_]]], optional):
                A map between material name and a boolean mask indicating which grid
                points are in that material. Defaults to None.
            origin (Optional[list[float]], optional): The location of the origin of the
                scenario (in meters). Defaults to None.
            sources (Optional[list[Source]], optional): The list of sources in the
                scenario. Defaults to None.
            material_outline_upsample_factor (Optional[int], optional): The factor by
                which to upsample the material outline when rendering the scenario.
                Defaults to None.
            target (Optional[Target], optional): The target in the scenario. Defaults
                to None.
            problem (Optional[Problem], optional): The problem definition for the
                scenario. Defaults to None.
            grid (Optional[Grid], optional): The grid for the scenario. Defaults to
                None.
            slice_axis (Optional[SliceAxis], optional): The axis along which to slice
                the 3D field to be recorded. If None, then the complete field will be
                recorded. Use 0 for X axis, 1 for Y axis and 2 for Z axis. Defaults to
                None.
            slice_position (Optional[float], optional): The position (in meters) along
                the slice axis at which the slice of the 3D field should be made.
                Defaults to None.
            viewer_config_3d (Optional[rendering.ViewerConfig3D], optional): The
                configuration to use when rendering the 3D scenario. Defaults to None.
        """
        if slice_axis is not None:
            self.slice_axis = slice_axis
        if slice_position is not None:
            self.slice_position = slice_position
        if viewer_config_3d is not None:
            self.viewer_config_3d = viewer_config_3d
        super().__init__(
            center_frequency=center_frequency,
            material_properties=material_properties,
            material_masks=material_masks,
            origin=origin,
            sources=sources,
            material_outline_upsample_factor=material_outline_upsample_factor,
            target=target,
            problem=problem,
            grid=grid,
        )

    def get_target_mask(self) -> npt.NDArray[np.bool_]:
        """Return the mask for the target region."""
        target_mask = create_grid_spherical_mask(
            grid=self.problem.grid,
            origin=np.array(self.origin, dtype=float),
            center=self.target_center,
            radius=self.target_radius,
        )
        return target_mask

    def simulate_pulse(
        self,
        points_per_period: int = 24,
        simulation_time: float | None = None,
        recording_time_undersampling: int = 4,
        n_jobs: int | None = None,
        slice_axis: int | None = None,
        slice_position: float | None = None,
    ) -> results.PulsedResult3D:
        """Execute a pulsed simulation in 3D.

        In this simulation, the sources will emit a pulse containing a few cycles of
        oscillation and then let the pulse propagate out to all edges of the scenario.

        !!! warning
            A poor choice of arguments to this function can lead to a failed
            simulation. Make sure you understand the impact of supplying parameter
            values other than the default if you chose to do so.

        Args:
            points_per_period: the number of points in time to simulate for each cycle
                of the wave.
            simulation_time: the amount of time (in seconds) the simulation should run.
                If the value is None, this time will automatically be set to the amount
                of time it would take to propagate from one corner to the opposite in
                the medium with the slowest speed of sound in the scenario.
            recording_time_undersampling: the undersampling factor to apply to the time
                axis when recording simulation results. One out of every this many
                consecutive time points will be recorded and all others will be dropped.
            n_jobs: the number of threads to be used for the computation. Use None to
                leverage Devito automatic tuning.
            slice_axis: the axis along which to slice the 3D field to be recorded. If
                None, then the complete field will be recorded. Use 0 for X axis, 1 for
                Y axis and 2 for Z axis. Only valid if `slice_position` is not None.
            slice_position: the position (in meters) along the slice axis at
                which the slice of the 3D field should be made. Only valid if
                `slice_axis` is not None.

        Returns:
            An object containing the result of the 3D pulsed simulation.
        """
        result = self._simulate_pulse(
            points_per_period=points_per_period,
            simulation_time=simulation_time,
            recording_time_undersampling=recording_time_undersampling,
            n_jobs=n_jobs,
            slice_axis=slice_axis,
            slice_position=slice_position,
        )
        assert isinstance(result, results.PulsedResult3D)
        return result

    def render_layout(
        self,
        show_sources: bool = True,
        show_target: bool = True,
        show_material_outlines: bool = False,
    ) -> None:
        """Create a matplotlib figure showing a 2D slice of the scenario layout.

        In order to visualize the 3D scenario in a 2D plot, a slice through the scenario
        needs to be specified via `slice_axis` and `slice_position`. Eg. to take a slice
        at z=0.01 m, use `slice_axis=2` and `slice_position=0.01`.

        The grid can be turned on via: `plt.grid(True)`

        Args:
            show_sources: whether or not to show the source transducer layer.
            show_target: whether or not to show the target layer.
            show_material_outlines: whether or not to display a thin white outline of
                the transition between different materials.
        """
        slice_axis = self.slice_axis
        slice_position = self.slice_position
        color_sequence = list(self.material_colors.values())
        field = slice_field(self.material_layer_ids, self, slice_axis, slice_position)
        extent = drop_element(self.extent, slice_axis)
        origin = drop_element(np.array(self.origin, dtype=float), slice_axis)
        fig, ax = rendering.create_layout_fig(extent, origin, color_sequence, field)

        # add layers
        if show_material_outlines:
            rendering.draw_material_outlines(
                ax=ax,
                material_field=field,
                dx=self.dx,
                origin=origin,
                upsample_factor=self.material_outline_upsample_factor,
            )
        if show_target:
            target_loc = drop_element(self.target_center, slice_axis)
            rendering.draw_target(ax, target_loc, self.target_radius)
        if show_sources:
            assert self.sources
            for source in self.sources:
                drawing_params = rendering.SourceDrawingParams(
                    position=drop_element(source.position, slice_axis),
                    direction=drop_element(source.unit_direction, slice_axis),
                    aperture=source.aperture,
                    focal_length=source.focal_length,
                    source_type=rendering.SourceRenderType.from_source(source),
                )
                rendering.draw_source(ax, drawing_params)

        axis_names = np.array(["X", "Y", "Z"])
        vert_name, horz_name = drop_element(axis_names, slice_axis)
        slice_name = axis_names[slice_axis]
        rendering.configure_layout_plot(
            fig=fig,
            ax=ax,
            color_sequence=color_sequence,
            layer_labels=self.material_layers,
            show_sources=show_sources,
            show_target=show_target,
            extent=extent,
            origin=origin,
            vertical_label=vert_name,
            horizontal_label=horz_name,
            title=f"Scenario Layout\nSlice: {slice_name} = {slice_position} m",
        )

    def render_layout_3d(
        self,
    ) -> None:
        """Render the scenario layout in 3D using napari.

        This function requires the napari package to be installed.

        !!! warning
            Integration with napari is experimental, so do not be surprised if you
            encounter issues calling this function.

        This will open up the napari interactive GUI in a separate window. The GUI
        contains many different controls for controlling the view of the data as well as
        the rendering of the layers. Among these, you can drag the scenario to view it
        from different angles, zoom in our out, and turn layers on or off.

        See napari documentation for more information on the GUI:
        [documentation](https://napari.org/stable/tutorials/fundamentals/viewer.html)
        """
        rendering.render_layout_3d_with_napari(self)
