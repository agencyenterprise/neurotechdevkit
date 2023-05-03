from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass
from typing import Mapping

import nest_asyncio
import numpy as np
import numpy.typing as npt
import stride
from frozenlist import FrozenList
from mosaic.types import Struct

from .. import rendering, scenarios
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
    choose_wavelet_for_mode,
    create_grid_circular_mask,
    create_grid_spherical_mask,
    drop_element,
    slice_field,
    wavelet_helper,
)

nest_asyncio.apply()


@dataclass
class Target:
    """A class for containing metadata for a target.

    Attributes:
        target_id: the string id of the target.
        center: the location of the center of the target (in meters).
        radius: the radius of the target (in meters).
        description: a text describing the target.
    """

    target_id: str
    center: npt.NDArray[np.float_]
    radius: float
    description: str


class Scenario(abc.ABC):
    """The base scenario"""

    _SCENARIO_ID: str
    _TARGET_OPTIONS: dict[str, Target]

    def __init__(
        self,
        origin: npt.NDArray[np.float_],
        complexity: str = "fast",
    ):
        """Initializes the scenario"""
        self._complexity = complexity
        if self._complexity != "fast":
            raise ValueError("the only complexity currently supported is 'fast'")

        self._origin = origin
        self._problem = self._compile_problem()
        self._sources: FrozenList[Source] = FrozenList()
        self._target_id: str

    @property
    def scenario_id(self) -> str:
        """The ID for this scenario."""
        return self._SCENARIO_ID

    @property
    def complexity(self) -> str:
        """The complexity level to use when simulating this scenario.

        Note: the only currently supported complexity is `fast`.

        Options are:
        * `fast`: uses a small grid size (large grid spacing) so that simulations are
            fast.
        * `accurate`: uses a large grid size (small grid spacing) so that simulation
            results are accurate.
        * `balanced`: a grid size and grid spacing balanced between `fast` and
            `accurate`.
        """
        return self._complexity

    @property
    def problem(self) -> stride.Problem:
        """The stride Problem object."""
        return self._problem

    @property
    def extent(self) -> npt.NDArray[np.float_]:
        """The extent of the spatial grid (in meters)."""
        return np.array(self.problem.space.size, dtype=float)

    @property
    def origin(self) -> npt.NDArray[np.float_]:
        """The spatial coordinates of grid position (0, 0, 0)."""
        return self._origin

    @property
    def shape(self) -> npt.NDArray[np.int_]:
        """The shape of the spatial grid (in number of grid points)."""
        return np.array(self.problem.space.shape, dtype=int)

    @property
    def dx(self) -> float:
        """The spacing (in meters) between spatial grid points.

        Spacing is the same in each spatial direction.
        """
        return self.problem.space.spacing[0]

    @property
    def ppw(self) -> float:
        # maybe choose lowest speed of sound?
        raise NotImplementedError()

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
    def ppp(self) -> float:
        raise NotImplementedError()

    @property
    def current_target_id(self) -> str:
        """Get or set the id of the currently selected target."""
        return self._target_id

    @current_target_id.setter
    def current_target_id(self, target_id: str) -> None:
        if target_id not in self._TARGET_OPTIONS:
            raise ValueError(
                f"{target_id} is not a valid target id."
                f" Options are: {self._TARGET_OPTIONS.keys()}."
            )
        self._target_id = target_id

    @property
    def target_options(self) -> dict[str, str]:
        """Information about each of the available targets for the scenario."""
        return {
            scenario_id: target.description
            for scenario_id, target in self._TARGET_OPTIONS.items()
        }

    @property
    def target(self) -> Target:
        """Details about the current target."""
        return self._TARGET_OPTIONS[self._target_id]

    @property
    def target_center(self) -> npt.NDArray[np.float_]:
        """The coordinates of the center of the target region (in meters)."""
        return self.target.center

    @property
    def target_radius(self) -> float:
        """The radius of the target region (in meters)."""
        return self.target.radius

    @property
    def materials(self) -> Mapping[str, Struct]:
        """A map between material name and material properties.

        vp: the speed of sound (in m/s).
        rho: the mass density (in kg/m³).
        alpha: the absorption (in dB/cm).
        render_color: the color used when rendering this material in the scenario layout
            plot.
        """
        return {name: material for name, material in self._material_layers}

    @property
    def layer_ids(self) -> Mapping[str, int]:
        """A map between material names and their layer id."""
        return {name: n for n, (name, _) in enumerate(self._material_layers)}

    @property
    def ordered_layers(self) -> list[str]:
        """An list of material names in order of their layer id."""
        return [name for name, _ in self._material_layers]

    @property
    @abc.abstractmethod
    def _material_layers(self) -> list[tuple[str, Struct]]:
        pass

    @property
    def material_properties(self):
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def _material_outline_upsample_factor(self) -> int:
        """The value of upsample_factor to use for this scenario when drawing material
        outlines.

        This parameter is internal to ndk, is not intended to be used directly.
        """
        pass

    @property
    def sources(self) -> FrozenList[Source]:
        """The list of sources currently defined.

        The source list can be edited only before a simulation is run. Once a simulation
        is run, the source list will be frozen and can no longer be modified.
        """
        return self._sources

    def _freeze_sources(self) -> None:
        """Freeze the sources list so that it can no longer be modified.

        This should be done before any simulation is run because the scenario and
        simulation are highly coupled and any changes to sources after the simulation
        could cause problems.
        """
        if not self._sources.frozen:
            self._sources.freeze()

    def get_layer_mask(self, layer_name: str) -> npt.NDArray[np.bool_]:
        """Returns the mask for the desired layer.

        The mask is `True` at each gridpoint where the requested layer exists,
        and False elsewhere.

        Args:
            layer_name: The name of the desired layer.

        Raises:
            ValueError: if `layer_name` does not match the name of one of the existing
                layers.

        Returns:
            A boolean array indicating which gridpoints correspond to the desired
                layer.
        """
        if layer_name not in self.layer_ids:
            raise ValueError(f"Layer '{layer_name}' does not exist.")
        layer = self.layer_ids[layer_name]
        layers_field = self.get_field_data("layer")
        return layers_field == layer

    @abc.abstractmethod
    def get_target_mask(self) -> npt.NDArray[np.bool_]:
        """Returns the mask for the target region.

        Returns:
            A boolean array indicating which gridpoints correspond to the target region.
        """
        pass

    def get_field_data(self, field: str) -> npt.NDArray[np.float_]:
        """Returns the array of field values across the scenario for a particular field.

        Common fields include:

        - vp: the speed of sound (in m/s)
        - rho: the density (in kg/m³)
        - alpha: absorption (in dB/cm)
        - layer: the layer id at each point over the grid

        Args:
            field: the name of the field to return.

        Returns:
            An array containing the field data.
        """
        return self.problem.medium.fields[field].data

    @abc.abstractmethod
    def _compile_problem(self) -> stride.Problem:
        pass

    def reset(self) -> None:
        """resets the scenario to initial state"""
        raise NotImplementedError()

    def add_source(self, source: Source) -> None:
        """Adds the specified source to the scenario.

        Sources can also added or removed by modifying the Scenario.sources list.

        Changes can only be made to sources before a simulation has started.

        Args:
            source: the source to add to the scenario.
        """
        self._sources.append(source)

    @abc.abstractmethod
    def get_default_source(self) -> Source:
        """Creates and returns a default source for this scenario.

        Returns:
            The default source.
        """
        pass

    def _ensure_source(self) -> None:
        """Ensures the scenario includes at least one source.

        If no source is pre-defined, the default source is included.
        """
        if len(self.sources) == 0:
            self.add_source(self.get_default_source())

    def simulate_steady_state(
        self,
        center_frequency: float = 5.0e5,
        points_per_period: int = 24,
        n_cycles_steady_state: int = 10,
        time_to_steady_state: float | None = None,
        recording_time_undersampling: int = 4,
        n_jobs: int | None = None,
    ) -> scenarios.SteadyStateResult:
        """Execute a steady-state simulation.

        In this simulation, the sources will emit pressure waves with a continuous
        waveform until steady-state has been reached. The steady-state wave amplitude is
        found by taking the Fourier transform of the last `n_cycles_steady_state` cycles
        of data and taking the amplitude of the component at the `center_frequency`.

        !!! note
            The only supported frequency currently supported is 500kHz. Any other
            value will raise a NotImplementedError.

        !!! warning
            A poor choice of arguments to this function can lead to a failed
            simulation. Make sure you understand the impact of supplying parameter
            values other than the default if you chose to do so.

        Args:
            center_frequency: The center frequency (in hertz) to use for the
                continuous-wave source output. No other value besides 500kHz (the
                default) is currently supported.
            points_per_period: The number of points in time to simulate for each cycle
                of the wave.
            n_cycles_steady_state: The number of complete cycles to use when calculating
                the steady-state wave amplitudes.
            time_to_steady_state: The amount of time (in seconds) the simulation should
                run before measuring the steady-state amplitude. If the value is None,
                this time will automatically be set to the amount of time it would take
                to propagate from one corner to the opposite and back in the medium with
                the slowest speed of sound in the scenario.
            recording_time_undersampling: The undersampling factor to apply to the time
                axis when recording simulation results. One out of every this many
                consecutive time points will be recorded and all others will be dropped.
            n_jobs: The number of threads to be used for the computation. Use None to
                leverage Devito automatic tuning.

        Raises:
            NotImplementedError: if a `center_frequency` other than 500kHz is provided.

        Returns:
            An object containing the result of the steady-state simulation.
        """

        if center_frequency != 5.0e5:
            raise NotImplementedError(
                "500kHz is the only currently supported center frequency. Support for"
                " other frequencies will be implemented once material properties as a"
                " function of frequency has been implemented."
            )

        problem = self.problem
        sim_time = select_simulation_time_for_steady_state(
            grid=problem.grid,
            materials=self.materials,
            freq_hz=center_frequency,
            time_to_steady_state=time_to_steady_state,
            n_cycles_steady_state=n_cycles_steady_state,
            delay=find_largest_delay_in_sources(self.sources),
        )
        problem.grid.time = create_time_grid(
            freq_hz=center_frequency, ppp=points_per_period, sim_time=sim_time
        )

        budget_time_and_memory_resources(
            grid_shape=self.shape,
            recording_time_undersampling=recording_time_undersampling,
            n_cycles_steady_state=n_cycles_steady_state,
            time_steps=problem.grid.time.grid.shape[0],
        )

        sub_problem = self._setup_sub_problem(center_frequency, "steady-state")
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

        # put the time axis last and remove the empty last frame
        wavefield = np.moveaxis(pde.wavefield.data[:-1], 0, -1)

        return scenarios.create_steady_state_result(
            scenario=self,
            center_frequency=center_frequency,
            effective_dt=self.dt * recording_time_undersampling,
            pde=pde,
            shot=sub_problem.shot,
            wavefield=wavefield,
            traces=traces,
        )

    def simulate_pulse(
        self,
        center_frequency: float = 5.0e5,
        points_per_period: int = 24,
        simulation_time: float | None = None,
        recording_time_undersampling: int = 4,
        n_jobs: int | None = None,
    ) -> scenarios.PulsedResult:
        """Execute a pulsed simulation in 2D.

        In this simulation, the sources will emit a pulse containing a few cycles of
        oscillation and then let the pulse propagate out to all edges of the scenario.

        !!! note
            The only supported frequency currently supported is 500kHz. Any other
            value will raise a NotImplementedError.

        !!! warning
            A poor choice of arguments to this function can lead to a failed
            simulation. Make sure you understand the impact of supplying parameter
            values other than the default if you chose to do so.

        Args:
            center_frequency: The center frequency (in hertz) to use for the
                continuous-wave source output. No other value besides
                500kHz (the default) is currently supported.
            points_per_period: The number of points in time to simulate for each cycle
                of the wave.
            simulation_time: The amount of time (in seconds) the simulation should run.
                If the value is None, this time will automatically be set to the amount
                of time it would take to propagate from one corner to the opposite in
                the medium with the slowest speed of sound in the scenario.
            recording_time_undersampling: The undersampling factor to apply to the time
                axis when recording simulation results. One out of every this many
                consecutive time points will be recorded and all others will be dropped.
            n_jobs: The number of threads to be used for the computation. Use None to
                leverage Devito automatic tuning.

        Raises:
            NotImplementedError: if a `center_frequency` other than 500kHz is provided.

        Returns:
            An object containing the result of the 2D pulsed simulation.
        """
        return self._simulate_pulse(
            center_frequency=center_frequency,
            points_per_period=points_per_period,
            simulation_time=simulation_time,
            recording_time_undersampling=recording_time_undersampling,
            n_jobs=n_jobs,
            slice_axis=None,
            slice_position=None,
        )

    def _simulate_pulse(
        self,
        center_frequency: float = 5.0e5,
        points_per_period: int = 24,
        simulation_time: float | None = None,
        recording_time_undersampling: int = 4,
        n_jobs: int | None = None,
        slice_axis: int | None = None,
        slice_position: float | None = None,
    ) -> scenarios.PulsedResult:
        """Execute a pulsed simulation.

        In this simulation, the sources will emit a pulse containing a few cycles of
        oscillation and then let the pulse propagate out to all edges of the scenario.

        Note: the only supported frequency currently supported is 500kHz. Any other
        value will raise a NotImplementedError.

        Warning: A poor choice of arguments to this function can lead to a failed
        simulation. Make sure you understand the impact of supplying parameter values
        other than the default if you chose to do so.

        Args:
            center_frequency: The center frequency (in hertz) to use for the
                continuous-wave source output. No other value besides
                500kHz (the default) is currently supported.
            points_per_period: The number of points in time to simulate for each cycle
                of the wave.
            simulation_time: The amount of time (in seconds) the simulation should run.
                If the value is None, this time will automatically be set to the amount
                of time it would take to propagate from one corner to the opposite in
                the medium with the slowest speed of sound in the scenario.
            recording_time_undersampling: The undersampling factor to apply to the time
                axis when recording simulation results. One out of every this many
                consecutive time points will be recorded and all others will be dropped.
            n_jobs: The number of threads to be used for the computation. Use None to
                leverage Devito automatic tuning.
            slice_axis: the axis along which to slice the 3D field to be recorded. If
                None, then the complete field wil be recorded. Use 0 for X axis, 1 for Y
                axis and 2 for Z axis. Only valid if `slice_position` is not None.
            slice_position: the position (in meters) along the slice axis at
                which the slice of the 3D field should be made. Only valid if
                `slice_axis` is not None.

        Raises:
            NotImplementedError: if a `center_frequency` other than 500kHz is provided.

        Returns:
            An object containing the result of the pulsed simulation.
        """

        if center_frequency != 5.0e5:
            raise NotImplementedError(
                "500kHz is the only currently supported center frequency. Support for"
                " other frequencies will be implemented once material properties as a"
                " function of frequency has been implemented."
            )

        problem = self.problem
        if simulation_time is None:
            simulation_time = select_simulation_time_for_pulsed(
                grid=problem.grid,
                materials=self.materials,
                delay=find_largest_delay_in_sources(self.sources),
            )
        problem.grid.time = create_time_grid(
            freq_hz=center_frequency, ppp=points_per_period, sim_time=simulation_time
        )

        if slice_axis is not None and slice_position is not None:
            recorded_slice = (slice_axis, slice_position)
        else:
            recorded_slice = None

        print(
            "Memory and time requirement estimations do not currently support pulsed"
            " simulations, so none will be provided."
        )

        sub_problem = self._setup_sub_problem(center_frequency, "pulsed")
        pde = self._create_pde()
        traces = self._execute_pde(
            pde=pde,
            sub_problem=sub_problem,
            save_bounds=self._get_pulsed_recording_time_bounds(),
            save_undersampling=recording_time_undersampling,
            wavefield_slice=self._wavefield_slice(slice_axis, slice_position),
            n_jobs=n_jobs,
        )

        # put the time axis last and remove the empty last frame
        wavefield = np.moveaxis(pde.wavefield.data[:-1], 0, -1)

        return scenarios.create_pulsed_result(
            scenario=self,
            center_frequency=center_frequency,
            effective_dt=self.dt * recording_time_undersampling,
            pde=pde,
            shot=sub_problem.shot,
            wavefield=wavefield,
            traces=traces,
            recorded_slice=recorded_slice,
        )

    def _setup_sub_problem(
        self, center_frequency: float, simulation_mode: str
    ) -> stride.SubProblem:
        """Sets up a stride `SubProblem` for the simulation.

        A SubProblem requires at least one source transducer. If no source is defined, a
        default source is used.

        Args:
            center_frequency: the center frequency (in hertz) of the source transducer.
            simulation_mode: the type of simulation which will be run.

        Returns:
            the `SubProblem` to use for the simulation.
        """
        self._ensure_source()
        self._freeze_sources()

        # create an actual list to avoid needing to use FrozenList as the type
        source_list = list(self.sources)
        shot = self._setup_shot(source_list, center_frequency, simulation_mode)
        return self.problem.sub_problem(shot.id)

    def _setup_shot(
        self, sources: list[Source], freq_hz: float, simulation_mode: str
    ) -> stride.Shot:
        """Creates the stride `Shot` for the simulation.

        Args:
            sources: the source transducers to use within the shot.
            freq_hz: the center frequency (in hertz) to use for the source wavelet.
            simulation_mode: the type of simulation which will be run.

        Returns:
            the `Shot` to use for the simulation.
        """
        problem = self.problem

        wavelet_name = choose_wavelet_for_mode(simulation_mode)
        wavelet = wavelet_helper(
            name=wavelet_name, freq_hz=freq_hz, time=problem.grid.time
        )
        return create_shot(problem, sources, self.origin, wavelet, self.dx)

    def _create_pde(self) -> stride.Operator:
        """Instantiates the stride `Operator` representing the PDE for the scenario.

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
        """Defines the region of of the grid that should be recorded.

        The first element of the tuple is for time, while all remaining elements are for
        space and should match the dimensionality of space.

        The returned time slice selects all time points.

        The returned space slices select only the data inside the defined extent of the
        scenario and drop data from the boundary layers.

        Args:
            slice_axis: the axis along which to slice the 3D field to be recorded. If
                None, then the complete field wil be recorded. Use 0 for X axis, 1 for Y
                axis and 2 for Z axis.
            slice_position: the position (in meters) along the slice axis at
                which the slice of the 3D field should be made.

        Returns:
            A tuple of slices defining the region of the grid to record.
        """
        space = self.problem.space

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

        offset_distance = slice_position - self.origin[slice_axis]
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
        """Validates that slicing axis and position are within scenario range.

        `slice_axis` should be either 0, 1, or 2 (for X, Y, Z).
        `slice_position` must be within boundaries for `slice_axis` extent.

        Args:
            slice_axis: the axis along which to slice the 3D field to be recorded. If
                None, then the complete field wil be recorded. Use 0 for X axis, 1 for Y
                axis and 2 for Z axis.
            slice_position: the position (in meters) along the slice axis at
                which the slice of the 3D field should be made.

        Raises:
            ValueError if axis is not 0, 1, 2.
            ValueError if `slice_position` falls outside the current range of
                `slice_axis`.
            ValueError if  `slice_axis` is None but `slice_position` is not None and
                vice versa.
        """
        if slice_axis not in (0, 1, 2):
            raise ValueError(
                "Unexpected value received for `slice_axis`. ",
                "Expected axis are 0 (X), 1 (Y) and/or 2 (Z).",
            )
        if (slice_axis is None and slice_position is not None) or (
            slice_axis is not None and slice_position is None
        ):
            raise ValueError(
                "Both `slice_axis` and `slice_position` must be passed together "
                "to correctly define how to slice the field. "
            )

        origin = self.origin
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
        """Defines the indices bounding the period of time to be recorded.

        For steady-state simulations, we only want to keep the last few cycles of the
        simulation.

        Args:
            ppp: The number of points in time per phase to simulate for each cycle of
                the wave.
            n_cycles: The number of complete cycles to record at the end of the
                simulation.

        Returns:
            A tuple containing the time indices bounding the period of time which should
                be recorded for a steady-state simulation.
        """
        n_frames = ppp * n_cycles
        time = self.problem.time
        return (time.num - n_frames, time.num - 1)

    def _get_pulsed_recording_time_bounds(self) -> tuple[int, int]:
        """Defines the indices bounding the period of time to be recorded.

        For pulsed simulations, we want to keep the data from all timesteps.

        Returns:
            A tuple containing the time indices bounding the period of time which should
                be recorded for a pulsed simulation.
        """
        time = self.problem.time
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
        """Executes the PDE for the simulation.

        Args:
            pde: The `Operator` containing the PDE to execute.
            sub_problem: The `SubProblem` containing details of the source and waveform
                for the simulation.
            save_bounds: The time indices bounding the period of time to be recorded.
            save_undersampling: The undersampling factor to apply to the time axis when
                recording simulation results.
            wavefield_slice: A tuple of slices defining the region of the grid to
                record.
            n_jobs: The number of threads to be used for the computation. Use None to
                leverage Devito automatic tuning.
        Returns:
            The `Traces` which are produced by the simulation.
        """
        problem = self.problem

        devito_args = {}
        if n_jobs is not None:
            devito_args = dict(nthreads=n_jobs)
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
                save_bounds=save_bounds,
                save_undersampling=save_undersampling,
                wavefield_slice=wavefield_slice,
                devito_args=devito_args,
            )
        )

    @abc.abstractmethod
    def render_material_property(
        self,
        name,
        show_orientation: bool = True,
        show_sources: bool = True,
        show_target: bool = True,
    ) -> None:
        # speed of sound, density, and absorption
        # maybe split these out into 3 separate functions?
        pass


class Scenario2D(Scenario):
    def get_target_mask(self) -> npt.NDArray[np.bool_]:
        target_mask = create_grid_circular_mask(
            grid=self.problem.grid,
            origin=self.origin,
            center=self.target_center,
            radius=self.target_radius,
        )
        return target_mask

    def render_layout(
        self,
        show_sources: bool = True,
        show_target: bool = True,
        show_material_outlines: bool = False,
    ) -> None:
        """Creates a matplotlib figure showing the 2D scenario layout.

        The grid can be turned on via:

        `plt.grid(True)`

        Args:
            show_sources: whether or not to show the source transducer layer.
            show_target: whether or not to show the target layer.
            show_material_outlines: whether or not to display a thin white outline of
                the transition between different materials.
        """
        color_sequence = [
            self.materials[name].render_color for name in self.ordered_layers
        ]
        field = self.get_field_data("layer").astype(int)
        fig, ax = rendering.create_layout_fig(
            self.extent, self.origin, color_sequence, field
        )

        # add layers
        if show_material_outlines:
            rendering.draw_material_outlines(
                ax=ax,
                material_field=field,
                dx=self.dx,
                origin=self.origin,
                upsample_factor=self._material_outline_upsample_factor,
            )
        if show_target:
            rendering.draw_target(ax, self.target_center, self.target_radius)
        if show_sources:
            self._ensure_source()
            for source in self.sources:
                drawing_params = rendering.SourceDrawingParams(
                    position=source.position,
                    direction=source.unit_direction,
                    aperture=source.aperture,
                    focal_length=source.focal_length,
                    source_is_flat=rendering.source_should_be_flat(source),
                )
                rendering.draw_source(ax, drawing_params)

        rendering.configure_layout_plot(
            fig=fig,
            ax=ax,
            color_sequence=color_sequence,
            layer_labels=self.ordered_layers,
            show_sources=show_sources,
            show_target=show_target,
            extent=self.extent,
            origin=self.origin,
        )


class Scenario3D(Scenario):
    @abc.abstractmethod
    def get_default_slice_axis(self) -> int:
        """Returns the default slice_axis for this scenario.

        This field is used if the slice_axis is not specified when plotting 3D data in
        2D.

        Returns:
            The default slice_axis for this scenario.
        """
        pass

    @abc.abstractmethod
    def get_default_slice_position(self, axis: int) -> float:
        """Returns the default slice_position (in meters) for this scenario.

        This field is used if the slice_position is not specified when plotting 3D data
        in 2D.

        Args:
            axis: the slice axis.

        Returns:
            The default slice_position for this scenario along the given axis.
        """
        pass

    @property
    @abc.abstractmethod
    def viewer_config_3d(self) -> rendering.ViewerConfig3D:
        """Configuration parameters for 3D visualization of this scenario."""
        pass

    def get_target_mask(self) -> npt.NDArray[np.bool_]:
        target_mask = create_grid_spherical_mask(
            grid=self.problem.grid,
            origin=self.origin,
            center=self.target_center,
            radius=self.target_radius,
        )
        return target_mask

    def simulate_pulse(
        self,
        center_frequency: float = 5.0e5,
        points_per_period: int = 24,
        simulation_time: float | None = None,
        recording_time_undersampling: int = 4,
        n_jobs: int | None = None,
        slice_axis: int | None = None,
        slice_position: float | None = None,
    ) -> scenarios.PulsedResult:
        """Execute a pulsed simulation in 3D.

        In this simulation, the sources will emit a pulse containing a few cycles of
        oscillation and then let the pulse propagate out to all edges of the scenario.

        Note: the only supported frequency currently supported is 500kHz. Any other
        value will raise a NotImplementedError.

        Warning: A poor choice of arguments to this function can lead to a failed
        simulation. Make sure you understand the impact of supplying parameter values
        other than the default if you chose to do so.

        Args:
            center_frequency: The center frequency (in hertz) to use for the
                continuous-wave source output. No other value besides
                500kHz (the default) is currently supported.
            points_per_period: The number of points in time to simulate for each cycle
                of the wave.
            simulation_time: The amount of time (in seconds) the simulation should run.
                If the value is None, this time will automatically be set to the amount
                of time it would take to propagate from one corner to the opposite in
                the medium with the slowest speed of sound in the scenario.
            recording_time_undersampling: The undersampling factor to apply to the time
                axis when recording simulation results. One out of every this many
                consecutive time points will be recorded and all others will be dropped.
            n_jobs: The number of threads to be used for the computation. Use None to
                leverage Devito automatic tuning.
            slice_axis: the axis along which to slice the 3D field to be recorded. If
                None, then the complete field wil be recorded. Use 0 for X axis, 1 for Y
                axis and 2 for Z axis. Only valid if `slice_position` is not None.
            slice_position: the position (in meters) along the slice axis at
                which the slice of the 3D field should be made. Only valid if
                `slice_axis` is not None.

        Raises:
            NotImplementedError: if a `center_frequency` other than 500kHz is provided.

        Returns:
            An object containing the result of the 3D pulsed simulation.
        """
        return self._simulate_pulse(
            center_frequency=center_frequency,
            points_per_period=points_per_period,
            simulation_time=simulation_time,
            recording_time_undersampling=recording_time_undersampling,
            n_jobs=n_jobs,
            slice_axis=slice_axis,
            slice_position=slice_position,
        )

    def render_layout(
        self,
        slice_axis: int | None = None,
        slice_position: float | None = None,
        show_sources: bool = True,
        show_target: bool = True,
        show_material_outlines: bool = False,
    ) -> None:
        """Creates a matplotlib figure showing a 2D slice of the scenario layout.

        In order to visualize the 3D scenario in a 2D plot, a slice through the scenario
        needs to be specified via `slice_axis` and `slice_position`. Eg. to take a slice
        at z=0.01 m, use `slice_axis=2` and `slice_position=0.01`.

        The grid can be turned on via:

        `plt.grid(True)`

        Args:
            slice_axis: the axis along which to slice. If None, then the value returned
                by `scenario.get_default_slice_axis()` is used.
            slice_position: the position (in meters) along the slice axis at
                which the slice should be made. If None, then the value returned by
                `scenario.get_default_slice_position()` is used.
            show_sources: whether or not to show the source transducer layer.
            show_target: whether or not to show the target layer.
            show_material_outlines: whether or not to display a thin white outline of
                the transition between different materials.
        """
        if slice_axis is None:
            slice_axis = self.get_default_slice_axis()
        if slice_position is None:
            slice_position = self.get_default_slice_position(slice_axis)

        color_sequence = [
            self.materials[name].render_color for name in self.ordered_layers
        ]
        field = self.get_field_data("layer").astype(int)
        field = slice_field(field, self, slice_axis, slice_position)
        extent = drop_element(self.extent, slice_axis)
        origin = drop_element(self.origin, slice_axis)
        fig, ax = rendering.create_layout_fig(extent, origin, color_sequence, field)

        # add layers
        if show_material_outlines:
            rendering.draw_material_outlines(
                ax=ax,
                material_field=field,
                dx=self.dx,
                origin=origin,
                upsample_factor=self._material_outline_upsample_factor,
            )
        if show_target:
            target_loc = drop_element(self.target_center, slice_axis)
            rendering.draw_target(ax, target_loc, self.target_radius)
        if show_sources:
            self._ensure_source()
            for source in self.sources:
                drawing_params = rendering.SourceDrawingParams(
                    position=drop_element(source.position, slice_axis),
                    direction=drop_element(source.unit_direction, slice_axis),
                    aperture=source.aperture,
                    focal_length=source.focal_length,
                    source_is_flat=rendering.source_should_be_flat(source),
                )
                rendering.draw_source(ax, drawing_params)

        axis_names = np.array(["X", "Y", "Z"])
        vert_name, horz_name = drop_element(axis_names, slice_axis)
        slice_name = axis_names[slice_axis]
        rendering.configure_layout_plot(
            fig=fig,
            ax=ax,
            color_sequence=color_sequence,
            layer_labels=self.ordered_layers,
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
        https://napari.org/stable/tutorials/fundamentals/viewer.html
        """
        rendering.render_layout_3d_with_napari(self)
