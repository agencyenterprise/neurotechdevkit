from __future__ import annotations

import abc
import gzip
import os
import pathlib
import pickle
import shutil
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import stride
from matplotlib.animation import FuncAnimation

from .. import rendering, scenarios
from . import _metrics as metrics
from ._utils import drop_element, slice_field


@dataclass
class Result(abc.ABC):
    """A base container for holding the results of a simulation.

    This class should not be instantiated, use SteadyStateResult2D, SteadyStateResult3D,
    PulsedResult2D, or PulsedResult3D.

    Attributes:
        scenario: The scenario from which this result came.
        center_frequency: the center frequency (in hertz) of the sources.
        effective_dt: the effective time step (in seconds) along the time axis of the
            wavefield. This can differ from the simulation dt if the recording
            undersampling factor is larger than 1.
        pde: the stride Operator that was executed to run the simulation.
        shot: the stride Shot which was used for the simulation.
        wavefield: an array containing the resulting simulation data.
        traces: the stride Traces object returned from executing the pde.
    """

    scenario: scenarios.Scenario
    center_frequency: float
    effective_dt: float
    pde: stride.Operator
    shot: stride.Shot
    wavefield: npt.NDArray[np.float_]
    traces: stride.Traces

    def save_to_disk(self, filepath: str | pathlib.Path) -> None:
        """Saves the result to disk to a gzip compressed file.

        The gzip compressed file is a pickle object.

        !!! warning
            This functionality is experimental, so do do not be surprised if you
            encounter issues calling this function.

        This function is particularly useful if simulation is performed in the cloud but
        the user would like to download the results in order to visualize them locally
        in 3D.

        Args:
            filepath: The path to the file where the results should be exported. Usually
                a .pkl.gz file.
        """
        try:
            save_data = self._generate_save_data()
            with gzip.open(filepath, "wb") as f:
                pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"A result artifact has been saved to {filepath}.")
        except Exception as e:
            raise Exception("Unable to save result artifact to disk.") from e

    @abc.abstractmethod
    def _generate_save_data(self) -> dict:
        """Collects objects to be saved to disk for simulation results.

        The result data saved to disk will depend on the type of simulation. Currently
        `PulsedResults` and `SteadyStateResults` results.

        For `PulsedResults` the entire wavefield will be saved.
        For `SteadyStateResults` only the steady state will be saved.

        Returns:
            A dictionary with the objects to be saved to disk.
        """
        ...


@dataclass
class SteadyStateResult(Result):
    """A base container for holding the results of a steady-state simulation.

    This class should not be instantiated, use SteadyStateResult2D or
    SteadyStateResult3D.

    Attributes:
        scenario: The scenario from which this result came.
        center_frequency: the center frequency (in hertz) of the sources.
        effective_dt: the effective time step (in seconds) along the time axis of the
            wavefield. This can differ from the simulation dt if the recording
            undersampling factor is larger than 1.
        pde: the stride Operator that was executed to run the simulation.
        shot: the stride Shot which was used for the simulation.
        wavefield: an array containing the resulting simulation data.
        traces: the stride Traces object returned from executing the pde.
    """

    steady_state: npt.NDArray[np.float_] | None = None

    def _extract_steady_state(
        self, by_slice: bool | None = None
    ) -> npt.NDArray[np.float_]:
        """Extracts the steady state results from the simulation wavefield.

        Args:
            by_slice: If False, the fft is executed over the entire wavefield array at
                once, which is faster but memory intensive. If True, the fft is
                executed over the wavefield slice by slice, which is slower but uses
                less memory. If None (the default), the fft will be over the entire
                wavefield if the wavefield has 2 spatial dimensions and by slice if it
                has 3 spatial dimensions.

        Returns:
            A numpy array containing the steady-state wave amplitudes (in pascals).
        """
        ppp = int(round(1 / self.center_frequency / self.effective_dt))
        n_cycles = int(self.wavefield.shape[-1] / ppp)
        data = self.wavefield[..., -n_cycles * ppp :]

        if by_slice is None:
            by_slice = len(data.shape) == 4

        return _extract_steady_state_amplitude(
            data, self.center_frequency, self.effective_dt, by_slice
        )

    def get_steady_state(self) -> npt.NDArray[np.float_]:
        """Returns the steady-state array and while computing it if necessary.

        Returns:
            An array containing steady-state pressure wave amplitudes (in pascals).
        """
        if self.steady_state is None:
            self.steady_state = self._extract_steady_state()
        return self.steady_state

    @property
    def metrics(self) -> dict[str, dict[str, str | float]]:
        """A dictionary containing metrics and their descriptions.

        The keys for the dictionary are the names of the metrics. The value for each
        metric is another dictionary containing the following:
            value: The value of the metric.
            unit-of-measurement: The unit of measurement for the metric.
            description: A text description of the metric.
        """
        self.get_steady_state()
        return metrics.calculate_all_metrics(self)

    def _generate_save_data(self) -> dict:
        """Collects objects to be saved to disk for steady-state simulation results.

        Returns:
            A dictionary with the objects to be saved to disk.
        """
        save_data = {
            "result_type": type(self).__name__,
            "scenario_id": self.scenario.scenario_id,
            "sources": self.scenario.sources,
            "target_id": self.scenario.current_target_id,
            "center_frequency": self.center_frequency,
            "effective_dt": self.effective_dt,
            "steady_state": self.get_steady_state(),
        }

        return save_data


class SteadyStateResult2D(SteadyStateResult):
    """A container for holding the results of a 2D steady-state simulation.

    Attributes:
        scenario: The 2D scenario from which this result came.
        center_frequency: the center frequency (in hertz) of the sources.
        effective_dt: the effective time step (in seconds) along the time axis of the
            wavefield. This can differ from the simulation dt if the recording
            downsampling factor is larger than 1.
        pde: the stride Operator that was executed to run the simulation.
        shot: the stride Shot which was used for the simulation.
        wavefield: a 3 dimensional array (two axes for space and one for time)
            containing the resulting simulation data.
        traces: the stride Traces object returned from executing the pde.
    """

    scenario: scenarios.Scenario2D

    def render_steady_state_amplitudes(
        self,
        show_sources: bool = True,
        show_target: bool = True,
        show_material_outlines: bool = True,
    ) -> None:
        """Create a matplotlib figure with the steady-state pressure wave amplitude.

        The grid can be turned on via:

        `plt.grid(True)`

        Args:
            show_sources: whether or not to show the source transducer layer.
            show_target: whether or not to show the target layer.
            show_material_outlines: whether or not to display a thin white outline of
                the transition between different materials.
        """
        field = self.get_steady_state()

        fig, ax = rendering.create_steady_state_figure(
            self.scenario.extent,
            self.scenario.origin,
            field,
        )

        # add layers
        if show_material_outlines:
            material_field = self.scenario.get_field_data("layer").astype(int)
            rendering.draw_material_outlines(
                ax=ax,
                material_field=material_field,
                dx=self.scenario.dx,
                origin=self.scenario.origin,
                upsample_factor=self.scenario._material_outline_upsample_factor,
            )
        if show_target:
            rendering.draw_target(
                ax, self.scenario.target_center, self.scenario.target_radius
            )
        if show_sources:
            for source in self.scenario.sources:
                drawing_params = rendering.SourceDrawingParams(
                    position=source.position,
                    direction=source.unit_direction,
                    aperture=source.aperture,
                    focal_length=source.focal_length,
                    source_is_flat=rendering.source_should_be_flat(source),
                )
                rendering.draw_source(ax, drawing_params)

        rendering.configure_result_plot(
            fig=fig,
            ax=ax,
            show_sources=show_sources,
            show_target=show_target,
            extent=self.scenario.extent,
            origin=self.scenario.origin,
            vertical_label="X",
            horizontal_label="Y",
            title="Steady-State Wave Amplitude",
        )


class SteadyStateResult3D(SteadyStateResult):
    """A container for holding the results of a 3D steady-state simulation.

    Attributes:
        scenario: The 3D scenario from which this result came.
        center_frequency: the center frequency (in hertz) of the sources.
        effective_dt: the effective time step (in seconds) along the time axis of the
            wavefield. This can differ from the simulation dt if the recording
            downsampling factor is larger than 1.
        pde: the stride Operator that was executed to run the simulation.
        shot: the stride Shot which was used for the simulation.
        wavefield: a 4 dimensional array (three axes for space and one for time)
            containing the resulting simulation data.
        traces: the stride Traces object returned from executing the pde.
    """

    scenario: scenarios.Scenario3D

    def render_steady_state_amplitudes(
        self,
        slice_axis: int | None = None,
        slice_position: float | None = None,
        show_sources: bool = True,
        show_target: bool = True,
        show_material_outlines: bool = True,
    ) -> None:
        """Create a matplotlib figure with the steady-state pressure wave amplitude.

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
            slice_axis = self.scenario.get_default_slice_axis()
        if slice_position is None:
            slice_position = self.scenario.get_default_slice_position(slice_axis)

        field = self.get_steady_state()

        field = slice_field(field, self.scenario, slice_axis, slice_position)
        extent = drop_element(self.scenario.extent, slice_axis)
        origin = drop_element(self.scenario.origin, slice_axis)

        fig, ax = rendering.create_steady_state_figure(extent, origin, field)

        # add layers
        if show_material_outlines:
            material_field = self.scenario.get_field_data("layer").astype(int)
            material_field_2d = slice_field(
                material_field, self.scenario, slice_axis, slice_position
            )
            rendering.draw_material_outlines(
                ax=ax,
                material_field=material_field_2d,
                dx=self.scenario.dx,
                origin=origin,
                upsample_factor=self.scenario._material_outline_upsample_factor,
            )
        if show_target:
            target_loc = drop_element(self.scenario.target_center, slice_axis)
            rendering.draw_target(ax, target_loc, self.scenario.target_radius)
        if show_sources:
            for source in self.scenario.sources:
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
        rendering.configure_result_plot(
            fig=fig,
            ax=ax,
            show_sources=show_sources,
            show_target=show_target,
            extent=extent,
            origin=origin,
            vertical_label=vert_name,
            horizontal_label=horz_name,
            title=(
                "Steady-State Wave Amplitude"
                f"\nSlice: {slice_name} = {slice_position} m"
            ),
        )

    def render_steady_state_amplitudes_3d(
        self,
    ) -> None:
        """Render the steady-state simulation results in 3D using napari.

        This function requires the napari package to be installed.

        !!! warning
            Integration with napari is experimental, and do not be surprised if you
            encounter issues calling this function.

        This will open up the napari interactive GUI in a separate window. The GUI
        contains many different controls for controlling the view of the data as well as
        the rendering of the layers. Among these, you can drag the scenario to view it
        from different angles, zoom in our out, and turn layers on or off.

        See the
        [napari](https://napari.org/stable/tutorials/fundamentals/viewer.html)
        documentation for more information on the GUI.
        """
        rendering.render_amplitudes_3d_with_napari(self)


def _extract_steady_state_amplitude(
    data: npt.NDArray[np.float_], freq_hz: float, dt: float, by_slice: bool
) -> npt.NDArray[np.float_]:
    """Extracts the amplitude of steady-state waves using an FFT.

    Note: in order to get the best results, dt should fit evenly into one cycle and we
    need to integrate over integer number of cycles.

    Args:
        data: The wave data to process. The array should contain 2 or 3 space dimensions
            followed by the time dimension.
        freq_hz: The frequency (in hertz) to extract from the FFT.
        dt: The time axis spacing (in seconds).
        by_slice: If False, the fft is executed over the entire data array at once,
            which is faster but memory intensive. If True, the fft is executed over the
            data slice by slice, which is slower but uses less memory.

    Returns:
        The steady-state wave amplitudes over the spatial dimensions (in pascals).
    """

    freqs = np.fft.fftfreq(data.shape[-1], d=dt)
    freq_idx = np.argwhere(np.abs(freqs - freq_hz) < 1e-5).item()
    scaling = data.shape[-1] / 2

    if not by_slice:
        fft_res = np.fft.fft(data, axis=-1)
        amplitudes = np.abs(fft_res)[..., freq_idx]
        return amplitudes / scaling

    amplitudes = np.zeros(data.shape[:-1])

    slice_axis = np.argmax(data.shape[:-1])
    n_space = len(data.shape[:-1])
    space_slices: list[Any] = [slice(0, None) for _ in range(n_space)]
    time_slice = [slice(0, None)]

    for idx in range(data.shape[slice_axis]):
        space_slices[slice_axis] = idx
        measurement_slice = data[tuple(space_slices + time_slice)]
        fft_res = np.fft.fft(measurement_slice, axis=-1)
        amplitudes[tuple(space_slices)] = np.abs(fft_res)[..., freq_idx]

    return amplitudes / scaling


@dataclass
class PulsedResult(Result):
    """A base container for holding the results of a pulsed simulation.

    This class should not be instantiated, use PulsedResult2D or PulsedResult3D.

    Attributes:
        scenario: The scenario from which this result came.
        center_frequency: the center frequency (in hertz) of the sources.
        effective_dt: the effective time step (in seconds) along the time axis of the
            wavefield. This can differ from the simulation dt if the recording
            undersampling factor is larger than 1.
        pde: the stride Operator that was executed to run the simulation.
        shot: the stride Shot which was used for the simulation.
        wavefield: an array containing the resulting simulation data.
        traces: the stride Traces object returned from executing the pde.
    """

    recorded_slice: tuple[int, float] | None = None

    def _recording_times(self) -> npt.NDArray[np.float_]:
        """Computes the time (in seconds) for each recorded frame in the wavefield.

        Returns:
            A 1D array with the time in seconds for each step.
        """
        start_time = 0
        times = (
            np.arange(start=start_time, stop=self.wavefield.shape[-1])
            * self.effective_dt
        )
        return times

    def _validate_time_lim(self, time_lim: tuple[np.float_, np.float_]) -> None:
        """Validates the input time limit for the animation.

        Args:
            time_lim: the input time limit tuple to validate. The expected format is
                (minimum_time, maximum_time).

        Raises:
            ValueError if the input time limit isn't valid.
        """
        if (
            (len(time_lim) != 2)
            or (time_lim[0] < 0)
            or (time_lim[1] <= time_lim[0])
            or (time_lim[1] > self._recording_times()[-1])
        ):
            raise ValueError("Wrong value for time_lim")

    @staticmethod
    def _check_ffmpeg_is_installed() -> None:
        """Checks that ffmpeg command is available.

        It is required to save the animations to disk.

        Raises:
            ModuleNotFoundError if `ffmpeg` is not installed.
        """
        if shutil.which("ffmpeg") is None:
            raise ModuleNotFoundError(
                "ffmpeg not found. `ffmpeg` is needed to create the animation "
                "To install it in unix run: $ sudo apt install ffmpeg "
                "To install it in MacOS (using brew) run: brew install ffmpeg "
                "See https://ffmpeg.org/download.html for less automated installation "
                "instructions."
            )

    @staticmethod
    def _validate_file_name(file_name, overwrite) -> None:
        """Verify that a valid `file_name` is passed.

        Only mp4 format is supported. `file_name` must have mp4 extension.
        If `file_name` exists, `overwrite` must set to true.

        Raises:
            ValueError if the format extension is not `mp4`.
            FileExistsError if the file that is attempting to create exists and
                `overwrite` is set to False.
        """
        if not file_name.lower().endswith(".mp4"):
            raise ValueError(
                "Only mp4 format supported currently. Please provide a filename with "
                " .mp4 as the extension."
            )

        if os.path.exists(file_name) and not overwrite:
            raise FileExistsError(
                f"File {file_name} exists. Pass `overwrite=True` to replace it ",
                "or change the value for `fname`",
            )

    def _generate_save_data(self) -> dict:
        """Collects objects to be saved to disk for pulsed simulation results.

        Returns:
            A dictionary with the objects to be saved to disk.
        """
        save_data = {
            "result_type": type(self).__name__,
            "scenario_id": self.scenario.scenario_id,
            "sources": self.scenario.sources,
            "target_id": self.scenario.current_target_id,
            "center_frequency": self.center_frequency,
            "effective_dt": self.effective_dt,
            "wavefield": self.wavefield,
        }

        return save_data


class PulsedResult2D(PulsedResult):
    """A container for holding the results of a 2D pulsed simulation.

    Attributes:
        scenario: The 2D scenario from which this result came.
        center_frequency: the center frequency (in hertz) of the sources.
        effective_dt: the effective time step (in seconds) along the time axis of the
            wavefield. This can differ from the simulation dt if the recording
            downsampling factor is larger than 1.
        pde: the stride Operator that was executed to run the simulation.
        shot: the stride Shot which was used for the simulation.
        wavefield: a 3 dimensional array (two axes for space and one for time)
            containing the resulting simulation data.
        traces: the stride Traces object returned from executing the pde.
    """

    scenario: scenarios.Scenario2D

    def render_pulsed_simulation_animation(
        self,
        show_sources: bool = True,
        show_target: bool = True,
        show_material_outlines: bool = True,
        n_frames_undersampling: int = 1,
        time_lim: tuple[np.float_, np.float_] | None = None,
        norm: str = "linear",
    ) -> FuncAnimation:
        """Creates a matplotlib animation with the time evolution of the wavefield.

        The created animation will be displayed as an interactive widget in a IPython or
        Jupyter Notebook environment.
        In a non-interactive environment (script) the result of this animation would be
        lost. Use `create_video_file` method instead.

        Args:
            show_sources: whether or not to show the source transducer layer.
            show_target: whether or not to show the target layer.
            show_material_outlines: whether or not to display a thin white outline of
                the transition between different materials.
            n_frames_undersampling: the number of time steps to be skipped when creating
                the animation.
            time_lim: the input time limit tuple to validate. The expected format is
                (minimum_time, maximum_time).
            norm: The normalization method used to scale scalar data to the [0, 1]
                range before mapping to colors using cmap. For a list
                of available scales, call `matplotlib.scale.get_scale_names()`.

        Returns:
            An matplotlib animation object.
        """
        animation = self._build_animation(
            show_sources=show_sources,
            show_target=show_target,
            show_material_outlines=show_material_outlines,
            time_lim=time_lim,
            n_frames_undersampling=n_frames_undersampling,
            norm=norm,
        )
        rendering.configure_matplotlib_for_embedded_animation()
        return animation

    def create_video_file(
        self,
        file_name: str,
        show_sources: bool = True,
        show_target: bool = True,
        show_material_outlines: bool = True,
        n_frames_undersampling: int = 1,
        time_lim: tuple[np.float_, np.float_] | None = None,
        norm: str = "linear",
        fps: int = 25,
        dpi: int = 100,
        bitrate: int = 2500,
        overwrite: bool = False,
    ) -> None:
        """Saves a `mp4` animation file to disk with the results of the simulation.

        Currently only mp4 format supported.
        `ffmpeg` command line tools needs to be installed.

        Args:
            file_name: the file with path an extension where the animation would be
                saved. Currently only supports mp4 extension.
            show_sources: whether or not to show the source transducer layer.
            show_target: whether or not to show the target layer.
            show_material_outlines: whether or not to display a thin white outline of
                the transition between different materials.
            n_frames_undersampling: the number of time steps to be skipped when creating
                the animation.
            time_lim: the input time limit tuple to validate. The expected format is
                (minimum_time, maximum_time).
            norm: The normalization method used to scale scalar data to the [0, 1]
                range before mapping to colors using cmap. For a list of available
                scales, call `matplotlib.scale.get_scale_names()`.
            fps: the frames per second in the animation.
            dpi: the number of dots per inch in the frames of the animation.
            bitrate: the bitrate for the saved movie file, which is one way to control
                the output file size and quality.
            overwrite: a boolean that allows the animation to be saved with the same
                file name that is already exists.
        """
        self._check_ffmpeg_is_installed()
        self._validate_file_name(file_name, overwrite)

        animation = self._build_animation(
            show_sources=show_sources,
            show_target=show_target,
            show_material_outlines=show_material_outlines,
            time_lim=time_lim,
            n_frames_undersampling=n_frames_undersampling,
            norm=norm,
        )

        rendering.save_animation(
            animation, file_name, fps=fps, dpi=dpi, bitrate=bitrate
        )
        print(f"Saved to {file_name} file.")

    @rendering.video_only_output
    def _build_animation(
        self,
        time_lim: tuple[np.float_, np.float_] | None = None,
        n_frames_undersampling: int = 1,
        show_sources: bool = True,
        show_target: bool = True,
        show_material_outlines: bool = True,
        norm: str = "linear",
    ) -> FuncAnimation:
        """Create a matplotlib animation with the time evolution of the pressure waves.

        Raises:
            ValueError if the wave field is not in 2D.

        Args:
            time_lim: the input time limit tuple to validate. The expected format is
                (minimum_time, maximum_time).
            n_frames_undersampling: the number of time steps to be skipped when creating
                the animation.
            show_sources: whether or not to show the source transducer layer.
            show_target: whether or not to show the target layer.
            show_material_outlines: whether or not to display a thin white outline of
                the transition between different materials.
            norm: The normalization method used to scale scalar data to the [0, 1]
                range before mapping to colors using cmap. For a list of available
                scales, call `matplotlib.scale.get_scale_names()`.

        Returns:
            A matplotlib animation object.
        """

        extent = self.scenario.extent
        origin = self.scenario.origin
        wavefield = self.wavefield

        assert len(extent) == 2, "The rendering only supports 2D fields."

        if time_lim is not None:
            self._validate_time_lim(time_lim)
            times = self._recording_times()
            time_mask = np.logical_and(times >= time_lim[0], times <= time_lim[1])
            wavefield = wavefield[:, :, time_mask]

        min_pressure = wavefield.min()
        max_pressure = wavefield.max()

        # create base figure
        fig, ax = rendering.create_pulsed_figure(origin, extent, wavefield, norm)

        # add layers
        if show_material_outlines:
            material_field = self.scenario.get_field_data("layer").astype(int)
            rendering.draw_material_outlines(
                ax=ax,
                material_field=material_field,
                dx=self.scenario.dx,
                origin=self.scenario.origin,
                upsample_factor=self.scenario._material_outline_upsample_factor,
            )

        if show_target:
            rendering.draw_target(
                ax, self.scenario.target_center, self.scenario.target_radius
            )

        if show_sources:
            for source in self.scenario.sources:
                drawing_params = rendering.SourceDrawingParams(
                    position=source.position,
                    direction=source.unit_direction,
                    aperture=source.aperture,
                    focal_length=source.focal_length,
                    source_is_flat=rendering.source_should_be_flat(source),
                )
                rendering.draw_source(ax, drawing_params)

        rendering.configure_result_plot(
            fig,
            ax,
            extent=extent,
            origin=origin,
            show_sources=show_sources,
            show_target=show_target,
            clim=(min_pressure, max_pressure),
            vertical_label="X",
            horizontal_label="Y",
            title="Pulsed Wave Amplitude",
        )
        animation = rendering.make_animation(
            fig,
            ax,
            wavefield=wavefield,
            n_frames_undersampling=n_frames_undersampling,
        )
        return animation


class PulsedResult3D(PulsedResult):
    """A container for holding the results of a 3D pulsed simulation.

    Attributes:
        scenario: The 3D scenario from which this result came.
        center_frequency: the center frequency (in hertz) of the sources.
        effective_dt: the effective time step (in seconds) along the time axis of the
            wavefield. This can differ from the simulation dt if the recording
            downsampling factor is larger than 1.
        pde: the stride Operator that was executed to run the simulation.
        shot: the stride Shot which was used for the simulation.
        wavefield: a 4 dimensional array (three axes for space and one for time)
            containing the resulting simulation data.
        traces: the stride Traces object returned from executing the pde.
    """

    scenario: scenarios.Scenario3D

    def _validate_slicing_options(
        self, slice_axis: int | None, slice_position: float | None
    ) -> None:
        """Checks that the slicing arguments are consistent with the recorded field.

        Only one slicing of the field is permitted, either at recording time or at
        rendering time. If the user recorded only a slice of the 3D field, slicing
        arguments at rendering time should be the default `None`.

        Args:
            slice_axis: the axis along which to slice the 3D field to be recorded. If
                None, then the complete field wil be recorded.
            slice_position: the position (in meters) along the slice axis at
                which the slice of the 3D field should be made.

        Raises:
            ValueError if the user passes not None arguments for `slice_axis` and/or
            `slice_position` when only a plane of the 3D field was recorded.
        """
        if self.recorded_slice is not None:
            if slice_axis is not None or slice_position is not None:
                raise ValueError(
                    "Recorded results are already a 2D slice of a 3D simulation, it "
                    "can't be sliced further. `slice_axis` and `slice_position` must "
                    "be None."
                )

    def render_pulsed_simulation_animation(
        self,
        show_sources: bool = True,
        show_target: bool = True,
        show_material_outlines: bool = True,
        n_frames_undersampling: int = 1,
        slice_axis: int | None = None,
        slice_position: float | None = None,
        time_lim: tuple[np.float_, np.float_] | None = None,
        norm: str = "linear",
    ) -> FuncAnimation:
        """Creates a matplotlib animation with the time evolution of the wavefield.

        The created animation will be displayed as an interactive widget in a IPython or
        Jupyter Notebook environment.
        In a non-interactive environment (script) the result of this animation would be
        lost. Use `create_video_file` method instead.

        Args:
            show_sources: whether or not to show the source transducer layer.
            show_target: whether or not to show the target layer.
            show_material_outlines: whether or not to display a thin white outline of
                the transition between different materials.
            n_frames_undersampling: the number of time steps to be skipped when creating
                the animation.
            slice_axis: the axis along which to slice. If None, then the value returned
                by `scenario.get_default_slice_axis()` is used.
            slice_position: the position (in meters) along the slice axis at
                which the slice should be made. If None, then the value returned by
                `scenario.get_default_slice_position()` is used.
            time_lim: the input time limit tuple to validate. The expected format is
                (minimum_time, maximum_time).
            norm: The normalization method used to scale scalar data to the [0, 1]
                range before mapping to colors using cmap. For a list
                of available scales, call `matplotlib.scale.get_scale_names()`.

        Returns:
            An matplotlib animation object.
        """

        animation = self._build_animation(
            show_sources=show_sources,
            show_target=show_target,
            show_material_outlines=show_material_outlines,
            time_lim=time_lim,
            n_frames_undersampling=n_frames_undersampling,
            norm=norm,
            slice_axis=slice_axis,
            slice_position=slice_position,
        )
        rendering.configure_matplotlib_for_embedded_animation()
        return animation

    def create_video_file(
        self,
        file_name: str,
        show_sources: bool = True,
        show_target: bool = True,
        show_material_outlines: bool = True,
        n_frames_undersampling: int = 1,
        slice_axis: int | None = None,
        slice_position: float | None = None,
        time_lim: tuple[np.float_, np.float_] | None = None,
        norm: str = "linear",
        fps: int = 25,
        dpi: int = 100,
        bitrate: int = 2500,
        overwrite: bool = False,
    ) -> None:
        """Saves a `mp4` animation file to disk with the results of the simulation.

        Currently only mp4 format supported.
        `ffmpeg` command line tools needs to be installed.

        Args:
            file_name: the file with path an extension where the animation would be
                saved. Currently only supports mp4 extension.
            show_sources: whether or not to show the source transducer layer.
            show_target: whether or not to show the target layer.
            show_material_outlines: whether or not to display a thin white outline of
                the transition between different materials.
            n_frames_undersampling: the number of time steps to be skipped when creating
                the animation.
            slice_axis: the axis along which to slice. If None, then the value returned
                by `scenario.get_default_slice_axis()` is used.
            slice_position: the position (in meters) along the slice axis at
                which the slice should be made. If None, then the value returned by
                `scenario.get_default_slice_position()` is used.
            time_lim: the input time limit tuple to validate. The expected format is
                (minimum_time, maximum_time).
            norm: The normalization method used to scale scalar data to the [0, 1]
                range before mapping to colors using cmap. For a list of available
                scales, call `matplotlib.scale.get_scale_names()`.
            fps: the frames per second in the animation.
            dpi: the number of dots per inch in the frames of the animation.
            bitrate: the bitrate for the saved movie file, which is one way to control
                the output file size and quality.
            overwrite: a boolean that allows the animation to be saved with the same
                file name that is already exists.
        """
        self._check_ffmpeg_is_installed()
        self._validate_file_name(file_name, overwrite)

        animation = self._build_animation(
            show_sources=show_sources,
            show_target=show_target,
            show_material_outlines=show_material_outlines,
            time_lim=time_lim,
            n_frames_undersampling=n_frames_undersampling,
            norm=norm,
            slice_axis=slice_axis,
            slice_position=slice_position,
        )

        rendering.save_animation(
            animation, file_name, fps=fps, dpi=dpi, bitrate=bitrate
        )
        print(f"Saved to {file_name} file.")

    @rendering.video_only_output
    def _build_animation(
        self,
        time_lim: tuple[np.float_, np.float_] | None = None,
        n_frames_undersampling: int = 1,
        slice_axis: int | None = None,
        slice_position: float | None = None,
        show_sources: bool = True,
        show_target: bool = True,
        show_material_outlines: bool = True,
        norm: str = "linear",
    ) -> FuncAnimation:
        """Create a matplotlib animation with the time evolution of the pressure waves.

        In order to visualize the 3D scenario in a 2D plot, a slice through the scenario
        needs to be specified via `slice_axis` and `slice_position`. Eg. to take a slice
        at z=0.01 m, use `slice_axis=2` and `slice_position=0.01`.

        Raises:
            ValueError if the wave field is not in 2D.

        Args:
            time_lim: the input time limit tuple to validate. The expected format is
                (minimum_time, maximum_time).
            n_frames_undersampling: the number of time steps to be skipped when creating
                the animation.
            slice_axis: the axis along which to slice. If None, then the value returned
                by `scenario.get_default_slice_axis()` is used.
            slice_position: the position (in meters) along the slice axis at
                which the slice should be made. If None, then the value returned by
                `scenario.get_default_slice_position()` is used.
            show_sources: whether or not to show the source transducer layer.
            show_target: whether or not to show the target layer.
            show_material_outlines: whether or not to display a thin white outline of
                the transition between different materials.
            norm: The normalization method used to scale scalar data to the [0, 1]
                range before mapping to colors using cmap. For a list of available
                scales, call `matplotlib.scale.get_scale_names()`.

        Returns:
            A matplotlib animation object.
        """
        self._validate_slicing_options(slice_axis, slice_position)

        wavefield = self.wavefield

        if self.recorded_slice is None:
            if slice_axis is None:
                slice_axis = self.scenario.get_default_slice_axis()
            if slice_position is None:
                slice_position = self.scenario.get_default_slice_position(slice_axis)
            wavefield = slice_field(
                wavefield, self.scenario, slice_axis, slice_position
            )
        else:
            wavefield = np.squeeze(wavefield, axis=slice_axis)
            slice_axis, slice_position = self.recorded_slice

        if time_lim is not None:
            self._validate_time_lim(time_lim)
            times = self._recording_times()
            time_start_idx = (np.abs(times - time_lim[0])).argmin()
            time_end_idx = (np.abs(times - time_lim[1])).argmin()
            time_slice = slice(time_start_idx, time_end_idx + 1, None)
            wavefield = wavefield[..., time_slice]

        extent = drop_element(self.scenario.extent, slice_axis)
        origin = drop_element(self.scenario.origin, slice_axis)

        min_pressure = wavefield.min()
        max_pressure = wavefield.max()

        fig, ax = rendering.create_pulsed_figure(origin, extent, wavefield, norm)

        # add layers
        if show_material_outlines:
            material_field = self.scenario.get_field_data("layer").astype(int)
            material_field_2d = slice_field(
                material_field, self.scenario, slice_axis, slice_position
            )
            rendering.draw_material_outlines(
                ax=ax,
                material_field=material_field_2d,
                dx=self.scenario.dx,
                origin=origin,
                upsample_factor=self.scenario._material_outline_upsample_factor,
            )
        if show_target:
            target_loc = drop_element(self.scenario.target_center, slice_axis)
            rendering.draw_target(ax, target_loc, self.scenario.target_radius)

        if show_sources:
            for source in self.scenario.sources:
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

        rendering.configure_result_plot(
            fig,
            ax,
            extent=extent,
            origin=origin,
            show_sources=show_sources,
            show_target=show_target,
            clim=(min_pressure, max_pressure),
            vertical_label=vert_name,
            horizontal_label=horz_name,
            title=(
                "Pulsed Wave Amplitude" f"\nSlice: {slice_name} = {slice_position} m"
            ),
        )

        animation = rendering.make_animation(
            fig,
            ax,
            wavefield=wavefield,
            n_frames_undersampling=n_frames_undersampling,
        )
        return animation

    def render_pulsed_simulation_animation_3d(self):
        raise NotImplementedError("Currently not supported.")


def create_steady_state_result(
    scenario: scenarios.Scenario,
    center_frequency: float,
    effective_dt: float,
    pde: stride.Operator,
    shot: stride.Shot,
    wavefield: npt.NDArray[np.float_],
    traces: stride.Traces,
) -> SteadyStateResult:
    """Utility function for creating a steady state result.

    Creates a SteadyStateResult2D or SteadyStateResult3D depending on the number of
    wavefield spatial dimensions. If the ndim of the wavefield is N, then the wavefield
    has N-1 spatial dimensions and 1 time dimension.

    Args:
        scenario: The scenario from which this result came.
        center_frequency: the center frequency (in hertz) of the sources.
        effective_dt: the effective time step (in seconds) along the time axis of the
            wavefield. This can differ from the simulation dt if the recording
            downsampling factor is larger than 1.
        pde: the stride Operator that was executed to run the simulation.
        shot: the stride Shot which was used for the simulation.
        wavefield: an array containing the resulting simulation data.
        traces: the stride Traces object returned from executing the pde.

    Raises:
        ValueError: if the ndim of the wavefield is less than 3 or more than 4.

    Returns:
        Result: a SteadyStateResult2D or SteadyStateResult3D, depending on the wavefield
            shape.
    """
    if wavefield.ndim == 3:
        return SteadyStateResult2D(
            scenario=scenario,
            center_frequency=center_frequency,
            effective_dt=effective_dt,
            pde=pde,
            shot=shot,
            wavefield=wavefield,
            traces=traces,
        )
    elif wavefield.ndim == 4:
        return SteadyStateResult3D(
            scenario=scenario,
            center_frequency=center_frequency,
            effective_dt=effective_dt,
            pde=pde,
            shot=shot,
            wavefield=wavefield,
            traces=traces,
        )
    else:
        raise ValueError(
            "Expected ndim of wavefield to be either"
            f" 3 (2D) or 4 (3D). Got {wavefield.ndim}."
        )


def create_pulsed_result(
    scenario: scenarios.Scenario,
    center_frequency: float,
    effective_dt: float,
    pde: stride.Operator,
    shot: stride.Shot,
    wavefield: npt.NDArray[np.float_],
    traces: stride.Traces,
    recorded_slice: tuple[int, float] | None = None,
) -> PulsedResult:
    """Utility function for creating results from pulsed simulations.

    Creates a PulsedResult2D or PulsedResult3D depending on the number of wavefield
    spatial dimensions. If the ndim of the wavefield is N, then the wavefield has N-1
    spatial dimensions and 1 time dimension.

    Args:
        scenario: The scenario from which this result came.
        center_frequency: the center frequency (in hertz) of the sources.
        effective_dt: the effective time step (in seconds) along the time axis of the
            wavefield. This can differ from the simulation dt if the recording
            downsampling factor is larger than 1.
        pde: the stride Operator that was executed to run the simulation.
        shot: the stride Shot which was used for the simulation.
        wavefield: an array containing the resulting simulation data.
        traces: the stride Traces object returned from executing the pde.

    Raises:
        ValueError: if the ndim of the wavefield is less than 3 or more than 4.

    Returns:
        Result: a PulsedResult2D or PulsedResult3D, depending on the wavefield shape.
    """
    if wavefield.ndim == 3:
        return PulsedResult2D(
            scenario=scenario,
            center_frequency=center_frequency,
            effective_dt=effective_dt,
            pde=pde,
            shot=shot,
            wavefield=wavefield,
            traces=traces,
        )
    elif wavefield.ndim == 4:
        return PulsedResult3D(
            scenario=scenario,
            center_frequency=center_frequency,
            effective_dt=effective_dt,
            pde=pde,
            shot=shot,
            wavefield=wavefield,
            traces=traces,
            recorded_slice=recorded_slice,
        )
    else:
        raise ValueError(
            "Expected ndim of wavefield to be either"
            f" 3 (2D) or 4 (3D). Got {wavefield.ndim}."
        )


def load_result_from_disk(filepath: str | pathlib.Path) -> Result:
    """Loads a result from disk from a gzip compressed pickle file.

    !!! warning
        This functionality is experimental, so do do not be surprised if you
        encounter issues calling this function.

    Load a file that was saved to disk via Result.save_to_disk.

    If the object saved in `filepath` is the result from a steady-state simulation
    the results will contain only the steady-state amplitudes. Instead, for pulsed
    simulations the result object will contain the original wavefield.

    This function is particularly useful if simulation is performed in the cloud but
    the user would like to download the results in order to visualize them locally
    in 3D.

    Args:
        filepath: The path to an existing result file previously saved via
            Result.save_to_disk.

    Returns:
        A Results object (SteadyStateResult or PulsedResult)
    """
    try:
        with gzip.open(filepath, "rb") as f:
            save_data = pickle.load(f)

        import neurotechdevkit as ndk

        print("Recreating the scenario for the result from saved metadata...")
        scenario = ndk.make(save_data["scenario_id"])

        for source in save_data["sources"]:
            scenario.add_source(source)

        scenario.current_target_id = save_data["target_id"]
        result_type = getattr(ndk.scenarios, save_data["result_type"])

        fields_kwargs = dict(
            scenario=scenario,
            center_frequency=save_data["center_frequency"],
            effective_dt=save_data["effective_dt"],
            pde=None,
            shot=None,
            wavefield=save_data.get("wavefield"),
            traces=None,
        )

        if save_data.get("steady_state") is not None:
            fields_kwargs.update(steady_state=save_data["steady_state"])

        return result_type(**fields_kwargs)

    except Exception as e:
        raise Exception("Unable to load result artifact from disk.") from e
