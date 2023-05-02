from __future__ import annotations

import numpy as np
import numpy.typing as npt
import stride
from stride.problem.geometry import TransducerLocation

from .. import sources


def create_shot(
    problem: stride.Problem,
    sources: list[sources.Source],
    origin: npt.NDArray[np.float_],
    wavelet: npt.NDArray[np.float_],
    dx: float,
) -> stride.Shot:
    """Compiles and returns a shot for a problem.

    This function will add the point sources for each source to the problem geometry,
    build the shot object, build the wavelet array for the shot, and then add the
    shot to the acquisitions for the problem.

    Note: support for receivers in a shot is not currently implemented in ndk.

    Args:
        problem: the problem to which the shot will be be added.
        sources: the list of sources containing all sources used in the shot.
        origin: the coordinates of the scenario gridpoint with index (0, 0)
            or (0, 0, 0).
        wavelet: a 1D array with shape (num_time_steps,) containing the pressure
            amplitude for a point source without any delays.
        dt: the time step (in seconds) used in the simulation.

    Returns:
        The newly-created shot for the simulation.
    """
    point_transducers = _add_sources_to_geometry(problem, sources, origin)

    shot = stride.Shot(
        id=0,
        problem=problem,
        sources=point_transducers,
        receivers=[],
        geometry=problem.geometry,
    )

    shot.wavelets.data[:] = _build_shot_wavelets_array(
        wavelet=wavelet,
        sources=sources,
        dx=dx,
        dt=problem.time.step,
    )

    problem.acquisitions.add(shot)
    return shot


def _add_sources_to_geometry(
    problem: stride.Problem,
    sources: list[sources.Source],
    origin: npt.NDArray[np.float_],
) -> list[TransducerLocation]:
    """Adds and returns source transducers at locations specified by all sources.

    Every point sources from all sources is added to the problem geometry.

    Before being added, the scenario origin is applied to translate the point source
    coordinates into the coordinates used by stride.

    Args:
        problem: the problem to which the point sources should be added.
        sources: the list of sources containing all sources used in the shot.
        origin: the coordinates of the scenario gridpoint with index (0, 0)
            or (0, 0, 0).

    Returns:
        A list containing the stride location object for each added point source.
    """
    point_transducers = []

    for source in sources:
        source_coords = source.coordinates - origin
        source_transducers = _add_points_for_source_to_geometry(problem, source_coords)
        point_transducers.extend(source_transducers)

    return point_transducers


def _add_points_for_source_to_geometry(
    problem: stride.Problem, coords: npt.NDArray[np.float_]
) -> list[TransducerLocation]:
    """Adds and returns source transducers at locations specified by coords.

    Each point source in a source is added to the stride problem geometry, and then the
    TransducerLocation for each point source is returned so that it can be added to the
    shot.

    Args:
        problem: the problem to which the point sources should be added.
        coords: an array containing the coordinates for each point source in the
            scenario. The array should have shape (N, D) where N is the number of point
            sources and D is the number of spatial dimensions in the scenario.

    Returns:
        A list containing the stride location object for each added point source.
    """
    if problem.transducers.num_transducers == 0:
        problem.transducers.default()
    transducer = problem.transducers.get(0)

    source_locations = []
    offset = problem.geometry.num_locations
    for n in range(coords.shape[0]):
        problem.geometry.add(offset + n, transducer, coords[n, :])
        source_locations.append(problem.geometry.get(offset + n))

    return source_locations


def _build_shot_wavelets_array(
    wavelet: npt.NDArray[np.float_],
    sources: list[sources.Source],
    dx: float,
    dt: float,
) -> npt.NDArray[np.float_]:
    """Returns the scaled and delayed wavelet array for all sources.

    The scaling and delays are determined by each individual source.

    The only delays currently implemented are for phased-arrays which have a delay
    between the individual elements comprising the transducer. Functionality for
    individual sources to have a global delay before wave emission starts is on the
    backlog.

    Args:
        wavelet: a 1D array with shape (num_time_steps,) containing the pressure
            amplitude for a point source without any delays.
        sources: the list of sources containing all sources used in the shot.
        dx: the spacing (in meters) between grid points in the simulation.
        dt: the time step (in seconds) used in the simulation.

    Returns:
        A 2D array of shape (num_points, num_time_steps) containing the wavelet
            for all point sources within all sources.
    """
    source_wavelets = [
        _get_wavelets_for_source(wavelet, source, dx, dt) for source in sources
    ]
    return np.concatenate(source_wavelets)


def _get_wavelets_for_source(
    wavelet: npt.NDArray[np.float_], source: sources.Source, dx: float, dt: float
) -> npt.NDArray[np.float_]:
    """Returns the scaled and delayed wavelet array for a single source.

    Args:
        wavelet: a 1D array with shape (num_time_steps,) containing the pressure
            amplitude for a point source without any delays.
        source: the source for which the wavelet array should be created.
        dx: the spacing (in meters) between grid points in the simulation.
        dt: the time step (in seconds) used in the simulation.

    Returns:
        A 2D array of shape (num_points, num_time_steps) containing the wavelet
            for each point source comprising the source.
    """
    unscaled_source_wavelets = _create_delayed_source_wavelets(
        wavelet, source.point_source_delays, dt
    )
    scaling = source.calculate_waveform_scale(dx)
    return scaling * unscaled_source_wavelets


def _create_delayed_source_wavelets(
    wavelet: npt.NDArray[np.float_], delays: npt.NDArray[np.float_], dt: float
) -> npt.NDArray[np.float_]:
    """
    Applies time delays to a provided source wavelet for each point source.

    Args:
        wavelet: a 1D array with shape (num_time_steps,) containing the pressure
            amplitude for a point source without any delays.
        delays: a 1D array with shape (num_points,) containing the delay (in seconds) to
            apply to each point source.
        dt: the time step (in seconds) used in the simulation.

    Returns:
        A 2D array of shape (num_points, num_time_steps) containing the delayed wavelet
            for each point source.
    """
    n_time_steps = wavelet.shape[0]
    n_sources = delays.shape[0]

    start_idx = (delays // dt).astype(int)
    delayed_wavelets = np.zeros(shape=(n_sources, n_time_steps))

    for n in range(n_sources):
        n_shift = start_idx[n]
        delayed_wavelets[n] = np.pad(wavelet, (n_shift, 0), "constant")[0:n_time_steps]
    return delayed_wavelets
