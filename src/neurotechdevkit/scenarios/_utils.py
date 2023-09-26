from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt
import stride
from stride.utils import wavelets


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
    center: list[float]
    radius: float
    description: str


class SliceAxis(IntEnum):
    """Axis along which to slice the 3D field to be recorded."""

    X = 0
    Y = 1
    Z = 2


def choose_wavelet_for_mode(simulation_mode: str) -> str:
    """Return the appropriate wavelet name for a given simulation mode.

    Args:
        simulation_mode: the type of simulation which will be run.

    Raises:
        ValueError: if the provided `simulation_mode` is not recognized.

    Returns:
        The name of the wavelet to be used for the given simulation mode.
    """
    if simulation_mode == "steady-state":
        return "continuous_wave"
    elif simulation_mode == "pulsed":
        return "tone_burst"
    else:
        raise ValueError(f"Unknown simulation mode: {simulation_mode}")


def wavelet_helper(
    name: str,
    time: stride.Time,
    freq_hz: float = 5.0e5,
    pressure: float = 6.0e4,
) -> npt.NDArray[np.float_]:
    """Create an array corresponding to the requested wavelet.

    The wavelet returned from this function is intended to be applied to stride point
    sources. It should be scaled and delayed as needed in order to simulate a source
    transducer with physical extent.

    Currently supported wavelet types (by name) are:
        "tone_burst": a pulse of waves composed of a gaussian envelop applied to a
            continuous waveform.
        "ricker": a [Ricker wavelet](https://en.wikipedia.org/wiki/Ricker_wavelet).
        "continuous_wave": a sinusoidal wave at a single center frequency with a short
            startup transient where the amplitude ramps up.

    These are the only values allowed for `name`.

    Args:
        name: the name of the type of wavelet desired.
        time: the stride Time object for the simulation.
        freq_hz: the center frequency of the wavelet (in hertz). Defaults to 5.0e5.
        pressure: the pressure wave amplitude desired at the transducer surface (in
            pascals). Defaults to 6.0e4.

    Raises:
        ValueError: if the value of `name` is not recognized as a supported wavelet
            type.

    Returns:
        A 1D numpy array representing the wavelet to apply to a stride point source.
    """ """"""
    if name == "tone_burst":
        wavelet = wavelets.tone_burst(
            freq_hz, n_cycles=3, n_samples=time.num, dt=time.step
        )
    elif name == "ricker":
        wavelet = wavelets.ricker(freq_hz, n_samples=time.num, dt=time.step)
    elif name == "continuous_wave":
        wavelet = wavelets.continuous_wave(freq_hz, n_samples=time.num, dt=time.step)
    else:
        raise ValueError(f"wavelet {name} unknown")
    return pressure * wavelet


def slice_field(
    field: npt.NDArray, scenario, slice_axis: int, slice_position: float
) -> npt.NDArray:
    """Return a slice of a field at a desired position along an axis.

    If `slice_position` does not align exactly with the scenario grid, the closest
    gridpoint will be used. If `slice_position` is outside of the bounds of the grid,
    then the slice will be taken along the boundary of the grid.

    Args:
        field: the N-dimensional field to slice.
        scenario: the scenario to which the field belongs.
        slice_axis: the axis along which to slice.
        slice_position (float): the position (in meters) along the slice axis at which
            the slice should be made.

    Returns:
        The N-1-dimensional field corresponding to `slice_position` along axis
            `slice_axis`.
    """
    offset_position = slice_position - scenario.origin[slice_axis]
    dx = scenario.dx
    slice_idx = int(np.round(offset_position / dx))
    slice_idx = np.clip(slice_idx, 0, scenario.shape[slice_axis] - 1)
    slices_init = np.array([np.s_[slice_idx], np.s_[:], np.s_[:]])
    array_slice = tuple(np.roll(slices_init, slice_axis))
    return field[array_slice]


def drop_element(arr: npt.NDArray, drop_idx: int) -> npt.NDArray:
    """Drop the element of a vector which corresponds to slice_axis.

    Args:
        arr: a 1D numpy array containing the vector to be sliced.
        drop_idx: the index of the element of the vector to drop.

    Returns:
        `arr` with the element corresponding to `drop_idx` removed.
    """
    mask = np.ones(arr.shape[0], dtype=bool)
    mask[drop_idx] = False
    return arr[mask]


def drop_column(arr: npt.NDArray, drop_idx: int) -> npt.NDArray:
    """Drop the column of a 2D array which corresponds to slice_axis.

    Args:
        arr: a 2D numpy array containing the data to be sliced.
        drop_idx: the index of the column of the array to drop.

    Returns:
        `arr` with the column corresponding to `drop_idx` removed.
    """
    mask = np.ones(arr.shape[1], dtype=bool)
    mask[drop_idx] = False
    return arr[:, mask]


def create_grid_elliptical_mask(
    grid: stride.Grid,
    origin: npt.NDArray[np.float_],
    center: npt.NDArray[np.float_],
    a: float,
    b: float,
) -> npt.NDArray[np.bool_]:
    """Return a 2D mask array for an ellipse with the specified parameters.

    Array elements are True for the gridpoints within the ellipse and False otherwise.

    Args:
        grid: the simulation grid.
        origin: the coordinates (in meters) of the grid element (0, 0).
        center: the coordinates (in meters) of the center of the ellipse.
        a: the radius (in meters) of the ellipse along the X-axis.
        b: the radius (in meters) of the ellipse along the Y-axis.

    Returns:
        The 2D boolean mask where gridpoints within the ellipse are True.
    """
    radii = np.array([a, b])
    return _create_nd_ellipse_mask(grid, origin, center, radii)


def create_grid_circular_mask(
    grid: stride.Grid,
    origin: npt.NDArray[np.float_],
    center: npt.NDArray[np.float_],
    radius: float,
) -> npt.NDArray[np.bool_]:
    """Return a 2D mask array for a circle with the specified parameters.

    Array elements are True for the gridpoints within the circle and False otherwise.

    Args:
        grid: the simulation grid.
        origin: the coordinates (in meters) of the grid element (0, 0).
        center: the coordinates (in meters) of the center of the circle.
        a: the radius (in meters) of the circle.

    Returns:
        The 2D boolean mask where gridpoints within the circle are True.
    """
    radii = np.array([radius] * 2)
    return _create_nd_ellipse_mask(grid, origin, center, radii)


def create_grid_spherical_mask(
    grid: stride.Grid,
    origin: npt.NDArray[np.float_],
    center: npt.NDArray[np.float_],
    radius: float,
) -> npt.NDArray[np.bool_]:
    """Return a 3D mask array for a sphere with the specified parameters.

    Array elements are True for the gridpoints within the sphere and False otherwise.

    Args:
        grid: the simulation grid.
        origin: the coordinates (in meters) of the grid element (0, 0, 0).
        center: the coordinates (in meters) of the center of the circle.
        a: the radius (in meters) of the circle.

    Returns:
        The 3D boolean mask where gridpoints within the circle are True.
    """
    radii = np.array([radius] * 3)
    return _create_nd_ellipse_mask(grid, origin, center, radii)


def _create_nd_ellipse_mask(
    grid: stride.Grid,
    origin: npt.NDArray[np.float_],
    center: npt.NDArray[np.float_],
    radii: npt.NDArray[np.float_],
) -> npt.NDArray[np.bool_]:
    """Return a mask array for an N-D ellipse with the specified parameters.

    If the grid is 2D, then a 2D ellipse will be returned. If the grid is 3D, then a 3D
    ellipsoid will be returned.

    Array elements are True for the gridpoints within the ellipse and False otherwise.

    Args:
        grid: the simulation grid.
        origin: the coordinates (in meters) of the grid element (0, 0, ...).
        center: the coordinates (in meters) of the center of the ellipse.
        radii: the radius (in meters) along each axis of the ellipse.

    Returns:
        The 2D or 3D boolean mask where gridpoints within the ellipse are True.
    """
    assert grid.space is not None
    shape = grid.space.shape
    spacing = grid.space.spacing

    X = np.meshgrid(
        *[np.arange(n) * d + p for n, d, p in zip(shape, spacing, origin)],
        indexing="ij",
    )

    distance_from_center_sq = sum(
        ((X_i - x_i) / R_i) ** 2 for X_i, x_i, R_i in zip(X, center, radii)
    )
    in_ellipse = distance_from_center_sq <= 1

    return in_ellipse
