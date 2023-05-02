from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import numpy.typing as npt
import stride
from mosaic.types import Struct
from stride.utils import wavelets


def make_grid(
    extent: npt.NDArray[np.float_],
    dx: float,
    extra: int | Iterable[int] = 50,
    absorbing: int | Iterable[int] = 40,
) -> stride.Grid:
    """Creates a stride Grid.

    Note that the time component of the grid is not defined here. That is created
    at simulation time because it depends on simulation parameters.

    Args:
        extent: A 2-tuple or 3-tuple containing the dimensions (in meters) of the
            simulation.
        dx: A float describing the distance (in meters) between grid points.
        extra: The number of gridpoints to add as boundary layers on each side of the
            grid. extras are added both before and after the grid on each axis.
        absorbing: The number of gridpoints within the boundary layers that are
            absorbing.

    Returns:
        The stride Grid object.
    """
    n_dims = len(extent)
    shape = compute_shape(extent, dx)

    if isinstance(extra, int):
        extra = (extra,) * n_dims
    else:
        extra = tuple(extra)
    if isinstance(absorbing, int):
        absorbing = (absorbing,) * n_dims
    else:
        absorbing = tuple(absorbing)

    print(f"creating a grid with shape: {shape} for extent: {extent} m")
    space = stride.Space(shape=shape, extra=extra, absorbing=absorbing, spacing=dx)
    return stride.Grid(space=space, time=None)


def compute_shape(extent: npt.NDArray[np.float_], dx: float) -> tuple[int, ...]:
    # TODO: verify that extent is a multiple of dx
    # but using modulus doesn't work due to floating point
    # numerical error
    n_steps = [int(np.round(ext / dx)) for ext in extent]
    return tuple(steps + 1 for steps in n_steps)


def add_material_fields_to_problem(
    problem: stride.Problem,
    materials: Mapping[str, Struct],
    layer_ids: Mapping[str, int],
    masks: Mapping[str, npt.NDArray[np.bool_]],
) -> stride.Problem:
    """Adds material fields as media to the problem.

    Included fields are:
    * the speed of sound (in m/s)
    * density (in kg/m^3)
    * absorption (in dB/cm)
    """

    grid = problem.grid

    vp = stride.ScalarField(name="vp", grid=grid)  # [m/s]
    rho = stride.ScalarField(name="rho", grid=grid)  # [kg/m^3]
    alpha = stride.ScalarField(name="alpha", grid=grid)  # [dB/cm]
    layer = stride.ScalarField(name="layer", grid=grid)  # integers

    for name, material in materials.items():
        material_mask = masks[name]
        vp.data[material_mask] = material.vp
        rho.data[material_mask] = material.rho
        alpha.data[material_mask] = material.alpha
        layer.data[material_mask] = layer_ids[name]

    vp.pad()
    rho.pad()
    alpha.pad()
    layer.pad()

    problem.medium.add(vp)
    problem.medium.add(rho)
    problem.medium.add(alpha)
    problem.medium.add(layer)

    return problem


def choose_wavelet_for_mode(simulation_mode: str) -> str:
    """Returns the appropriate wavelet name for a given simulation mode.

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
    """Creates an array corresponding to the requested wavelet.

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
    """Returns a slice of a field at a desired position along an axis.

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
    """Drops the element of a vector which corresponds to slice_axis.

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
    """Drops the column of a 2D array which corresponds to slice_axis.

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
    """Returns a 2D mask array for an ellipse with the specified parameters.

    Array elements are True for the gridpoints within the ellipse and False otherwise.

    Args:
        grid: The simulation grid.
        origin: The coordinates (in meters) of the grid element (0, 0).
        center: The coordinates (in meters) of the center of the ellipse.
        a: The radius (in meters) of the ellipse along the X-axis.
        b: The radius (in meters) of the ellipse along the Y-axis.

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
    """Returns a 2D mask array for a circle with the specified parameters.

    Array elements are True for the gridpoints within the circle and False otherwise.

    Args:
        grid: The simulation grid.
        origin: The coordinates (in meters) of the grid element (0, 0).
        center: The coordinates (in meters) of the center of the circle.
        a: The radius (in meters) of the circle.

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
    """Returns a 3D mask array for a sphere with the specified parameters.

    Array elements are True for the gridpoints within the sphere and False otherwise.

    Args:
        grid: The simulation grid.
        origin: The coordinates (in meters) of the grid element (0, 0, 0).
        center: The coordinates (in meters) of the center of the circle.
        a: The radius (in meters) of the circle.

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
    """Returns a mask array for an N-D ellipse with the specified parameters.

    If the grid is 2D, then a 2D ellipse will be returned. If the grid is 3D, then a 3D
    ellipsoid will be returned.

    Array elements are True for the gridpoints within the ellipse and False otherwise.

    Args:
        grid: The simulation grid.
        origin: The coordinates (in meters) of the grid element (0, 0, ...).
        center: The coordinates (in meters) of the center of the ellipse.
        radii: The radius (in meters) along each axis of the ellipse.

    Returns:
        The 2D or 3D boolean mask where gridpoints within the ellipse are True.
    """
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
