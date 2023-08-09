# noqa: D100
# preventing package docstring to be rendered in documentation
from typing import Iterable, Tuple, Union

import numpy as np
import numpy.typing as npt
import stride


def _compute_grid_shape(extent: npt.NDArray[np.float_], dx: float) -> tuple[int, ...]:
    """Compute the shape of the grid for a given extent and dx.

    Args:
        extent: a numpy array of floats representing the dimensions (in meters)
            of the simulation.
        dx: a float representing the axis-wise spacing of the grid, in meters.

    Returns:
        tuple[int, ...]: a tuple of integers representing the shape of the grid.
    """
    n_steps_float = extent / dx
    n_steps = n_steps_float.round().astype(int)
    np.testing.assert_allclose(n_steps_float, n_steps, rtol=1e-5)
    return tuple(steps + 1 for steps in n_steps)


class Grid(stride.Grid):
    """
    Grid class for neurotechdevkit. It is a subclass of stride.Grid.

    The grid is a container for the spatial and temporal grids. It is used to
    define the simulation domain and the grid spacing.
    """

    @staticmethod
    def make_shaped_grid(
        shape: Tuple[int, int],
        spacing: float,
    ) -> "Grid":
        """Create an NDK Grid with the given shape.

        Args:
            shape: a tuple of two integers representing the shape of the grid.
            spacing: a float representing the axis-wise spacing of the grid, in
                meters.

        Returns:
            Grid: the Grid object.
        """
        space = stride.Space(shape=shape, spacing=spacing)
        return Grid(space=space, time=None)

    @staticmethod
    def make_grid(
        extent: Union[Tuple[float, float], Tuple[float, float, float]],
        speed_water: float,
        center_frequency: float,
        ppw: int,
        extra: Union[int, Iterable[int]] = 50,
        absorbing: Union[int, Iterable[int]] = 40,
    ) -> "Grid":
        """Create an NDK Grid.

        Note that the time component of the grid is not defined here. That is
        created at simulation time because it depends on simulation parameters.

        Args:
            extent: a tuple of two or three floats representing the dimensions
                (in meters) of the simulation.
            speed_water: a float representing the speed of sound in water (in
                m/s).
            center_frequency: a float representing the center frequency of the
                source (in Hz).
            ppw: an integer representing the number of points per wavelength.
            extra: an integer or an iterable of integers representing the number
                of gridpoints to add as boundary layers on each side of the grid.
                Extras are added both before and after the grid on each axis.
                Default is 50.
            absorbing: an integer or an iterable of integers representing the
                number of gridpoints within the boundary layers that are
                absorbing. Default is 40.

        Returns:
            Grid: the Grid object.
        """
        _extent = np.array(extent, dtype=float)
        n_dims = len(_extent)
        dx = speed_water / center_frequency / ppw  # m
        shape = _compute_grid_shape(_extent, dx)

        if isinstance(extra, int):
            extra = (extra,) * n_dims
        else:
            extra = tuple(extra)
        if isinstance(absorbing, int):
            absorbing = (absorbing,) * n_dims
        else:
            absorbing = tuple(absorbing)

        space = stride.Space(shape=shape, extra=extra, absorbing=absorbing, spacing=dx)
        return Grid(space=space, time=None)
