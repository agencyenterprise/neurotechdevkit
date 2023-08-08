# noqa: D100
# preventing package docstring to be rendered in documentation
from typing import Iterable, Tuple, Union

import numpy as np
import numpy.typing as npt
import stride


def _compute_grid_shape(extent: npt.NDArray[np.float_], dx: float) -> tuple[int, ...]:
    """Compute the shape of the grid for a given extent and dx."""
    n_steps_float = extent / dx
    n_steps = n_steps_float.round().astype(int)
    np.testing.assert_allclose(n_steps_float, n_steps, rtol=1e-5)
    return tuple(steps + 1 for steps in n_steps)


class Grid(stride.Grid):
    """
    Grid class for neurotechdevkit. It is a subclass of stride.Grid.

    The grid is a container for the spatial and temporal grids. It is used to define
    the simulation domain and the grid spacing.
    """

    @staticmethod
    def make_shaped_grid(
        shape: Tuple[int, int],
        spacing: float,
    ) -> "Grid":
        """
        Create a NDK Grid with the given shape.

        Args:
            shape: The shape of the grid.
            spacing: Axis-wise spacing of the grid, in meters.

        Returns:
            Grid: The Grid object.
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
        """Create a NDK Grid.

        Note that the time component of the grid is not defined here. That is created
        at simulation time because it depends on simulation parameters.

        Args:
            extent: a 2-tuple or 3-tuple containing the dimensions (in meters) of the
                simulation.
            speed_water: the speed of sound in water (in m/s).
            center_frequency: the center frequency of the source (in Hz).
            ppw: the number of points per wavelength.
            extra: the number of gridpoints to add as boundary layers on each side of
                the grid. extras are added both before and after the grid on each axis.
            absorbing: the number of gridpoints within the boundary layers that are
                absorbing.

        Returns:
            The Grid object.
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
