"""Grid class for neurotechdevkit."""

from typing import Iterable, Union

import numpy as np
import numpy.typing as npt
import stride


def _compute_grid_shape(extent: npt.NDArray[np.float_], dx: float) -> tuple[int, ...]:
    """Compute the shape of the grid for a given extent and dx."""
    # TODO: verify that extent is a multiple of dx
    # but using modulus doesn't work due to floating point
    # numerical error
    n_steps = [int(np.round(ext / dx)) for ext in extent]
    return tuple(steps + 1 for steps in n_steps)


class Grid(stride.Grid):
    """Grid class for neurotechdevkit. It is a subclass of stride.Grid."""

    @staticmethod
    def make_grid(
        extent: npt.NDArray[np.float_],
        dx: float,
        extra: Union[int, Iterable[int]] = 50,
        absorbing: Union[int, Iterable[int]] = 40,
    ) -> "Grid":
        """Create a NDK Grid.

        Note that the time component of the grid is not defined here. That is created
        at simulation time because it depends on simulation parameters.

        Args:
            extent: a 2-tuple or 3-tuple containing the dimensions (in meters) of the
                simulation.
            dx: the grid spacing in meters. If None, it is computed based on the
                center frequency and the desired resolution.
            extra: the number of gridpoints to add as boundary layers on each side of
                the grid. extras are added both before and after the grid on each axis.
            absorbing: the number of gridpoints within the boundary layers that are
                absorbing.

        Returns:
            The Grid object.
        """
        n_dims = len(extent)
        shape = _compute_grid_shape(extent, dx)

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
