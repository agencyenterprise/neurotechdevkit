"""Grid class for neurotechdevkit."""

from typing import Iterable, Optional, Union

import numpy as np
import numpy.typing as npt
import stride


class Grid(stride.Grid):
    """Grid class for neurotechdevkit. It is a subclass of stride.Grid."""

    def __init__(self, center_frequency: float, *args, **kwargs):
        """
        Initialize a Grid.

        Args:
            center_frequency (float): the center frequency of the simulation
                in Hz.
        """
        self.center_frequency = center_frequency
        super().__init__(*args, **kwargs)

    @staticmethod
    def make_grid(
        center_frequency: float,
        extent: npt.NDArray[np.float_],
        dx: Optional[float] = None,
        extra: Union[int, Iterable[int]] = 50,
        absorbing: Union[int, Iterable[int]] = 40,
    ) -> "Grid":
        """Create a stride Grid.

        Note that the time component of the grid is not defined here. That is created
        at simulation time because it depends on simulation parameters.

        Args:
            center_frequency: the center frequency of the simulation in Hz.
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

        def compute_shape(extent: npt.NDArray[np.float_], dx: float) -> tuple[int, ...]:
            """Compute the shape of the grid for a given extent and dx."""
            # TODO: verify that extent is a multiple of dx
            # but using modulus doesn't work due to floating point
            # numerical error
            n_steps = [int(np.round(ext / dx)) for ext in extent]
            return tuple(steps + 1 for steps in n_steps)

        if dx is None:
            # scenario constants
            speed_water = 1500  # m/s

            # desired resolution for complexity=fast
            ppw = 6
            # compute resolution
            dx = speed_water / center_frequency / ppw  # m

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

        space = stride.Space(shape=shape, extra=extra, absorbing=absorbing, spacing=dx)
        return Grid(center_frequency=center_frequency, space=space, time=None)
