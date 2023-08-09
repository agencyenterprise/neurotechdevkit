# noqa: D100
# preventing package docstring to be rendered in documentation
from typing import Mapping

import numpy as np
import numpy.typing as npt
import stride
from mosaic.types import Struct

from neurotechdevkit.grid import Grid


class Problem(stride.Problem):
    """Problem class for NDK. It is a subclass of stride.Problem.

    The problem defines a medium with a set of fields (such as Vp or density), some
    transducers (such as a series of scalar point transducers), a geometry where those
    transducers are located in space, and the acquisitions that happen given that
    geometry.
    """

    def __init__(self, grid: Grid, *args, **kwargs):
        """
        Initialize a problem with a grid.

        Args:
            grid (Grid): the grid to use for the problem.
        """
        super().__init__(name="neurotechdevkit", grid=grid, *args, **kwargs)

    def add_material_fields(
        self,
        materials: Mapping[str, Struct],
        masks: Mapping[str, npt.NDArray[np.bool_]],
    ):
        """Add material fields as media to the problem.

        Included fields are:

        - the speed of sound (in m/s)
        - density (in kg/m^3)
        - absorption (in dB/cm)

        Args:
            materials (Mapping[str, Struct]): a mapping from material names
                to Structs containing the material properties.
            masks (Mapping[str, npt.NDArray[np.bool_]]): a mapping from material
                names to boolean masks indicating the gridpoints.
        """
        vp = stride.ScalarField(name="vp", grid=self.grid)  # [m/s]
        rho = stride.ScalarField(name="rho", grid=self.grid)  # [kg/m^3]
        alpha = stride.ScalarField(name="alpha", grid=self.grid)  # [dB/cm]

        for name, material in materials.items():
            material_mask = masks[name]
            vp.data[material_mask] = material.vp
            rho.data[material_mask] = material.rho
            alpha.data[material_mask] = material.alpha

        vp.pad()
        rho.pad()
        alpha.pad()

        self.medium.add(vp)
        self.medium.add(rho)
        self.medium.add(alpha)
