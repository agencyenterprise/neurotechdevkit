# noqa: D100
# preventing package docstring to be rendered in documentation
from typing import Mapping

import numpy as np
import numpy.typing as npt
import stride
from mosaic.types import Struct


class Problem(stride.Problem):
    """
    Problem class for neurotechdevkit. It is a subclass of stride.Problem.

    The problem defines a medium with a set of fields (such as Vp or density), some
    transducers (such as a series of scalar point transducers), a geometry where those
    transducers are located in space, and the acquisitions that happen given that
    geometry.
    """

    def __init__(self, *args, **kwargs):
        """Initialize a Problem."""
        super().__init__(name="neurotechdevkit", *args, **kwargs)

    def add_material_fields(
        self,
        materials: Mapping[str, Struct],
        layer_ids: Mapping[str, int],
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
            layer_ids (Mapping[str, int]): a mapping from material names to
                integers representing the layer number for each material.
            masks (Mapping[str, npt.NDArray[np.bool_]]): a mapping from material
                names to boolean masks indicating the gridpoints.
        """
        vp = stride.ScalarField(name="vp", grid=self.grid)  # [m/s]
        rho = stride.ScalarField(name="rho", grid=self.grid)  # [kg/m^3]
        alpha = stride.ScalarField(name="alpha", grid=self.grid)  # [dB/cm]
        layer = stride.ScalarField(name="layer", grid=self.grid)  # integers

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

        self.medium.add(vp)
        self.medium.add(rho)
        self.medium.add(alpha)
        self.medium.add(layer)
