"""Materials for the neurotechdevkit scenarios."""
from dataclasses import dataclass

from mosaic.types import Struct


@dataclass
class Material:
    """A material with properties used in the neurotechdevkit scenarios."""

    vp: float
    rho: float
    alpha: float
    render_color: str

    def to_struct(self) -> Struct:
        """Return a Struct representation of the material.

        Returns:
            Struct: a Struct representation of the material.
        """
        struct = Struct()
        struct.vp = self.vp
        struct.rho = self.rho
        struct.alpha = self.alpha
        struct.render_color = self.render_color
        return struct


water = Material(1500.0, 1000.0, 0.0, "#2E86AB")
skin = Material(1610.0, 1090.0, 0.2, "#FA8B53")
cortical_bone = Material(2800.0, 1850.0, 4.0, "#FAF0CA")
trabecular_bone = Material(2300.0, 1700.0, 8.0, "#EBD378")
brain = Material(1560.0, 1040.0, 0.3, "#DB504A")

# these numbers are completely made up
# TODO: research reasonable values
tumor = Material(1650.0, 1150.0, 0.8, "#94332F")
