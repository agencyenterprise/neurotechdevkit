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


# The values below consider a center frequency of 500kHz
SUPPORTED_MATERIALS = {
    "water": Material(1500.0, 1000.0, 0.0, "#2E86AB"),
    "skin": Material(1610.0, 1090.0, 0.2, "#FA8B53"),
    "cortical_bone": Material(2800.0, 1850.0, 4.0, "#FAF0CA"),
    "trabecular_bone": Material(2300.0, 1700.0, 8.0, "#EBD378"),
    "brain": Material(1560.0, 1040.0, 0.3, "#DB504A"),
    # these numbers are completely made up
    # TODO: research reasonable values
    "tumor": Material(1650.0, 1150.0, 0.8, "#94332F"),
}


def _calculate_absorption(material_name: str) -> float:
    # TODO: calculate alpha
    return SUPPORTED_MATERIALS[material_name].alpha


def get_render_color(material_name: str) -> str:
    """
    Get the render color for a material.

    Args:
        material_name (str): the name of the material. E.g. "water"

    Raises:
        ValueError: raised if the material is not supported.

    Returns:
        str: the render color of the material.
    """
    if material_name not in SUPPORTED_MATERIALS:
        raise ValueError(f"Unsupported material: {material_name}")

    return SUPPORTED_MATERIALS[material_name].render_color


def get_material(material_name: str, center_frequency: float = 5.0e5) -> Material:
    """
    Get a material with properties used in the neurotechdevkit scenarios.

    Args:
        material_name (str): the name of the material. E.g. "water"
        center_frequency (float, optional): The center frequency of the
            transducer. Defaults to 5.0e5.

    Raises:
        ValueError: raised if the material is not supported.

    Returns:
        Material: a material with properties used in the neurotechdevkit.
    """
    if material_name not in SUPPORTED_MATERIALS:
        raise ValueError(f"Unsupported material: {material_name}")

    base_material = SUPPORTED_MATERIALS[material_name]
    if center_frequency == 5.0e5:
        return base_material

    return Material(
        vp=base_material.vp,
        rho=base_material.rho,
        alpha=_calculate_absorption(material_name),
        render_color=base_material.render_color,
    )
