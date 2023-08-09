# noqa: D100
# preventing package docstring to be rendered in documentation
from dataclasses import dataclass

from mosaic.types import Struct

NEPER_TO_DECIBEL = 8.6860000037


@dataclass
class _BaseMaterial:
    """A material with properties used in the neurotechdevkit scenarios."""

    vp: float
    rho: float
    render_color: str


@dataclass
class Material(_BaseMaterial):
    """An NDK Material with an attenuation coefficient."""

    alpha: float

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
DEFAULT_MATERIALS = {
    "water": _BaseMaterial(vp=1500.0, rho=1000.0, render_color="#2E86AB"),
    "skin": _BaseMaterial(vp=1610.0, rho=1090.0, render_color="#FA8B53"),
    "cortical_bone": _BaseMaterial(vp=2800.0, rho=1850.0, render_color="#FAF0CA"),
    "trabecular_bone": _BaseMaterial(vp=2300.0, rho=1700.0, render_color="#EBD378"),
    "brain": _BaseMaterial(vp=1560.0, rho=1040.0, render_color="#DB504A"),
    # these numbers for tumor material are made up
    # TODO: research reasonable values
    "tumor": _BaseMaterial(vp=1650.0, rho=1150.0, render_color="#94332F"),
}


@dataclass
class AttenuationConstant:
    """The parameters of the attenuation constant."""

    a0: float  # α0 [Np/m/MHz]
    b: float

    def calculate_absorption(self, frequency: float) -> float:
        """
        Calculate the absorption coefficient for a given center frequency.

        The absorption coefficient is calculated using the following formula:
            α(f) = α0 * f^b

        We convert the absorption coefficient from Np/m/Hz to dB/m/Hz
        multiplying it by 8.6860000037.

        Args:
            frequency (float): in Hz

        Returns:
            float: attenuation in dB/cm/MHz
        """
        attenuation = self.a0 * (frequency / 1e6) ** self.b  # Np/m
        db_attenuation = attenuation * NEPER_TO_DECIBEL  # dB/m
        return db_attenuation / 100  # convert from dB/m to dB/cm


ATTENUATION_CONSTANTS = {
    "trabecular_bone": AttenuationConstant(a0=47, b=1.2),
    "cortical_bone": AttenuationConstant(a0=54.553, b=1),
    "skin": AttenuationConstant(a0=21.158, b=1),
    "water": AttenuationConstant(a0=0.025328436, b=1),
    "brain": AttenuationConstant(a0=6.8032, b=1.3),
    # these numbers are completely made up
    # TODO: research reasonable values
    "tumor": AttenuationConstant(a0=8, b=1),
}


def _calculate_absorption(material_name: str, center_frequency: float) -> float:
    """Calculate the absorption coefficient for a given center frequency.

    Args:
        material_name (str): the name of the material. E.g. "water"
        center_frequency (float): the center frequency of the transducer in Hz

    Raises:
        ValueError: raised if the material is not supported.

    Returns:
        float: the absorption coefficient in dB/cm/Hz
    """
    try:
        return ATTENUATION_CONSTANTS[material_name].calculate_absorption(
            center_frequency
        )
    except KeyError:
        raise ValueError(f"Undefined material: {material_name}")


def get_render_color(material_name: str) -> str:
    """Get the render color for a material.

    Args:
        material_name (str): the name of the material. E.g. "water"

    Raises:
        ValueError: raised if the material is not supported.

    Returns:
        str: the render color of the material.
    """
    if material_name not in DEFAULT_MATERIALS:
        raise ValueError(f"Undefined material: {material_name}")

    return DEFAULT_MATERIALS[material_name].render_color


def get_material(material_name: str, center_frequency: float = 5.0e5) -> Material:
    """Get a material with properties used in the neurotechdevkit scenarios.

    Args:
        material_name (str): the name of the material. E.g. "water"
        center_frequency (float, optional): The center frequency of the
            transducer. Defaults to 5.0e5.

    Raises:
        ValueError: raised if the material is not supported.

    Returns:
        Material: a material with properties used in the neurotechdevkit.
    """
    if material_name not in DEFAULT_MATERIALS:
        raise ValueError(f"Undefined material: {material_name}")

    base_material = DEFAULT_MATERIALS[material_name]

    return Material(
        vp=base_material.vp,
        rho=base_material.rho,
        alpha=_calculate_absorption(material_name, center_frequency),
        render_color=base_material.render_color,
    )
