"""Pydantic models for the material properties."""
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, create_model

from neurotechdevkit.materials import Material as NDKMaterial
from neurotechdevkit.scenarios import Scenario2D, Scenario3D


class Material(BaseModel):
    """Material model for the material properties."""

    vp: float
    rho: float
    alpha: float
    renderColor: str

    @classmethod
    def from_ndk_material(cls, material: NDKMaterial) -> "Material":
        """Instantiate the web material from an NDKMaterial."""
        return cls(
            vp=material.vp,
            rho=material.rho,
            alpha=material.alpha,
            renderColor=material.render_color,
        )

    def to_ndk_material(self) -> NDKMaterial:
        """Instantiate the NDKMaterial from the web material."""
        return NDKMaterial(
            vp=self.vp,
            rho=self.rho,
            alpha=self.alpha,
            render_color=self.renderColor,
        )


class MaterialName(Enum):
    """Enum to map the material names between the web and NDK."""

    water = "water"
    brain = "brain"
    trabecularBone = "trabecular_bone"
    corticalBone = "cortical_bone"
    skin = "skin"
    tumor = "tumor"
    agarHydrogel = "agar_hydrogel"

    @classmethod
    def get_material_titles(cls) -> List[Tuple[str, str]]:
        """Get the material names and titles.

        Returns:
            The list of material names and titles.
        """
        titles = []
        for material_name in cls:
            title = material_name.value.replace("_", " ").title()
            titles.append((material_name.name, title))
        return titles

    @classmethod
    def get_material_name(cls, ndk_material_name: str) -> str:
        """Get the material name from the NDK material name."""
        for material_name in MaterialName:
            if material_name.value == ndk_material_name:
                return material_name.name
        raise ValueError(f"Material name {ndk_material_name} not found.")

    @classmethod
    def get_ndk_material_name(cls, material_name: str) -> str:
        """Get the NDK material name from the material name."""
        for material in MaterialName:
            if material.name == material_name:
                return material.value
        raise ValueError(f"Material name {material_name} not found.")


_MaterialProperties: BaseModel = create_model(  # type: ignore
    "MaterialProperties",
    **{
        material_name.name: (Optional[Material], None) for material_name in MaterialName
    },
)


class MaterialProperties(_MaterialProperties):  # type: ignore
    """Material properties model for the material properties."""

    def to_ndk_material_properties(self) -> Dict[str, NDKMaterial]:
        """Instantiate the NDKMaterialProperties from the web material properties."""
        ndk_material_properties = {}
        for material_name in self.__fields__:
            if material := getattr(self, material_name):
                ndk_material = material.to_ndk_material()
                ndk_material_name = MaterialName.get_ndk_material_name(material_name)
                ndk_material_properties[ndk_material_name] = ndk_material
        return ndk_material_properties

    @classmethod
    def from_scenario(
        cls, scenario: Union[Scenario2D, Scenario3D]
    ) -> "MaterialProperties":
        """Instantiate the web material properties from a scenario."""
        material_properties = {}
        ndk_material_properties = scenario.material_properties
        for ndk_material_name, ndk_material in ndk_material_properties.items():
            material = Material.from_ndk_material(ndk_material)
            material_name = MaterialName.get_material_name(ndk_material_name)
            material_properties[material_name] = material
        return cls(**material_properties)
