import pytest
from mosaic.types import Struct

from neurotechdevkit.materials import Material
from neurotechdevkit.scenarios import Scenario2D


def compare_structs(struct1: Struct, struct2: Struct):
    """Compare two Structs."""
    assert struct1.vp == struct2.vp
    assert struct1.rho == struct2.rho
    assert struct1.alpha == struct2.alpha
    assert struct1.render_color == struct2.render_color


def test_custom_material_property():
    """Test that a custom material property is used."""

    class ScenarioWithCustomMaterialProperties(Scenario2D):
        material_layers = ["brain"]
        material_properties = {
            "brain": Material(vp=1600.0, rho=1100.0, alpha=0.0, render_color="#2E86AB")
        }

    scenario = ScenarioWithCustomMaterialProperties()
    assert scenario.material_colors == {"brain": "#2E86AB"}

    materials = scenario.get_materials(500e3)
    assert list(materials.keys()) == ["brain"]
    compare_structs(
        materials["brain"],
        Material(vp=1600.0, rho=1100.0, alpha=0.0, render_color="#2E86AB").to_struct(),
    )


def test_new_material():
    """Test that a new material is used."""

    class ScenarioWithCustomMaterial(Scenario2D):
        material_layers = ["brain", "eye"]
        material_properties = {
            "eye": Material(vp=1600.0, rho=1100.0, alpha=0.0, render_color="#2E86AB")
        }

    scenario = ScenarioWithCustomMaterial()
    assert scenario.material_colors == {"brain": "#DB504A", "eye": "#2E86AB"}

    materials = scenario.get_materials(500e3)

    assert list(materials.keys()) == ["brain", "eye"]
    compare_structs(
        materials["eye"],
        Material(vp=1600.0, rho=1100.0, alpha=0.0, render_color="#2E86AB").to_struct(),
    )


def test_material_absorption_is_calculated():
    """Test that the material absorption is calculated for a frequency !=500e3."""

    class ScenarioWithBrainMaterial(Scenario2D):
        material_layers = ["brain"]
        material_properties = {}

    scenario = ScenarioWithBrainMaterial()

    materials = scenario.get_materials(600e3)

    assert list(materials.keys()) == ["brain"]
    assert materials["brain"].alpha == 0.3041793231753331


def test_unknown_material_without_properties():
    """Test that an unknown material without properties raises an error."""

    class ScenarioWithCustomMaterial(Scenario2D):
        material_layers = ["unknown_material"]
        material_properties = {}

    scenario = ScenarioWithCustomMaterial()
    with pytest.raises(ValueError):
        scenario.material_colors
    with pytest.raises(ValueError):
        scenario.get_materials(500e3)
