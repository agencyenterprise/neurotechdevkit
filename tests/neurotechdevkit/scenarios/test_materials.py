import numpy as np
import pytest
from mosaic.types import Struct

from neurotechdevkit.materials import DEFAULT_MATERIALS, Material
from neurotechdevkit.scenarios import Scenario


def compare_structs(struct1: Struct, struct2: Struct):
    """Compare two Structs."""
    assert struct1.vp == struct2.vp
    assert struct1.rho == struct2.rho
    assert struct1.alpha == struct2.alpha
    assert struct1.render_color == struct2.render_color


class BaseScenario(Scenario):
    """A scenario for testing the materials module."""

    _SCENARIO_ID = "scenario-tester"
    _TARGET_OPTIONS = {}

    def __init__(self, complexity="fast"):
        self._target_id = "target_1"
        super().__init__(
            origin=np.array([-0.1, -0.1, 0.0]),
            complexity=complexity,
        )

    def _compile_problem(self, center_frequency: float):
        pass

    def _material_outline_upsample_factor(self):
        pass

    def get_default_source(self):
        pass

    def get_target_mask(self):
        pass


def test_default_materials():
    """Test that the default materials are used."""

    class ScenarioWithDefaultMaterials(BaseScenario):
        material_layers = ["brain", "skin"]

    scenario = ScenarioWithDefaultMaterials()
    assert scenario.material_colors == {"brain": "#DB504A", "skin": "#FA8B53"}

    materials = scenario.get_materials(500e3)
    assert list(materials.keys()) == ["brain", "skin"]
    compare_structs(materials["brain"], DEFAULT_MATERIALS["brain"].to_struct())
    compare_structs(materials["skin"], DEFAULT_MATERIALS["skin"].to_struct())


def test_custom_material_property():
    """Test that a custom material property is used."""

    class ScenarioWithCustomMaterialProperties(BaseScenario):
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

    class ScenarioWithCustomMaterial(BaseScenario):
        material_layers = ["brain", "eye"]
        material_properties = {
            "eye": Material(vp=1600.0, rho=1100.0, alpha=0.0, render_color="#2E86AB")
        }

    scenario = ScenarioWithCustomMaterial()
    assert scenario.material_colors == {"brain": "#DB504A", "eye": "#2E86AB"}

    materials = scenario.get_materials(500e3)

    assert list(materials.keys()) == ["brain", "eye"]
    compare_structs(materials["brain"], DEFAULT_MATERIALS["brain"].to_struct())
    compare_structs(
        materials["eye"],
        Material(vp=1600.0, rho=1100.0, alpha=0.0, render_color="#2E86AB").to_struct(),
    )


def test_material_absorption_is_calculated():
    """Test that the material absorption is calculated for a frequency !=500e3."""

    class ScenarioWithBrainMaterial(BaseScenario):
        material_layers = ["brain"]

    scenario = ScenarioWithBrainMaterial()

    materials = scenario.get_materials(600e3)

    assert list(materials.keys()) == ["brain"]
    assert materials["brain"].alpha == 0.3041793231753331


def test_unknown_material_without_properties():
    """Test that an unknown material without properties raises an error."""

    class ScenarioWithCustomMaterial(BaseScenario):
        material_layers = ["unknown_material"]

    scenario = ScenarioWithCustomMaterial()
    with pytest.raises(ValueError):
        scenario.material_colors
    with pytest.raises(ValueError):
        scenario.get_materials(500e3)
