"""Materials for the neurotechdevkit scenarios.

Each material has the following characteristics:
- vp: the speed of sound (in m/s).
- rho: the mass density (in kg/m³).
- alpha: the absorption (in dB/cm).
- render_color: the color used when rendering this material in the scenario layout
    plot.
"""
from mosaic.types import Struct

# TODO: encapsulate mosaic struct behind an NDK materials type


water = Struct()
water.vp = 1500.0
water.rho = 1000.0
water.alpha = 0.0
water.render_color = "#2E86AB"


skin = Struct()
skin.vp = 1610.0
skin.rho = 1090.0
skin.alpha = 0.2
skin.render_color = "#FA8B53"


cortical_bone = Struct()
cortical_bone.vp = 2800.0
cortical_bone.rho = 1850.0
cortical_bone.alpha = 4.0
cortical_bone.render_color = "#FAF0CA"


trabecular_bone = Struct()
trabecular_bone.vp = 2300.0
trabecular_bone.rho = 1700.0
trabecular_bone.alpha = 8.0
trabecular_bone.render_color = "#EBD378"


brain = Struct()
brain.vp = 1560.0
brain.rho = 1040.0
brain.alpha = 0.3
brain.render_color = "#DB504A"


# these numbers are completely made up
# TODO: research reasonable values
tumor = Struct()
tumor.vp = 1650.0
tumor.rho = 1150.0
tumor.alpha = 0.8
tumor.render_color = "#94332F"


# Based on Table 1 of: https://doi.org/10.1016/j.ultrasmedbio.2020.03.011
agar_hydrogel = Struct()
agar_hydrogel.vp = 1570.0
agar_hydrogel.rho = 1016.0
agar_hydrogel.alpha = 0.1
agar_hydrogel.render_color = "#E9E6C9"
