# -*- coding: utf-8 -*-
"""
Plot customized center frequency
====================================

This example demonstrates how to use a customized center frequency
using ndk
"""
# %%
import neurotechdevkit as ndk

CENTER_FREQUENCY = 6e5

scenario = ndk.make('scenario-0-v0')

# Customizing material layers with random values
scenario._material_layers = [
  ("water", ndk.materials.Material(1600.0, 1100.0, 0.0, "#2E86AB")),
  ("skull", ndk.materials.Material(3000.0, 1850.0, 4.0, "#FAF0CA")),
  ("brain", ndk.materials.Material(1660.0, 1140.0, 0.3, "#DB504A")),
  ("tumor", ndk.materials.Material(1850.0, 1250.0, 0.8, "#94332F")),
]

result = scenario.simulate_steady_state(center_frequency=CENTER_FREQUENCY)
result.render_steady_state_amplitudes(show_material_outlines=False)

# %%
