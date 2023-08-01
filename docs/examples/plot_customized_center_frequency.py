# -*- coding: utf-8 -*-
"""
Custom center frequency
====================================

This example demonstrates how to use a customized center frequency
using ndk
"""
# %%
import neurotechdevkit as ndk

CENTER_FREQUENCY = 6e5

scenario = ndk.BUILTIN_SCENARIOS.SCENARIO_0.value()

# using default material layers
scenario.material_layers = ["water", "cortical_bone", "brain", "tumor"]

# Customizing material properties
scenario.material_properties = {
    "tumor": ndk.materials.Material(
        vp=1850.0, rho=1250.0, alpha=0.8, render_color="#94332F"
    ),
}
scenario.make_grid(center_frequency=CENTER_FREQUENCY)
scenario.compile_problem()
result = scenario.simulate_steady_state()
assert isinstance(result, ndk.results.SteadyStateResult2D)
result.render_steady_state_amplitudes(show_material_outlines=False)

# %%
