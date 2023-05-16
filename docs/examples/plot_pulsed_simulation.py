# -*- coding: utf-8 -*-
"""
Plot pulsed simulation
====================================

This example demonstrates how to execute a pulsed simulation
using ndk
"""
# %%
import neurotechdevkit as ndk

scenario = ndk.make('scenario-0-v0')
result = scenario.simulate_pulse()
result.render_pulsed_simulation_animation()

# %%
