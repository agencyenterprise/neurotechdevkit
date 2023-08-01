# -*- coding: utf-8 -*-
"""
Plot scenarios
====================================

This example demonstrates how to execute a steady state simulation
using ndk
"""
# %%
import neurotechdevkit as ndk


def plot_scenario(scenario):
    print(f"Simulating scenario: {scenario.name}")
    scenario = scenario.value()
    scenario.make_grid(center_frequency=5e5)
    scenario.compile_problem()
    result = scenario.simulate_steady_state()
    result.render_steady_state_amplitudes(show_material_outlines=False)


# %%
# Simulating scenario: scenario 0
# ===================================
plot_scenario(ndk.BUILTIN_SCENARIOS.SCENARIO_0)

# %%
# Simulating scenario: scenario 1 2D
# ===================================
plot_scenario(ndk.BUILTIN_SCENARIOS.SCENARIO_1_2D)

# %%
# Simulating scenario: scenario 2 2D
# ===================================
plot_scenario(ndk.BUILTIN_SCENARIOS.SCENARIO_2_2D)

# %%
