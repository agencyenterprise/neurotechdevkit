# -*- coding: utf-8 -*-
"""
Plot scenarios
====================================

This example demonstrates how to execute a steady state simulation
using ndk
"""
# %%
import neurotechdevkit as ndk


def plot_scenario(chosen_scenario):
    print(f"Simulating scenario: {chosen_scenario.__name__}")
    scenario = chosen_scenario()
    scenario.make_grid()
    scenario.compile_problem()
    result = scenario.simulate_steady_state()
    result.render_steady_state_amplitudes(show_material_outlines=False)


# %%
# Simulating scenario: scenario-simple
# ===================================
plot_scenario(ndk.scenarios.built_in.ScenarioSimple)

# %%
# Simulating scenario: scenario-flat-skull 2D
# ===================================
plot_scenario(ndk.scenarios.built_in.ScenarioFlatSkull_2D)

# %%
# Simulating scenario: scenario-realistic-skull 2D
# ===================================
plot_scenario(ndk.scenarios.built_in.ScenarioRealisticSkull_2D)

# %%
