# -*- coding: utf-8 -*-
"""

Reading simulation metrics
====================================

!!! note
    NDK and its examples are under constant development, more information and content will be added to this example soon!

This example demonstrates how to display the metrics collected from the simulation.
"""
# %%
# ## Rendering scenario
import neurotechdevkit as ndk


scenario = ndk.make('scenario-0-v0')
result = scenario.simulate_steady_state()
result.render_steady_state_amplitudes(show_material_outlines=False)

# %%
# ## Printing metrics
for metric, metric_value in result.metrics.items():
    print(f"{metric}:")
    print(f"\t {metric_value['description']}")
    print(f"\t Unit: {metric_value['unit-of-measurement']} Value: {metric_value['value']}")
    print()
