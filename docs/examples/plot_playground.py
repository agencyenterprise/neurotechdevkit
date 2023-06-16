# -*- coding: utf-8 -*-
"""
Plot playground
====================================

This is a walkthrough of NDK capabilities. It is not meant to be run as a
script, but rather to be used as a playground to test out different
features of NDK.
"""
# %%
# First let's import the neurotechdevkit package and create a scenario
import neurotechdevkit as ndk
import numpy as np

scenario = ndk.make('scenario-0-v0')

# %%
# Then we can simulate the steady state of the scenario
result = scenario.simulate_steady_state()

# %%
# We can then render the steady state amplitudes
result.render_steady_state_amplitudes(show_material_outlines=False)

# %%
# We can easily print the focal gain of the executed simulation
result.metrics['focal_gain']['value']

# %%
# We can also print all available metrics for this simulation
for metric, metric_value in result.metrics.items():
    print(f"{metric}:")
    print(f"\t {metric_value['description']}")
    print(f"\t Unit: {metric_value['unit-of-measurement']} Value: {metric_value['value']}")
    print()

# %%
# Now let's try to move the source to a different position
# and simulate the steady state again
scenario = ndk.make('scenario-0-v0')
source = ndk.sources.FocusedSource2D(
    position=np.array([0.04, -0.002]),
    direction=np.array([-0.85, 0.35]),
    aperture=0.01,
    focal_length=0.01,
    num_points=1000,
)
scenario.add_source(source)
result = scenario.simulate_steady_state()
result.render_steady_state_amplitudes(show_material_outlines=False)

# %%
# Let's display the focal gain of the executed simulation
result.metrics['focal_gain']['value']

# %%
# Now let's try to move the target to a different position
scenario = ndk.make('scenario-0-v0')
scenario._TARGET_OPTIONS = {
    "target_1": ndk.scenarios.Target(
        target_id="target_1",
        center=np.array([0.02, -0.0024]),
        radius=0.0012,
        description="Represents a simulated tumor.",
    ),
  }
result = scenario.simulate_steady_state()
result.render_steady_state_amplitudes(show_material_outlines=False)

# %%
# Let's see how the changed target affects the focal gain
result.metrics['focal_gain']['value']

# %%
# Are you able to move the source and target to maximize the focal gain?

# add your solution here

# %%
# Now, let's add a second source to the scenario
scenario = ndk.make('scenario-0-v0')
s1 = ndk.sources.FocusedSource2D(
            position=np.array([0.01, 0.0]),
            direction=np.array([0.92, 0.25]),
            aperture=0.01,
            focal_length=0.022,
            num_points=1000,
        )

s2 = ndk.sources.FocusedSource2D(
            position=np.array([0.04, -0.002]),
            direction=np.array([-0.85, 0.35]),
            aperture=0.01,
            focal_length=0.011,
            num_points=1000,
            delay=5.1e-6,
        )

scenario.add_source(s1)
scenario.add_source(s2)
result = scenario.simulate_steady_state()
result.render_steady_state_amplitudes(show_material_outlines=False)

# %%
# Let's see how the changed target affects the focal gain
result.metrics['focal_gain']['value']
# %%
# Can you add a third source to the scenario to maximize the focal gain?

# add your solution here

# %%
# Now let's customize the tumor of the scenario
scenario = ndk.make('scenario-0-v0')
scenario._material_layers = [
  ("water", ndk.materials.water),
  ("skull", ndk.materials.cortical_bone),
  ("brain", ndk.materials.brain),
  ("tumor", ndk.materials.Material(1650.0, 1150.0, 0.8, "#ff00ff")),
]
result = scenario.simulate_steady_state()
result.render_steady_state_amplitudes(show_target=False, show_material_outlines=False)

# %%
