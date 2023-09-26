# -*- coding: utf-8 -*-
"""
Adding multiple sources
====================================

!!! note
    NDK and its examples are under constant development, more information
    and content will be added to this example soon!

Adding multiple sources for transcranial ultrasound stimulation enables
greater precision and control in targeting specific areas of the brain.

By choosing the phase of ultrasound waves for each source, a combined beam
can be created that is focused on the desired target precisely. This allows
for complex wave patterns that open up new possibilities for therapies.
"""
# %%
import neurotechdevkit as ndk

scenario = ndk.built_in.ScenarioSimple()

s1 = ndk.sources.FocusedSource2D(
    position=[0.01, 0.0],
    direction=[0.92, 0.25],
    aperture=0.01,
    focal_length=0.022,
    num_points=1000,
)

s2 = ndk.sources.FocusedSource2D(
    position=[0.04, -0.002],
    direction=[-0.85, 0.35],
    aperture=0.01,
    focal_length=0.011,
    num_points=1000,
    delay=5.1e-6,
)

scenario.sources = [s1, s2]
scenario.make_grid()
scenario.compile_problem()
result = scenario.simulate_pulse()
assert isinstance(result, ndk.results.PulsedResult2D)
result.render_pulsed_simulation_animation()

# %%
