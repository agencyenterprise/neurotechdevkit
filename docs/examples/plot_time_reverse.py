# -*- coding: utf-8 -*-
"""
Time-reverse simulation for phased array
====================================================================

The skull adds aberrations to the beam propagation; phased arrays can compensate
for those by having different delays for each element, but estimating these
delays can be challenging. One method shared by Sergio Jiménez-Gambín and
Samuel Blackman is to run a "time reverse" simulation to calculate the
necessary delays.

This notebook shows an example of a "time reverse" simulation. The notebook
sets up a scenario with a phased array source and a target and then runs
a simulation with the source and target reversed to calculate the delays.
Finally, it uses the calculated delays to perform a forward-time simulation.

Note: In this notebook, we refer to the "true" target as the eventual brain
region we would like to stimulate, and the "true" source as the placement of
the ultrasound probes. We refer to the "reversed" or "simulated" target and
point-source as the values defined in our simulation, which are reversed from
the physical setup to help calculate values.
"""

# %%

import matplotlib.pyplot as plt
import numpy as np

import neurotechdevkit as ndk

# Parameters
SCENARIO_NAME = "scenario-2-2d-v0"
NUM_ELEMENTS = 20
ELEMENT_WIDTH = 1.2e-3


# %%
# Helper function to make the scenario with a PhasedArraySource
def make_scenario(element_delays=None) -> ndk.scenarios.Scenario:
    true_scenario = ndk.make(SCENARIO_NAME)

    # define a phased-array source
    default_source = true_scenario.get_default_source()
    true_source = ndk.sources.PhasedArraySource2D(
        element_delays=element_delays,
        position=default_source.position,
        direction=default_source.unit_direction,
        num_elements=NUM_ELEMENTS,
        pitch=default_source.aperture / NUM_ELEMENTS,
        element_width=ELEMENT_WIDTH,
        num_points=1000,
    )

    true_scenario.add_source(true_source)
    return true_scenario


# %%
# ## Set up and visualize the forward scenario
true_scenario = make_scenario()
assert isinstance(true_scenario, ndk.scenarios.Scenario2D)
true_scenario.render_layout()


# %%
# ## Simulate the time-reverse scenario
# Place a point source at the true target, and simulate a pulse.
# The point source is visualized as a gray dot.

# Reinitialize the scenario
reversed_scenario = ndk.make(SCENARIO_NAME)
# and reverse the source
point_source = ndk.sources.PointSource2D(
    position=true_scenario.target.center,
)
reversed_scenario.add_source(point_source)

assert isinstance(reversed_scenario, ndk.scenarios.Scenario2D)
reversed_scenario.render_layout()


# %%
result = reversed_scenario.simulate_pulse()
assert isinstance(result, ndk.results.PulsedResult2D)
result.render_pulsed_simulation_animation()


# %% Calculate the time-reverse delays
# We calculate how long it took for the point-source pulse to reach each of
# the true array elements. Here, we coarsely approximate these delays by
# finding the pressure argmax at each element's nearest-neighbor coordinates.

# Map array elements onto the nearest pixels in our simulation
def map_coordinates_to_indices(coordinates, origin, dx):
    indices = np.round((coordinates - origin) / dx).astype(int)
    return indices


# Get the pressure time-series of these elements
[true_source] = true_scenario.sources
assert isinstance(true_source, ndk.sources.PhasedArraySource2D)
element_indices = map_coordinates_to_indices(
    true_source.element_positions,
    reversed_scenario.origin,
    reversed_scenario.dx,
)
pressure_at_elements = result.wavefield[element_indices[:, 0], element_indices[:, 1]]

# Calculate the time of arrival for each element
element_reverse_delays = np.argmax(pressure_at_elements, axis=1) * result.effective_dt
plt.plot(element_reverse_delays, marker="o")
plt.xlabel("element index")
plt.ylabel("delay [s]")


# %%
# Visually inspecting the earlier scenario layout, these results seem reasonable.
# The expected delay \(t_d\) is approximately:
#
# $$
# t_d \approx \frac{||x_{source} - x_{target}||_2}{c_{water}} \approx
# \frac{0.07 \text{ m}}{1500 \text{ m/s}} \approx 47 \mu s
# $$
#


# %%
# ## Use delays in forward-time simulation
# Next, let's validate these delays by using them in a normal forward-time
# simulation.
# We simulate the original scenario, setting the pulse delays as calculated.

# Elements that took longer to reach should now be pulsed first,
# so we invert the values
element_delays = element_reverse_delays.max() - element_reverse_delays

true_scenario = make_scenario(element_delays=element_delays)
result = true_scenario.simulate_pulse()
assert isinstance(result, ndk.results.PulsedResult2D)
result.render_pulsed_simulation_animation()


# %%
# The pulse should focus on the true target.


# %%
# ### Simulate steady-state
# Another way to visualize the simulation is to check that the steady-state
# pressure (within the skull) peaks near the target.

# Re-initialize scenario to clear previous simulation
true_scenario = make_scenario(element_delays=element_delays)
steady_state_result = true_scenario.simulate_steady_state()
assert isinstance(steady_state_result, ndk.results.SteadyStateResult2D)
steady_state_result.render_steady_state_amplitudes()


# %%
# ## Future directions for improvement
# The time-reverse simulation is not an exact solution for the forward-time
# design. Other factors, like the angle of incidence at the boundary of two
# materials, will be different in the time reverse vs forward-time.
