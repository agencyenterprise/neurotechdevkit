# -*- coding: utf-8 -*-
"""
Time-reverse simulation for phased array
====================================================================

The skull adds aberrations to the beam propagation; phased arrays can compensate
for those by having different delays for each element, but estimating these
delays can be challenging.
One method to estimate the delays is a "time reverse" simulation as described in
this [Article](https://koreascience.kr/article/JAKO200612242715181.pdf).
This notebook demonstrates the "time reverse" method to estimate the delays. The
notebook sets up a scenario with a phased array source and a target and then
runs a simulation with the source and target reversed to calculate the delays.
Finally, it uses the calculated delays to perform a forward-time simulation.

!!! note
    In this notebook, we refer to the "true" target as the eventual brain
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
NUM_ELEMENTS = 20
ELEMENT_WIDTH = 1.2e-3


# %%
# Helper function to make the scenario with a PhasedArraySource
def make_scenario(element_delays=None):
    true_scenario = ndk.scenarios.built_in.Scenario2_2D()

    # define a phased-array source
    default_source = true_scenario.sources[0]
    true_source = ndk.sources.PhasedArraySource2D(
        element_delays=element_delays,
        position=default_source.position,
        direction=default_source.unit_direction,
        num_elements=NUM_ELEMENTS,
        pitch=default_source.aperture / NUM_ELEMENTS,
        element_width=ELEMENT_WIDTH,
        num_points=1000,
    )

    true_scenario.sources = [true_source]
    return true_scenario


# %%
# ## Set up and visualize the forward scenario
true_scenario = make_scenario()
true_scenario.make_grid()
true_scenario.compile_problem()
true_scenario.render_layout()


# %%
# ## Simulate the time-reverse scenario
# Place a point source at the true target, and simulate a pulse.
# The point source is visualized as a gray dot.

# Reinitialize the scenario
reversed_scenario = ndk.scenarios.built_in.Scenario2_2D()
# and reverse the source
point_source = ndk.sources.PointSource2D(
    position=true_scenario.target.center,
)
reversed_scenario.sources = [point_source]

reversed_scenario.make_grid()
reversed_scenario.compile_problem()
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
true_scenario.make_grid()
true_scenario.compile_problem()
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
true_scenario.make_grid()
true_scenario.compile_problem()
steady_state_result = true_scenario.simulate_steady_state()
assert isinstance(steady_state_result, ndk.results.SteadyStateResult2D)
steady_state_result.render_steady_state_amplitudes()


# %%
# We want to visualize and find the maximum pressure within the brain, so let's
# mask out everything else.
steady_state_pressure = steady_state_result.get_steady_state()
# Only consider the brain region
steady_state_pressure[~true_scenario.material_masks["brain"]] = np.nan
steady_state_result.steady_state = steady_state_pressure

steady_state_result.render_steady_state_amplitudes()


# %%
# We can also calculate how far the "time reverse" estimate is from the true
# target.
max_pressure_idx = steady_state_result.metrics["focal_position"]["value"]
grid = steady_state_result.traces.grid.space.grid
focal_point = np.array(
    [
        grid[0][max_pressure_idx[0]],
        grid[1][max_pressure_idx[1]],
    ]
)
# The backend grid is in different coordinates from the scenario grid, so we
# need to shift it.
focal_point += true_scenario.origin

print("target center:", true_scenario.target.center)
print("beam focal point:", focal_point)
error_distance = np.linalg.norm(true_scenario.target.center - focal_point)
print("error [m]:", error_distance)
print("error [mm]:", error_distance * 1000)


# %%
# ## Reasons for target mismatch
# The time-reverse simulation is not an exact solution for the forward-time
# design. Other factors, like the angle of incidence at the boundary of two
# materials, will be different in the time reverse vs forward-time.
#
# ### Exercise
# Do you think the time-reverse simulation will work better or worse for deeper
# targets? How about if the transducer was positioned next to a different part
# of the skull that is flatter?


# %%
# ### Acknowledgments
# Thanks to Sergio Jiménez-Gambín and Samuel Blackman for pointing us to the
# "time reverse" simulation method.
