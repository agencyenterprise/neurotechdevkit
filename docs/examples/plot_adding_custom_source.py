# -*- coding: utf-8 -*-
"""

Custom source
====================================

!!! note
    NDK and its examples are under constant development, more information
    and content will be added to this example soon!

This example demonstrates how to add a source to the simulation.
"""
# %%
# A Source receives the following parameters:
#
#
# - position `(npt.NDArray[np.float_])`: a numpy float array indicating the
#   coordinates (in meters) of the point at the center of the source.
# - direction `(npt.NDArray[np.float_])`: a numpy float array representing a vector
#   located at position and pointing towards the focal point. Only the
#   orientation of `direction` affects the source, the length of the vector has
#   no affect. See the `unit_direction` property.
# - aperture `(float)`: the width (in meters) of the source.
# - focal_length `(float)`: the distance (in meters) from `position` to the focal
#   point.
# - num_points `(int)`: the number of point sources to use when simulating the source.
# - delay `(float, optional)`: the delay (in seconds) that the source will wait before
#   emitting. Defaults to 0.0.

import neurotechdevkit as ndk

source = ndk.sources.FocusedSource2D(
    position=[0.00, 0.0],
    direction=[0.9, 0.0],
    aperture=0.01,
    focal_length=0.01,
    num_points=1000,
)

scenario = ndk.built_in.ScenarioSimple()
scenario.sources = [source]
scenario.make_grid()
scenario.compile_problem()
result = scenario.simulate_steady_state()
assert isinstance(result, ndk.results.SteadyStateResult2D)
result.render_steady_state_amplitudes()

# %%
