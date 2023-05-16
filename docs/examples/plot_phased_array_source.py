# -*- coding: utf-8 -*-
"""
Phased array source
====================================

This example demonstrates how to add a phased array source to the simulation.
"""
# %%
# A PhasedArraySource receives the following parameters:
#
#
# - position `(npt.NDArray[np.float_])`: a numpy float array indicating
#     the coordinates (in meters) of the point at the center of the
#     source, which is the point that bisects the line segment source.
# - direction `(npt.NDArray[np.float_])`: a numpy float array representing
#     a vector located at position that is perpendicular to the plane
#     of the source. Only the orientation of `direction` affects the
#     source, the length of the vector has no affect. See the
#     `unit_direction` property.
# - num_points `(int)`: the number of point sources to use when simulating
#     the source.
# - num_elements `(int)`: the number of elements of the phased array.
# - pitch `(float)`: the distance (in meters) between the centers of neighboring
#     elements in the phased array.
# - element_width `(float)`: the width (in meters) of each individual element of the array.
# - tilt_angle `(float)`: the desired tilt angle (in degrees) of the wavefront. The angle is
#     measured between the direction the wavefront travels and the normal to the
#     surface of the transducer, with positive angles resulting in a
#     counter-clockwise tilt away from the normal.
# - focal_length `(float)`: the distance (in meters) from `position` to the focal
#     point.
# - delay `(float, optional)`: the delay (in seconds) that the source will wait
#     before emitting.
# - element_delays: an 1D array with the delays (in seconds) for each element of the
#     phased array. Delays from `element_delays` take precedence; No other
#     argument affected the delays (`tilt_angle`, `focal_length` or `delay`)
#     would be considered. ValueError will be raised if provided values for either
#     `tilt_angle`, `focal_length` or `delay` are non-default.

import numpy as np
import neurotechdevkit as ndk

source = ndk.sources.PhasedArraySource2D(
    position=np.array([0., 0.]), # center for all n_elements
    direction=np.array([1., 0.]), # unit vector pointing downwards
    num_points=1000, # per all of the lines/rectangles
    pitch=0.0015,
    num_elements=16,
    tilt_angle=30.,
    element_width=0.0005,
)

scenario = ndk.make('scenario-0-v0')
scenario.add_source(source)
result = scenario.simulate_steady_state()
result.render_steady_state_amplitudes()

# %%
