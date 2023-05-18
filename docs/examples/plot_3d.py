# -*- coding: utf-8 -*-
"""
Running a 3D simulation
====================================

This example demonstrates how to execute a steady state
simulation for a 3D source.

Running such simulations is computationally expensive and
can take a long time to complete. For this reason, we
recommend running this example on an external machine, store
the results in a file and then load them on your local machine
for visualization.

Check the gallery example
[Save and load results](./plot_store_results.md) to learn how
to save and load results.
"""
# %%
# The following step is computationally expensive:
import neurotechdevkit as ndk

scenario = ndk.make('scenario-1-3d-v0')
result = scenario.simulate_steady_state()

# %%
# In order to render the 3D results you will need to install
# napari with: `pip install "napari[all]"`.

try:
    import napari
    result.render_steady_state_amplitudes_3d()
except ImportError:
    print("napari has not been installed. Please install it with: pip install napari[all]")

# %%
# If you have napari installed you should have been able to
# see an image like the following:
# ![3d-visualization](../../images/3d_visualization.gif)