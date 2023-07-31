# -*- coding: utf-8 -*-
"""
Visualizing 3D results with Napari
====================================

This example demonstrates how to render a steady state result for
a 3D scenario using [napari](https://napari.org/stable/).

Running such simulations is computationally expensive and can take a long time to
complete. For this reason, we recommend running this simulation on an external machine,
store the results in a file and then load them on your local machine for visualization.

Check the gallery example
[Save and load results](./plot_store_results.md) to learn how to save and load results.
"""
# %%
# The following step downloads and loads a simulation executed on an external machine.
import pooch

import neurotechdevkit as ndk

URL = "https://neurotechdevkit.s3.us-west-2.amazonaws.com/result-scenario-2-3d-v2.tz"
known_hash = "0d1eb7684b67702d0951ff4436cf3db736c253181c34c6b4e92a49a72ad0a0c8"
downloaded_file_path = pooch.retrieve(url=URL, known_hash=known_hash, progressbar=True)
result = ndk.load_result_from_disk(downloaded_file_path)

# %%
# In order to render the 3D results you will need to install
# Install `napari` via pip:
#
# ```
# pip install "napari[all]"
# ```
#
# Note for Mac M1 users: Qt5 does not support Mac M1, so you will need to
# install the Qt6 backend instead:
# ```
# pip install "napari[pyqt6_experimental]"
# ```
#
# You can also follow the `napari` installation instructions:
# [link](https://napari.org/stable/tutorials/fundamentals/installation.html).

try:
    import napari  # noqa: F401

    assert isinstance(result, ndk.results.SteadyStateResult3D)
    result.render_steady_state_amplitudes_3d()
except ImportError:
    print(
        "napari has not been installed. Please install it with: pip install napari[all]"
    )

# %%
# If you have napari installed you should see an output like the following:
#
# ```
# Opening the napari viewer. The window might not show up on top of your notebook;
# look through your open applications if it does not.
# ```

# %%
# If you have napari installed you should have been able to see an image like
# the following:
# ![3d-visualization](../../images/3d_visualization.gif)
