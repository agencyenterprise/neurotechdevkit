# -*- coding: utf-8 -*-
"""
Plot pulsed simulation
====================================

This example demonstrates how to execute a pulsed simulation
using ndk
"""
# %%
import neurotechdevkit as ndk

scenario = ndk.make("scenario-0-v0")
result = scenario.simulate_pulse()
result.render_pulsed_simulation_animation()

# %%
#
# Generating a video file
# =======================
# You can also generate a video file of the simulation (which requires [ffmpeg](https://ffmpeg.org/download.html) installed).
#
# To create and save the video as `animation.mp4` in the current folder, all you need is the execute following command:
result.create_video_file("animation.mp4", fps=25, overwrite=True)
