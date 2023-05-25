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
# You can also generate a video file of the simulation.
#
# For this you will have to install [ffmpeg](https://ffmpeg.org/download.html).
import os

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation

import neurotechdevkit as ndk
from neurotechdevkit import rendering
from neurotechdevkit.rendering._formatting import (
    configure_axis_labels,
    configure_axis_ticks,
    configure_grid,
    configure_title,
)
from neurotechdevkit.rendering.font import (
    AXIS_LABEL_FONT_PROPS,
    AXIS_TICK_FONT_PROPS,
    LEGEND_FONT_PROPS,
)
from neurotechdevkit.rendering.legends import (
    LegendConfig,
    SourceHandle,
    SourceHandler,
    TargetHandle,
    TargetHandler,
)

matplotlib.rcParams["animation.embed_limit"] = 2**100  # bytes, 1.5 GB


def make_animation(
    fig,
    ax,
    wavefield,
    fname="./pulsed_simulation.mp4",
    overwrite=False,
    n_frames_undersample=2,
    bitrate=2500,
    fps=25,
    dpi=100,
    verbose=False,
):
    """Creates an animation of a wavefield.

    Args:
        fig, matplotlib figure that would act as template for the animation.
        ax, matplotlib figure axes where the actual wavelet is displayed.
        wavefield, np.array with the evolution of the wavefield. Time component should
            be in the last index.
        fname, str, the file name (path included and format) where the animation will be
            saved.
        overwrite: bool, whether to overwrite an animation stored in disk with the same
            file name.
        n_frames_undersample: int, the number of wavefield time steps to skip during the creation
            of the animation. 2 means that every second step is considered.
        bitrate: int, bitrate for the saved movie file (controls the output file size and quality).
        fps: int, the number of frames per second to show during the animation.
        dpi: int, the resolution of the animation in dots per inch (controls the output file size and quality).
        verbose: bool, whether to print to stdout aux information.

    Returns:
        None - Animation saved to disk.
    """
    if os.path.exists(fname) and not overwrite:
        print(
            f"File {fname} exists. Pass `overwrite=True` to replace it ",
            "or change the value for `fname`",
        )
        return

    wavefield = wavefield[:, :, ::n_frames_undersample].copy()

    im = ax.get_images()[0]
    im.set_array(wavefield[:, :, 0])

    wavefield = wavefield[:, :, 1:]
    n_frames = wavefield.shape[-1]

    def init():
        return [im]

    def animate(i):
        if i % 10 == 0 and verbose:
            print(f"Processing frame #{i} out of {n_frames+1}")
        data = wavefield[:, :, i]
        im.set_array(data)
        return [im]

    anim = FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=n_frames,
        repeat=False,
        save_count=n_frames,
        blit=True,
    )

    FFwriter = FFMpegWriter(
        fps=fps, codec="h264", metadata=dict(artist="neurotechdevkit"), bitrate=bitrate
    )

    anim.save(fname, writer=FFwriter, dpi=dpi)


def show_target_material_sources(fig, ax, scenario):
    # add layers
    material_field = scenario.get_field_data("layer").astype(int)
    rendering.draw_material_outlines(
        ax=ax,
        material_field=material_field,
        dx=scenario.dx,
        origin=scenario.origin,
        upsample_factor=scenario._material_outline_upsample_factor,
    )
    rendering.draw_target(ax, scenario.target_center, scenario.target_radius)
    for source in scenario.sources:
        drawing_params = rendering.SourceDrawingParams(
            source_is_flat=True,
            position=source.position,
            direction=source.unit_direction,
            aperture=source.aperture,
            focal_length=source.focal_length,
        )
        rendering.draw_source(ax, drawing_params)
    return fig, ax


def configure_pulsed_plot(
    fig,
    ax,
    extent,
    origin,
    vmin,
    vmax,
    vertical_label="X",
    horizontal_label="Y",
    title="Pulsed Wave Amplitude",
):
    configure_title(fig, title, x_pos=0.63)
    _configure_colorbar(fig, ax, vmin, vmax)

    _configure_legend(ax)

    configure_grid(ax)
    ax.set_xlim(origin[1], origin[1] + extent[1])
    ax.set_ylim(origin[0] + extent[0], origin[0])
    configure_axis_labels(ax, horizontal_label, vertical_label)
    configure_axis_ticks(ax)

    fig.tight_layout()

    return fig, ax


def _configure_colorbar(fig: plt.Figure, ax: plt.Axes, vmin, vmax) -> None:
    im = ax.get_images()[0]
    cb = fig.colorbar(im, pad=0.01)
    cb.mappable.set_clim(vmin, vmax)
    cb.set_label("Pressure (Pa)", fontproperties=AXIS_LABEL_FONT_PROPS, color="#4F4F4F")

    for label in cb.ax.get_yticklabels():
        label.set_font_properties(AXIS_TICK_FONT_PROPS)


def _configure_legend(ax: plt.Axes) -> None:
    config = LegendConfig()

    config.append_item("target", TargetHandle(), TargetHandler())
    config.append_item("source\ntransducer", SourceHandle(), SourceHandler())

    ax.legend(
        handles=config.get_handles(),
        labels=config.get_labels(),
        handler_map=config.get_custom_handlers(),
        bbox_to_anchor=(1.25, 1.0),
        loc="upper left",
        frameon=True,
        prop=LEGEND_FONT_PROPS,
        labelcolor="#4F4F4F",
        edgecolor="#E7E7E7",
    )


def create_bidirectional_cmap(vmin, vmax):
    """
    Extends the NDK default colormap (viridis) to support
    negative numbers while keeping the non negative numbers the same color.
    """
    ratio = np.abs(vmin / vmax)
    n_points = 128
    colors1 = plt.cm.viridis(np.linspace(0, 1, int(n_points * ratio)))
    colors2 = plt.cm.viridis(np.linspace(0.0, 1, n_points))

    def modify_viridis(color, it, dc=1 / int(n_points * ratio)):
        R, G, B, opa = color
        B = min(1, B + it * dc)
        opa = max(0, opa - it * dc)
        return R, G, B, opa

    colors1 = [modify_viridis(c, i) for i, c in enumerate(colors1)]

    # combine them and build a new colormap
    colors = np.vstack((colors1[::-1], colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list("pulsed_cm", colors)

    return mymap


def create_pulsed_base_figure(scenario, wavefield):
    extent = scenario.extent
    origin = scenario.origin

    min_pressure_cliped = wavefield.min()
    max_pressure_cliped = wavefield.max()

    fig = plt.figure()
    ax = fig.gca()

    assert len(extent) == 2, "The rendering only supports 2D fields."

    # the extent for imshow is "(left, right, bottom, top)"
    imshow_extent = np.array(
        [origin[1], origin[1] + extent[1], origin[0] + extent[0], origin[0]]
    )

    cmap = create_bidirectional_cmap(vmin=min_pressure_cliped, vmax=max_pressure_cliped)

    # get the image in the middle as reference
    ax.imshow(
        wavefield[:, :, wavefield.shape[-1] // 2], cmap=cmap, extent=imshow_extent
    )

    fig, ax = show_target_material_sources(fig, ax, scenario)
    fig, ax = configure_pulsed_plot(
        fig,
        ax,
        extent,
        origin,
        vmin=min_pressure_cliped,
        vmax=max_pressure_cliped,
    )

    return fig, ax

# %%
# Creating and storing the animation into `./animation.mp4`.
fig, ax = create_pulsed_base_figure(scenario=scenario, wavefield=result.wavefield)
make_animation(
    fig,
    ax,
    result.wavefield,
    fname="animation.mp4",
    overwrite=True,
    n_frames_undersample=2,
    fps=25,
)
