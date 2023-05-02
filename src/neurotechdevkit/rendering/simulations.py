from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap

from ._formatting import (
    configure_axis_labels,
    configure_axis_ticks,
    configure_grid,
    configure_title,
)
from .font import AXIS_LABEL_FONT_PROPS, AXIS_TICK_FONT_PROPS, LEGEND_FONT_PROPS
from .legends import (
    LegendConfig,
    SourceHandle,
    SourceHandler,
    TargetHandle,
    TargetHandler,
)


def create_steady_state_figure(
    extent: npt.NDArray[np.float_],
    origin: npt.NDArray[np.float_],
    amplitudes: npt.NDArray[np.float_],
) -> tuple[plt.Figure, plt.Axes]:
    """Create an unformatted figure containing the steady-state pressure amplitude.

    Unformatted means that the data has been plotted, but things like axes ticks and
    labels, titles, fonts, legend, etc have not been adjusted.

    The x-axis is plotted in decreasing manner on the vertical axis, and the y-axis is
    plotted in increasing manner on the horizontal axis.

    Args:
        extent: An array of shape (2,) which indicates the spatial size (in meters) of
            the field along each axis.
        origin: An array of shape (2,) which contains the spatial coordinates (in
            meters) of the field element with indices (0, 0).
        amplitudes: A 2D integer array containing the steady-state wave amplitudes
            over the entire field.

    Returns:
        A tuple with the figure and the axis that contain the steady-state data plot.
    """

    assert len(extent) == 2, "The rendering only supports 2D fields."

    fig = plt.figure()
    ax = fig.gca()
    # the extent for imshow is "(left, right, bottom, top)"
    imshow_extent = np.array(
        [origin[1], origin[1] + extent[1], origin[0] + extent[0], origin[0]]
    )

    ax.imshow(amplitudes, cmap="viridis", extent=imshow_extent)
    return fig, ax


def create_pulsed_figure(
    origin: npt.NDArray[np.float_],
    extent: npt.NDArray[np.float_],
    wavefield: npt.NDArray[np.float_],
    norm: str = "linear",
) -> tuple[plt.Figure, plt.Axes]:
    """Creates a base figure containing the pulsed wavefield pressures.

    The figure is used as a template to created frames for an animation. The colorbar
    is adjusted to the minimum and maximum pressure observed in the wavefield.

    The x-axis is plotted in decreasing manner on the vertical axis, and the y-axis is
    plotted in increasing manner on the horizontal axis.

    The scale of the pressure can be controlled with the `norm` argument. For a list of
    available scales, call matplotlib.scale.get_scale_names().

    Args:
        extent: An array of shape (2,) which indicates the spatial size (in meters) of
            the field along each axis.
        origin: An array of shape (2,) which contains the spatial coordinates (in
            meters) of the field element with indices (0, 0).
        wavefield: An array in 2 spacial dimensions and one temporal dimension with the
            recorded pressures during the simulation. The temporal dimension should
            be the last one.
        norm: The normalization method used to scale scalar data to the [0, 1]
            range before mapping to colors using cmap. For a list of available scales,
            call `matplotlib.scale.get_scale_names()`.

    Returns:
        A tuple with the figure and the axis that contain the pulsed data plot.
    """
    fig = plt.figure()
    ax = fig.gca()

    min_pressure = wavefield.min()
    max_pressure = wavefield.max()

    cmap = _create_centered_bidirectional_cmap(vmin=min_pressure, vmax=max_pressure)
    imshow_extent = np.array(
        [origin[1], origin[1] + extent[1], origin[0] + extent[0], origin[0]]
    )
    # reference image, it would be replaced when creating the animation.
    mid_point = wavefield.shape[-1] // 2
    ax.imshow(wavefield[:, :, mid_point], cmap=cmap, norm=norm, extent=imshow_extent)
    return fig, ax


def configure_result_plot(
    fig: plt.Figure,
    ax: plt.Axes,
    show_sources: bool,
    show_target: bool,
    extent: npt.NDArray[np.float_],
    origin: npt.NDArray[np.float_],
    vertical_label: str,
    horizontal_label: str,
    title: str,
    clim: tuple[float, float] | None = None,
) -> None:
    """Configures a results plot figure.

      Configuration includes: axes, title, colorbar, and legend.

    Args:
        fig: The layout plot figure to configure.
        ax: The axes which contains the layout plot.
        show_sources: Whether or not to include the source transducer layer in the
        legend.
        show_target: Whether or not to include the the target layer in the legend.
        extent: An array of shape (2,) which indicates the spatial size (in meters) of
            the field along each axis.
        origin: An array of shape (2,) which contains the spatial coordinates (in
            meters) of the field element with indices (0, 0).
        vertical_label: The label to apply to the vertical axis.
        horizontal_label: The label to apply to the horizontal axis.
        title: The title to give the figure.
        clim: A tuple with (min, max) values for the limits for the colorbar.
    """

    configure_title(fig, title, x_pos=0.63)
    _configure_colorbar(fig, ax, clim=clim)

    show_legend = show_sources or show_target
    if show_legend:
        _configure_legend(ax, show_sources, show_target)

    configure_grid(ax)
    ax.set_xlim(origin[1], origin[1] + extent[1])
    ax.set_ylim(origin[0] + extent[0], origin[0])
    configure_axis_labels(ax, horizontal_label, vertical_label)
    configure_axis_ticks(ax)

    fig.tight_layout()
    return


def _configure_colorbar(
    fig: plt.Figure, ax: plt.Axes, clim: tuple[float, float] | None = None
) -> None:
    """Configure the colorbar for the steady-state amplitude plot.

    Args:
        fig: The figure containing the plot to be configured.
        ax: The axes containing the plot to be configured.
        clim: A tuple with (min, max) values for the limits for the colorbar.
    """
    im = ax.get_images()[0]
    cb = fig.colorbar(im, pad=0.01)
    if clim is not None:
        cb.mappable.set_clim(*clim)
    cb.set_label("Pressure (Pa)", fontproperties=AXIS_LABEL_FONT_PROPS, color="#4F4F4F")

    for label in cb.ax.get_yticklabels():
        label.set_font_properties(AXIS_TICK_FONT_PROPS)


def _configure_legend(ax: plt.Axes, show_sources: bool, show_target: bool) -> None:
    """Configure the legend for the steady-state amplitude plot.

    Args:
        ax: The axes containing the legend to be configured.
        show_target: Whether or not to show the target marker.
        show_sources: Whether or not to show the source markers.
    """

    config = LegendConfig()

    if show_target:
        config.append_item("target", TargetHandle(), TargetHandler())

    if show_sources:
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


def _create_centered_bidirectional_cmap(
    vmin: float, vmax: float
) -> LinearSegmentedColormap:
    """
    Extends the default colormap (viridis) to support negative values.

    Values close to zero in absolute value have almost identical color.
    Values diverge as they approach the extremes values `vmin` and `vmax`.

    Args:
        vmin: minimum value that the colormap scale should reach.
        vmax: maximum value that the colormap scale should reach.

    Returns:
        A colormap where zero has always the same color (LinearSegmentedColorMap).
    """

    ratio = np.abs(vmin / vmax)
    n_points = 128
    colors1 = cm.viridis(np.linspace(0, 1, int(n_points * ratio)))
    colors2 = cm.viridis(np.linspace(0.0, 1, n_points))

    def modify_viridis(color, it, delta_color=1 / int(n_points * ratio)):
        """Gradually modifies the color of the original color map."""
        R, G, B, opacity = color
        B = min(1, B + it * delta_color)
        opa = max(0, opacity - it * delta_color)
        return R, G, B, opa

    colors1 = [modify_viridis(c, i) for i, c in enumerate(colors1)]

    # combine them and build a new colormap
    colors = np.vstack((colors1[::-1], colors2))
    mymap = LinearSegmentedColormap.from_list("pulsed_cm", colors)

    return mymap
