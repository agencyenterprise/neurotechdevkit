import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from ._formatting import (
    configure_axis_labels,
    configure_axis_ticks,
    configure_grid,
    configure_title,
)
from .font import LEGEND_FONT_PROPS
from .legends import (
    LegendConfig,
    SourceHandle,
    SourceHandler,
    TargetHandle,
    TargetHandler,
)


def create_layout_fig(
    extent: npt.NDArray[np.float_],
    origin: npt.NDArray[np.float_],
    color_sequence: list[str],
    field: npt.NDArray[np.int_],
) -> tuple[plt.Figure, plt.Axes]:
    """Create an unformatted figure showing the layout of a scenario.

    Unformatted means that the data has been plotted, but things like axes ticks and
    labels, titles, fonts, legend, etc have not been adjusted.

    The x-axis is plotted in decreasing manner on the vertical axis, and the y-axis is
    plotted in increasing manner on the horizontal axis.

    Args:
        extent: An array of shape (2,) which indicates the spatial size (in meters) of
            the field along each axis.
        origin: An array of shape (2,) which contains the spatial coordinates (in
            meters) of the field element with indices (0, 0).
        color_sequence: The list of colors which correspond to each material in the
            field.
        field: A 2D integer array indicating which material is located at each point on
            the grid.

    Returns:
        fig: The new matplotlib figure.
        ax: The axes containing the plotted data.
    """

    assert len(extent) == 2, "The rendering only supports 2D fields."

    fig = plt.figure()
    ax = fig.gca()
    # the extent for imshow is "(left, right, bottom, top)"
    imshow_extent = np.array(
        [origin[1], origin[1] + extent[1], origin[0] + extent[0], origin[0]]
    )

    cmap = mpl.colors.ListedColormap(color_sequence)
    clim = (-0.5, len(color_sequence) - 0.5)
    ax.imshow(field, cmap=cmap, extent=imshow_extent, clim=clim)

    return fig, ax


def configure_layout_plot(
    fig: plt.Figure,
    ax: plt.Axes,
    color_sequence: list[str],
    layer_labels: list[str],
    show_sources: bool,
    show_target: bool,
    extent: npt.NDArray[np.float_],
    origin: npt.NDArray[np.float_],
    vertical_label: str = "X",
    horizontal_label: str = "Y",
    title: str = "Scenario Layout",
) -> None:
    """Configures a layout plot figure including axes, title, and legend.

    Args:
        fig: The layout plot figure to configure.
        ax: The axes which contains the layout plot.
        color_sequence: The list of colors which correspond to each material in the
            field.
        layer_labels: A list of labels to apply to the material layers in the plot.
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
    """

    configure_title(fig, title)
    _configure_legend(ax, layer_labels, color_sequence, show_target, show_sources)
    configure_grid(ax)

    ax.set_xlim(origin[1], origin[1] + extent[1])
    ax.set_ylim(origin[0] + extent[0], origin[0])
    configure_axis_labels(ax, horizontal_label, vertical_label)
    configure_axis_ticks(ax)

    fig.tight_layout()


def _configure_legend(
    ax: plt.Axes,
    layer_labels: list[str],
    color_sequence: list[str],
    show_target: bool,
    show_sources: bool,
) -> None:
    """Configure the legend for the figure.

    Args:
        axes: The axes containing the legend to be configured.
        layer_labels: A list of labels to give the material layers.
        color_sequence: A list of colors to give the material layers.
        show_target: Whether or not to show the target marker.
        show_sources: Whether or not to show the source markers.
    """

    config = LegendConfig()

    for label, color in zip(layer_labels, color_sequence):
        config.append_item(label, _get_material_layer_handle(color))

    if show_target:
        config.append_item("target", TargetHandle(), TargetHandler())

    if show_sources:
        config.append_item("source\ntransducer", SourceHandle(), SourceHandler())

    ax.legend(
        handles=config.get_handles(),
        labels=config.get_labels(),
        handler_map=config.get_custom_handlers(),
        bbox_to_anchor=(1.0, 1.0),
        loc="upper left",
        frameon=True,
        prop=LEGEND_FONT_PROPS,
        labelcolor="#4F4F4F",
        edgecolor="#E7E7E7",
    )


def _get_material_layer_handle(color: str) -> plt.Line2D:
    """Creates a legend handle for a material layer in the layout figure.

    Args:
        color: the color to use for the layer in the legend.

    Returns:
        A handle for the layer linestyle.
    """
    return plt.Line2D([], [], color=color, linestyle="", marker="s", markersize=12)
