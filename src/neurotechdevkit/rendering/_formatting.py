from itertools import chain

import matplotlib.pyplot as plt

from .font import AXIS_LABEL_FONT_PROPS, AXIS_TICK_FONT_PROPS, TITLE_FONT_PROPS


def configure_title(fig: plt.Figure, title: str, x_pos: float = 0.5) -> None:
    """Configures the title of the plot.

    Note: we might expect that 0.5 in figure coordinates would place the text centered
    above the plot, but this does not seem to always be the case. The `x_pos` parameter
    is provided to make manual adjustments to the horizontal position of the suptitle if
    needed. This will hopefully be resolved in the future by better controlling figure
    and axes dimensions for reproducibility.

    Args:
        fig: The figure where the title should be configured.
        title: The text for the title to give the plot.
        x_pos: The x position in figure coordinates to pass into `fig.suptitle`.
    """
    fig.suptitle(
        title,
        x=x_pos,
        fontproperties=TITLE_FONT_PROPS,
        color="#4F4F4F",
    )


def configure_axis_labels(
    ax: plt.Axes, horizontal_label: str, vertical_label: str
) -> None:
    """Configures the labels for the X and Y axes.

    Args:
        ax: The axes containing the labels to be configured.
        vertical_label: The label to apply to the vertical axis.
        horizontal_label: The label to apply to the horizontal axis.
    """
    ax.set_xlabel(
        f"{horizontal_label} (m)",
        fontproperties=AXIS_LABEL_FONT_PROPS,
        color="#4F4F4F",
    )
    ax.set_ylabel(
        f"{vertical_label} (m)",
        fontproperties=AXIS_LABEL_FONT_PROPS,
        color="#4F4F4F",
    )


def configure_axis_ticks(ax: plt.Axes) -> None:
    """Configures the ticks and tick labels for the X and Y axes.

    Args:
        ax: The axes containing the ticks to be configured.
    """
    ax.tick_params(axis="x", rotation=45)
    for label in chain(ax.get_xticklabels(), ax.get_yticklabels()):
        label.set_font_properties(AXIS_TICK_FONT_PROPS)


def configure_grid(ax: plt.Axes) -> None:
    """Configures the grid for the plot.

    Args:
        ax: The axes containing the grid to be configured.
    """
    ax.grid(
        color="#E7E7E7",
        linewidth=0.5,
    )
    ax.grid(False)
