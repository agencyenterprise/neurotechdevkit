import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from ._source import create_source_legend_artist
from ._target import create_target_legend_artist


class LegendConfig:
    """A utility class for keeping track of the components needed for a legend."""

    def __init__(self):
        self._labels = []
        self._handles = []
        self._custom_handlers = {}

    def append_item(
        self,
        label: str,
        handle: object,
        custom_handler: mpl.legend_handler.HandlerBase = None,
    ) -> None:
        """Add a new legend item to the config.

        Args:
            label: the label to use for the legend item.
            handle: the handle for the legend item. Most often this is an Artist object,
                but can be any arbitrary object if a custom_handler is provided.
            custom_handler: A custom handler to be used for drawing the legend element
                for this legend item.
        """
        self._labels.append(label)
        self._handles.append(handle)
        if custom_handler is not None:
            self._custom_handlers[type(handle)] = custom_handler

    def get_labels(self, title_case: bool = True) -> list[str]:
        """Returns the list of configured labels.

        Args:
            title_case: If True, all labels will be converted to title case. If False,
                no changes will be made to the labels.

        Returns:
            The list of configured labels for the legend.
        """
        if title_case:
            return [label.title() for label in self._labels]
        return self._labels.copy()

    def get_handles(self) -> list[object]:
        """Returns the list of configured legend handles.

        Returns:
            The list of configured legend handles.
        """
        return self._handles.copy()

    def get_custom_handlers(self) -> dict[type, mpl.legend_handler.HandlerBase]:
        """Returns a map containing custom legend handlers.

        Returns:
            A map from handle type to legend handler instance.
        """
        return self._custom_handlers.copy()


class TargetHandle:
    """A handle for the target symbol.

    This class is used to trigger the TargetHandler in order to draw the correct icon
    for the target in the legend.
    """

    pass


class TargetHandler:
    """A legend handler to draw the target symbol in a legend.

    This class implements the required `legend_artist` . See
    https://matplotlib.org/stable/api/legend_handler_api.html#module-matplotlib.legend_handler
    for more information.
    """

    def legend_artist(
        self,
        legend: mpl.legend.Legend,
        orig_handle: TargetHandle,
        fontsize: float,
        handlebox: mpl.offsetbox.OffsetBox,
    ) -> plt.Artist:
        """Returns the artist that draws the target in the legend.

        Args:
            legend: The legend for which these legend artists are being created.
            orig_handle: The object for which these legend artists are being created.
            fontsize: The fontsize in pixels. The artists being created should be scaled
                according to the given fontsize.
            handlebox: The box which has been created to hold this legend entry's
                artists. Artists created in the legend_artist method must be added to
                this handlebox inside this method.

        Returns:
            An artist that draws the target in the legend.
        """
        TARGET_LEGEND_SCALE_FACTOR = 1.5

        center = np.array(  # in scenario coordinates
            [
                handlebox.xdescent + handlebox.width / 2,
                handlebox.ydescent + handlebox.height / 2,
            ]
        )
        target_radius = (
            np.min([handlebox.width, handlebox.height]) / 2 * TARGET_LEGEND_SCALE_FACTOR
        )

        target_artist = create_target_legend_artist(
            center, target_radius, transform=handlebox.get_transform()
        )
        handlebox.add_artist(target_artist)
        return target_artist


class SourceHandle:
    """A handle for the source symbol.

    This class is used to trigger the SourceHandler in order to draw the correct icon
    for the source in the legend.
    """

    pass


class SourceHandler:
    """A legend handler to draw the source symbol in a legend.

    This class implements the required `legend_artist` . See
    https://matplotlib.org/stable/api/legend_handler_api.html#module-matplotlib.legend_handler
    for more information.
    """

    def legend_artist(
        self,
        legend: mpl.legend.Legend,
        orig_handle: TargetHandle,
        fontsize: float,
        handlebox: mpl.offsetbox.OffsetBox,
    ) -> tuple[plt.Artist, ...]:
        """Returns the artist that draws the source in the legend.

        Args:
            legend: The legend for which these legend artists are being created.
            orig_handle: The object for which these legend artists are being created.
            fontsize: The fontsize in pixels. The artists being created should be scaled
                according to the given fontsize.
            handlebox: The box which has been created to hold this legend entry's
                artists. Artists created in the legend_artist method must be added to
                this handlebox inside this method.

        Returns:
            An artist that draws the source in the legend.
        """
        SOURCE_LEGEND_SCALE_FACTOR = 1.0

        center = np.array(  # in display coordinates (I think)
            [
                handlebox.xdescent + handlebox.width / 2,
                handlebox.ydescent + handlebox.height / 2,
            ]
        )
        max_width = handlebox.width * SOURCE_LEGEND_SCALE_FACTOR

        source_artist = create_source_legend_artist(
            center, max_width, transform=handlebox.get_transform()
        )
        handlebox.add_artist(source_artist)
        return source_artist
