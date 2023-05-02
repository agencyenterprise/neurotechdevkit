import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from ._source import SourceDrawingParams, create_source_drawing_artist
from ._target import create_target_drawing_artist


def draw_source(ax: plt.Axes, source: SourceDrawingParams) -> None:
    """Draw a layer showing the scenario sources on top of a figure.

    This layer can be added to any scenario figure in 2D.

    Args:
        ax: The axes to draw the sources on.
        source: The Source object to be drawn.
    """
    source_artist = create_source_drawing_artist(source, transform=ax.transData)
    ax.add_artist(source_artist)


def draw_target(
    ax: plt.Axes, target_loc: npt.NDArray[np.float_], target_radius: float
) -> None:
    """Draw a layer showing the scenario target on top of a figure.

    This layer can be added to any scenario figure in 2D.

    Args:
        ax: The axes to draw the target on.
        target_loc: An array of shape (2,) indicating the location (in meters) of the
            target within the scenario.
        target_radius: The radius (in meters) of the target.
    """

    target_artist = create_target_drawing_artist(
        target_loc, target_radius, transform=ax.transData
    )
    ax.add_artist(target_artist)


def draw_material_outlines(
    ax: plt.Axes,
    material_field: npt.NDArray[np.int_],
    dx: float,
    origin: npt.NDArray[np.float_],
    upsample_factor: int = 1,
) -> None:
    """Draw an outline of the transitions between materials.

    The outline will have white pixels wherever there is an interface between more than
    one material (i.e. two neighboring pixels have different material ids).

    The alignment of the transition outline and the number of outline points are both
    affected by upsample_factor. A factor of 2 means that pixel density will be doubled
    along each dimension, so there will be twice the density of outline pixels. An
    appropriate value for upsample_factor has been chosen for each scenario so that
    ideally there are no gaps between pixels.

    Args:
        ax: The axes to draw the target on.
        material_field: A 2D array with the material id of the material at each
            gridpoint.
        dx: The spacing (in meters) between gridpoints in the scenario.
        origin: The 2D origin (in meters) of the scenario slice to be drawn.
        upsample_factor: The factor to use when upsampling the material field before
            detecting transitions between materials. If the factor is N, then each pixel
            will be split into N^2 pixels. Defaults to 1 (no resampling).
    """
    field = _upsample_field(material_field, upsample_factor)
    outline_mask = _get_outline_mask(field)

    shape = outline_mask.shape
    spacing = np.array([dx / upsample_factor] * 2)
    X, Y = np.meshgrid(
        np.arange(shape[0]) * spacing[0] + origin[0] - dx / 2,
        np.arange(shape[1]) * spacing[1] + origin[1] - dx / 2,
        indexing="ij",
    )
    mask = outline_mask.reshape((-1,))
    x = X.reshape((-1,))[mask]
    y = Y.reshape((-1,))[mask]
    ax.plot(y, x, ",w", markersize=1.0)


def _upsample_field(
    field: npt.NDArray[np.int_], upsample_factor: int
) -> npt.NDArray[np.int_]:
    """Upsample a 2D array using closest interpolation.

    Returns:
        The upsampled array.
    """
    upsample_kernel = np.ones((upsample_factor, upsample_factor))
    return np.kron(field, upsample_kernel).astype(field.dtype)


def _get_outline_mask(field: npt.NDArray[np.int_]) -> npt.NDArray[np.bool_]:
    """Returns a mask indicating where there is a transition between materials.

    Transitions are detected by comparing the input field by itself but shifted
    one pixel horizontally or vertically. Detected horizontal and vertical transitions
    are superimposed to determine where to return white pixels.

    Args:
        field: The input field from which outlines should be detected.

    Returns:
        A boolean mask marking the transition between materials in the input field.
    """
    h_diff = field[:, :-1] != field[:, 1:]
    v_diff = field[:-1] != field[1:]
    h_edges = np.pad(h_diff, [(0, 0), (0, 1)])
    v_edges = np.pad(v_diff, [(0, 1), (0, 0)])

    return h_edges | v_edges
