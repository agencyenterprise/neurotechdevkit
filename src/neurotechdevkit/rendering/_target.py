import pathlib

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, Transform, TransformedBbox

_COMPONENT_DIR = pathlib.Path(__file__).parent / "components"


def create_target_drawing_artist(
    target_loc: npt.NDArray[np.float_],
    target_radius: float,
    transform: Transform,
) -> plt.Artist:
    """Creates the matplotlib artist for the target symbol.

    The caller is expected to provide the transform that takes the coordinates for the
    center and radius and map them into display coordinates. Eg. for placing the marker
    on the plot, that can be `ax.transData`.

    Args:
        target_loc: The location of the center of the target (in meters) in scenario
            coordinates.
        target_radius: The radius of the target (in meters).
        transform: A Transform function which maps from plot data coordinates into
            display coordinates.

    Returns:
        A matplotlib artist containing the target symbol.
    """
    target_data = _load_target("light")

    data_location = np.flip(target_loc)  # transform into plot data coordinates
    upper_left = data_location - target_radius
    lower_right = data_location + target_radius
    bbox = Bbox([upper_left, lower_right])
    bbox = TransformedBbox(bbox, transform)

    img_box = BboxImage(
        bbox, resample=True, interpolation="antialiased", interpolation_stage="rgba"
    )
    img_box.set_data(target_data)
    return img_box


def create_target_legend_artist(
    center_loc: npt.NDArray[np.float_],
    target_radius: float,
    transform: Transform,
) -> plt.Artist:
    """Creates the matplotlib artist for the target symbol.

    The caller is expected to provide the transform that takes the coordinates for the
    center and radius and map them into display coordinates. Eg. for placing the marker
    on the plot, that can be `ax.transData`.

    Args:
        center_loc: The location of the center of the legend canvas (in canvas
            coordinates).
        target_radius: The radius of the target (in canvas coordinates).
        transform: A Transform function which maps from canvas coordinates into
            display coordinates.

    Returns:
        A matplotlib artist containing the target symbol.
    """
    target_data = _load_target("dark")

    upper_left = center_loc - target_radius
    lower_right = center_loc + target_radius
    bbox = Bbox([upper_left, lower_right])
    bbox = TransformedBbox(bbox, transform)

    img_box = BboxImage(
        bbox, resample=True, interpolation="antialiased", interpolation_stage="rgba"
    )
    img_box.set_data(target_data)
    return img_box


def _load_target(version: str) -> npt.NDArray[np.int_]:
    """Loads a png of the target and returns it as a numpy array.

    The array is returned in RGBA channel order.

    Args:
        version: The version of the target symbol to load. Options are 'dark' or
            'light'.

    Raises:
        ValueError: if the requested version is no found.

    Returns:
        An RGBA array containing pixel data for the target symbol.
    """
    version_options = ("light", "dark")
    if version not in version_options:
        raise ValueError(
            f"Target version '{version}' not found. Options are: {version_options}"
        )

    return plt.imread(_COMPONENT_DIR / f"target-{version}.png")
