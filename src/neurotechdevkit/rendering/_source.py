import pathlib
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy
from matplotlib.image import BboxImage
from matplotlib.transforms import Bbox, Transform, TransformedBbox

from neurotechdevkit.sources import PhasedArrayMixin, Source

_COMPONENT_DIR = pathlib.Path(__file__).parent / "components"

_ANGLE_OPTION_FILENAMES = {
    2: "Angle=2°.png",
    5: "Angle=5°.png",
    10: "Angle=10°.png",
    20: "Angle=20°.png",
    40: "Angle=40°.png",
    60: "Angle=60°.png",
    90: "Angle=90°.png",
    120: "Angle=120°.png",
    150: "Angle=150°.png",
    180: "Angle=180°.png",
}


class SourceDrawingParams(NamedTuple):
    """A container for the parameters needed to draw a source.

    Args:
        position: the 2D coordinates (in meters) of the source.
        direction: a 2D vector indicating the direction the source is pointing.
        aperture: the aperture (in meters) of the source.
        focal_length: the focal length (in meters) of the source.
        source_is_flat: whether the source should be rendered as a flat object.
    """

    position: npt.NDArray[np.float_]
    direction: npt.NDArray[np.float_]
    aperture: float
    focal_length: float
    source_is_flat: bool


def create_source_drawing_artist(
    source_params: SourceDrawingParams, transform: Transform
) -> plt.Artist:
    """Create a matplotlib artist for a source rendered inside a scenario.

    Note that the source coordinates are in scenario coordinates, and not plot
    coordinates.

    Args:
        source_params: The SourceDrawingParams that describe the source.
        transform: A Transform function which maps from plot data coordinates into
            display coordinates.

    Returns:
        A matplotlib artist containing the rendered source.
    """
    raw_img = _load_most_similar_source_image(
        source_params.aperture, source_params.focal_length, source_params.source_is_flat
    )
    transformed_img = _translate_and_rotate(raw_img, source_params.direction)
    # now the center of the img corresponds to the source position, and it is
    # rotated in the correct direction

    if source_params.source_is_flat:
        data_width = (
            source_params.aperture * transformed_img.shape[1] / raw_img.shape[1]
        )
    else:
        data_width = source_params.focal_length * transformed_img.shape[0] / 360
    hw = data_width / 2
    position = np.flip(source_params.position)  # into plot data coordinates
    upper_left = position - hw
    lower_right = position + hw

    bbox = Bbox([upper_left, lower_right])
    bbox = TransformedBbox(bbox, transform)

    img_box = BboxImage(
        bbox, resample=True, interpolation="antialiased", interpolation_stage="rgba"
    )
    img_box.set_data(transformed_img)
    return img_box


def create_source_legend_artist(
    loc: npt.NDArray[np.float_], width: float, transform: Transform
) -> plt.Artist:
    """Create a matplotlib artist for a source icon used in a legend.

    Note that `loc` and `width` should  be in legend canvas coordinates. Ideally, they
    come from the offsetbox passed to a legend handler.

    Args:
        loc: The location (in legend canvas coordinates) of the center of the canvas for
            this icon.
        width: The width (in legend canvas coordinates) of the canvas for this icon.
        transform: A Transformation object that will translate from the legend handlebox
            to display coordinates. This should come from the handlebox passed to the
            legend handler.

    Returns:
        A matplotlib artist containing the source icon for the legend.
    """
    raw_img = _load_most_similar_source_image(0.7, 1.0, False)  # 40°

    # we don't need to do any rotation here, just crop it appropriately
    cropped_img = raw_img[260:]

    hw = width / 2
    hh = hw * cropped_img.shape[0] / cropped_img.shape[1]
    delta = np.array([hw, hh])
    upper_left = loc - delta
    lower_right = loc + delta

    bbox = Bbox([upper_left, lower_right])
    bbox = TransformedBbox(bbox, transform)

    img_box = BboxImage(
        bbox, resample=True, interpolation="antialiased", interpolation_stage="rgba"
    )
    img_box.set_data(cropped_img)
    return img_box


def _load_most_similar_source_image(
    aperture: float,
    focal_length: float,
    source_is_flat: bool,
) -> npt.NDArray[np.float_]:
    """Loads the source image which best matches the specified aperture and focus.

    Args:
        aperture: The aperture of the source (in meters).
        focal_length: The focal length of the source (in meters). For planar sources,
            this value should equal np.inf.
        source_is_flat: A boolean indicating that the source should be represented flat.

    Returns:
        A numpy array containing the source image data.
    """
    src_file = _select_image_file(aperture, focal_length, source_is_flat)
    src_img = plt.imread(src_file)

    if src_file.name == "Angle=180°.png":
        # this image file has an inconsistent size
        src_img = src_img[28:]

    if src_file.name == "Angle=Flat.png":
        # set the source position at X=360 pixels
        new_img = np.zeros((376, src_img.shape[1], 4), dtype=src_img.dtype)
        new_img[new_img.shape[0] - src_img.shape[0] :] = src_img
        src_img = new_img

    return src_img


def source_should_be_flat(source: Source) -> bool:
    """Determines if a source is flat based on the source type and focal length.

    A source should be represented flat if the source is unfocused (as determined by
    `focal_length`) or the source is a phased array


    Args:
        source: the source instance.

    Returns:
        True if the source is a flat transducer, and False otherwise.
    """
    is_phased_array = isinstance(source, PhasedArrayMixin)
    return is_phased_array or np.isinf(source.focal_length)


def _select_image_file(
    aperture: float, focal_length: float, source_is_flat: bool
) -> pathlib.Path:
    """Selects the image file to load based on aperture and focal length.

    For focused transducers, we select the image file based on the angle subtended. For
    planar transducers, there is only one option.

    Args:
        aperture: The aperture of the source (in meters).
        focal_length: The focal length of the source (in meters). For planar sources,
            this value should equal np.inf.
        source_is_flat: A boolean indicating if a source should be represented flat.

    Returns:
        The path to the file containing the selected image.
    """
    if source_is_flat:
        return _COMPONENT_DIR / "Angle=Flat.png"

    angle_subtended = 2 * np.arcsin(aperture / (2 * focal_length)) * 180 / np.pi

    options = list(_ANGLE_OPTION_FILENAMES.keys())
    closest_angle = _choose_nearest(angle_subtended, options)
    return _COMPONENT_DIR / _ANGLE_OPTION_FILENAMES[closest_angle]


def _choose_nearest(desired: float, options: list[int]) -> int:
    """Returns the closest option to a desired value from a list of options.

    Args:
        desired: The value for which we want the closest match.
        options: The list of options to chose from.

    Returns:
        The selected closest value.
    """
    idx = np.argmin([np.abs(opt - desired) for opt in options])
    return options[idx]


def _translate_and_rotate(
    raw_img: npt.NDArray[np.float_], direction: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """Applies the required transformations to a source image.

    This function encapsulates three operations:
    1. Pad the image array so that a rotation will not cut off parts of the image.
    2. Translate the source so that the point of the transducer indicated by its
        position (the point which bisects the focused or planar source) is located at
        the center of the image.
    3. Rotates the source around the center of the image to point in the desired
        direction.

    Args:
        raw_img: The array containing the raw image data loaded from disk.
        direction: A vector indicating the direction that the transducer is aimed.

    Returns:
        An array containing the transformed image data.
    """
    rotation = (np.pi + np.arctan2(direction[1], direction[0])) * 180 / np.pi

    padded_width = 1200  # should be large enough for any of the source icons

    map_from = np.array([360, raw_img.shape[1] // 2])
    map_to = np.array([padded_width // 2, padded_width // 2])
    trans = map_to - map_from

    padded_img = np.zeros([padded_width, padded_width, 4], dtype=float)

    new_ul = np.array([0, 0]) + trans
    new_lr = np.array(raw_img.shape[:-1]) + trans

    padded_img[new_ul[0] : new_lr[0], new_ul[1] : new_lr[1]] = raw_img
    transformed_img = scipy.ndimage.rotate(
        padded_img, angle=rotation, reshape=False, mode="constant", cval=0
    )

    return np.clip(transformed_img, 0.0, 1.0)
