from math import ceil
from pathlib import Path

from skimage import color
from skimage.io import imread

from neurotechdevkit.scenarios import ct_loader


def test_get_dicom_masks():
    """Test the get_masks function with a DICOM file."""
    dicom_path = Path(__file__).parent / "test_data/ct_scan.dcm"
    brain_mask, skull_mask = ct_loader.get_masks(dicom_path)
    assert brain_mask.shape == (451, 341)
    assert skull_mask.shape == (451, 341)


def _load_image_mask(image_path):
    image = imread(image_path)
    image = color.rgba2rgb(image)
    image = color.rgb2gray(image)
    image[image > 0] = -1
    image[image == 0] = True
    image[image == -1] = False
    return image


def test_rotate_angle_misaligned_image():
    """Test the get_rotation_angle function."""
    misaligned_image_path = Path(__file__).parent / "test_data/misaligned_image.png"
    misaligned_image = _load_image_mask(misaligned_image_path)
    angle = ct_loader.get_rotation_angle(misaligned_image)
    assert ceil(angle) == -13
