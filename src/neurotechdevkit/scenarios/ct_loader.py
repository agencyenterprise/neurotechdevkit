"""Functions for loading and preprocessing CT scans."""
from pathlib import Path

import numpy as np
import nibabel as nib
from nibabel.nicom import dicomwrappers
from scipy import ndimage
from skimage import morphology
from skimage.measure import find_contours
from sklearn.decomposition import PCA

BONE_HU_MIN = 400


def load_dicom(file_path: Path) -> np.ndarray:
    """Load a DICOM file and return a numpy array with the image data."""
    img = dicomwrappers.wrapper_from_file(file_path)
    hu_image = img.get_data()
    return hu_image


def load_nifti(file_path: Path) -> np.ndarray:
    """Load a NIfTI file and return a numpy array with the image data."""
    full_img = nib.load(file_path)
    middle = int(full_img.get_fdata().shape[0]/2)
    image = full_img.get_fdata()[middle]
    return image


def get_skull_mask(image: np.ndarray) -> np.ndarray:
    """Get a mask of the skull from a CT scan."""
    image = image.copy()
    image[image < BONE_HU_MIN] = 0
    image[image > 0] = 1
    return image


def get_brain_mask(image: np.ndarray) -> np.ndarray:
    """Get a mask of the brain from a CT scan."""
    image = image.copy()
    image[image > BONE_HU_MIN] = 0
    image[image < 0] = 0
    image[image > 0] = 1
    return image


def get_rotation_angle(mask: np.ndarray) -> float:
    """Get the rotation angle of the skull from a CT scan."""
    c, r = np.where(mask != 0)
    X = np.array([c, r]).T
    pca = PCA(n_components=2)
    pca.fit_transform(X)
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])
    angle = np.degrees(angle)
    return -angle


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image by a given angle."""
    rotated_img = ndimage.rotate(image, angle, reshape=False)
    return rotated_img


def crop(
    image: np.ndarray, min_y: int, min_x: int, max_y: int, max_x: int
) -> np.ndarray:
    """Crop an image to a given bounding box."""
    cropped_image = image[min_y:max_y, min_x:max_x]
    return cropped_image


def get_contours(image: np.ndarray) -> tuple[int, int, int, int]:
    """Get the bounding box of the skull from a CT scan."""
    contours = find_contours(image)

    # find the biggest contour (c) by the area
    largest_contour = max(contours, key=len)
    min_y, min_x = np.min(largest_contour, axis=0)
    max_y, max_x = np.max(largest_contour, axis=0)

    # Convert bounding box coordinates to integers
    min_y, min_x, max_y, max_x = int(min_y), int(min_x), int(max_y), int(max_x)
    return min_y, min_x, max_y, max_x


def to_mask(image: np.ndarray) -> np.ndarray:
    """Convert an image to a binary mask."""
    segmentation = morphology.dilation(image, np.ones((1, 1)))
    labels, label_nb = ndimage.label(segmentation)

    label_count = np.bincount(labels.ravel().astype(np.int32))
    label_count[0] = 0

    mask = labels == label_count.argmax()
    mask = morphology.dilation(mask, np.ones((1, 1)))
    mask = morphology.dilation(mask, np.ones((3, 3)))
    return mask


def add_pad(
    image: np.ndarray, new_height: int = 451, new_width: int = 341
) -> np.ndarray:
    """Add padding to an image."""
    height, width = image.shape

    final_image = np.zeros((new_height, new_width))

    pad_left = int((new_width - width) // 2)
    pad_top = int((new_height - height) // 2)
    final_image[pad_top : pad_top + height, pad_left : pad_left + width] = image

    return final_image


def get_masks(ct_path: Path, convert_2d=False) -> tuple[np.ndarray, np.ndarray]:
    """Get the brain and skull masks from a CT scan."""
    assert convert_2d is True, "Only 2d is supported"
    if ct_path.suffix == ".dcm":
        image = load_dicom(ct_path)
    elif ct_path.suffix in [".nii", ".gz"]:
        image = load_nifti(ct_path)
    else:
        raise Exception("Unknown file format")

    skull_image = get_skull_mask(image)
    rotation_angle = get_rotation_angle(skull_image)

    skull_image = skull_image.astype(np.uint8)
    skull_image = rotate_image(skull_image, rotation_angle)
    contours = get_contours(skull_image)
    skull_image = crop(skull_image, *contours)
    skull_image = add_pad(skull_image)
    skull_mask = to_mask(skull_image)

    brain_image = get_brain_mask(image)
    brain_image = brain_image.astype(np.uint8)
    brain_image = rotate_image(brain_image, rotation_angle)
    brain_image = crop(brain_image, *contours)
    brain_image = add_pad(brain_image)
    brain_mask = to_mask(brain_image)

    return brain_mask, skull_mask
