"""Functions for loading CT scans."""
import glob
import os
from collections import namedtuple
from pathlib import Path

import numpy as np
from nibabel.nicom import dicomwrappers

MaterialMap = namedtuple("MaterialMap", ["brain_mask_id", "skull_mask_id"])


def read_dicom_dir(dicom_path: Path) -> np.ndarray:
    """
    Read a directory of DICOM files into a 3D numpy array.

    Args:
        dicom_path: Path to the directory containing the DICOM files.

    Raises:
        OSError: Raised if no files are found in the directory.

    Returns:
        The 3D numpy array containing the DICOM data.
    """
    full_globber = os.path.join(dicom_path, "*.dcm")
    filenames = sorted(glob.glob(full_globber))
    arrays = []
    if len(filenames) == 0:
        raise OSError(f'Found no files with "{full_globber}"')
    for fname in filenames:
        dcm_w = dicomwrappers.wrapper_from_file(fname)
        arrays.append(dcm_w.get_data()[..., None])

    return np.concatenate(arrays, -1)


def get_brain_mask(masks: np.ndarray, brain_mask_id: int) -> np.ndarray:
    """
    Get the brain mask from a 3D numpy array of masks.

    Args:
        masks: The 3D numpy array of masks.
        brain_mask_id: The id of the brain mask.

    Returns:
        The 3D numpy array of the brain mask.
    """
    brain = masks.copy()
    brain[brain != brain_mask_id] = 0
    brain[brain == brain_mask_id] = 1
    return brain.astype(np.bool_)


def get_skull_mask(masks: np.ndarray, skull_mask_id: int) -> np.ndarray:
    """
    Get the skull mask from a 3D numpy array of masks.

    Args:
        masks: The 3D numpy array of masks.
        skull_mask_id: The id of the skull mask.

    Returns:
        The 3D numpy array of the skull mask.
    """
    skull = masks.copy()
    skull[skull != skull_mask_id] = 0
    skull[skull == skull_mask_id] = 1
    return skull.astype(np.bool_)


def get_masks(
    ct_path: Path, masks_map: MaterialMap, convert_2d: bool
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get the brain and skull masks from a dicom directory containing masks.

    The dicom directory should contain a 3D numpy array of masks, where each
    mask is a different integer value.

    Args:
        ct_path: the path to the dicom folder containing the segmentation masks
        masks_map: a map containing the ids of the brain and skull masks
        convert_2d: If only one slice of data should be returned

    Returns:
        The brain and skull masks.
    """
    masks = read_dicom_dir(ct_path)

    skull_mask = get_skull_mask(masks, masks_map.skull_mask_id)
    brain_mask = get_brain_mask(masks, masks_map.brain_mask_id)

    if convert_2d:
        middle = int(skull_mask.shape[2] // 2)
        skull_mask = skull_mask[:, :, middle]
        brain_mask = brain_mask[:, :, middle]

    return brain_mask, skull_mask
