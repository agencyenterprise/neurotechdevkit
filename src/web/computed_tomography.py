"""Contains functions for loading Computed Tomography images and their masks."""
import json
import pathlib
import tempfile
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import nibabel as nib
import nibabel.orientations as ornt
import numpy as np
import pydicom
from nibabel.nifti1 import Nifti1Image
from nibabel.nifti2 import Nifti2Image
from web.messages.settings import Axis

from neurotechdevkit.materials import DEFAULT_MATERIALS


@dataclass
class ComputedTomographyImage:
    """A Computed Tomography image with its voxel spacing and material masks."""

    data: np.ndarray
    spacing: Tuple[float, float, float]  # voxel sizes in millimeter
    material_masks: Dict[str, np.ndarray]

    @property
    def spacing_in_meters(self) -> Tuple[float, ...]:
        """The voxel spacing in meters."""
        return tuple(s / 1000 for s in self.spacing)


@dataclass
class ComputedTomographyInfo:
    """Information about a Computed Tomography."""

    filename: str
    shape: Tuple[int, ...]
    spacing: str  # We need to convert the tuple to a string to be able to serialize it.


def _assign_masks(mapping: dict[str, int], data: np.ndarray) -> Dict[str, np.ndarray]:
    """Assign the masks to the materials.

    Args:
        mapping: A dictionary mapping material names to label values.
        data: The Computed Tomography image data.

    Returns:
        A dictionary of material masks with material names as keys.
    """
    mask_materials: Dict[str, np.ndarray] = {}
    for material, label_value in mapping.items():
        label_mask = data == label_value
        mask_materials[material] = label_mask.astype(np.bool_)

    return mask_materials


def _load_nii(filepath: pathlib.Path) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load a NIfTI file.

    Args:
        filepath: Path to the NIfTI file.

    Returns:
        A tuple containing the image data and the voxel spacing.
    """
    image: Union[Nifti1Image, Nifti2Image] = nib.load(filepath)  # type: ignore

    data = image.get_fdata()
    data = np.swapaxes(data, 0, 2)

    # data = np.fliplr(data)
    # data = np.flipud(data)
    return data, image.header.get_zooms()


def _load_compacted_dicom(
    filepath: pathlib.Path,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Extract and load the dicom files from a zip file.

    Args:
        filepath: Path to the zip file.

    Returns:
        A tuple containing the image data and the voxel spacing.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(tmpdir)
        dicom_files = list(pathlib.Path(tmpdir).glob("*.dcm"))
        dicom_files.sort()
        dicom_data = []
        for dicom_file in dicom_files:
            data = pydicom.dcmread(dicom_file, force=True)
            dicom_data.append(data)

        image = np.stack([dicom.pixel_array for dicom in dicom_data])
        spacing = (
            float(dicom_data[0].PixelSpacing[0]),
            float(dicom_data[0].PixelSpacing[1]),
            float(dicom_data[0].SliceThickness),
        )
        image = np.fliplr(image)
        image = np.flipud(image)
        return image, spacing


def _load_ct_file(
    ct_path: pathlib.Path,
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Load a CT image file.

    Args:
        ct_path: Path to the CT image file.

    Returns:
        A tuple containing the image data and the voxel spacing.
    """
    if ct_path.suffix == ".nii":
        return _load_nii(ct_path)
    elif ct_path.suffix == ".zip":
        return _load_compacted_dicom(ct_path)
    else:
        raise ValueError(f"Unsupported CT file type: {ct_path.suffix}")


def _load_mapping_file(mapping_path: pathlib.Path) -> dict[str, int]:
    """Load the mapping file.

    Args:
        mapping_path: Path to the mapping file.

    Returns:
        A dictionary mapping material names to label values.
    """
    with open(mapping_path, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            raise ValueError(
                f"Could not load masks file {mapping_path.name} as JSON."
            ) from None


def get_ct_image(
    ct_path: pathlib.Path,
    slice_axis: Optional[Axis],
    slice_position: Optional[float],
) -> ComputedTomographyImage:
    """Load a CT image and its masks.

    Args:
        ct_path: Path to the CT image file.
        slice_axis: The axis along which to slice the image.
        slice_position: The position along the slice axis.

    Returns:
        A ComputedTomographyImage object.
    """
    array, spacing = _load_ct_file(ct_path)
    mapping = _load_mapping_file(ct_path.with_suffix(".json"))

    if slice_axis is not None and slice_position is not None:
        if slice_axis == Axis.x:
            position = int(slice_position / spacing[0] * 1000)
            array = array[position, :, :]
        elif slice_axis == Axis.y:
            position = int(slice_position / spacing[1] * 1000)
            array = array[:, position, :]
        elif slice_axis == Axis.z:
            position = int(slice_position / spacing[2] * 1000)
            array = array[:, :, position]

    material_masks = _assign_masks(mapping, array)
    return ComputedTomographyImage(
        data=array, spacing=spacing, material_masks=material_masks
    )


def get_available_cts(cts_folder: pathlib.Path) -> List[ComputedTomographyInfo]:
    """Get the list of available Computed Tomography's in the given directory.

    Args:
        cts_folder: Path to the folder containing the Computed Tomography's.

    Returns:
        The list of available Computed Tomography's.
    """
    files: List[ComputedTomographyInfo] = []
    for file in cts_folder.glob("**/*"):
        if file.is_file() and file.suffix in [".nii", ".zip"]:
            ct, spacing = _load_ct_file(file)
            ct_info = ComputedTomographyInfo(
                filename=file.name, shape=ct.shape, spacing=str(list(spacing))
            )
            files.append(ct_info)
    return files


def validate_ct(directory: pathlib.Path, files: list[str]) -> ComputedTomographyInfo:
    """Validate the CT image and its masks.

    Args:
        directory: Path to the directory containing the CT image and its masks.
        files: List of files in the directory to be checked.

    Returns:
        The ComputedTomographyInfo object.
    """
    assert (
        len(files) == 2
    ), f"Expected exactly two files but found {len(files)}: {files}"
    ct_file, mapping_file = (
        (files[0], files[1]) if files[1].endswith(".json") else (files[1], files[0])
    )
    data, spacing = _load_ct_file(directory / ct_file)
    mapping = _load_mapping_file(directory / mapping_file)
    data_layers = set(np.unique(data).astype(np.int_))
    assert data_layers == set(mapping.values()), (
        "The labels in the CT image do not match the labels in the masks file."
        f"Expected labels: {data_layers}"
    )

    for material in mapping.keys():
        if material not in DEFAULT_MATERIALS:
            raise ValueError(f"Material {material} is not supported.")
    return ComputedTomographyInfo(
        filename=ct_file, shape=data.shape, spacing=str(list(spacing))
    )
