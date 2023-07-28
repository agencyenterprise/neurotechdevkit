from pathlib import Path

from neurotechdevkit.scenarios import ct_loader


def test_get_masks_3d():
    """Test that the masks are loaded correctly."""
    dicom_path = Path(__file__).parent / "test_data/dicom_dir/"
    brain_mask, skull_mask = ct_loader.get_masks(
        dicom_path,
        ct_loader.MaterialMap(brain_mask_id=1, skull_mask_id=2),
        convert_2d=False,
    )
    assert brain_mask.shape == (452, 375, 5)
    assert skull_mask.shape == (452, 375, 5)


def test_get_masks_2d():
    """Test that only a single slice is returned."""
    dicom_path = Path(__file__).parent / "test_data/dicom_dir/"
    brain_mask, skull_mask = ct_loader.get_masks(
        dicom_path,
        ct_loader.MaterialMap(brain_mask_id=1, skull_mask_id=2),
        convert_2d=True,
    )
    assert brain_mask.shape == (452, 375)
    assert skull_mask.shape == (452, 375)
