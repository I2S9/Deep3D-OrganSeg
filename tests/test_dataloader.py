"""Tests for dataloader module."""

import numpy as np
from pathlib import Path
import tempfile
import torch
import pytest
from monai.data import DataLoader

from src.data import dataloader


def create_test_nifti_pair(
    volume_path: Path,
    mask_path: Path,
    shape: tuple = (128, 128, 64),
    spacing: tuple = (1.0, 1.0, 1.0),
) -> None:
    """Create a test NIfTI volume and mask pair."""
    import nibabel as nib

    volume = np.random.randn(*shape).astype(np.float32) * 100 + 0
    volume = np.clip(volume, -1000, 1000)

    mask = np.zeros(shape, dtype=np.float32)
    mask[32:96, 32:96, 16:48] = 1.0

    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]

    vol_img = nib.Nifti1Image(volume, affine)
    mask_img = nib.Nifti1Image(mask, affine)

    nib.save(vol_img, str(volume_path))
    nib.save(mask_img, str(mask_path))


def test_organ_segmentation_dataset():
    """Test OrganSegmentationDataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        vol_path = tmpdir / "volume.nii.gz"
        mask_path = tmpdir / "mask.nii.gz"
        create_test_nifti_pair(vol_path, mask_path)

        transform = dataloader.get_validation_transforms()
        dataset = dataloader.OrganSegmentationDataset(
            volume_paths=[vol_path],
            mask_paths=[mask_path],
            transform=transform,
        )

        assert len(dataset) == 1

        sample = dataset[0]
        assert "image" in sample
        assert "label" in sample
        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["label"], torch.Tensor)


def test_training_transforms():
    """Test training transforms."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        vol_path = tmpdir / "volume.nii.gz"
        mask_path = tmpdir / "mask.nii.gz"
        create_test_nifti_pair(vol_path, mask_path)

        transform = dataloader.get_training_transforms()
        dataset = dataloader.OrganSegmentationDataset(
            volume_paths=[vol_path],
            mask_paths=[mask_path],
            transform=transform,
        )

        sample = dataset[0]
        assert sample["image"].shape[0] == 1
        assert sample["label"].shape[0] == 1
        assert sample["image"].dtype == torch.float32
        assert sample["label"].dtype == torch.float32


def test_patch_dataset():
    """Test PatchDataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        vol_path = tmpdir / "volume.nii.gz"
        mask_path = tmpdir / "mask.nii.gz"
        create_test_nifti_pair(vol_path, mask_path, shape=(256, 256, 128))

        transform = dataloader.get_patch_transforms()
        dataset = dataloader.PatchDataset(
            volume_paths=[vol_path],
            mask_paths=[mask_path],
            patch_size=(128, 128, 64),
            transform=transform,
        )

        assert len(dataset) > 0

        sample = dataset[0]
        assert "image" in sample
        assert "label" in sample
        assert sample["image"].shape[1:] == (128, 128, 64)
        assert sample["label"].shape[1:] == (128, 128, 64)


def test_dataloader_batch():
    """Test that a batch is generated correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        vol_paths = []
        mask_paths = []

        for i in range(4):
            vol_path = tmpdir / f"volume_{i}.nii.gz"
            mask_path = tmpdir / f"mask_{i}.nii.gz"
            create_test_nifti_pair(vol_path, mask_path)
            vol_paths.append(vol_path)
            mask_paths.append(mask_path)

        transform = dataloader.get_validation_transforms()
        loader = dataloader.create_dataloader(
            volume_paths=vol_paths,
            mask_paths=mask_paths,
            batch_size=2,
            transform=transform,
            num_workers=0,
            shuffle=False,
        )

        batch = next(iter(loader))
        assert "image" in batch
        assert "label" in batch
        assert batch["image"].shape[0] == 2
        assert batch["label"].shape[0] == 2


def test_batch_contains_volume_and_mask():
    """Test that each item in batch contains volume and mask."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        vol_paths = []
        mask_paths = []

        for i in range(3):
            vol_path = tmpdir / f"volume_{i}.nii.gz"
            mask_path = tmpdir / f"mask_{i}.nii.gz"
            create_test_nifti_pair(vol_path, mask_path)
            vol_paths.append(vol_path)
            mask_paths.append(mask_path)

        transform = dataloader.get_validation_transforms()
        loader = dataloader.create_dataloader(
            volume_paths=vol_paths,
            mask_paths=mask_paths,
            batch_size=1,
            transform=transform,
            num_workers=0,
            shuffle=False,
        )

        for batch in loader:
            assert "image" in batch
            assert "label" in batch
            assert batch["image"].shape[0] == 1
            assert batch["label"].shape[0] == 1
            assert batch["image"].ndim == 5
            assert batch["label"].ndim == 5


def test_batch_dimensions():
    """Test that batch dimensions are correct."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        vol_paths = []
        mask_paths = []

        for i in range(4):
            vol_path = tmpdir / f"volume_{i}.nii.gz"
            mask_path = tmpdir / f"mask_{i}.nii.gz"
            create_test_nifti_pair(vol_path, mask_path, shape=(128, 128, 64))
            vol_paths.append(vol_path)
            mask_paths.append(mask_path)

        transform = dataloader.get_validation_transforms()
        loader = dataloader.create_dataloader(
            volume_paths=vol_paths,
            mask_paths=mask_paths,
            batch_size=2,
            transform=transform,
            num_workers=0,
            shuffle=False,
        )

        batch = next(iter(loader))

        assert batch["image"].shape[0] == 2
        assert batch["image"].shape[1] == 1
        assert batch["label"].shape[0] == 2
        assert batch["label"].shape[1] == 1
        assert batch["image"].shape[2:] == batch["label"].shape[2:]


def test_patch_dataloader():
    """Test DataLoader with patches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        vol_path = tmpdir / "volume.nii.gz"
        mask_path = tmpdir / "mask.nii.gz"
        create_test_nifti_pair(vol_path, mask_path, shape=(256, 256, 128))

        transform = dataloader.get_patch_transforms()
        loader = dataloader.create_dataloader(
            volume_paths=[vol_path],
            mask_paths=[mask_path],
            batch_size=2,
            patch_size=(128, 128, 64),
            transform=transform,
            use_patches=True,
            num_workers=0,
            shuffle=False,
        )

        batch = next(iter(loader))
        assert batch["image"].shape[0] == 2
        assert batch["image"].shape[1:] == (1, 128, 128, 64)
        assert batch["label"].shape[1:] == (1, 128, 128, 64)


def test_oversampling():
    """Test that positive patches are oversampled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        vol_path = tmpdir / "volume.nii.gz"
        mask_path = tmpdir / "mask.nii.gz"
        create_test_nifti_pair(vol_path, mask_path, shape=(256, 256, 128))

        transform = dataloader.get_validation_transforms()
        dataset = dataloader.PatchDataset(
            volume_paths=[vol_path],
            mask_paths=[mask_path],
            patch_size=(128, 128, 64),
            transform=transform,
            oversample_ratio=2.0,
        )

        positive_count = sum(1 for p in dataset.patches if p["is_positive"])
        total_count = len(dataset.patches)

        assert positive_count > 0
        assert total_count > positive_count

