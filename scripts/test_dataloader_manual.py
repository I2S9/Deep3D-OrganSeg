"""Manual verification script for dataloader.

This script verifies:
- Batch is generated correctly
- Each item in batch contains volume + mask
- Batch dimensions are correct
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import dataloader, preprocessing


def create_test_data():
    """Create test volume and mask pairs."""
    data_dir = Path("data/samples")
    data_dir.mkdir(parents=True, exist_ok=True)

    volume_paths = []
    mask_paths = []

    for i in range(4):
        vol_path = data_dir / f"volume_{i}.nii.gz"
        mask_path = data_dir / f"mask_{i}.nii.gz"

        if not vol_path.exists():
            volume = np.random.randn(128, 128, 64).astype(np.float32) * 100 + 0
            volume = np.clip(volume, -1000, 1000)

            mask = np.zeros((128, 128, 64), dtype=np.float32)
            mask[32:96, 32:96, 16:48] = 1.0

            preprocessing.save_preprocessed_volume(
                volume,
                vol_path,
                {"spacing": np.array([1.0, 1.0, 1.0]), "origin": np.array([0.0, 0.0, 0.0]), "direction": np.eye(3)}
            )
            preprocessing.save_preprocessed_volume(
                mask,
                mask_path,
                {"spacing": np.array([1.0, 1.0, 1.0]), "origin": np.array([0.0, 0.0, 0.0]), "direction": np.eye(3)}
            )

        volume_paths.append(vol_path)
        mask_paths.append(mask_path)

    return volume_paths, mask_paths


def verify_batch_generation():
    """Verify that a batch is generated correctly."""
    print("\n=== Testing Batch Generation ===")

    volume_paths, mask_paths = create_test_data()

    transform = dataloader.get_validation_transforms()
    loader = dataloader.create_dataloader(
        volume_paths=volume_paths,
        mask_paths=mask_paths,
        batch_size=2,
        transform=transform,
        num_workers=0,
        shuffle=False,
    )

    batch = next(iter(loader))

    assert "image" in batch, "Batch must contain 'image' key"
    assert "label" in batch, "Batch must contain 'label' key"
    assert batch["image"].shape[0] == 2, f"Batch size should be 2, got {batch['image'].shape[0]}"
    assert batch["label"].shape[0] == 2, f"Batch size should be 2, got {batch['label'].shape[0]}"

    print(f"✓ Batch generated correctly")
    print(f"  Image batch shape: {batch['image'].shape}")
    print(f"  Label batch shape: {batch['label'].shape}")


def verify_batch_items():
    """Verify that each item in batch contains volume and mask."""
    print("\n=== Testing Batch Items ===")

    volume_paths, mask_paths = create_test_data()

    transform = dataloader.get_validation_transforms()
    loader = dataloader.create_dataloader(
        volume_paths=volume_paths,
        mask_paths=mask_paths,
        batch_size=1,
        transform=transform,
        num_workers=0,
        shuffle=False,
    )

    for i, batch in enumerate(loader):
        assert "image" in batch, f"Batch {i} must contain 'image' key"
        assert "label" in batch, f"Batch {i} must contain 'label' key"
        assert batch["image"].shape[0] == 1, f"Batch {i} image batch size should be 1"
        assert batch["label"].shape[0] == 1, f"Batch {i} label batch size should be 1"
        assert batch["image"].ndim == 5, f"Batch {i} image should be 5D (B, C, D, H, W)"
        assert batch["label"].ndim == 5, f"Batch {i} label should be 5D (B, C, D, H, W)"
        assert batch["image"].shape[2:] == batch["label"].shape[2:], \
            f"Batch {i} image and label spatial dimensions must match"

    print(f"✓ All batch items contain volume and mask")
    print(f"  Processed {len(volume_paths)} samples")


def verify_batch_dimensions():
    """Verify that batch dimensions are correct."""
    print("\n=== Testing Batch Dimensions ===")

    volume_paths, mask_paths = create_test_data()

    transform = dataloader.get_validation_transforms()
    loader = dataloader.create_dataloader(
        volume_paths=volume_paths,
        mask_paths=mask_paths,
        batch_size=2,
        transform=transform,
        num_workers=0,
        shuffle=False,
    )

    batch = next(iter(loader))

    expected_batch_size = 2
    expected_channels = 1

    assert batch["image"].shape[0] == expected_batch_size, \
        f"Image batch dimension 0 should be {expected_batch_size}, got {batch['image'].shape[0]}"
    assert batch["image"].shape[1] == expected_channels, \
        f"Image channel dimension should be {expected_channels}, got {batch['image'].shape[1]}"
    assert batch["label"].shape[0] == expected_batch_size, \
        f"Label batch dimension 0 should be {expected_batch_size}, got {batch['label'].shape[0]}"
    assert batch["label"].shape[1] == expected_channels, \
        f"Label channel dimension should be {expected_channels}, got {batch['label'].shape[1]}"
    assert batch["image"].shape[2:] == batch["label"].shape[2:], \
        "Image and label spatial dimensions must match"

    print(f"✓ Batch dimensions are correct")
    print(f"  Image shape: {batch['image'].shape}")
    print(f"  Label shape: {batch['label'].shape}")
    print(f"  Spatial dimensions match: {batch['image'].shape[2:] == batch['label'].shape[2:]}")


def verify_patch_dataloader():
    """Verify patch dataloader."""
    print("\n=== Testing Patch DataLoader ===")

    volume_paths, mask_paths = create_test_data()

    transform = dataloader.get_patch_transforms()
    loader = dataloader.create_dataloader(
        volume_paths=volume_paths[:1],
        mask_paths=mask_paths[:1],
        batch_size=2,
        patch_size=(64, 64, 32),
        transform=transform,
        use_patches=True,
        num_workers=0,
        shuffle=False,
    )

    batch = next(iter(loader))

    assert batch["image"].shape[0] == 2, "Patch batch size should be 2"
    assert batch["image"].shape[1:] == (1, 64, 64, 32), "Patch shape should be (1, 64, 64, 32)"
    assert batch["label"].shape[1:] == (1, 64, 64, 32), "Patch label shape should match"

    print(f"✓ Patch DataLoader works correctly")
    print(f"  Patch batch shape: {batch['image'].shape}")


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("DataLoader Verification")
    print("=" * 60)

    try:
        verify_batch_generation()
        verify_batch_items()
        verify_batch_dimensions()
        verify_patch_dataloader()

        print("\n" + "=" * 60)
        print("✓ All verifications passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()



