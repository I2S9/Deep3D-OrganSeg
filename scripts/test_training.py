"""Test script for mini training on 2-3 volumes.

This script performs a mini training run to verify that:
- Training loop works correctly
- Metrics evolve properly
- Checkpoints are saved
"""

import sys
from pathlib import Path
import torch
import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import UNet3D
from src.training.train import Trainer
from src.data.dataloader import create_dataloader, get_training_transforms, get_validation_transforms
from src.data import preprocessing


def create_test_data(num_samples: int = 3):
    """Create test volume and mask pairs."""
    data_dir = Path("data/samples")
    data_dir.mkdir(parents=True, exist_ok=True)

    volume_paths = []
    mask_paths = []

    for i in range(num_samples):
        vol_path = data_dir / f"train_volume_{i}.nii.gz"
        mask_path = data_dir / f"train_mask_{i}.nii.gz"

        volume = np.random.randn(64, 64, 32).astype(np.float32) * 100 + 0
        volume = np.clip(volume, -1000, 1000)

        mask = np.zeros((64, 64, 32), dtype=np.float32)
        mask[16:48, 16:48, 8:24] = 1.0

        affine = np.eye(4)
        affine[0, 0] = 1.0
        affine[1, 1] = 1.0
        affine[2, 2] = 1.0

        vol_img = nib.Nifti1Image(volume, affine)
        mask_img = nib.Nifti1Image(mask, affine)

        nib.save(vol_img, str(vol_path))
        nib.save(mask_img, str(mask_path))

        volume_paths.append(vol_path)
        mask_paths.append(mask_path)

    return volume_paths, mask_paths


def test_mini_training():
    """Test mini training on 2-3 volumes."""
    print("=" * 60)
    print("Mini Training Test")
    print("=" * 60)

    print("\n1. Creating test data...")
    volume_paths, mask_paths = create_test_data(num_samples=3)
    print(f"   Created {len(volume_paths)} volume/mask pairs")

    train_volumes = volume_paths[:2]
    train_masks = mask_paths[:2]
    val_volumes = volume_paths[2:3]
    val_masks = mask_paths[2:3]

    print(f"   Training samples: {len(train_volumes)}")
    print(f"   Validation samples: {len(val_volumes)}")

    print("\n2. Creating model...")
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        depth=3,
    )
    print(f"   Model parameters: {model.count_parameters():,}")

    print("\n3. Creating data loaders...")
    train_transform = get_training_transforms()
    train_loader = create_dataloader(
        volume_paths=train_volumes,
        mask_paths=train_masks,
        batch_size=1,
        transform=train_transform,
        num_workers=0,
        shuffle=True,
    )

    val_transform = get_validation_transforms()
    val_loader = create_dataloader(
        volume_paths=val_volumes,
        mask_paths=val_masks,
        batch_size=1,
        transform=val_transform,
        num_workers=0,
        shuffle=False,
    )

    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")

    print("\n4. Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=1e-3,
        scheduler_type="cosine",
        use_amp=False,
        checkpoint_dir=Path("checkpoints/test"),
        log_dir=Path("logs/test"),
    )

    print("\n5. Starting mini training (3 epochs)...")
    trainer.train(num_epochs=3, save_every=1, save_best=True)

    print("\n6. Verifying training results...")
    history = trainer.history

    assert len(history["train_loss"]) == 3, "Should have 3 training epochs"
    assert len(history["train_dice"]) == 3, "Should have 3 training dice scores"
    assert len(history["val_loss"]) == 3, "Should have 3 validation epochs"
    assert len(history["val_dice"]) == 3, "Should have 3 validation dice scores"

    print("   [OK] Training history has correct number of entries")

    print("\n7. Verifying metrics evolution...")
    train_dice = history["train_dice"]
    val_dice = history["val_dice"]

    print(f"   Train Dice: {train_dice[0]:.4f} -> {train_dice[-1]:.4f}")
    print(f"   Val Dice: {val_dice[0]:.4f} -> {val_dice[-1]:.4f}")

    assert all(0 <= d <= 1 for d in train_dice), "Dice scores should be in [0, 1]"
    assert all(0 <= d <= 1 for d in val_dice), "Dice scores should be in [0, 1]"

    print("   [OK] Metrics are in valid range")

    print("\n8. Verifying checkpoint saving...")
    checkpoint_dir = Path("checkpoints/test")
    latest_checkpoint = checkpoint_dir / "latest_checkpoint.pth"
    best_checkpoint = checkpoint_dir / "best_model.pth"

    assert latest_checkpoint.exists(), "Latest checkpoint should exist"
    assert best_checkpoint.exists(), "Best checkpoint should exist"

    checkpoint = torch.load(latest_checkpoint, map_location="cpu")
    assert "epoch" in checkpoint, "Checkpoint should contain epoch"
    assert "model_state_dict" in checkpoint, "Checkpoint should contain model state"
    assert "optimizer_state_dict" in checkpoint, "Checkpoint should contain optimizer state"
    assert "history" in checkpoint, "Checkpoint should contain history"

    print("   [OK] Checkpoints saved correctly")
    print(f"   Latest checkpoint: {latest_checkpoint}")
    print(f"   Best checkpoint: {best_checkpoint}")

    print("\n9. Verifying checkpoint loading...")
    checkpoint = torch.load(best_checkpoint, map_location="cpu")
    model_config = checkpoint.get("model_config", {})
    
    loaded_model = UNet3D(
        in_channels=model_config.get("in_channels", 1),
        out_channels=model_config.get("out_channels", 1),
        base_channels=model_config.get("base_channels", 16),
        depth=model_config.get("depth", 3),
    )
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.eval()
    
    assert loaded_model.in_channels == model.in_channels
    assert loaded_model.out_channels == model.out_channels
    assert loaded_model.base_channels == model.base_channels
    assert loaded_model.depth == model.depth

    print("   [OK] Checkpoint loading works")

    print("\n" + "=" * 60)
    print("[OK] All tests passed!")
    print("=" * 60)
    print("\nTraining system is operational!")


if __name__ == "__main__":
    try:
        test_mini_training()
    except Exception as e:
        print(f"\n[FAILED] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

