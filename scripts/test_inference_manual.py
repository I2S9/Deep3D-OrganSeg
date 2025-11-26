"""Manual verification script for inference pipeline.

This script verifies:
- Prediction on multiple volumes
- Output dimensions are correct
- Segmentation superposition with ground truth
"""

import sys
import tempfile
from pathlib import Path
import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import UNet3D
from src.inference import predict, visualize
from src.data import preprocessing
from src.training.metrics import dice_score
import torch


def create_test_data(num_samples: int = 3):
    """Create test volume and mask pairs."""
    data_dir = Path("data/samples")
    data_dir.mkdir(parents=True, exist_ok=True)

    volume_paths = []
    mask_paths = []

    for i in range(num_samples):
        vol_path = data_dir / f"inf_volume_{i}.nii.gz"
        mask_path = data_dir / f"inf_mask_{i}.nii.gz"

        volume = np.random.randn(64, 64, 32).astype(np.float32) * 100 + 0
        volume = np.clip(volume, -1000, 1000)

        mask = np.zeros((64, 64, 32), dtype=np.float32)
        mask[16:48, 16:48, 8:24] = 1.0

        affine = np.eye(4)
        preprocessing.save_preprocessed_volume(
            volume, vol_path,
            {"spacing": np.array([1.0, 1.0, 1.0]), "origin": np.array([0.0, 0.0, 0.0]), "direction": np.eye(3)}
        )
        preprocessing.save_preprocessed_volume(
            mask, mask_path,
            {"spacing": np.array([1.0, 1.0, 1.0]), "origin": np.array([0.0, 0.0, 0.0]), "direction": np.eye(3)}
        )

        volume_paths.append(vol_path)
        mask_paths.append(mask_path)

    return volume_paths, mask_paths


def test_prediction_multiple_volumes():
    """Test prediction on multiple volumes."""
    print("\n=== Testing Prediction on Multiple Volumes ===")

    volume_paths, mask_paths = create_test_data(num_samples=3)

    model = UNet3D(in_channels=1, out_channels=1, base_channels=16, depth=3)

    checkpoint_dir = Path("checkpoints/test_inference")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model.pth"
    model.save(checkpoint_path)

    print(f"Created {len(volume_paths)} test volumes")

    results = []
    for i, vol_path in enumerate(volume_paths):
        print(f"  Predicting volume {i+1}/{len(volume_paths)}...")
        segmentation, metadata = predict.predict_from_file(
            checkpoint_path, vol_path, threshold=0.5, apply_morphology=True
        )
        results.append(segmentation)
        print(f"    Segmentation shape: {segmentation.shape}")

    assert len(results) == len(volume_paths), "Should have predictions for all volumes"
    assert all(r.shape == (64, 64, 32) for r in results), "All predictions should have same shape"

    print(f"[OK] Successfully predicted {len(results)} volumes")


def test_output_dimensions():
    """Test that output dimensions are correct."""
    print("\n=== Testing Output Dimensions ===")

    model = UNet3D(in_channels=1, out_channels=1, base_channels=16, depth=3)

    checkpoint_dir = Path("checkpoints/test_inference")
    checkpoint_path = checkpoint_dir / "model.pth"
    model.save(checkpoint_path)

    test_shapes = [(64, 64, 32), (96, 96, 48), (128, 128, 64)]

    for shape in test_shapes:
        volume = np.random.rand(*shape).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            vol_path = tmpdir / "test_volume.nii.gz"
            preprocessing.save_preprocessed_volume(
                volume, vol_path,
                {"spacing": np.array([1.0, 1.0, 1.0]), "origin": np.array([0.0, 0.0, 0.0]), "direction": np.eye(3)}
            )

            segmentation, metadata = predict.predict_from_file(
                checkpoint_path, vol_path
            )

            assert segmentation.shape == shape, \
                f"Output shape {segmentation.shape} should match input shape {shape}"

            print(f"  [OK] Input shape {shape} -> Output shape {segmentation.shape}")

    print("[OK] All output dimensions are correct")


def test_segmentation_superposition():
    """Test segmentation superposition with ground truth."""
    print("\n=== Testing Segmentation Superposition ===")

    volume_paths, mask_paths = create_test_data(num_samples=2)

    model = UNet3D(in_channels=1, out_channels=1, base_channels=16, depth=3)

    checkpoint_dir = Path("checkpoints/test_inference")
    checkpoint_path = checkpoint_dir / "model.pth"
    model.save(checkpoint_path)

    output_dir = Path("outputs/inference_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    dice_scores = []

    for i, (vol_path, mask_path) in enumerate(zip(volume_paths, mask_paths)):
        print(f"  Processing volume {i+1}...")

        segmentation, metadata = predict.predict_from_file(
            checkpoint_path, vol_path, threshold=0.5
        )

        mask_img = nib.load(str(mask_path))
        ground_truth = mask_img.get_fdata()

        assert segmentation.shape == ground_truth.shape, \
            "Segmentation and ground truth should have same shape"

        pred_tensor = torch.from_numpy(segmentation).unsqueeze(0).unsqueeze(0).float()
        gt_tensor = torch.from_numpy(ground_truth).unsqueeze(0).unsqueeze(0).float()
        dice = dice_score(pred_tensor, gt_tensor).item()
        dice_scores.append(dice)

        print(f"    Dice score: {dice:.4f}")

        if i == 0:
            visualize.visualize_overlay(
                preprocessing.load_nifti_volume(vol_path)[0],
                segmentation,
                ground_truth=ground_truth,
                output_path=output_dir / f"superposition_{i}.png"
            )
            print(f"    Superposition visualization saved")

    avg_dice = np.mean(dice_scores)
    print(f"\n  Average Dice: {avg_dice:.4f}")
    print(f"  Dice range: [{min(dice_scores):.4f}, {max(dice_scores):.4f}]")

    assert all(0 <= d <= 1 for d in dice_scores), "Dice scores should be in [0, 1]"

    print("[OK] Segmentation superposition verified")


def test_complete_pipeline():
    """Test complete inference pipeline."""
    print("\n=== Testing Complete Inference Pipeline ===")

    volume_paths, mask_paths = create_test_data(num_samples=1)

    model = UNet3D(in_channels=1, out_channels=1, base_channels=16, depth=3)

    checkpoint_dir = Path("checkpoints/test_inference")
    checkpoint_path = checkpoint_dir / "model.pth"
    model.save(checkpoint_path)

    output_dir = Path("outputs/inference_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    vol_path = volume_paths[0]
    mask_path = mask_paths[0]

    print("  Running complete pipeline...")
    segmentation, metadata = predict.predict_from_file(
        checkpoint_path,
        vol_path,
        output_path=output_dir / "segmentation.nii.gz",
        threshold=0.5,
        apply_morphology=True,
    )

    volume, _ = preprocessing.load_nifti_volume(vol_path)
    ground_truth, _ = preprocessing.load_nifti_volume(mask_path)

    print("  Creating visualizations...")
    visualize.visualize_2d_slices(
        volume,
        segmentation,
        ground_truth=ground_truth,
        output_path=output_dir / "2d_slices.png"
    )

    visualize.visualize_overlay(
        volume,
        segmentation,
        ground_truth=ground_truth,
        output_path=output_dir / "overlay.png"
    )

    print("  [OK] Complete pipeline executed successfully")
    print(f"  Output files saved to: {output_dir}")


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Inference Pipeline Verification")
    print("=" * 60)

    try:
        test_prediction_multiple_volumes()
        test_output_dimensions()
        test_segmentation_superposition()
        test_complete_pipeline()

        print("\n" + "=" * 60)
        print("[OK] All verifications passed!")
        print("=" * 60)
        print("\nInference pipeline is fully functional!")

    except Exception as e:
        print(f"\n[FAILED] Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

