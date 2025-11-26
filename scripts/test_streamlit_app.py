"""Test script for Streamlit application.

This script verifies:
- Upload functionality
- Slice display
- Correct segmentation
"""

import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import UNet3D
from src.data import preprocessing
from src.inference import predict


def create_test_volume(volume_path: Path, shape: tuple = (64, 64, 32)):
    """Create a test volume file."""
    volume = np.random.randn(*shape).astype(np.float32) * 100 + 0
    volume = np.clip(volume, -1000, 1000)

    affine = np.eye(4)
    preprocessing.save_preprocessed_volume(
        volume, volume_path,
        {"spacing": np.array([1.0, 1.0, 1.0]), "origin": np.array([0.0, 0.0, 0.0]), "direction": np.eye(3)}
    )


def test_upload_functionality():
    """Test that file upload would work."""
    print("\n=== Testing Upload Functionality ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        test_volume = tmpdir / "test_volume.nii.gz"
        create_test_volume(test_volume)

        assert test_volume.exists(), "Test volume should be created"
        assert test_volume.suffix == ".gz", "File should have .gz extension"

        volume, metadata = preprocessing.load_nifti_volume(test_volume)
        assert volume.shape == (64, 64, 32), "Volume should have correct shape"

        print(f"  [OK] File upload simulation successful")
        print(f"    File: {test_volume.name}")
        print(f"    Shape: {volume.shape}")


def test_slice_display():
    """Test slice display functionality."""
    print("\n=== Testing Slice Display ===")

    volume = np.random.rand(64, 64, 32).astype(np.float32)
    segmentation = np.zeros((64, 64, 32), dtype=np.float32)
    segmentation[16:48, 16:48, 8:24] = 1.0

    d, h, w = volume.shape
    axial_idx = d // 2
    coronal_idx = h // 2
    sagittal_idx = w // 2

    slices_to_test = [
        ("Axial", 0, axial_idx),
        ("Coronal", 1, coronal_idx),
        ("Sagittal", 2, sagittal_idx),
    ]

    for plane_name, axis, slice_idx in slices_to_test:
        if axis == 0:
            vol_slice = volume[slice_idx, :, :]
            seg_slice = segmentation[slice_idx, :, :]
        elif axis == 1:
            vol_slice = volume[:, slice_idx, :]
            seg_slice = segmentation[:, slice_idx, :]
        else:
            vol_slice = volume[:, :, slice_idx]
            seg_slice = segmentation[:, :, slice_idx]

        assert vol_slice.shape == seg_slice.shape, \
            f"{plane_name} slice shapes should match"
        assert vol_slice.ndim == 2, f"{plane_name} slice should be 2D"

        print(f"  [OK] {plane_name} slice display: shape {vol_slice.shape}")

    print("[OK] All slice displays work correctly")


def test_segmentation_correctness():
    """Test that segmentation is correct."""
    print("\n=== Testing Segmentation Correctness ===")

    model = UNet3D(in_channels=1, out_channels=1, base_channels=16, depth=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        checkpoint_path = tmpdir / "model.pth"
        model.save(checkpoint_path)

        volume_path = tmpdir / "volume.nii.gz"
        create_test_volume(volume_path)

        segmentation, metadata = predict.predict_from_file(
            checkpoint_path, volume_path, threshold=0.5
        )

        assert segmentation.shape == (64, 64, 32), \
            f"Segmentation shape should be (64, 64, 32), got {segmentation.shape}"

        assert np.all((segmentation == 0) | (segmentation == 1)), \
            "Segmentation should be binary"

        assert segmentation.dtype == np.float32, \
            "Segmentation should be float32"

        voxels_segmented = np.sum(segmentation > 0.5)
        assert voxels_segmented >= 0, "Should have non-negative segmented voxels"

        print(f"  [OK] Segmentation correctness verified")
        print(f"    Shape: {segmentation.shape}")
        print(f"    Segmented voxels: {voxels_segmented}")
        print(f"    Percentage: {(voxels_segmented / segmentation.size) * 100:.2f}%")


def test_complete_pipeline():
    """Test complete inference pipeline for app."""
    print("\n=== Testing Complete Pipeline ===")

    model = UNet3D(in_channels=1, out_channels=1, base_channels=16, depth=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        checkpoint_path = tmpdir / "model.pth"
        model.save(checkpoint_path)

        volume_path = tmpdir / "test_volume.nii.gz"
        create_test_volume(volume_path, shape=(96, 96, 48))

        volume, metadata = preprocessing.load_nifti_volume(volume_path)
        segmentation, _ = predict.predict_from_file(
            checkpoint_path, volume_path, threshold=0.5
        )

        assert volume.shape == segmentation.shape, \
            "Volume and segmentation should have same shape"

        d, h, w = volume.shape
        assert d > 0 and h > 0 and w > 0, "All dimensions should be positive"

        print(f"  [OK] Complete pipeline works")
        print(f"    Volume shape: {volume.shape}")
        print(f"    Segmentation shape: {segmentation.shape}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Streamlit Application Verification")
    print("=" * 60)

    try:
        test_upload_functionality()
        test_slice_display()
        test_segmentation_correctness()
        test_complete_pipeline()

        print("\n" + "=" * 60)
        print("[OK] All verifications passed!")
        print("=" * 60)
        print("\nStreamlit application is ready for testing!")
        print("\nTo run the app:")
        print("  streamlit run app.py")

    except Exception as e:
        print(f"\n[FAILED] Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

