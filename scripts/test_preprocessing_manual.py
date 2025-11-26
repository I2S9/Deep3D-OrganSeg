"""Manual verification script for preprocessing pipeline.

This script tests the preprocessing pipeline with sample data and verifies:
- Volume sizes are consistent
- Volumes can be read and displayed (slice)
- Pipeline supports large volumes without memory issues
"""

import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import preprocessing


def create_test_nifti(output_path: Path, shape: tuple = (128, 128, 64), spacing: tuple = (2.0, 2.0, 3.0)):
    """Create a test NIfTI file with realistic CT-like values."""
    import nibabel as nib
    
    volume = np.random.randn(*shape).astype(np.float32) * 200 - 500
    volume = np.clip(volume, -1000, 1000)
    
    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    
    nii_img = nib.Nifti1Image(volume, affine)
    nib.save(nii_img, str(output_path))
    print(f"Created test NIfTI: {output_path}")
    print(f"  Shape: {shape}, Spacing: {spacing} mm")


def verify_volume_sizes():
    """Verify that volume sizes are consistent after preprocessing."""
    print("\n=== Testing Volume Size Consistency ===")
    
    test_dir = Path("data/samples")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = test_dir / "test_volume.nii.gz"
    create_test_nifti(test_file, shape=(128, 128, 64), spacing=(2.0, 2.0, 3.0))
    
    volume, metadata = preprocessing.load_nifti_volume(test_file)
    print(f"Original volume shape: {volume.shape}")
    print(f"Original spacing: {metadata['spacing']} mm")
    
    preprocessed, new_metadata = preprocessing.preprocess_volume(
        test_file,
        target_spacing=(1.0, 1.0, 1.0),
        auto_crop_enabled=True
    )
    
    print(f"Preprocessed volume shape: {preprocessed.shape}")
    print(f"New spacing: {new_metadata['spacing']} mm")
    print(f"Original shape (from metadata): {new_metadata['original_shape']}")
    print(f"Original spacing (from metadata): {new_metadata['original_spacing']} mm")
    
    assert preprocessed.ndim == 3, "Volume must be 3D"
    assert all(s > 0 for s in preprocessed.shape), "All dimensions must be positive"
    print("✓ Volume sizes are consistent")


def verify_volume_reading_and_display():
    """Verify that volumes can be read and displayed."""
    print("\n=== Testing Volume Reading and Display ===")
    
    test_dir = Path("data/samples")
    test_file = test_dir / "test_volume.nii.gz"
    
    if not test_file.exists():
        create_test_nifti(test_file)
    
    volume, metadata = preprocessing.load_nifti_volume(test_file)
    print(f"Loaded volume shape: {volume.shape}")
    print(f"Volume dtype: {volume.dtype}")
    print(f"Volume range: [{volume.min():.2f}, {volume.max():.2f}]")
    
    mid_slice = volume[:, :, volume.shape[2] // 2]
    print(f"Middle slice shape: {mid_slice.shape}")
    print(f"Middle slice range: [{mid_slice.min():.2f}, {mid_slice.max():.2f}]")
    
    preprocessed, _ = preprocessing.preprocess_volume(test_file, auto_crop_enabled=True)
    preprocessed_slice = preprocessed[:, :, preprocessed.shape[2] // 2]
    print(f"Preprocessed middle slice shape: {preprocessed_slice.shape}")
    print(f"Preprocessed slice range: [{preprocessed_slice.min():.2f}, {preprocessed_slice.max():.2f}]")
    
    assert preprocessed_slice.min() >= 0.0 and preprocessed_slice.max() <= 1.0, \
        "Preprocessed values should be normalized to [0, 1]"
    
    print("✓ Volumes can be read and displayed")


def verify_memory_efficiency():
    """Verify that the pipeline supports large volumes without memory issues."""
    print("\n=== Testing Memory Efficiency ===")
    
    test_dir = Path("data/samples")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    large_file = test_dir / "large_volume.nii.gz"
    large_shape = (256, 256, 128)
    create_test_nifti(large_file, shape=large_shape, spacing=(1.5, 1.5, 2.0))
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024
    
    volume, metadata = preprocessing.load_nifti_volume(large_file)
    mem_after_load = process.memory_info().rss / 1024 / 1024
    
    preprocessed, _ = preprocessing.preprocess_volume(
        large_file,
        target_spacing=(1.0, 1.0, 1.0),
        auto_crop_enabled=True
    )
    mem_after_process = process.memory_info().rss / 1024 / 1024
    
    print(f"Memory before: {mem_before:.2f} MB")
    print(f"Memory after load: {mem_after_load:.2f} MB (+{mem_after_load - mem_before:.2f} MB)")
    print(f"Memory after process: {mem_after_process:.2f} MB (+{mem_after_process - mem_before:.2f} MB)")
    print(f"Large volume shape: {volume.shape}")
    print(f"Preprocessed shape: {preprocessed.shape}")
    
    memory_increase = mem_after_process - mem_before
    assert memory_increase < 2000, f"Memory increase ({memory_increase:.2f} MB) should be reasonable"
    
    print("✓ Pipeline handles large volumes efficiently")


def verify_save_preprocessed():
    """Verify that preprocessed volumes can be saved."""
    print("\n=== Testing Save Preprocessed Volume ===")
    
    test_dir = Path("data/samples")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    test_file = test_dir / "test_volume.nii.gz"
    if not test_file.exists():
        create_test_nifti(test_file)
    
    output_file = processed_dir / "test_preprocessed.nii.gz"
    
    volume, metadata = preprocessing.preprocess_volume(
        test_file,
        output_path=output_file,
        target_spacing=(1.0, 1.0, 1.0)
    )
    
    assert output_file.exists(), "Output file should be created"
    
    loaded_volume, loaded_metadata = preprocessing.load_nifti_volume(output_file)
    assert np.allclose(volume, loaded_volume, atol=1e-5), "Saved and loaded volumes should match"
    
    print(f"✓ Preprocessed volume saved to {output_file}")
    print(f"  Saved volume shape: {loaded_volume.shape}")
    print(f"  Saved volume spacing: {loaded_metadata['spacing']} mm")


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Preprocessing Pipeline Verification")
    print("=" * 60)
    
    try:
        verify_volume_sizes()
        verify_volume_reading_and_display()
        verify_memory_efficiency()
        verify_save_preprocessed()
        
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


