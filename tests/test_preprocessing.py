"""Tests for preprocessing module."""

import numpy as np
from pathlib import Path
import tempfile
import shutil
import pytest
from src.data import preprocessing


def create_dummy_nifti(shape: tuple, spacing: tuple, output_path: Path) -> None:
    """Create a dummy NIfTI file for testing."""
    import nibabel as nib
    
    volume = np.random.randn(*shape).astype(np.float32) * 100 + 0
    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    
    nii_img = nib.Nifti1Image(volume, affine)
    nib.save(nii_img, str(output_path))


def test_load_nifti_volume():
    """Test loading NIfTI volume."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_path = Path(tmpdir) / "test.nii.gz"
        create_dummy_nifti((64, 64, 32), (2.0, 2.0, 3.0), nifti_path)
        
        volume, metadata = preprocessing.load_nifti_volume(nifti_path)
        
        assert volume.shape == (64, 64, 32)
        assert volume.dtype == np.float32
        assert "spacing" in metadata
        assert "origin" in metadata
        assert "direction" in metadata


def test_resample_isotropic():
    """Test isotropic resampling."""
    volume = np.random.randn(32, 32, 16).astype(np.float32)
    spacing = np.array([2.0, 2.0, 3.0])
    target_spacing = (1.0, 1.0, 1.0)
    
    resampled, new_spacing = preprocessing.resample_isotropic(volume, spacing, target_spacing)
    
    assert resampled.dtype == np.float32
    assert np.allclose(new_spacing, target_spacing, atol=0.1)
    assert resampled.shape[0] > volume.shape[0]
    assert resampled.shape[1] > volume.shape[1]
    assert resampled.shape[2] > volume.shape[2]


def test_auto_crop():
    """Test automatic cropping."""
    volume = np.zeros((100, 100, 50), dtype=np.float32)
    volume[30:70, 30:70, 10:40] = 100.0
    
    cropped, crop_slices = preprocessing.auto_crop(volume, threshold=50.0)
    
    assert cropped.shape[0] <= volume.shape[0]
    assert cropped.shape[1] <= volume.shape[1]
    assert cropped.shape[2] <= volume.shape[2]
    assert len(crop_slices) == 3


def test_normalize_hu():
    """Test HU normalization."""
    volume = np.array([-1500, -500, 0, 500, 1500], dtype=np.float32)
    
    normalized = preprocessing.normalize_hu(volume, hu_min=-1000.0, hu_max=1000.0)
    
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0
    assert normalized.dtype == np.float32


def test_preprocess_volume():
    """Test complete preprocessing pipeline."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / "test.nii.gz"
        output_path = Path(tmpdir) / "preprocessed.nii.gz"
        
        create_dummy_nifti((64, 64, 32), (2.0, 2.0, 3.0), input_path)
        
        volume, metadata = preprocessing.preprocess_volume(
            input_path,
            output_path,
            target_spacing=(1.0, 1.0, 1.0),
            auto_crop_enabled=True
        )
        
        assert volume.dtype == np.float32
        assert volume.min() >= 0.0
        assert volume.max() <= 1.0
        assert "spacing" in metadata
        assert "original_shape" in metadata
        assert output_path.exists()


def test_memory_efficient_loading():
    """Test that large volumes can be loaded without memory issues."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_path = Path(tmpdir) / "large.nii.gz"
        create_dummy_nifti((256, 256, 128), (1.0, 1.0, 1.0), nifti_path)
        
        volume, metadata = preprocessing.load_nifti_volume(nifti_path)
        
        assert volume.shape == (256, 256, 128)
        assert volume.nbytes < 1e9


def test_save_preprocessed_volume():
    """Test saving preprocessed volume."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "saved.nii.gz"
        volume = np.random.rand(32, 32, 16).astype(np.float32)
        metadata = {
            "spacing": np.array([1.0, 1.0, 1.0]),
            "origin": np.array([0.0, 0.0, 0.0]),
            "direction": np.eye(3)
        }
        
        preprocessing.save_preprocessed_volume(volume, output_path, metadata)
        
        assert output_path.exists()
        
        loaded_volume, loaded_metadata = preprocessing.load_nifti_volume(output_path)
        assert np.allclose(volume, loaded_volume, atol=1e-5)

