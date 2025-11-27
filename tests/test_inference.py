"""Tests for inference pipeline."""

import numpy as np
from pathlib import Path
import tempfile
import torch
import pytest
import nibabel as nib

from src.inference import predict, visualize
from src.models import UNet3D
from src.data import preprocessing


def create_test_volume_and_mask(volume_path: Path, mask_path: Path, shape: tuple = (64, 64, 32)):
    """Create test volume and mask files."""
    volume = np.random.randn(*shape).astype(np.float32) * 100 + 0
    volume = np.clip(volume, -1000, 1000)

    mask = np.zeros(shape, dtype=np.float32)
    mask[16:48, 16:48, 8:24] = 1.0

    affine = np.eye(4)
    affine[0, 0] = 1.0
    affine[1, 1] = 1.0
    affine[2, 2] = 1.0

    vol_img = nib.Nifti1Image(volume, affine)
    mask_img = nib.Nifti1Image(mask, affine)

    nib.save(vol_img, str(volume_path))
    nib.save(mask_img, str(mask_path))


def test_load_model():
    """Test model loading from checkpoint."""
    model = UNet3D(in_channels=1, out_channels=1, base_channels=16, depth=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_model.pth"
        model.save(checkpoint_path)

        loaded_model = predict.load_model(checkpoint_path)
        assert loaded_model.in_channels == model.in_channels
        assert loaded_model.out_channels == model.out_channels


def test_preprocess_for_inference():
    """Test volume preprocessing for inference."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        volume_path = tmpdir / "test_volume.nii.gz"
        mask_path = tmpdir / "test_mask.nii.gz"
        create_test_volume_and_mask(volume_path, mask_path)

        volume, metadata = predict.preprocess_for_inference(volume_path)

        assert volume.ndim == 3
        assert volume.dtype == np.float32
        assert volume.min() >= 0.0
        assert volume.max() <= 1.0
        assert "spacing" in metadata


def test_predict_volume():
    """Test volume prediction."""
    model = UNet3D(in_channels=1, out_channels=1, base_channels=16, depth=3)
    model.eval()

    volume = np.random.rand(32, 32, 16).astype(np.float32)

    prediction = predict.predict_volume(model, volume)

    assert prediction.shape == volume.shape
    assert prediction.dtype == np.float32


def test_postprocess_segmentation():
    """Test post-processing of segmentation."""
    prediction = np.random.rand(32, 32, 16).astype(np.float32)

    segmentation = predict.postprocess_segmentation(
        prediction, threshold=0.5, apply_morphology=True
    )

    assert segmentation.shape == prediction.shape
    assert segmentation.dtype == np.float32
    assert np.all((segmentation == 0) | (segmentation == 1))


def test_predict_from_file():
    """Test complete prediction pipeline from file."""
    model = UNet3D(in_channels=1, out_channels=1, base_channels=16, depth=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        checkpoint_path = tmpdir / "model.pth"
        model.save(checkpoint_path)

        volume_path = tmpdir / "volume.nii.gz"
        mask_path = tmpdir / "test_mask.nii.gz"
        create_test_volume_and_mask(volume_path, mask_path)

        output_path = tmpdir / "segmentation.nii.gz"

        segmentation, metadata = predict.predict_from_file(
            checkpoint_path, volume_path, output_path
        )

        assert segmentation.shape == (64, 64, 32)
        assert output_path.exists()


def test_predict_batch():
    """Test batch prediction."""
    model = UNet3D(in_channels=1, out_channels=1, base_channels=16, depth=3)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        checkpoint_path = tmpdir / "model.pth"
        model.save(checkpoint_path)

        volume_paths = []
        for i in range(3):
            vol_path = tmpdir / f"volume_{i}.nii.gz"
            mask_path = tmpdir / f"mask_{i}.nii.gz"
            create_test_volume_and_mask(vol_path, mask_path)
            volume_paths.append(vol_path)

        results = predict.predict_batch(
            checkpoint_path, volume_paths, output_dir=tmpdir / "outputs"
        )

        assert len(results) == 3
        assert all(r.shape == (64, 64, 32) for r in results)


def test_visualize_2d_slices():
    """Test 2D slice visualization."""
    volume = np.random.rand(64, 64, 32).astype(np.float32)
    segmentation = np.zeros((64, 64, 32), dtype=np.float32)
    segmentation[16:48, 16:48, 8:24] = 1.0

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "visualization.png"
        visualize.visualize_2d_slices(volume, segmentation, output_path=output_path)

        assert output_path.exists()


def test_visualize_overlay():
    """Test overlay visualization."""
    volume = np.random.rand(64, 64, 32).astype(np.float32)
    segmentation = np.zeros((64, 64, 32), dtype=np.float32)
    segmentation[16:48, 16:48, 8:24] = 1.0

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "overlay.png"
        visualize.visualize_overlay(volume, segmentation, output_path=output_path)

        assert output_path.exists()


def test_output_dimensions():
    """Test that output dimensions are correct."""
    model = UNet3D(in_channels=1, out_channels=1, base_channels=16, depth=3)
    model.eval()

    input_shape = (64, 64, 32)
    volume = np.random.rand(*input_shape).astype(np.float32)

    prediction = predict.predict_volume(model, volume)
    segmentation = predict.postprocess_segmentation(prediction)

    assert prediction.shape == input_shape
    assert segmentation.shape == input_shape


def test_segmentation_superposition():
    """Test segmentation superposition with ground truth."""
    volume = np.random.rand(64, 64, 32).astype(np.float32)
    ground_truth = np.zeros((64, 64, 32), dtype=np.float32)
    ground_truth[16:48, 16:48, 8:24] = 1.0

    segmentation = ground_truth.copy()
    segmentation[20:44, 20:44, 10:22] = 1.0

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "superposition.png"
        visualize.visualize_overlay(
            volume, segmentation, ground_truth=ground_truth, output_path=output_path
        )

        assert output_path.exists()

        from src.training.metrics import dice_score

        pred_tensor = torch.from_numpy(segmentation).unsqueeze(0).unsqueeze(0).float()
        gt_tensor = torch.from_numpy(ground_truth).unsqueeze(0).unsqueeze(0).float()
        dice = dice_score(pred_tensor, gt_tensor).item()

        assert dice > 0.0
        assert dice <= 1.0



