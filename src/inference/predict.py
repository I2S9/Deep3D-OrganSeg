"""Inference pipeline for 3D organ segmentation.

This module provides functions to load a trained model, preprocess volumes,
perform 3D segmentation, and apply post-processing.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import torch
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import binary_closing, binary_opening, binary_erosion, binary_dilation

from src.models import UNet3D
from src.data import preprocessing
from src.data.dataloader import get_validation_transforms


def load_model(
    checkpoint_path: Path,
    device: Optional[torch.device] = None,
) -> UNet3D:
    """Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        device: Device to load model on (default: cuda if available, else cpu).

    Returns:
        Loaded UNet3D model in evaluation mode.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = checkpoint.get("model_config", {})

    model_config_filtered = {
        k: v
        for k, v in model_config.items()
        if k in ["in_channels", "out_channels", "base_channels", "depth", "bilinear"]
    }

    if not model_config_filtered:
        model_config_filtered = {
            "in_channels": checkpoint.get("in_channels", 1),
            "out_channels": checkpoint.get("out_channels", 1),
            "base_channels": checkpoint.get("base_channels", 64),
            "depth": checkpoint.get("depth", 4),
        }

    model = UNet3D(**model_config_filtered)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    model.to(device)

    return model


def preprocess_for_inference(
    volume_path: Path,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    hu_min: float = -1000.0,
    hu_max: float = 1000.0,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Preprocess volume for inference.

    Args:
        volume_path: Path to input volume (DICOM dir or NIfTI file).
        target_spacing: Target isotropic spacing in mm (default: 1x1x1).
        hu_min: Minimum HU value for clamping (default: -1000).
        hu_max: Maximum HU value for clamping (default: 1000).

    Returns:
        Tuple of (preprocessed_volume, metadata_dict).
    """
    volume_path = Path(volume_path)

    if volume_path.is_dir():
        volume, metadata = preprocessing.load_dicom_volume(volume_path)
    elif volume_path.suffix in [".nii", ".gz"]:
        volume, metadata = preprocessing.load_nifti_volume(volume_path)
    else:
        raise ValueError(f"Unsupported file format: {volume_path}")

    original_shape = volume.shape
    original_spacing = metadata["spacing"]

    volume, new_spacing = preprocessing.resample_isotropic(
        volume, original_spacing, target_spacing
    )
    metadata["spacing"] = new_spacing
    metadata["original_shape"] = original_shape
    metadata["original_spacing"] = original_spacing

    volume = preprocessing.normalize_hu(volume, hu_min, hu_max)

    return volume, metadata


def predict_volume(
    model: UNet3D,
    volume: np.ndarray,
    device: Optional[torch.device] = None,
    batch_size: int = 1,
    overlap: float = 0.5,
) -> np.ndarray:
    """Perform 3D segmentation on a volume.

    Args:
        model: Trained UNet3D model.
        volume: Preprocessed volume array (D, H, W).
        device: Device to run inference on (default: model device).
        batch_size: Batch size for inference (default: 1).
        overlap: Overlap ratio for sliding window if volume is too large (default: 0.5).

    Returns:
        Segmentation logits array (D, H, W).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(volume_tensor)
        prediction = output.cpu().squeeze().numpy()

    return prediction


def postprocess_segmentation(
    prediction: np.ndarray,
    threshold: float = 0.5,
    apply_morphology: bool = True,
    closing_iterations: int = 2,
    opening_iterations: int = 1,
) -> np.ndarray:
    """Apply post-processing to segmentation prediction.

    Args:
        prediction: Prediction logits or probabilities (D, H, W).
        threshold: Threshold for binarization (default: 0.5).
        apply_morphology: Whether to apply morphological operations (default: True).
        closing_iterations: Number of closing iterations (default: 2).
        opening_iterations: Number of opening iterations (default: 1).

    Returns:
        Post-processed binary mask (D, H, W).
    """
    if prediction.max() > 1.0 or prediction.min() < 0.0:
        import torch
        prediction = torch.sigmoid(torch.from_numpy(prediction)).numpy()

    binary_mask = (prediction > threshold).astype(np.float32)

    if apply_morphology:
        if closing_iterations > 0:
            binary_mask = binary_closing(
                binary_mask, structure=np.ones((3, 3, 3)), iterations=closing_iterations
            ).astype(np.float32)

        if opening_iterations > 0:
            binary_mask = binary_opening(
                binary_mask, structure=np.ones((3, 3, 3)), iterations=opening_iterations
            ).astype(np.float32)

    return binary_mask


def predict_from_file(
    checkpoint_path: Path,
    volume_path: Path,
    output_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
    apply_morphology: bool = True,
    save_probabilities: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Complete inference pipeline from file to segmentation.

    Args:
        checkpoint_path: Path to model checkpoint.
        volume_path: Path to input volume (DICOM dir or NIfTI file).
        output_path: Optional path to save segmentation mask.
        device: Device to run inference on (default: cuda if available, else cpu).
        threshold: Threshold for binarization (default: 0.5).
        apply_morphology: Whether to apply morphological operations (default: True).
        save_probabilities: Whether to save probability map instead of binary mask (default: False).

    Returns:
        Tuple of (segmentation_mask, metadata_dict).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device=device)

    print(f"Preprocessing volume from {volume_path}...")
    volume, metadata = preprocess_for_inference(volume_path)

    print(f"Performing segmentation...")
    prediction = predict_volume(model, volume, device=device)

    print(f"Post-processing segmentation...")
    segmentation = postprocess_segmentation(
        prediction, threshold=threshold, apply_morphology=apply_morphology
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if save_probabilities:
            output_data = prediction
        else:
            output_data = segmentation

        spacing = metadata.get("spacing", np.array([1.0, 1.0, 1.0]))
        origin = metadata.get("origin", np.array([0.0, 0.0, 0.0]))
        direction = metadata.get("direction", np.eye(3))

        affine = np.eye(4)
        affine[:3, :3] = direction * spacing[:, None]
        affine[:3, 3] = origin

        nii_img = nib.Nifti1Image(output_data.astype(np.float32), affine)
        nib.save(nii_img, str(output_path))
        print(f"Segmentation saved to {output_path}")

    return segmentation, metadata


def predict_batch(
    checkpoint_path: Path,
    volume_paths: list,
    output_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
    apply_morphology: bool = True,
) -> list:
    """Predict segmentation for multiple volumes.

    Args:
        checkpoint_path: Path to model checkpoint.
        volume_paths: List of paths to input volumes.
        output_dir: Optional directory to save segmentation masks.
        device: Device to run inference on (default: cuda if available, else cpu).
        threshold: Threshold for binarization (default: 0.5).
        apply_morphology: Whether to apply morphological operations (default: True).

    Returns:
        List of segmentation masks.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(checkpoint_path, device=device)
    results = []

    for vol_path in volume_paths:
        vol_path = Path(vol_path)
        output_path = None

        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{vol_path.stem}_segmentation.nii.gz"

        segmentation, metadata = predict_from_file(
            checkpoint_path=checkpoint_path,
            volume_path=vol_path,
            output_path=output_path,
            device=device,
            threshold=threshold,
            apply_morphology=apply_morphology,
        )

        results.append(segmentation)

    return results

