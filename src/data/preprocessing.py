"""Preprocessing pipeline for medical imaging data.

This module provides functions to load, normalize, and preprocess 3D medical volumes
from DICOM and NIfTI formats.
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pydicom


def load_dicom_volume(dicom_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a 3D volume from a directory of DICOM files.

    Args:
        dicom_dir: Path to directory containing DICOM files.

    Returns:
        Tuple of (volume_array, metadata_dict) where:
        - volume_array: 3D numpy array of the volume
        - metadata_dict: Dictionary containing spacing, origin, direction, etc.
    """
    dicom_files = sorted(dicom_dir.glob("*.dcm"))
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    slices = []
    for dcm_file in dicom_files:
        ds = pydicom.dcmread(str(dcm_file))
        if not hasattr(ds, "ImagePositionPatient") or len(ds.ImagePositionPatient) < 3:
            raise ValueError(f"DICOM file {dcm_file} missing ImagePositionPatient attribute")
        slices.append(ds)

    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    volume = np.stack([s.pixel_array for s in slices])
    volume = volume.astype(np.float32)

    first_slice = slices[0]
    metadata = {
        "spacing": np.array([
            float(first_slice.PixelSpacing[0]),
            float(first_slice.PixelSpacing[1]),
            float(first_slice.SliceThickness) if hasattr(first_slice, "SliceThickness") else
            abs(float(slices[1].ImagePositionPatient[2]) - float(first_slice.ImagePositionPatient[2]))
        ]),
        "origin": np.array([float(x) for x in first_slice.ImagePositionPatient]),
        "direction": np.eye(3),
        "modality": getattr(first_slice, "Modality", "CT"),
    }

    return volume, metadata


def load_nifti_volume(nifti_path: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Load a 3D volume from a NIfTI file.

    Args:
        nifti_path: Path to NIfTI file (.nii or .nii.gz).

    Returns:
        Tuple of (volume_array, metadata_dict) where:
        - volume_array: 3D numpy array of the volume
        - metadata_dict: Dictionary containing spacing, origin, direction, etc.
    """
    nii_img = nib.load(str(nifti_path))
    volume = nii_img.get_fdata().astype(np.float32)

    affine = nii_img.affine
    spacing = np.array([
        np.linalg.norm(affine[:3, 0]),
        np.linalg.norm(affine[:3, 1]),
        np.linalg.norm(affine[:3, 2])
    ])
    origin = affine[:3, 3]
    direction = affine[:3, :3] / spacing[:, None]

    metadata = {
        "spacing": spacing,
        "origin": origin,
        "direction": direction,
        "modality": "CT",
    }

    return volume, metadata


def resample_isotropic(
    volume: np.ndarray,
    spacing: np.ndarray,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample volume to isotropic spacing using SimpleITK.

    Args:
        volume: 3D numpy array of the volume.
        spacing: Current spacing (x, y, z) in mm.
        target_spacing: Target isotropic spacing (default: 1x1x1 mm).

    Returns:
        Tuple of (resampled_volume, new_spacing).
    """
    sitk_image = sitk.GetImageFromArray(volume)
    sitk_image.SetSpacing(spacing[::-1])

    target_spacing_array = np.array(target_spacing)
    new_size = [
        int(size * old_sp / new_sp)
        for size, old_sp, new_sp in zip(volume.shape[::-1], spacing[::-1], target_spacing_array[::-1])
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing_array[::-1])
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(sitk_image.GetDirection())
    resampler.SetOutputOrigin(sitk_image.GetOrigin())

    resampled = resampler.Execute(sitk_image)
    resampled_array = sitk.GetArrayFromImage(resampled).astype(np.float32)

    return resampled_array, target_spacing_array


def auto_crop(volume: np.ndarray, threshold: float = -1000.0) -> Tuple[np.ndarray, Tuple[slice, slice, slice]]:
    """Automatically crop volume to remove background.

    Args:
        volume: 3D numpy array of the volume.
        threshold: Threshold value below which pixels are considered background.

    Returns:
        Tuple of (cropped_volume, crop_slices) where crop_slices can be used
        to crop other volumes with the same geometry.
    """
    mask = volume > threshold
    coords = np.argwhere(mask)

    if len(coords) == 0:
        return volume, (slice(None), slice(None), slice(None))

    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1

    crop_slices = tuple(slice(min_c, max_c) for min_c, max_c in zip(min_coords, max_coords))
    cropped = volume[crop_slices]

    return cropped, crop_slices


def normalize_hu(volume: np.ndarray, hu_min: float = -1000.0, hu_max: float = 1000.0) -> np.ndarray:
    """Normalize HU values and clamp extreme values.

    Args:
        volume: 3D numpy array in HU units.
        hu_min: Minimum HU value to clamp (default: -1000).
        hu_max: Maximum HU value to clamp (default: 1000).

    Returns:
        Normalized volume with values clamped and optionally normalized.
    """
    volume_clamped = np.clip(volume, hu_min, hu_max)
    volume_normalized = (volume_clamped - hu_min) / (hu_max - hu_min)

    return volume_normalized.astype(np.float32)


def preprocess_volume(
    input_path: Path,
    output_path: Optional[Path] = None,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    hu_min: float = -1000.0,
    hu_max: float = 1000.0,
    auto_crop_enabled: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Complete preprocessing pipeline for a medical volume.

    Args:
        input_path: Path to input file or directory (DICOM dir or NIfTI file).
        output_path: Optional path to save preprocessed volume as NIfTI.
        target_spacing: Target isotropic spacing in mm (default: 1x1x1).
        hu_min: Minimum HU value for clamping (default: -1000).
        hu_max: Maximum HU value for clamping (default: 1000).
        auto_crop_enabled: Whether to apply automatic cropping (default: True).

    Returns:
        Tuple of (preprocessed_volume, metadata_dict).
    """
    input_path = Path(input_path)

    if input_path.is_dir():
        volume, metadata = load_dicom_volume(input_path)
    elif input_path.suffix in [".nii", ".gz"]:
        volume, metadata = load_nifti_volume(input_path)
    else:
        raise ValueError(f"Unsupported file format: {input_path}")

    original_shape = volume.shape
    original_spacing = metadata["spacing"]

    volume, new_spacing = resample_isotropic(volume, original_spacing, target_spacing)
    metadata["spacing"] = new_spacing
    metadata["original_shape"] = original_shape
    metadata["original_spacing"] = original_spacing

    if auto_crop_enabled:
        volume, crop_slices = auto_crop(volume)
        metadata["crop_slices"] = crop_slices

    volume = normalize_hu(volume, hu_min, hu_max)
    metadata["hu_range"] = (hu_min, hu_max)

    if output_path is not None:
        save_preprocessed_volume(volume, output_path, metadata)

    return volume, metadata


def save_preprocessed_volume(
    volume: np.ndarray,
    output_path: Path,
    metadata: Dict[str, Any]
) -> None:
    """Save preprocessed volume as NIfTI file.

    Args:
        volume: 3D numpy array of preprocessed volume.
        output_path: Path to save NIfTI file.
        metadata: Metadata dictionary containing spacing, origin, direction.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    spacing = metadata.get("spacing", np.array([1.0, 1.0, 1.0]))
    origin = metadata.get("origin", np.array([0.0, 0.0, 0.0]))
    direction = metadata.get("direction", np.eye(3))

    affine = np.eye(4)
    affine[:3, :3] = direction * spacing[:, None]
    affine[:3, 3] = origin

    nii_img = nib.Nifti1Image(volume, affine)
    nib.save(nii_img, str(output_path))


def process_dataset(
    input_dir: Path,
    output_dir: Path,
    target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    hu_min: float = -1000.0,
    hu_max: float = 1000.0,
    auto_crop_enabled: bool = True
) -> None:
    """Process an entire dataset of medical volumes.

    Args:
        input_dir: Directory containing input volumes (DICOM dirs or NIfTI files).
        output_dir: Directory to save preprocessed volumes.
        target_spacing: Target isotropic spacing in mm (default: 1x1x1).
        hu_min: Minimum HU value for clamping (default: -1000).
        hu_max: Maximum HU value for clamping (default: 1000).
        auto_crop_enabled: Whether to apply automatic cropping (default: True).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = []
    if Path(input_dir).is_dir():
        input_paths.extend(Path(input_dir).glob("*.nii"))
        input_paths.extend(Path(input_dir).glob("*.nii.gz"))
        input_paths.extend([d for d in Path(input_dir).iterdir() if d.is_dir()])

    for input_path in input_paths:
        output_filename = input_path.stem if input_path.is_file() else input_path.name
        if output_filename.endswith(".nii"):
            output_filename = output_filename[:-4]
        output_path = output_dir / f"{output_filename}_preprocessed.nii.gz"

        try:
            preprocess_volume(
                input_path,
                output_path,
                target_spacing=target_spacing,
                hu_min=hu_min,
                hu_max=hu_max,
                auto_crop_enabled=auto_crop_enabled
            )
            print(f"Processed: {input_path.name} -> {output_path.name}")
        except Exception as e:
            print(f"Error processing {input_path.name}: {e}")

