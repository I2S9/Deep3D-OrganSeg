"""Dataset and DataLoader for 3D medical imaging using MONAI.

This module provides Dataset classes and transforms for training 3D segmentation models.
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import torch
from monai.data import Dataset, DataLoader, CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    Orientationd,
    ScaleIntensityRanged,
    RandRotate90d,
    RandFlipd,
    RandZoomd,
    RandGaussianNoised,
    RandShiftIntensityd,
    NormalizeIntensityd,
    EnsureTyped,
    AsDiscreted,
    ToTensord,
)
from monai.data.utils import pad_list_data_collate


class OrganSegmentationDataset(Dataset):
    """Dataset for 3D organ segmentation with volumes and masks.

    This dataset loads pairs of volumes and masks from NIfTI files and applies
    data augmentation transforms during training.
    """

    def __init__(
        self,
        volume_paths: List[Path],
        mask_paths: List[Path],
        transform: Optional[Compose] = None,
        cache: bool = False,
    ):
        """Initialize the dataset.

        Args:
            volume_paths: List of paths to volume NIfTI files.
            mask_paths: List of paths to mask NIfTI files (same order as volumes).
            transform: Optional MONAI Compose transform to apply.
            cache: Whether to cache loaded data in memory (default: False).
        """
        if len(volume_paths) != len(mask_paths):
            raise ValueError(
                f"Number of volumes ({len(volume_paths)}) must match "
                f"number of masks ({len(mask_paths)})"
            )

        self.data = [
            {"image": str(vol_path), "label": str(mask_path)}
            for vol_path, mask_path in zip(volume_paths, mask_paths)
        ]

        if cache:
            self.dataset = CacheDataset(
                data=self.data,
                transform=transform,
                cache_rate=1.0,
                num_workers=4,
            )
        else:
            self.dataset = Dataset(data=self.data, transform=transform)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.

        Args:
            idx: Index of the sample.

        Returns:
            Dictionary with 'image' and 'label' keys containing tensors.
        """
        return self.dataset[idx]


def get_training_transforms(
    patch_size: Tuple[int, int, int] = (128, 128, 128),
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    intensity_range: Tuple[float, float] = (0.0, 1.0),
    prob_rotation: float = 0.5,
    prob_flip: float = 0.5,
    prob_zoom: float = 0.3,
    prob_noise: float = 0.2,
    prob_shift: float = 0.2,
) -> Compose:
    """Get training transforms with data augmentation.

    Args:
        patch_size: Size of patches to extract (D, H, W).
        spacing: Target spacing in mm (default: 1x1x1).
        intensity_range: Range for intensity normalization (default: [0, 1]).
        prob_rotation: Probability of applying rotation (default: 0.5).
        prob_flip: Probability of applying flip (default: 0.5).
        prob_zoom: Probability of applying zoom (default: 0.3).
        prob_noise: Probability of adding Gaussian noise (default: 0.2).
        prob_shift: Probability of shifting intensity (default: 0.2).

    Returns:
        MONAI Compose transform for training.
    """
    keys = ["image", "label"]

    transforms = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Spacingd(
            keys=keys,
            pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=keys, axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=intensity_range[0],
            a_max=intensity_range[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        RandRotate90d(
            keys=keys,
            prob=prob_rotation,
            spatial_axes=(0, 1),
        ),
        RandFlipd(
            keys=keys,
            prob=prob_flip,
            spatial_axis=[0, 1, 2],
        ),
        RandZoomd(
            keys=keys,
            prob=prob_zoom,
            min_zoom=0.9,
            max_zoom=1.1,
            mode=("trilinear", "nearest"),
            padding_mode="constant",
        ),
        RandGaussianNoised(
            keys=["image"],
            prob=prob_noise,
            mean=0.0,
            std=0.1,
        ),
        RandShiftIntensityd(
            keys=["image"],
            prob=prob_shift,
            offsets=0.1,
        ),
        EnsureTyped(keys=keys, data_type="tensor", dtype=torch.float32),
    ]

    return Compose(transforms)


def get_patch_transforms(
    prob_rotation: float = 0.5,
    prob_flip: float = 0.5,
    prob_zoom: float = 0.3,
    prob_noise: float = 0.2,
    prob_shift: float = 0.2,
) -> Compose:
    """Get transforms for patches (data already loaded as arrays).

    Args:
        prob_rotation: Probability of applying rotation (default: 0.5).
        prob_flip: Probability of applying flip (default: 0.5).
        prob_zoom: Probability of applying zoom (default: 0.3).
        prob_noise: Probability of adding Gaussian noise (default: 0.2).
        prob_shift: Probability of shifting intensity (default: 0.2).

    Returns:
        MONAI Compose transform for patches.
    """
    keys = ["image", "label"]

    transforms = [
        EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        RandRotate90d(
            keys=keys,
            prob=prob_rotation,
            spatial_axes=(0, 1),
        ),
        RandFlipd(
            keys=keys,
            prob=prob_flip,
            spatial_axis=[0, 1, 2],
        ),
        RandZoomd(
            keys=keys,
            prob=prob_zoom,
            min_zoom=0.9,
            max_zoom=1.1,
            mode=("trilinear", "nearest"),
            padding_mode="constant",
        ),
        RandGaussianNoised(
            keys=["image"],
            prob=prob_noise,
            mean=0.0,
            std=0.1,
        ),
        RandShiftIntensityd(
            keys=["image"],
            prob=prob_shift,
            offsets=0.1,
        ),
        EnsureTyped(keys=keys, data_type="tensor", dtype=torch.float32),
    ]

    return Compose(transforms)


def get_validation_transforms(
    spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    intensity_range: Tuple[float, float] = (0.0, 1.0),
) -> Compose:
    """Get validation transforms (no augmentation).

    Args:
        spacing: Target spacing in mm (default: 1x1x1).
        intensity_range: Range for intensity normalization (default: [0, 1]).

    Returns:
        MONAI Compose transform for validation.
    """
    keys = ["image", "label"]

    transforms = [
        LoadImaged(keys=keys, image_only=False),
        EnsureChannelFirstd(keys=keys),
        Spacingd(
            keys=keys,
            pixdim=spacing,
            mode=("bilinear", "nearest"),
        ),
        Orientationd(keys=keys, axcodes="RAS"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=intensity_range[0],
            a_max=intensity_range[1],
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        EnsureTyped(keys=keys, data_type="tensor", dtype=torch.float32),
    ]

    return Compose(transforms)


class PatchDataset(Dataset):
    """Dataset that extracts 3D patches from volumes with organ oversampling.

    This dataset extracts patches of a fixed size from volumes, with oversampling
    of patches that contain the target organ (positive patches).
    """

    def __init__(
        self,
        volume_paths: List[Path],
        mask_paths: List[Path],
        patch_size: Tuple[int, int, int] = (128, 128, 128),
        transform: Optional[Compose] = None,
        oversample_ratio: float = 2.0,
        organ_threshold: float = 0.1,
    ):
        """Initialize the patch dataset.

        Args:
            volume_paths: List of paths to volume NIfTI files.
            mask_paths: List of paths to mask NIfTI files.
            patch_size: Size of patches to extract (D, H, W).
            transform: Optional MONAI Compose transform to apply.
            oversample_ratio: Ratio of positive patches to negative patches (default: 2.0).
            organ_threshold: Minimum fraction of organ pixels in patch to be considered positive (default: 0.1).
        """
        if len(volume_paths) != len(mask_paths):
            raise ValueError(
                f"Number of volumes ({len(volume_paths)}) must match "
                f"number of masks ({len(mask_paths)})"
            )

        self.volume_paths = volume_paths
        self.mask_paths = mask_paths
        self.patch_size = patch_size
        self.transform = transform
        self.oversample_ratio = oversample_ratio
        self.organ_threshold = organ_threshold

        self.patches = self._generate_patches()

    def _generate_patches(self) -> List[Dict[str, Any]]:
        """Generate list of patches with oversampling of positive patches.

        Returns:
            List of patch dictionaries with 'volume_path', 'mask_path', 'patch_coords', and 'is_positive'.
        """
        patches = []

        for vol_path, mask_path in zip(self.volume_paths, self.mask_paths):
            import nibabel as nib

            mask_img = nib.load(str(mask_path))
            mask = mask_img.get_fdata()

            volume_shape = mask.shape
            patch_d, patch_h, patch_w = self.patch_size

            stride_d = patch_d // 2
            stride_h = patch_h // 2
            stride_w = patch_w // 2

            positive_patches = []
            negative_patches = []

            for d in range(0, volume_shape[0] - patch_d + 1, stride_d):
                for h in range(0, volume_shape[1] - patch_h + 1, stride_h):
                    for w in range(0, volume_shape[2] - patch_w + 1, stride_w):
                        patch_coords = (d, h, w)
                        patch_mask = mask[
                            d : d + patch_d, h : h + patch_h, w : w + patch_w
                        ]

                        organ_fraction = np.sum(patch_mask > 0) / (
                            patch_d * patch_h * patch_w
                        )

                        patch_info = {
                            "volume_path": str(vol_path),
                            "mask_path": str(mask_path),
                            "patch_coords": patch_coords,
                            "is_positive": organ_fraction >= self.organ_threshold,
                        }

                        if patch_info["is_positive"]:
                            positive_patches.append(patch_info)
                        else:
                            negative_patches.append(patch_info)

            patches.extend(positive_patches)

            num_negative = int(len(positive_patches) * self.oversample_ratio)
            if num_negative > 0:
                import random

                selected_negative = random.sample(
                    negative_patches, min(num_negative, len(negative_patches))
                )
                patches.extend(selected_negative)

        return patches

    def __len__(self) -> int:
        """Return the number of patches."""
        return len(self.patches)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a patch from the dataset.

        Args:
            idx: Index of the patch.

        Returns:
            Dictionary with 'image' and 'label' keys containing patch tensors.
        """
        patch_info = self.patches[idx]
        vol_path = patch_info["volume_path"]
        mask_path = patch_info["mask_path"]
        coords = patch_info["patch_coords"]

        import nibabel as nib

        volume_img = nib.load(vol_path)
        mask_img = nib.load(mask_path)

        volume = volume_img.get_fdata()
        mask = mask_img.get_fdata()

        d, h, w = coords
        patch_d, patch_h, patch_w = self.patch_size

        volume_patch = volume[
            d : d + patch_d, h : h + patch_h, w : w + patch_w
        ].astype(np.float32)
        mask_patch = mask[
            d : d + patch_d, h : h + patch_h, w : w + patch_w
        ].astype(np.float32)

        data = {
            "image": volume_patch,
            "label": mask_patch,
        }

        if self.transform is not None:
            data = self.transform(data)

        return data


def create_dataloader(
    volume_paths: List[Path],
    mask_paths: List[Path],
    batch_size: int = 2,
    patch_size: Optional[Tuple[int, int, int]] = None,
    transform: Optional[Compose] = None,
    use_patches: bool = False,
    oversample_ratio: float = 2.0,
    num_workers: int = 4,
    shuffle: bool = True,
    cache: bool = False,
) -> DataLoader:
    """Create a DataLoader for training or validation.

    Args:
        volume_paths: List of paths to volume NIfTI files.
        mask_paths: List of paths to mask NIfTI files.
        batch_size: Batch size (default: 2).
        patch_size: Patch size if using patches (D, H, W). If None, uses full volumes.
        transform: Optional MONAI Compose transform.
        use_patches: Whether to extract patches (default: False).
        oversample_ratio: Ratio for oversampling positive patches (default: 2.0).
        num_workers: Number of worker processes (default: 4).
        shuffle: Whether to shuffle data (default: True).
        cache: Whether to cache data in memory (default: False).

    Returns:
        MONAI DataLoader instance.
    """
    if use_patches:
        if patch_size is None:
            patch_size = (128, 128, 128)
        if transform is None:
            transform = get_patch_transforms()
        dataset = PatchDataset(
            volume_paths=volume_paths,
            mask_paths=mask_paths,
            patch_size=patch_size,
            transform=transform,
            oversample_ratio=oversample_ratio,
        )
    else:
        dataset = OrganSegmentationDataset(
            volume_paths=volume_paths,
            mask_paths=mask_paths,
            transform=transform,
            cache=cache,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pad_list_data_collate,
    )

    return dataloader

