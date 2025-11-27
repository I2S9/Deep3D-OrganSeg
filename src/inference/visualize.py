"""Visualization utilities for 3D segmentation results.

This module provides functions to visualize segmentation results in 2D and 3D.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import napari


def visualize_2d_slices(
    volume: np.ndarray,
    segmentation: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    slice_indices: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Visualize segmentation in 3 standard 2D planes (axial, coronal, sagittal).

    Args:
        volume: Input volume array (D, H, W).
        segmentation: Segmentation mask array (D, H, W).
        ground_truth: Optional ground truth mask for comparison (D, H, W).
        output_path: Optional path to save visualization.
        slice_indices: Optional tuple of (axial_idx, coronal_idx, sagittal_idx).
                       If None, uses middle slices.
    """
    d, h, w = volume.shape

    if slice_indices is None:
        axial_idx = d // 2
        coronal_idx = h // 2
        sagittal_idx = w // 2
    else:
        axial_idx, coronal_idx, sagittal_idx = slice_indices

    fig, axes = plt.subplots(3, 3 if ground_truth is not None else 2, figsize=(15, 12))
    fig.suptitle("Segmentation Visualization - 3 Standard Planes", fontsize=16)

    planes = [
        ("Axial", volume[axial_idx, :, :], segmentation[axial_idx, :, :]),
        ("Coronal", volume[:, coronal_idx, :], segmentation[:, coronal_idx, :]),
        ("Sagittal", volume[:, :, sagittal_idx], segmentation[:, :, sagittal_idx]),
    ]

    if ground_truth is not None:
        gt_planes = [
            ground_truth[axial_idx, :, :],
            ground_truth[:, coronal_idx, :],
            ground_truth[:, :, sagittal_idx],
        ]

    for row, (plane_name, vol_slice, seg_slice) in enumerate(planes):
        axes[row, 0].imshow(vol_slice, cmap="gray")
        axes[row, 0].set_title(f"{plane_name} - Volume")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(vol_slice, cmap="gray")
        axes[row, 1].imshow(seg_slice, cmap="Reds", alpha=0.5)
        axes[row, 1].set_title(f"{plane_name} - Volume + Segmentation")
        axes[row, 1].axis("off")

        if ground_truth is not None:
            axes[row, 2].imshow(vol_slice, cmap="gray")
            axes[row, 2].imshow(gt_planes[row], cmap="Blues", alpha=0.5)
            axes[row, 2].imshow(seg_slice, cmap="Reds", alpha=0.3)
            axes[row, 2].set_title(f"{plane_name} - Comparison")
            axes[row, 2].axis("off")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"2D visualization saved to {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_overlay(
    volume: np.ndarray,
    segmentation: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    slice_idx: Optional[int] = None,
    axis: int = 2,
) -> None:
    """Visualize overlay of volume and segmentation with optional ground truth.

    Args:
        volume: Input volume array (D, H, W).
        segmentation: Segmentation mask array (D, H, W).
        ground_truth: Optional ground truth mask (D, H, W).
        output_path: Optional path to save visualization.
        slice_idx: Slice index to visualize (default: middle slice).
        axis: Axis along which to slice (0=axial, 1=coronal, 2=sagittal, default: 2).
    """
    if slice_idx is None:
        slice_idx = volume.shape[axis] // 2

    if axis == 0:
        vol_slice = volume[slice_idx, :, :]
        seg_slice = segmentation[slice_idx, :, :]
        if ground_truth is not None:
            gt_slice = ground_truth[slice_idx, :, :]
    elif axis == 1:
        vol_slice = volume[:, slice_idx, :]
        seg_slice = segmentation[:, slice_idx, :]
        if ground_truth is not None:
            gt_slice = ground_truth[:, slice_idx, :]
    else:
        vol_slice = volume[:, :, slice_idx]
        seg_slice = segmentation[:, :, slice_idx]
        if ground_truth is not None:
            gt_slice = ground_truth[:, :, slice_idx]

    fig, axes = plt.subplots(1, 3 if ground_truth is not None else 2, figsize=(15, 5))
    fig.suptitle(f"Segmentation Overlay - Slice {slice_idx}", fontsize=16)

    axes[0].imshow(vol_slice, cmap="gray")
    axes[0].set_title("Input Volume")
    axes[0].axis("off")

    axes[1].imshow(vol_slice, cmap="gray")
    axes[1].imshow(seg_slice, cmap="Reds", alpha=0.6)
    axes[1].set_title("Volume + Segmentation")
    axes[1].axis("off")

    if ground_truth is not None:
        overlap = seg_slice * gt_slice
        axes[2].imshow(vol_slice, cmap="gray")
        axes[2].imshow(gt_slice, cmap="Blues", alpha=0.4)
        axes[2].imshow(seg_slice, cmap="Reds", alpha=0.4)
        axes[2].imshow(overlap, cmap="Greens", alpha=0.6)
        axes[2].set_title("Comparison (Green=Overlap)")
        axes[2].axis("off")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Overlay visualization saved to {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_3d_napari(
    volume: np.ndarray,
    segmentation: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
) -> None:
    """Visualize 3D volume and segmentation using Napari.

    Args:
        volume: Input volume array (D, H, W).
        segmentation: Segmentation mask array (D, H, W).
        ground_truth: Optional ground truth mask (D, H, W).
    """
    viewer = napari.Viewer()

    viewer.add_image(volume, name="Volume", colormap="gray")

    viewer.add_labels(
        (segmentation > 0.5).astype(np.uint8),
        name="Segmentation",
        opacity=0.6,
    )

    if ground_truth is not None:
        viewer.add_labels(
            (ground_truth > 0.5).astype(np.uint8),
            name="Ground Truth",
            opacity=0.4,
        )

    napari.run()


def visualize_3d_mesh(
    segmentation: np.ndarray,
    output_path: Optional[Path] = None,
    threshold: float = 0.5,
) -> None:
    """Visualize 3D segmentation as mesh using matplotlib.

    Args:
        segmentation: Segmentation mask array (D, H, W).
        output_path: Optional path to save visualization.
        threshold: Threshold for creating mesh (default: 0.5).
    """
    from skimage import measure

    binary_mask = (segmentation > threshold).astype(np.uint8)

    try:
        verts, faces, normals, values = measure.marching_cubes(
            binary_mask, level=0.5, spacing=(1.0, 1.0, 1.0)
        )

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_trisurf(
            verts[:, 0],
            verts[:, 1],
            verts[:, 2],
            triangles=faces,
            alpha=0.7,
            color="red",
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D Segmentation Mesh")

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"3D mesh visualization saved to {output_path}")
        else:
            plt.show()

        plt.close()

    except Exception as e:
        print(f"Could not create 3D mesh: {e}")
        print("Falling back to volume rendering...")
        visualize_3d_volume(segmentation, output_path)


def visualize_3d_volume(
    segmentation: np.ndarray,
    output_path: Optional[Path] = None,
) -> None:
    """Visualize 3D segmentation as volume rendering.

    Args:
        segmentation: Segmentation mask array (D, H, W).
        output_path: Optional path to save visualization.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    binary_mask = (segmentation > 0.5).astype(bool)
    coords = np.argwhere(binary_mask)

    if len(coords) > 0:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            c=coords[:, 2],
            cmap="Reds",
            alpha=0.3,
            s=1,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Segmentation Volume")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"3D volume visualization saved to {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_complete(
    volume: np.ndarray,
    segmentation: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
) -> None:
    """Create complete visualization suite.

    Args:
        volume: Input volume array (D, H, W).
        segmentation: Segmentation mask array (D, H, W).
        ground_truth: Optional ground truth mask (D, H, W).
        output_dir: Optional directory to save visualizations.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating 2D slice visualizations...")
    visualize_2d_slices(
        volume,
        segmentation,
        ground_truth,
        output_path=output_dir / "2d_slices.png" if output_dir else None,
    )

    print("Creating overlay visualizations...")
    visualize_overlay(
        volume,
        segmentation,
        ground_truth,
        output_path=output_dir / "overlay.png" if output_dir else None,
    )

    print("Creating 3D mesh visualization...")
    visualize_3d_mesh(
        segmentation,
        output_path=output_dir / "3d_mesh.png" if output_dir else None,
    )

    print("Visualization complete!")



