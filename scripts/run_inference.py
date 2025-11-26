"""Command-line script for running inference on medical volumes.

Usage:
    python scripts/run_inference.py <checkpoint_path> <volume_path> [--output <output_path>]
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import predict, visualize
from src.data import preprocessing


def main():
    """Main function for inference script."""
    parser = argparse.ArgumentParser(
        description="Run 3D organ segmentation inference on a medical volume"
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "volume_path",
        type=str,
        help="Path to input volume (DICOM directory or NIfTI file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save segmentation mask (default: <volume_name>_segmentation.nii.gz)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binarization (default: 0.5)"
    )
    parser.add_argument(
        "--no-morphology",
        action="store_true",
        help="Disable morphological post-processing"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization images"
    )
    parser.add_argument(
        "--visualize-dir",
        type=str,
        default=None,
        help="Directory to save visualizations (default: same as output)"
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint_path)
    volume_path = Path(args.volume_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if not volume_path.exists():
        raise FileNotFoundError(f"Volume not found: {volume_path}")

    if args.output is None:
        output_path = volume_path.parent / f"{volume_path.stem}_segmentation.nii.gz"
        if volume_path.suffix == ".gz":
            output_path = volume_path.parent / f"{volume_path.stem[:-4]}_segmentation.nii.gz"
    else:
        output_path = Path(args.output)

    print(f"Running inference...")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Input volume: {volume_path}")
    print(f"  Output: {output_path}")

    segmentation, metadata = predict.predict_from_file(
        checkpoint_path=checkpoint_path,
        volume_path=volume_path,
        output_path=output_path,
        threshold=args.threshold,
        apply_morphology=not args.no_morphology,
    )

    print(f"\nSegmentation complete!")
    print(f"  Output shape: {segmentation.shape}")
    print(f"  Output saved to: {output_path}")

    if args.visualize:
        visualize_dir = Path(args.visualize_dir) if args.visualize_dir else output_path.parent
        visualize_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating visualizations...")
        volume, _ = preprocessing.load_nifti_volume(volume_path)

        visualize.visualize_2d_slices(
            volume,
            segmentation,
            output_path=visualize_dir / "2d_slices.png"
        )

        visualize.visualize_overlay(
            volume,
            segmentation,
            output_path=visualize_dir / "overlay.png"
        )

        visualize.visualize_3d_mesh(
            segmentation,
            output_path=visualize_dir / "3d_mesh.png"
        )

        print(f"  Visualizations saved to: {visualize_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


