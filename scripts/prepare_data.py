"""Script to preprocess medical imaging data.

This script processes raw medical volumes (DICOM or NIfTI) and saves
preprocessed volumes ready for training.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import preprocessing


def main():
    """Main function to preprocess dataset."""
    parser = argparse.ArgumentParser(
        description="Preprocess medical imaging data for training"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing input volumes (DICOM dirs or NIfTI files)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save preprocessed volumes (default: data/processed)"
    )
    parser.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        default=[1.0, 1.0, 1.0],
        help="Target isotropic spacing in mm (default: 1.0 1.0 1.0)"
    )
    parser.add_argument(
        "--hu-min",
        type=float,
        default=-1000.0,
        help="Minimum HU value for clamping (default: -1000.0)"
    )
    parser.add_argument(
        "--hu-max",
        type=float,
        default=1000.0,
        help="Maximum HU value for clamping (default: 1000.0)"
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Disable automatic cropping"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    print(f"Processing dataset from: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Target spacing: {args.target_spacing} mm")
    print(f"HU range: [{args.hu_min}, {args.hu_max}]")
    print(f"Auto-crop: {not args.no_crop}")

    preprocessing.process_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        target_spacing=tuple(args.target_spacing),
        hu_min=args.hu_min,
        hu_max=args.hu_max,
        auto_crop_enabled=not args.no_crop
    )

    print(f"\nPreprocessing complete. Processed volumes saved to: {output_dir}")


if __name__ == "__main__":
    main()

