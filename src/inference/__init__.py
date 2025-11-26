"""Inference modules."""

from src.inference.predict import (
    load_model,
    preprocess_for_inference,
    predict_volume,
    postprocess_segmentation,
    predict_from_file,
    predict_batch,
)
from src.inference.visualize import (
    visualize_2d_slices,
    visualize_overlay,
    visualize_3d_napari,
    visualize_3d_mesh,
    visualize_3d_volume,
    visualize_complete,
)

__all__ = [
    "load_model",
    "preprocess_for_inference",
    "predict_volume",
    "postprocess_segmentation",
    "predict_from_file",
    "predict_batch",
    "visualize_2d_slices",
    "visualize_overlay",
    "visualize_3d_napari",
    "visualize_3d_mesh",
    "visualize_3d_volume",
    "visualize_complete",
]

