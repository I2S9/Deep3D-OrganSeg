"""Streamlit application for 3D organ segmentation.

Minimal clinical prototype for testing organ segmentation on medical volumes.
"""

import streamlit as st
import numpy as np
from pathlib import Path
import tempfile
import nibabel as nib
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.inference import predict, visualize
from src.data import preprocessing


st.set_page_config(
    page_title="3D Organ Segmentation",
    page_icon="🏥",
    layout="wide",
)

st.title("3D Organ Segmentation - Clinical Prototype")
st.markdown("Upload a medical volume (NIfTI) to perform automatic organ segmentation.")


@st.cache_resource
def load_cached_model(checkpoint_path: Path):
    """Load and cache model for faster inference."""
    return predict.load_model(checkpoint_path)


def process_uploaded_file(uploaded_file, temp_dir: Path) -> Path:
    """Save uploaded file to temporary directory."""
    file_path = temp_dir / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def display_slice(volume: np.ndarray, segmentation: np.ndarray, axis: int, slice_idx: int):
    """Display a single slice with overlay."""
    if axis == 0:
        vol_slice = volume[slice_idx, :, :]
        seg_slice = segmentation[slice_idx, :, :]
        title = f"Axial Slice {slice_idx}/{volume.shape[0]-1}"
    elif axis == 1:
        vol_slice = volume[:, slice_idx, :]
        seg_slice = segmentation[:, slice_idx, :]
        title = f"Coronal Slice {slice_idx}/{volume.shape[1]-1}"
    else:
        vol_slice = volume[:, :, slice_idx]
        seg_slice = segmentation[:, :, slice_idx]
        title = f"Sagittal Slice {slice_idx}/{volume.shape[2]-1}"

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=14)

    axes[0].imshow(vol_slice, cmap="gray")
    axes[0].set_title("Volume")
    axes[0].axis("off")

    axes[1].imshow(vol_slice, cmap="gray")
    axes[1].imshow(seg_slice, cmap="Reds", alpha=0.5)
    axes[1].set_title("Volume + Segmentation")
    axes[1].axis("off")

    return fig


def main():
    """Main Streamlit application."""

    checkpoint_path = st.sidebar.text_input(
        "Model Checkpoint Path",
        value="checkpoints/test_inference/model.pth",
        help="Path to the trained model checkpoint (.pth file)"
    )

    threshold = st.sidebar.slider(
        "Segmentation Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Threshold for binarizing the segmentation"
    )

    apply_morphology = st.sidebar.checkbox(
        "Apply Morphological Post-processing",
        value=True,
        help="Apply closing and opening operations to smooth the segmentation"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Instructions")
    st.sidebar.markdown("1. Upload a NIfTI volume (.nii or .nii.gz)")
    st.sidebar.markdown("2. Click 'Run Segmentation'")
    st.sidebar.markdown("3. View results in the main panel")

    uploaded_file = st.file_uploader(
        "Upload Medical Volume (NIfTI format)",
        type=["nii", "gz"],
        help="Upload a .nii or .nii.gz file"
    )

    if uploaded_file is not None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            volume_path = process_uploaded_file(uploaded_file, tmpdir)

            st.success(f"File uploaded: {uploaded_file.name}")

            if st.button("Run Segmentation", type="primary"):
                checkpoint = Path(checkpoint_path)

                if not checkpoint.exists():
                    st.error(f"Checkpoint not found: {checkpoint_path}")
                    st.stop()

                with st.spinner("Loading model..."):
                    try:
                        model = load_cached_model(checkpoint)
                        st.success("Model loaded successfully")
                    except Exception as e:
                        st.error(f"Error loading model: {e}")
                        st.stop()

                with st.spinner("Preprocessing volume..."):
                    try:
                        volume, metadata = predict.preprocess_for_inference(volume_path)
                        st.success(f"Volume preprocessed: shape {volume.shape}")
                    except Exception as e:
                        st.error(f"Error preprocessing volume: {e}")
                        st.stop()

                with st.spinner("Running segmentation..."):
                    try:
                        prediction = predict.predict_volume(model, volume)
                        segmentation = predict.postprocess_segmentation(
                            prediction,
                            threshold=threshold,
                            apply_morphology=apply_morphology
                        )
                        st.success("Segmentation complete")
                    except Exception as e:
                        st.error(f"Error during segmentation: {e}")
                        st.stop()

                st.markdown("---")
                st.header("Segmentation Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Volume Shape", f"{volume.shape[0]} x {volume.shape[1]} x {volume.shape[2]}")
                with col2:
                    voxels_segmented = np.sum(segmentation > 0.5)
                    total_voxels = segmentation.size
                    percentage = (voxels_segmented / total_voxels) * 100
                    st.metric("Segmented Voxels", f"{percentage:.2f}%")
                with col3:
                    st.metric("Segmentation Shape", f"{segmentation.shape[0]} x {segmentation.shape[1]} x {segmentation.shape[2]}")

                st.markdown("---")
                st.subheader("Main Slices - Axial, Coronal, Sagittal")

                d, h, w = volume.shape
                axial_idx = d // 2
                coronal_idx = h // 2
                sagittal_idx = w // 2

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("**Axial View**")
                    fig_axial = display_slice(volume, segmentation, axis=0, slice_idx=axial_idx)
                    st.pyplot(fig_axial)
                    plt.close(fig_axial)

                with col2:
                    st.markdown("**Coronal View**")
                    fig_coronal = display_slice(volume, segmentation, axis=1, slice_idx=coronal_idx)
                    st.pyplot(fig_coronal)
                    plt.close(fig_coronal)

                with col3:
                    st.markdown("**Sagittal View**")
                    fig_sagittal = display_slice(volume, segmentation, axis=2, slice_idx=sagittal_idx)
                    st.pyplot(fig_sagittal)
                    plt.close(fig_sagittal)

                st.markdown("---")
                st.subheader("Interactive Slice Navigation")

                slice_type = st.selectbox("Select View", ["Axial", "Coronal", "Sagittal"])

                if slice_type == "Axial":
                    max_slices = volume.shape[0]
                    axis = 0
                elif slice_type == "Coronal":
                    max_slices = volume.shape[1]
                    axis = 1
                else:
                    max_slices = volume.shape[2]
                    axis = 2

                slice_idx = st.slider(
                    "Slice Index",
                    min_value=0,
                    max_value=max_slices - 1,
                    value=max_slices // 2,
                    key=f"slice_{slice_type}"
                )

                fig_interactive = display_slice(volume, segmentation, axis=axis, slice_idx=slice_idx)
                st.pyplot(fig_interactive)
                plt.close(fig_interactive)

                st.markdown("---")
                st.subheader("Download Segmentation")

                output_buffer = io.BytesIO()
                spacing = metadata.get("spacing", np.array([1.0, 1.0, 1.0]))
                origin = metadata.get("origin", np.array([0.0, 0.0, 0.0]))
                direction = metadata.get("direction", np.eye(3))

                affine = np.eye(4)
                affine[:3, :3] = direction * spacing[:, None]
                affine[:3, 3] = origin

                nii_img = nib.Nifti1Image(segmentation.astype(np.float32), affine)
                nib.save(nii_img, output_buffer)
                output_buffer.seek(0)

                st.download_button(
                    label="Download Segmentation Mask",
                    data=output_buffer.getvalue(),
                    file_name=f"{uploaded_file.name}_segmentation.nii.gz",
                    mime="application/gzip"
                )


if __name__ == "__main__":
    main()

