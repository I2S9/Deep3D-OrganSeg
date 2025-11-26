# Deep3D-OrganSeg

3D U-Net-based deep learning pipeline for automatic organ segmentation in CT scans using PyTorch and MONAI.

## Table of Contents

- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Clinical Prototype](#clinical-prototype)
- [Project Structure](#project-structure)
- [Testing](#testing)

## Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM (for large volumes)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Deep3D-OrganSeg
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dependencies

- **PyTorch** (>=2.0.0): Deep learning framework
- **MONAI** (>=1.3.0): Medical imaging transforms and datasets
- **SimpleITK** (>=2.3.0): Medical image I/O
- **NiBabel** (>=5.0.0): NIfTI file handling
- **pydicom** (>=2.4.0): DICOM file handling
- **Streamlit** (>=1.28.0): Web interface for clinical prototype
- **matplotlib**, **napari**: Visualization

## Dataset Preparation

### Data Format

The pipeline supports:
- **NIfTI** (.nii, .nii.gz): Recommended format
- **DICOM**: Raw CT scan directories

### Directory Structure

Organize your data as follows:

```
data/
├── raw/
│   ├── volumes/
│   │   ├── patient_001/
│   │   │   └── [DICOM files or .nii.gz]
│   │   └── patient_002/
│   └── masks/
│       ├── patient_001.nii.gz
│       └── patient_002.nii.gz
└── processed/
    ├── volumes/
    └── masks/
```

### Preprocessing

Preprocess your dataset using the provided script:

```bash
python scripts/prepare_data.py \
    --input-dir data/raw/volumes \
    --mask-dir data/raw/masks \
    --output-dir data/processed \
    --target-spacing 1.0 1.0 1.0
```

**Preprocessing steps:**
- Isotropic resampling (default: 1×1×1 mm)
- Automatic cropping of empty regions
- HU normalization (clamped to [-1000, 1000])
- Z-score normalization

**Options:**
- `--target-spacing`: Target voxel spacing (default: 1.0 1.0 1.0)
- `--auto-crop`: Enable automatic cropping (default: True)
- `--hu-min`, `--hu-max`: HU value range (default: -1000, 1000)

## Training

### Quick Start

Run a complete training experiment:

```bash
python scripts/run_experiment.py
```

This script will:
1. Create a small synthetic dataset for testing
2. Train the model
3. Generate training curves
4. Visualize segmentation results

### Custom Training

For custom training, use the training module:

```python
from src.training import Trainer, train_model
from src.models import UNet3D
from src.data import dataloader

# Create model
model = UNet3D(
    in_channels=1,
    out_channels=1,
    base_channels=64,
    depth=4
)

# Create data loaders
train_loader = dataloader.create_dataloader(
    volume_paths=train_volumes,
    mask_paths=train_masks,
    batch_size=2,
    transform=dataloader.get_training_transforms(),
    shuffle=True
)

# Train
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=1e-4,
    use_amp=True
)

trainer.train(num_epochs=100, save_every=10)
```

### Training Configuration

**Model Architecture:**
- `in_channels`: Input channels (1 for grayscale CT)
- `out_channels`: Output channels (1 for binary segmentation)
- `base_channels`: Base channel count (64 recommended)
- `depth`: U-Net depth (4 recommended)

**Training Parameters:**
- Learning rate: 1e-4 (AdamW optimizer)
- Batch size: 2-4 (depending on GPU memory)
- Loss: Dice + Cross-Entropy (combined)
- Scheduler: CosineAnnealingLR or ReduceLROnPlateau
- AMP: Enabled by default (faster training)

**Data Augmentation:**
- Random rotations (90°)
- Random flips
- Random zoom (0.9-1.1)
- Gaussian noise
- Intensity shifts

### Monitoring

Training metrics are logged to:
- `logs/training_history.json`: Training history
- `checkpoints/`: Model checkpoints
- TensorBoard logs (optional)

## Inference

### Command Line

Run inference on a single volume:

```bash
python scripts/run_inference.py \
    checkpoints/best_model.pth \
    data/test/volume.nii.gz \
    --output outputs/segmentation.nii.gz \
    --threshold 0.5 \
    --visualize
```

**Options:**
- `--output`: Output path (default: `<volume_name>_segmentation.nii.gz`)
- `--threshold`: Segmentation threshold (default: 0.5)
- `--no-morphology`: Disable morphological post-processing
- `--visualize`: Generate visualization images

### Batch Inference

Process multiple volumes:

```python
from src.inference import predict

results = predict.predict_batch(
    checkpoint_path="checkpoints/best_model.pth",
    volume_paths=["vol1.nii.gz", "vol2.nii.gz"],
    output_dir="outputs/"
)
```

### Python API

```python
from src.inference import predict, visualize
from pathlib import Path

# Load model
model = predict.load_model(Path("checkpoints/best_model.pth"))

# Preprocess volume
volume, metadata = predict.preprocess_for_inference(
    Path("data/test/volume.nii.gz")
)

# Predict
prediction = predict.predict_volume(model, volume)

# Post-process
segmentation = predict.postprocess_segmentation(
    prediction,
    threshold=0.5,
    apply_morphology=True
)

# Visualize
visualize.visualize_2d_slices(
    volume,
    segmentation,
    output_path=Path("outputs/slices.png")
)
```

## Clinical Prototype

A Streamlit web application for easy clinical testing:

```bash
streamlit run app.py
```

**Features:**
- Upload NIfTI volumes via web interface
- Real-time segmentation
- Interactive slice navigation (Axial, Coronal, Sagittal)
- Segmentation overlay visualization
- Download segmentation masks

**Usage:**
1. Start the app: `streamlit run app.py`
2. Configure model checkpoint path in sidebar
3. Upload a volume (.nii or .nii.gz)
4. Click "Run Segmentation"
5. View results and download segmentation

## Project Structure

```
Deep3D-OrganSeg/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── .cursorrules              # Project guidelines
├── .gitignore               # Git ignore rules
│
├── src/                      # Source code
│   ├── data/
│   │   ├── preprocessing.py  # Data preprocessing (DICOM/NIfTI loading, resampling, normalization)
│   │   └── dataloader.py    # MONAI datasets and data loaders
│   │
│   ├── models/
│   │   ├── unet3d.py        # 3D U-Net architecture
│   │   └── layers/
│   │       └── blocks.py    # Building blocks (DoubleConv, Down, Up, OutConv)
│   │
│   ├── training/
│   │   ├── train.py         # Training loop (Trainer class)
│   │   └── metrics.py       # Loss functions and metrics (Dice, IoU, Hausdorff)
│   │
│   ├── inference/
│   │   ├── predict.py       # Inference pipeline (model loading, preprocessing, prediction)
│   │   └── visualize.py     # Visualization functions (2D slices, 3D rendering)
│   │
│   └── utils/               # Utility functions
│
├── scripts/                  # Executable scripts
│   ├── prepare_data.py      # Data preprocessing script
│   ├── run_experiment.py     # Complete training experiment
│   ├── run_inference.py     # Command-line inference
│   └── test_*.py            # Manual test scripts
│
├── tests/                    # Unit tests
│   ├── test_preprocessing.py
│   ├── test_dataloader.py
│   ├── test_unet3d.py
│   └── test_inference.py
│
├── configs/                  # Configuration files (optional)
│   ├── train_config.yaml
│   ├── model_config.yaml
│   └── data_config.yaml
│
├── data/                      # Data directories
│   ├── raw/                  # Raw data (DICOM, original NIfTI)
│   ├── processed/            # Preprocessed data
│   └── samples/              # Sample data for testing
│
├── checkpoints/              # Trained models
├── logs/                     # Training logs
├── outputs/                  # Inference outputs
│
├── notebooks/                 # Jupyter notebooks (exploration, visualization)
└── app.py                    # Streamlit clinical prototype
```

## Testing

Run all unit tests:

```bash
python -m pytest tests/ -v
```

**Test Coverage:**
- `test_preprocessing.py`: Data loading, resampling, normalization, cropping
- `test_dataloader.py`: Dataset creation, transforms, batch generation
- `test_unet3d.py`: Model architecture, forward pass, save/load
- `test_inference.py`: Model loading, preprocessing, prediction, visualization

**Expected Output:**
```
37 passed, 1 skipped (GPU test skipped if CUDA unavailable)
```

## Key Features

- **3D U-Net Architecture**: Encoder-decoder with skip connections
- **Medical Image Support**: DICOM and NIfTI formats
- **Robust Preprocessing**: Isotropic resampling, HU normalization, auto-cropping
- **Data Augmentation**: 3D transforms (rotation, flip, zoom, noise)
- **Training Pipeline**: AMP, learning rate scheduling, checkpointing
- **Inference Pipeline**: Complete preprocessing, prediction, post-processing
- **Visualization**: 2D slices, 3D rendering (Napari, matplotlib)
- **Clinical Prototype**: Streamlit web interface

## License

[Specify your license here]

## Citation

If you use this code, please cite:

```bibtex
@software{deep3d_organseg,
  title = {Deep3D-OrganSeg: 3D U-Net Pipeline for Organ Segmentation},
  author = {[Your Name]},
  year = {2024},
  url = {[Repository URL]}
}
```
