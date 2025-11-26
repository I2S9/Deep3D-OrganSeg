"""First experimentation script for 3D organ segmentation.

This script runs a complete training experiment on a reduced subset,
visualizes results, and documents hyperparameters.
"""

import sys
from pathlib import Path
from typing import Optional
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
import nibabel as nib

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import UNet3D
from src.training.train import Trainer
from src.data.dataloader import create_dataloader, get_training_transforms, get_validation_transforms
from src.data import preprocessing


def create_experiment_data(num_train: int = 5, num_val: int = 2):
    """Create experimental dataset."""
    data_dir = Path("data/samples")
    data_dir.mkdir(parents=True, exist_ok=True)

    train_volumes = []
    train_masks = []
    val_volumes = []
    val_masks = []

    for i in range(num_train + num_val):
        vol_path = data_dir / f"exp_volume_{i}.nii.gz"
        mask_path = data_dir / f"exp_mask_{i}.nii.gz"

        if not vol_path.exists():
            volume = np.random.randn(96, 96, 48).astype(np.float32) * 150 - 500
            volume = np.clip(volume, -1000, 1000)

            mask = np.zeros((96, 96, 48), dtype=np.float32)
            mask[24:72, 24:72, 12:36] = 1.0

            affine = np.eye(4)
            preprocessing.save_preprocessed_volume(
                volume, vol_path,
                {"spacing": np.array([1.0, 1.0, 1.0]), "origin": np.array([0.0, 0.0, 0.0]), "direction": np.eye(3)}
            )
            preprocessing.save_preprocessed_volume(
                mask, mask_path,
                {"spacing": np.array([1.0, 1.0, 1.0]), "origin": np.array([0.0, 0.0, 0.0]), "direction": np.eye(3)}
            )

        if i < num_train:
            train_volumes.append(vol_path)
            train_masks.append(mask_path)
        else:
            val_volumes.append(vol_path)
            val_masks.append(mask_path)

    return train_volumes, train_masks, val_volumes, val_masks


def plot_training_curves(history: dict, output_path: Path):
    """Plot training curves for loss and metrics."""
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Training Curves", fontsize=16)

    axes[0, 0].plot(epochs, history["train_loss"], label="Train Loss", marker="o")
    if history["val_loss"]:
        axes[0, 0].plot(epochs, history["val_loss"], label="Val Loss", marker="s")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(epochs, history["train_dice"], label="Train Dice", marker="o")
    if history["val_dice"]:
        axes[0, 1].plot(epochs, history["val_dice"], label="Val Dice", marker="s")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Dice Score")
    axes[0, 1].set_title("Dice Score")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_ylim([0, 1])

    axes[1, 0].plot(epochs, history["train_iou"], label="Train IoU", marker="o")
    if history["val_iou"]:
        axes[1, 0].plot(epochs, history["val_iou"], label="Val IoU", marker="s")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("IoU Score")
    axes[1, 0].set_title("IoU Score")
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_ylim([0, 1])

    axes[1, 1].plot(epochs, history["learning_rate"], marker="o", color="green")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Learning Rate")
    axes[1, 1].set_title("Learning Rate Schedule")
    axes[1, 1].grid(True)
    axes[1, 1].set_yscale("log")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_segmentation(
    volume: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    slice_idx: Optional[int] = None,
):
    """Visualize segmentation results vs ground truth."""
    if slice_idx is None:
        slice_idx = volume.shape[2] // 2

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Segmentation Results - Slice {slice_idx}", fontsize=16)

    axes[0, 0].imshow(volume[:, :, slice_idx], cmap="gray")
    axes[0, 0].set_title("Input Volume")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(ground_truth[:, :, slice_idx], cmap="Reds", alpha=0.7)
    axes[0, 1].set_title("Ground Truth")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(prediction[:, :, slice_idx], cmap="Blues", alpha=0.7)
    axes[0, 2].set_title("Prediction")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(volume[:, :, slice_idx], cmap="gray")
    axes[1, 0].imshow(ground_truth[:, :, slice_idx], cmap="Reds", alpha=0.5)
    axes[1, 0].set_title("Volume + Ground Truth")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(volume[:, :, slice_idx], cmap="gray")
    axes[1, 1].imshow(prediction[:, :, slice_idx], cmap="Blues", alpha=0.5)
    axes[1, 1].set_title("Volume + Prediction")
    axes[1, 1].axis("off")

    overlap = ground_truth[:, :, slice_idx] * prediction[:, :, slice_idx]
    axes[1, 2].imshow(volume[:, :, slice_idx], cmap="gray")
    axes[1, 2].imshow(overlap, cmap="Greens", alpha=0.7)
    axes[1, 2].set_title("Overlap (Green)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def check_gpu_memory():
    """Check GPU memory usage if available."""
    if not torch.cuda.is_available():
        return {"available": False, "message": "CUDA not available"}

    device = torch.device("cuda:0")
    memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
    memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
    memory_total = torch.cuda.get_device_properties(device).total_memory / 1024**3

    return {
        "available": True,
        "allocated_gb": memory_allocated,
        "reserved_gb": memory_reserved,
        "total_gb": memory_total,
        "usage_percent": (memory_reserved / memory_total) * 100,
    }


def document_hyperparameters(config: dict, output_path: Path):
    """Document hyperparameters used in the experiment."""
    doc = {
        "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hyperparameters": config,
        "model_config": config.get("model_config", {}),
        "training_config": {
            k: v
            for k, v in config.items()
            if k not in ["model_config"]
        },
    }

    with open(output_path, "w") as f:
        json.dump(doc, f, indent=2)


def run_experiment():
    """Run complete experimentation."""
    print("=" * 60)
    print("First Experimentation - 3D Organ Segmentation")
    print("=" * 60)

    experiment_dir = Path("experiments/exp_001")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("\n1. Creating experimental dataset...")
    train_volumes, train_masks, val_volumes, val_masks = create_experiment_data(num_train=5, num_val=2)
    print(f"   Training samples: {len(train_volumes)}")
    print(f"   Validation samples: {len(val_volumes)}")

    print("\n2. Checking GPU memory...")
    gpu_info = check_gpu_memory()
    if gpu_info["available"]:
        print(f"   GPU Memory: {gpu_info['allocated_gb']:.2f} GB allocated")
        print(f"   GPU Memory: {gpu_info['reserved_gb']:.2f} GB reserved")
        print(f"   GPU Memory: {gpu_info['total_gb']:.2f} GB total")
        print(f"   GPU Usage: {gpu_info['usage_percent']:.1f}%")
    else:
        print(f"   {gpu_info['message']}")

    print("\n3. Setting up model and training configuration...")
    hyperparameters = {
        "model_config": {
            "in_channels": 1,
            "out_channels": 1,
            "base_channels": 32,
            "depth": 3,
        },
        "num_epochs": 10,
        "batch_size": 2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-5,
        "scheduler_type": "cosine",
        "use_amp": False,
    }

    document_hyperparameters(hyperparameters, experiment_dir / "hyperparameters.json")
    print("   Hyperparameters documented")

    print("\n4. Creating model...")
    model = UNet3D(**hyperparameters["model_config"])
    print(f"   Model parameters: {model.count_parameters():,}")

    print("\n5. Creating data loaders...")
    train_transform = get_training_transforms()
    train_loader = create_dataloader(
        volume_paths=train_volumes,
        mask_paths=train_masks,
        batch_size=hyperparameters["batch_size"],
        transform=train_transform,
        num_workers=0,
        shuffle=True,
    )

    val_transform = get_validation_transforms()
    val_loader = create_dataloader(
        volume_paths=val_volumes,
        mask_paths=val_masks,
        batch_size=hyperparameters["batch_size"],
        transform=val_transform,
        num_workers=0,
        shuffle=False,
    )

    print("\n6. Creating trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=hyperparameters["learning_rate"],
        weight_decay=hyperparameters["weight_decay"],
        scheduler_type=hyperparameters["scheduler_type"],
        use_amp=hyperparameters["use_amp"],
        checkpoint_dir=experiment_dir / "checkpoints",
        log_dir=experiment_dir / "logs",
    )

    print("\n7. Starting training...")
    trainer.train(
        num_epochs=hyperparameters["num_epochs"],
        save_every=2,
        save_best=True,
    )

    print("\n8. Analyzing training results...")
    history = trainer.history

    train_dice_final = history["train_dice"][-1]
    val_dice_final = history["val_dice"][-1] if history["val_dice"] else None
    train_loss_final = history["train_loss"][-1]
    val_loss_final = history["val_loss"][-1] if history["val_loss"] else None

    print(f"   Final Train Loss: {train_loss_final:.4f}")
    print(f"   Final Train Dice: {train_dice_final:.4f}")
    if val_loss_final:
        print(f"   Final Val Loss: {val_loss_final:.4f}")
    if val_dice_final:
        print(f"   Final Val Dice: {val_dice_final:.4f}")

    convergence_check = train_loss_final < history["train_loss"][0]
    dice_improvement = train_dice_final > history["train_dice"][0] if len(history["train_dice"]) > 0 else False

    print(f"\n9. Convergence analysis:")
    print(f"   Loss decreased: {convergence_check}")
    print(f"   Dice improved: {dice_improvement}")
    if convergence_check and dice_improvement:
        print("   [OK] Training converged successfully")
    else:
        print("   [WARNING] Training may not have converged")

    print("\n10. Plotting training curves...")
    plot_training_curves(history, experiment_dir / "training_curves.png")
    print("   Training curves saved")

    print("\n11. Visualizing segmentation results and checking coherence...")
    best_model_path = experiment_dir / "checkpoints" / "best_model.pth"
    avg_dice = 0.0
    
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location="cpu")
        model_config = checkpoint.get("model_config", {})
        model_config_filtered = {
            k: v for k, v in model_config.items()
            if k in ["in_channels", "out_channels", "base_channels", "depth", "bilinear"]
        }
        model = UNet3D(**model_config_filtered)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to(trainer.device)

        val_transform = get_validation_transforms()
        coherence_scores = []
        
        for idx, (val_vol, val_mask) in enumerate(zip(val_volumes, val_masks)):
            data = val_transform({"image": str(val_vol), "label": str(val_mask)})
            
            with torch.no_grad():
                input_tensor = data["image"].unsqueeze(0).to(trainer.device)
                output = model(input_tensor)
                prediction = torch.sigmoid(output).cpu().squeeze().numpy()
                prediction_binary = (prediction > 0.5).astype(np.float32)
                
                mask_img = nib.load(str(val_mask))
                ground_truth = mask_img.get_fdata()
                
                from src.training.metrics import dice_score
                pred_tensor = torch.from_numpy(prediction_binary).unsqueeze(0).unsqueeze(0).float()
                target_tensor = torch.from_numpy(ground_truth).unsqueeze(0).unsqueeze(0).float()
                dice_val = dice_score(pred_tensor, target_tensor).item()
                coherence_scores.append(dice_val)
            
            if idx == 0:
                volume_img = nib.load(str(val_vol))
                volume = volume_img.get_fdata()
                
                visualize_segmentation(
                    volume,
                    ground_truth,
                    prediction_binary,
                    experiment_dir / "segmentation_visualization.png",
                )
                print(f"   Segmentation visualization saved (Dice: {dice_val:.4f})")
        
        avg_dice = np.mean(coherence_scores)
        print(f"\n   Segmentation coherence analysis:")
        print(f"   Average Dice on validation set: {avg_dice:.4f}")
        print(f"   Dice range: [{min(coherence_scores):.4f}, {max(coherence_scores):.4f}]")
        
        if avg_dice > 0.5:
            print("   [OK] Segmentation is coherent (Dice > 0.5)")
        elif avg_dice > 0.3:
            print("   [WARNING] Segmentation is partially coherent (Dice > 0.3)")
        else:
            print("   [WARNING] Segmentation may not be coherent (Dice < 0.3)")

    print("\n13. Final GPU memory check...")
    final_gpu_info = check_gpu_memory()
    if final_gpu_info["available"]:
        print(f"   Final GPU Memory: {final_gpu_info['allocated_gb']:.2f} GB allocated")
        print(f"   Final GPU Memory: {final_gpu_info['reserved_gb']:.2f} GB reserved")
        print(f"   Final GPU Usage: {final_gpu_info['usage_percent']:.1f}%")
        
        if final_gpu_info['usage_percent'] < 90:
            print("   [OK] GPU memory usage is reasonable")
        else:
            print("   [WARNING] GPU memory usage is high (>90%)")
    else:
        print(f"   {final_gpu_info['message']}")
        print("   [INFO] Running on CPU - memory check skipped")

    print("\n" + "=" * 60)
    print("Experiment Summary")
    print("=" * 60)
    print(f"\nTraining Results:")
    print(f"  - Final Train Dice: {train_dice_final:.4f}")
    print(f"  - Final Val Dice: {val_dice_final:.4f}")
    print(f"  - Convergence: {'Yes' if convergence_check and dice_improvement else 'Partial'}")
    print(f"  - Segmentation Coherence: {'Good' if avg_dice > 0.5 else 'Needs improvement'}")
    
    print(f"\nResults saved to: {experiment_dir}")
    print(f"  - Training curves: {experiment_dir / 'training_curves.png'}")
    print(f"  - Segmentation visualization: {experiment_dir / 'segmentation_visualization.png'}")
    print(f"  - Hyperparameters: {experiment_dir / 'hyperparameters.json'}")
    print(f"  - Training history: {experiment_dir / 'logs'}")
    print(f"  - Model checkpoints: {experiment_dir / 'checkpoints'}")
    
    print("\n" + "=" * 60)
    print("[OK] Experiment completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        run_experiment()
    except Exception as e:
        print(f"\n[FAILED] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

