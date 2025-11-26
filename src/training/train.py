"""Training loop for 3D U-Net organ segmentation.

This module provides a complete training pipeline with AMP, optimizers,
schedulers, and comprehensive metrics tracking.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm import tqdm
import json
from datetime import datetime

from src.models import UNet3D
from src.data.dataloader import create_dataloader, get_training_transforms, get_validation_transforms
from src.training.metrics import DiceCELoss, SegmentationMetrics


class Trainer:
    """Trainer class for 3D U-Net segmentation."""

    def __init__(
        self,
        model: UNet3D,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        device: Optional[torch.device] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        scheduler_type: str = "cosine",
        use_amp: bool = True,
        checkpoint_dir: Path = Path("checkpoints"),
        log_dir: Path = Path("logs"),
    ):
        """Initialize trainer.

        Args:
            model: UNet3D model instance.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader (optional).
            device: Device to train on (default: cuda if available, else cpu).
            learning_rate: Initial learning rate (default: 1e-4).
            weight_decay: Weight decay for AdamW (default: 1e-5).
            scheduler_type: Type of scheduler ('cosine' or 'plateau', default: 'cosine').
            use_amp: Whether to use Automatic Mixed Precision (default: True).
            checkpoint_dir: Directory to save checkpoints (default: checkpoints/).
            log_dir: Directory to save logs (default: logs/).
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = use_amp and torch.cuda.is_available()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)

        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.scheduler_type = scheduler_type
        if scheduler_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=len(train_loader) * 100,
                eta_min=1e-6,
            )
        elif scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                verbose=True,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        self.criterion = DiceCELoss(dice_weight=0.5, ce_weight=0.5)
        self.scaler = GradScaler() if self.use_amp else None

        self.train_metrics = SegmentationMetrics(include_hausdorff=False)
        self.val_metrics = SegmentationMetrics(include_hausdorff=True)

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.history = {
            "train_loss": [],
            "train_dice": [],
            "train_iou": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
            "val_hausdorff": [],
            "learning_rate": [],
        }

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary containing training metrics.
        """
        self.model.train()
        self.train_metrics.reset()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            if self.scheduler_type == "cosine":
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            self.train_metrics.update(outputs, labels)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        metrics = self.train_metrics.compute()

        return {
            "loss": avg_loss,
            "dice": metrics["dice"],
            "iou": metrics["iou"],
        }

    @torch.no_grad()
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model.

        Args:
            epoch: Current epoch number.

        Returns:
            Dictionary containing validation metrics.
        """
        if self.val_loader is None:
            return {}

        self.model.eval()
        self.val_metrics.reset()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            num_batches += 1

            self.val_metrics.update(outputs, labels)

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / num_batches
        metrics = self.val_metrics.compute()

        results = {
            "loss": avg_loss,
            "dice": metrics["dice"],
            "iou": metrics["iou"],
        }

        if "hausdorff" in metrics:
            results["hausdorff"] = metrics["hausdorff"]

        if self.scheduler_type == "plateau":
            self.scheduler.step(metrics["dice"])

        return results

    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "history": self.history,
            "model_config": self.model.get_model_info(),
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)

        latest_path = self.checkpoint_dir / "latest_checkpoint.pth"
        torch.save(checkpoint, latest_path)

    def train(self, num_epochs: int, save_every: int = 10, save_best: bool = True) -> None:
        """Train the model for multiple epochs.

        Args:
            num_epochs: Number of epochs to train.
            save_every: Save checkpoint every N epochs (default: 10).
            save_best: Whether to save best model based on validation Dice (default: True).
        """
        best_val_dice = 0.0

        print(f"Starting training on device: {self.device}")
        print(f"Using AMP: {self.use_amp}")
        print(f"Scheduler: {self.scheduler_type}")
        print(f"Model parameters: {self.model.count_parameters():,}")

        for epoch in range(1, num_epochs + 1):
            train_results = self.train_epoch(epoch)
            val_results = self.validate(epoch)

            self.history["train_loss"].append(train_results["loss"])
            self.history["train_dice"].append(train_results["dice"])
            self.history["train_iou"].append(train_results["iou"])
            self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])

            if val_results:
                self.history["val_loss"].append(val_results["loss"])
                self.history["val_dice"].append(val_results["dice"])
                self.history["val_iou"].append(val_results["iou"])
                if "hausdorff" in val_results:
                    self.history["val_hausdorff"].append(val_results["hausdorff"])

                print(
                    f"Epoch {epoch}/{num_epochs} - "
                    f"Train Loss: {train_results['loss']:.4f}, "
                    f"Train Dice: {train_results['dice']:.4f}, "
                    f"Val Loss: {val_results['loss']:.4f}, "
                    f"Val Dice: {val_results['dice']:.4f}"
                )

                if save_best and val_results["dice"] > best_val_dice:
                    best_val_dice = val_results["dice"]
                    self.save_checkpoint(epoch, is_best=True)
                    print(f"New best model saved! Val Dice: {best_val_dice:.4f}")
            else:
                print(
                    f"Epoch {epoch}/{num_epochs} - "
                    f"Train Loss: {train_results['loss']:.4f}, "
                    f"Train Dice: {train_results['dice']:.4f}"
                )

            if epoch % save_every == 0:
                self.save_checkpoint(epoch, is_best=False)

        self.save_history()

    def save_history(self) -> None:
        """Save training history to JSON file."""
        history_path = self.log_dir / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")


def train_model(
    train_volume_paths: list,
    train_mask_paths: list,
    val_volume_paths: Optional[list] = None,
    val_mask_paths: Optional[list] = None,
    model_config: Optional[Dict[str, Any]] = None,
    num_epochs: int = 100,
    batch_size: int = 2,
    learning_rate: float = 1e-4,
    use_patches: bool = False,
    patch_size: Optional[tuple] = None,
    **kwargs,
) -> Trainer:
    """Convenience function to train a model.

    Args:
        train_volume_paths: List of training volume paths.
        train_mask_paths: List of training mask paths.
        val_volume_paths: List of validation volume paths (optional).
        val_mask_paths: List of validation mask paths (optional).
        model_config: Model configuration dictionary (optional).
        num_epochs: Number of epochs to train (default: 100).
        batch_size: Batch size (default: 2).
        learning_rate: Learning rate (default: 1e-4).
        use_patches: Whether to use patches (default: False).
        patch_size: Patch size if using patches (default: None).
        **kwargs: Additional arguments for Trainer.

    Returns:
        Trained Trainer instance.
    """
    if model_config is None:
        model_config = {}

    model = UNet3D(**model_config)

    train_transform = get_training_transforms()
    train_loader = create_dataloader(
        volume_paths=train_volume_paths,
        mask_paths=train_mask_paths,
        batch_size=batch_size,
        transform=train_transform,
        use_patches=use_patches,
        patch_size=patch_size,
        shuffle=True,
    )

    val_loader = None
    if val_volume_paths and val_mask_paths:
        val_transform = get_validation_transforms()
        val_loader = create_dataloader(
            volume_paths=val_volume_paths,
            mask_paths=val_mask_paths,
            batch_size=batch_size,
            transform=val_transform,
            use_patches=use_patches,
            patch_size=patch_size,
            shuffle=False,
        )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate,
        **kwargs,
    )

    trainer.train(num_epochs=num_epochs)

    return trainer

