"""Metrics for 3D segmentation evaluation.

This module provides Dice, IoU, and Hausdorff distance metrics
for evaluating segmentation performance.
"""

import torch
import torch.nn as nn
import numpy as np
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.transforms import AsDiscrete


def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Calculate Dice score.

    Args:
        pred: Predicted binary mask (B, C, D, H, W) or (B, C, D, H, W) logits.
        target: Ground truth binary mask (B, C, D, H, W).
        smooth: Smoothing factor to avoid division by zero (default: 1e-5).

    Returns:
        Dice score tensor (B, C).
    """
    if pred.shape[1] == 1:
        pred_binary = torch.sigmoid(pred) > 0.5
    else:
        pred_binary = torch.argmax(pred, dim=1, keepdim=True)

    pred_flat = pred_binary.view(pred_binary.shape[0], pred_binary.shape[1], -1).float()
    target_flat = target.view(target.shape[0], target.shape[1], -1).float()

    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5) -> torch.Tensor:
    """Calculate IoU (Intersection over Union) score.

    Args:
        pred: Predicted binary mask (B, C, D, H, W) or (B, C, D, H, W) logits.
        target: Ground truth binary mask (B, C, D, H, W).
        smooth: Smoothing factor to avoid division by zero (default: 1e-5).

    Returns:
        IoU score tensor (B, C).
    """
    if pred.shape[1] == 1:
        pred_binary = torch.sigmoid(pred) > 0.5
    else:
        pred_binary = torch.argmax(pred, dim=1, keepdim=True)

    pred_flat = pred_binary.view(pred_binary.shape[0], pred_binary.shape[1], -1).float()
    target_flat = target.view(target.shape[0], target.shape[1], -1).float()

    intersection = (pred_flat * target_flat).sum(dim=2)
    union = pred_flat.sum(dim=2) + target_flat.sum(dim=2) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth: float = 1e-5):
        """Initialize Dice loss.

        Args:
            smooth: Smoothing factor (default: 1e-5).
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate Dice loss.

        Args:
            pred: Predicted logits (B, C, D, H, W).
            target: Ground truth mask (B, C, D, H, W).

        Returns:
            Dice loss scalar tensor.
        """
        if pred.shape[1] == 1:
            pred_probs = torch.sigmoid(pred)
        else:
            pred_probs = torch.softmax(pred, dim=1)

        pred_flat = pred_probs.view(pred_probs.shape[0], pred_probs.shape[1], -1)
        target_flat = target.view(target.shape[0], target.shape[1], -1)

        intersection = (pred_flat * target_flat).sum(dim=2)
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice.mean()

        return loss


class DiceCELoss(nn.Module):
    """Combined Dice and Cross-Entropy loss.

    This is the recommended loss for segmentation tasks as it combines
    the benefits of Dice loss (handles class imbalance) and Cross-Entropy
    (provides stable gradients).
    """

    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5, smooth: float = 1e-5):
        """Initialize Dice + Cross-Entropy loss.

        Args:
            dice_weight: Weight for Dice loss component (default: 0.5).
            ce_weight: Weight for Cross-Entropy loss component (default: 0.5).
            smooth: Smoothing factor for Dice loss (default: 1e-5).
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.ce_loss = nn.BCEWithLogitsLoss() if dice_weight + ce_weight == 1.0 else nn.CrossEntropyLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate combined Dice + Cross-Entropy loss.

        Args:
            pred: Predicted logits (B, C, D, H, W).
            target: Ground truth mask (B, C, D, H, W).

        Returns:
            Combined loss scalar tensor.
        """
        dice_loss = self.dice_loss(pred, target)

        if pred.shape[1] == 1:
            ce_loss = nn.functional.binary_cross_entropy_with_logits(pred, target)
        else:
            target_long = target.argmax(dim=1) if target.shape[1] > 1 else target.squeeze(1).long()
            ce_loss = nn.functional.cross_entropy(pred, target_long)

        total_loss = self.dice_weight * dice_loss + self.ce_weight * ce_loss
        return total_loss


class SegmentationMetrics:
    """Container for segmentation metrics."""

    def __init__(self, include_hausdorff: bool = False):
        """Initialize metrics container.

        Args:
            include_hausdorff: Whether to compute Hausdorff distance (default: False).
        """
        self.include_hausdorff = include_hausdorff
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.iou_scores = []
        self.dice_scores = []
        if include_hausdorff:
            self.hausdorff_metric = HausdorffDistanceMetric(
                include_background=False, percentile=95, reduction="mean"
            )
            self.hausdorff_scores = []

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Update metrics with a batch of predictions and targets.

        Args:
            pred: Predicted logits or probabilities (B, C, D, H, W).
            target: Ground truth mask (B, C, D, H, W).
        """
        with torch.no_grad():
            if pred.shape[1] == 1:
                pred_binary = (torch.sigmoid(pred) > 0.5).float()
            else:
                pred_binary = torch.argmax(pred, dim=1, keepdim=True).float()

            dice_batch = dice_score(pred, target)
            iou_batch = iou_score(pred, target)

            self.dice_scores.append(dice_batch.cpu().numpy())
            self.iou_scores.append(iou_batch.cpu().numpy())

            if self.include_hausdorff:
                self.hausdorff_metric(pred_binary, target)

    def compute(self) -> dict:
        """Compute and return all metrics.

        Returns:
            Dictionary containing mean Dice, IoU, and optionally Hausdorff distance.
        """
        dice_mean = np.mean(np.concatenate(self.dice_scores))
        iou_mean = np.mean(np.concatenate(self.iou_scores))

        results = {
            "dice": float(dice_mean),
            "iou": float(iou_mean),
        }

        if self.include_hausdorff:
            hausdorff_mean = self.hausdorff_metric.aggregate().item()
            results["hausdorff"] = float(hausdorff_mean)
            self.hausdorff_metric.reset()

        return results

    def reset(self) -> None:
        """Reset all metrics."""
        self.dice_scores.clear()
        self.iou_scores.clear()
        self.dice_metric.reset()
        if self.include_hausdorff:
            self.hausdorff_scores.clear()
            self.hausdorff_metric.reset()

