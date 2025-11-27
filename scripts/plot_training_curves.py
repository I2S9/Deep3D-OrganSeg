#!/usr/bin/env python3
"""Generate training analysis plots from training logs.

This script creates the standard R&D plots for training analysis:
- Training Loss / Validation Loss curves
- Training Dice / Validation Dice curves
- (Optional) Hausdorff distance by epoch

Usage:
    python scripts/plot_training_curves.py --log_path logs/training_history_20240101_120000.json
    python scripts/plot_training_curves.py --log_path logs/training_history_20240101_120000.json --output_dir outputs/
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution


def load_training_history(log_path: Path) -> Dict:
    """Load training history from JSON file.
    
    Args:
        log_path: Path to JSON file containing training history.
        
    Returns:
        Dictionary containing training history.
    """
    with open(log_path, 'r') as f:
        history = json.load(f)
    return history


def plot_loss_curves(
    history: Dict,
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 5),
) -> None:
    """Plot training and validation loss curves.
    
    Args:
        history: Training history dictionary.
        output_path: Optional path to save the figure.
        figsize: Figure size (width, height).
    """
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Training and Validation Loss', fontsize=16, fontweight='bold')
    
    # Training Loss
    if 'train_loss' in history and len(history['train_loss']) > 0:
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2, marker='o', markersize=3)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(fontsize=11)
        
        # Add statistics
        final_loss = history['train_loss'][-1]
        best_loss = min(history['train_loss'])
        axes[0].axhline(y=best_loss, color='green', linestyle='--', alpha=0.5, label=f'Best: {best_loss:.4f}')
        axes[0].text(0.02, 0.98, 
                    f'Initial: {history["train_loss"][0]:.4f}\nFinal: {final_loss:.4f}\nBest: {best_loss:.4f}',
                    transform=axes[0].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                    fontsize=10)
    else:
        axes[0].text(0.5, 0.5, 'No training loss data', 
                    ha='center', va='center', transform=axes[0].transAxes,
                    fontsize=12)
        axes[0].set_title('Training Loss', fontsize=14, fontweight='bold')
    
    # Validation Loss
    if 'val_loss' in history and len(history['val_loss']) > 0:
        axes[1].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, marker='s', markersize=3)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Validation Loss', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].legend(fontsize=11)
        
        # Add statistics
        final_loss = history['val_loss'][-1]
        best_loss = min(history['val_loss'])
        best_epoch = history['val_loss'].index(best_loss) + 1
        axes[1].axhline(y=best_loss, color='green', linestyle='--', alpha=0.5, label=f'Best: {best_loss:.4f}')
        axes[1].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)
        axes[1].text(0.02, 0.98, 
                    f'Initial: {history["val_loss"][0]:.4f}\nFinal: {final_loss:.4f}\nBest: {best_loss:.4f} (Epoch {best_epoch})',
                    transform=axes[1].transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                    fontsize=10)
    else:
        axes[1].text(0.5, 0.5, 'No validation loss data', 
                    ha='center', va='center', transform=axes[1].transAxes,
                    fontsize=12)
        axes[1].set_title('Validation Loss', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Loss curves saved to {output_path}")
    else:
        plt.savefig('loss_curves.png', dpi=300, bbox_inches='tight')
        print("Loss curves saved to loss_curves.png")
    
    plt.close()


def plot_dice_curves(
    history: Dict,
    output_path: Optional[Path] = None,
    figsize: tuple = (12, 5),
) -> None:
    """Plot training and validation Dice score curves.
    
    Args:
        history: Training history dictionary.
        output_path: Optional path to save the figure.
        figsize: Figure size (width, height).
    """
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle('Training and Validation Dice Score', fontsize=16, fontweight='bold')
    
    # Training Dice
    if 'train_dice' in history and len(history['train_dice']) > 0:
        axes[0].plot(epochs, history['train_dice'], 'b-', label='Training Dice', linewidth=2, marker='o', markersize=3)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Dice Score', fontsize=12)
        axes[0].set_title('Training Dice Score', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].legend(fontsize=11)
        axes[0].set_ylim([0, 1])
        
        # Add statistics
        final_dice = history['train_dice'][-1]
        best_dice = max(history['train_dice'])
        axes[0].axhline(y=best_dice, color='green', linestyle='--', alpha=0.5, label=f'Best: {best_dice:.4f}')
        axes[0].text(0.02, 0.02, 
                    f'Initial: {history["train_dice"][0]:.4f}\nFinal: {final_dice:.4f}\nBest: {best_dice:.4f}',
                    transform=axes[0].transAxes,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                    fontsize=10)
    else:
        axes[0].text(0.5, 0.5, 'No training Dice data', 
                    ha='center', va='center', transform=axes[0].transAxes,
                    fontsize=12)
        axes[0].set_title('Training Dice Score', fontsize=14, fontweight='bold')
        axes[0].set_ylim([0, 1])
    
    # Validation Dice
    if 'val_dice' in history and len(history['val_dice']) > 0:
        axes[1].plot(epochs, history['val_dice'], 'r-', label='Validation Dice', linewidth=2, marker='s', markersize=3)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Dice Score', fontsize=12)
        axes[1].set_title('Validation Dice Score', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].legend(fontsize=11)
        axes[1].set_ylim([0, 1])
        
        # Add statistics
        final_dice = history['val_dice'][-1]
        best_dice = max(history['val_dice'])
        best_epoch = history['val_dice'].index(best_dice) + 1
        axes[1].axhline(y=best_dice, color='green', linestyle='--', alpha=0.5, label=f'Best: {best_dice:.4f}')
        axes[1].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.5)
        axes[1].text(0.02, 0.02, 
                    f'Initial: {history["val_dice"][0]:.4f}\nFinal: {final_dice:.4f}\nBest: {best_dice:.4f} (Epoch {best_epoch})',
                    transform=axes[1].transAxes,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                    fontsize=10)
    else:
        axes[1].text(0.5, 0.5, 'No validation Dice data', 
                    ha='center', va='center', transform=axes[1].transAxes,
                    fontsize=12)
        axes[1].set_title('Validation Dice Score', fontsize=14, fontweight='bold')
        axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Dice curves saved to {output_path}")
    else:
        plt.savefig('dice_curves.png', dpi=300, bbox_inches='tight')
        print("Dice curves saved to dice_curves.png")
    
    plt.close()


def plot_combined_curves(
    history: Dict,
    output_path: Optional[Path] = None,
    figsize: tuple = (14, 10),
) -> None:
    """Plot combined training vs validation curves for Loss and Dice.
    
    Args:
        history: Training history dictionary.
        output_path: Optional path to save the figure.
        figsize: Figure size (width, height).
    """
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    fig.suptitle('Training vs Validation Performance', fontsize=16, fontweight='bold')
    
    # Loss comparison
    if 'train_loss' in history and 'val_loss' in history:
        if len(history['train_loss']) > 0 and len(history['val_loss']) > 0:
            axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2, alpha=0.8, marker='o', markersize=3)
            axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2, alpha=0.8, marker='s', markersize=3)
            axes[0].set_xlabel('Epoch', fontsize=12)
            axes[0].set_ylabel('Loss', fontsize=12)
            axes[0].set_title('Training vs Validation - Loss', fontsize=14, fontweight='bold')
            axes[0].grid(True, alpha=0.3, linestyle='--')
            axes[0].legend(fontsize=11, loc='upper right')
            
            # Add statistics
            train_final = history['train_loss'][-1]
            val_final = history['val_loss'][-1]
            val_best = min(history['val_loss'])
            axes[0].text(0.02, 0.98, 
                        f'Train Final: {train_final:.4f}\nVal Final: {val_final:.4f}\nVal Best: {val_best:.4f}',
                        transform=axes[0].transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                        fontsize=10)
    
    # Dice comparison
    if 'train_dice' in history and 'val_dice' in history:
        if len(history['train_dice']) > 0 and len(history['val_dice']) > 0:
            axes[1].plot(epochs, history['train_dice'], 'b-', label='Training Dice', linewidth=2, alpha=0.8, marker='o', markersize=3)
            axes[1].plot(epochs, history['val_dice'], 'r-', label='Validation Dice', linewidth=2, alpha=0.8, marker='s', markersize=3)
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Dice Score', fontsize=12)
            axes[1].set_title('Training vs Validation - Dice Score', fontsize=14, fontweight='bold')
            axes[1].grid(True, alpha=0.3, linestyle='--')
            axes[1].legend(fontsize=11, loc='lower right')
            axes[1].set_ylim([0, 1])
            
            # Add statistics
            train_final = history['train_dice'][-1]
            val_final = history['val_dice'][-1]
            val_best = max(history['val_dice'])
            gap = train_final - val_final
            axes[1].text(0.02, 0.98, 
                        f'Train Final: {train_final:.4f}\nVal Final: {val_final:.4f}\nVal Best: {val_best:.4f}\nGap: {gap:.4f}',
                        transform=axes[1].transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                        fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Combined curves saved to {output_path}")
    else:
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        print("Combined curves saved to training_curves.png")
    
    plt.close()


def plot_hausdorff_curve(
    history: Dict,
    output_path: Optional[Path] = None,
    figsize: tuple = (10, 6),
) -> None:
    """Plot Hausdorff distance curve (optional metric).
    
    Args:
        history: Training history dictionary.
        output_path: Optional path to save the figure.
        figsize: Figure size (width, height).
    """
    if 'val_hausdorff' not in history or len(history['val_hausdorff']) == 0:
        print("No Hausdorff distance data available. Skipping Hausdorff plot.")
        return
    
    epochs = range(1, len(history['val_hausdorff']) + 1)
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    fig.suptitle('Validation Hausdorff Distance', fontsize=16, fontweight='bold')
    
    ax.plot(epochs, history['val_hausdorff'], 'g-', label='Hausdorff Distance', linewidth=2, marker='o', markersize=3)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Hausdorff Distance (mm)', fontsize=12)
    ax.set_title('Validation Hausdorff Distance by Epoch', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=11)
    
    # Add statistics
    final_hd = history['val_hausdorff'][-1]
    best_hd = min(history['val_hausdorff'])
    best_epoch = history['val_hausdorff'].index(best_hd) + 1
    ax.axhline(y=best_hd, color='red', linestyle='--', alpha=0.5, label=f'Best: {best_hd:.4f}')
    ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5)
    ax.text(0.02, 0.98, 
            f'Initial: {history["val_hausdorff"][0]:.4f}\nFinal: {final_hd:.4f}\nBest: {best_hd:.4f} (Epoch {best_epoch})',
            transform=ax.transAxes,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
            fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Hausdorff curve saved to {output_path}")
    else:
        plt.savefig('hausdorff_curve.png', dpi=300, bbox_inches='tight')
        print("Hausdorff curve saved to hausdorff_curve.png")
    
    plt.close()


def main():
    """Main function to generate all training analysis plots."""
    parser = argparse.ArgumentParser(description='Generate training analysis plots')
    parser.add_argument('--log_path', type=str, required=True,
                       help='Path to training history JSON file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for plots (default: outputs)')
    parser.add_argument('--include_hausdorff', action='store_true',
                       help='Include Hausdorff distance plot (if available)')
    
    args = parser.parse_args()
    
    log_path = Path(args.log_path)
    if not log_path.exists():
        print(f"ERROR: Log file not found at {log_path}")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading training history from {log_path}...")
    history = load_training_history(log_path)
    
    print(f"Generating plots in {output_dir}...")
    
    # Generate all required plots
    plot_loss_curves(history, output_dir / 'loss_curves.png')
    plot_dice_curves(history, output_dir / 'dice_curves.png')
    plot_combined_curves(history, output_dir / 'training_curves.png')
    
    # Optional: Hausdorff distance
    if args.include_hausdorff:
        plot_hausdorff_curve(history, output_dir / 'hausdorff_curve.png')
    
    print("\nAll plots generated successfully!")
    print(f"Plots saved in: {output_dir}")


if __name__ == '__main__':
    main()

