"""3D U-Net architecture for organ segmentation.

This module implements a 3D U-Net with encoder-decoder architecture,
skip connections, and configurable channel dimensions.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Tuple

from src.models.layers.blocks import DoubleConv3d, Down3d, Up3d, OutConv3d


class UNet3D(nn.Module):
    """3D U-Net architecture for medical image segmentation.

    This implementation features:
    - Multi-resolution encoder with downsampling
    - Decoder with skip connections
    - Configurable channel dimensions
    - Weight initialization
    - GPU/CPU support
    - Model saving/loading
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        depth: int = 4,
        bilinear: bool = False,
    ):
        """Initialize the 3D U-Net model.

        Args:
            in_channels: Number of input channels (default: 1).
            out_channels: Number of output channels (default: 1 for binary segmentation).
            base_channels: Number of base channels in the first layer (default: 64).
            depth: Depth of the U-Net (number of downsampling levels, default: 4).
            bilinear: Whether to use bilinear upsampling instead of transposed conv (default: False).
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.depth = depth
        self.bilinear = bilinear

        factor = 2 if bilinear else 1

        self.inc = DoubleConv3d(in_channels, base_channels)

        self.down_layers = nn.ModuleList()
        channels = base_channels
        for i in range(depth):
            next_channels = channels * 2
            self.down_layers.append(Down3d(channels, next_channels))
            channels = next_channels

        self.up_layers = nn.ModuleList()
        for i in range(depth):
            if i == depth - 1:
                prev_channels = channels
            else:
                prev_channels = channels // factor
            next_channels = channels // 2
            self.up_layers.append(Up3d(channels, next_channels, bilinear=bilinear))
            channels = next_channels

        self.outc = OutConv3d(base_channels, out_channels)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Output tensor of shape (B, out_channels, D, H, W).
        """
        x1 = self.inc(x)

        skip_connections = [x1]

        for down in self.down_layers:
            x1 = down(x1)
            skip_connections.append(x1)

        skip_connections = skip_connections[:-1]
        skip_connections = skip_connections[::-1]

        for i, up in enumerate(self.up_layers):
            x1 = up(x1, skip_connections[i])

        logits = self.outc(x1)

        return logits

    def get_device(self) -> torch.device:
        """Get the device of the model parameters.

        Returns:
            Device where the model parameters are located.
        """
        return next(self.parameters()).device

    def to_device(self, device: torch.device) -> "UNet3D":
        """Move model to specified device.

        Args:
            device: Target device (cuda or cpu).

        Returns:
            Self for method chaining.
        """
        return self.to(device)

    def save(self, filepath: Path) -> None:
        """Save model state dictionary to file.

        Args:
            filepath: Path to save the model.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "base_channels": self.base_channels,
                "depth": self.depth,
                "bilinear": self.bilinear,
            },
            filepath,
        )

    @classmethod
    def load(
        cls,
        filepath: Path,
        device: Optional[torch.device] = None,
        strict: bool = True,
    ) -> "UNet3D":
        """Load model from saved checkpoint.

        Args:
            filepath: Path to the saved model.
            device: Device to load the model on (default: None, uses saved device).
            strict: Whether to strictly enforce that the keys match (default: True).

        Returns:
            Loaded UNet3D model instance.
        """
        filepath = Path(filepath)
        checkpoint = torch.load(filepath, map_location=device)

        model = cls(
            in_channels=checkpoint.get("in_channels", 1),
            out_channels=checkpoint.get("out_channels", 1),
            base_channels=checkpoint.get("base_channels", 64),
            depth=checkpoint.get("depth", 4),
            bilinear=checkpoint.get("bilinear", False),
        )

        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        if device is not None:
            model = model.to(device)

        return model

    def count_parameters(self) -> int:
        """Count the number of trainable parameters in the model.

        Returns:
            Total number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_info(self) -> dict:
        """Get model configuration information.

        Returns:
            Dictionary containing model configuration.
        """
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "base_channels": self.base_channels,
            "depth": self.depth,
            "bilinear": self.bilinear,
            "parameters": self.count_parameters(),
            "device": str(self.get_device()),
        }

