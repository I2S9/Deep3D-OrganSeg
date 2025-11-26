"""Building blocks for 3D U-Net architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3d(nn.Module):
    """Double 3D convolution block with batch normalization and ReLU.

    This block consists of two 3D convolutions, each followed by batch normalization
    and ReLU activation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        """Initialize the double convolution block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel (default: 3).
            padding: Padding size (default: 1).
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Output tensor of shape (B, out_channels, D, H, W).
        """
        return self.conv(x)


class Down3d(nn.Module):
    """Downsampling block with max pooling and double convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ):
        """Initialize the downsampling block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel (default: 3).
            padding: Padding size (default: 1).
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3d(in_channels, out_channels, kernel_size, padding),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Output tensor of shape (B, out_channels, D/2, H/2, W/2).
        """
        return self.maxpool_conv(x)


class Up3d(nn.Module):
    """Upsampling block with transposed convolution and double convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        bilinear: bool = False,
    ):
        """Initialize the upsampling block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolution kernel (default: 3).
            padding: Padding size (default: 1).
            bilinear: Whether to use bilinear upsampling instead of transposed conv (default: False).
        """
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = DoubleConv3d(in_channels, out_channels, kernel_size, padding)
        else:
            self.up = nn.ConvTranspose3d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv3d(in_channels, out_channels, kernel_size, padding)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection.

        Args:
            x1: Input tensor from decoder path (B, C, D, H, W).
            x2: Skip connection tensor from encoder path (B, C, D', H', W').

        Returns:
            Output tensor after upsampling and concatenation.
        """
        x1 = self.up(x1)

        diff_d = x2.size()[2] - x1.size()[2]
        diff_h = x2.size()[3] - x1.size()[3]
        diff_w = x2.size()[4] - x1.size()[4]

        x1 = F.pad(
            x1,
            [
                diff_w // 2,
                diff_w - diff_w // 2,
                diff_h // 2,
                diff_h - diff_h // 2,
                diff_d // 2,
                diff_d - diff_d // 2,
            ],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3d(nn.Module):
    """Output convolution layer for segmentation."""

    def __init__(self, in_channels: int, out_channels: int):
        """Initialize the output convolution layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels (typically 1 for binary segmentation).
        """
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, D, H, W).

        Returns:
            Output tensor of shape (B, out_channels, D, H, W).
        """
        return self.conv(x)

