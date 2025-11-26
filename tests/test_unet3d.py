"""Tests for 3D U-Net model."""

import torch
import pytest
from pathlib import Path
import tempfile

from src.models import UNet3D


def test_unet3d_initialization():
    """Test model initialization."""
    model = UNet3D(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        depth=4,
    )

    assert model.in_channels == 1
    assert model.out_channels == 1
    assert model.base_channels == 64
    assert model.depth == 4


def test_unet3d_forward():
    """Test forward pass with a batch."""
    model = UNet3D(in_channels=1, out_channels=1, base_channels=32, depth=3)
    model.eval()

    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 64, 64, 32)

    with torch.no_grad():
        output = model(input_tensor)

    assert output.shape[0] == batch_size
    assert output.shape[1] == 1
    assert output.ndim == 5


def test_unet3d_output_shape():
    """Test that model returns tensor [B, 1, D, H, W]."""
    model = UNet3D(in_channels=1, out_channels=1, base_channels=32, depth=3)
    model.eval()

    batch_size = 2
    d, h, w = 64, 64, 32
    input_tensor = torch.randn(batch_size, 1, d, h, w)

    with torch.no_grad():
        output = model(input_tensor)

    assert output.shape == (batch_size, 1, d, h, w), \
        f"Expected shape ({batch_size}, 1, {d}, {h}, {w}), got {output.shape}"


def test_unet3d_forward_no_crash():
    """Test that a batch passes through the model without crashing."""
    model = UNet3D(in_channels=1, out_channels=1, base_channels=32, depth=3)
    model.eval()

    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 128, 128, 64)

    try:
        with torch.no_grad():
            output = model(input_tensor)
        assert output is not None
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    except Exception as e:
        pytest.fail(f"Forward pass crashed: {e}")


def test_unet3d_gpu_forward():
    """Test forward pass on GPU if available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device("cuda:0")
    model = UNet3D(in_channels=1, out_channels=1, base_channels=32, depth=3)
    model = model.to(device)
    model.eval()

    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 64, 64, 32, device=device)

    with torch.no_grad():
        output = model(input_tensor)

    assert output.device.type == "cuda"
    assert output.shape[0] == batch_size
    assert output.shape[1] == 1


def test_unet3d_cpu_forward():
    """Test forward pass on CPU."""
    device = torch.device("cpu")
    model = UNet3D(in_channels=1, out_channels=1, base_channels=32, depth=3)
    model = model.to(device)
    model.eval()

    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 64, 64, 32, device=device)

    with torch.no_grad():
        output = model(input_tensor)

    assert output.device.type == "cpu"
    assert output.shape[0] == batch_size
    assert output.shape[1] == 1


def test_unet3d_save_load():
    """Test model saving and loading."""
    model = UNet3D(in_channels=1, out_channels=1, base_channels=32, depth=3)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test_model.pth"

        model.save(filepath)
        assert filepath.exists()

        loaded_model = UNet3D.load(filepath)
        loaded_model.eval()
        assert loaded_model.in_channels == model.in_channels
        assert loaded_model.out_channels == model.out_channels
        assert loaded_model.base_channels == model.base_channels
        assert loaded_model.depth == model.depth

        input_tensor = torch.randn(1, 1, 64, 64, 32)
        with torch.no_grad():
            output_original = model(input_tensor)
            output_loaded = loaded_model(input_tensor)

        assert torch.allclose(output_original, output_loaded, atol=1e-3)


def test_unet3d_weight_initialization():
    """Test that weights are properly initialized."""
    model = UNet3D(in_channels=1, out_channels=1, base_channels=32, depth=3)

    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d):
            assert m.weight is not None
            assert not torch.isnan(m.weight).any()
        elif isinstance(m, torch.nn.BatchNorm3d):
            assert m.weight is not None
            assert m.bias is not None


def test_unet3d_different_channels():
    """Test model with different channel configurations."""
    configs = [
        (1, 1, 32),
        (1, 2, 64),
        (3, 1, 128),
    ]

    for in_ch, out_ch, base_ch in configs:
        model = UNet3D(
            in_channels=in_ch,
            out_channels=out_ch,
            base_channels=base_ch,
            depth=3,
        )
        model.eval()

        input_tensor = torch.randn(1, in_ch, 64, 64, 32)
        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape[1] == out_ch


def test_unet3d_different_depths():
    """Test model with different depths."""
    depths = [2, 3, 4, 5]

    for depth in depths:
        model = UNet3D(
            in_channels=1,
            out_channels=1,
            base_channels=32,
            depth=depth,
        )
        model.eval()

        input_tensor = torch.randn(1, 1, 64, 64, 32)
        with torch.no_grad():
            output = model(input_tensor)

        assert output.shape[1] == 1


def test_unet3d_parameter_count():
    """Test parameter counting."""
    model = UNet3D(in_channels=1, out_channels=1, base_channels=32, depth=3)
    param_count = model.count_parameters()

    assert param_count > 0
    assert isinstance(param_count, int)


def test_unet3d_model_info():
    """Test model info retrieval."""
    model = UNet3D(in_channels=1, out_channels=1, base_channels=32, depth=3)
    info = model.get_model_info()

    assert "in_channels" in info
    assert "out_channels" in info
    assert "base_channels" in info
    assert "depth" in info
    assert "parameters" in info
    assert "device" in info


def test_unet3d_device_management():
    """Test device management methods."""
    model = UNet3D(in_channels=1, out_channels=1, base_channels=32, depth=3)

    device = model.get_device()
    assert device.type in ["cpu", "cuda"]

    if torch.cuda.is_available():
        model_gpu = model.to_device(torch.device("cuda:0"))
        assert model_gpu.get_device().type == "cuda"

