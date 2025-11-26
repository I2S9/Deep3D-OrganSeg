"""Manual verification script for 3D U-Net model.

This script verifies:
- Batch passes through model without crash
- Model returns tensor [B, 1, D, H, W]
- Forward pass works on GPU if available
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import UNet3D


def verify_batch_forward():
    """Verify that a batch passes through the model without crash."""
    print("\n=== Testing Batch Forward Pass ===")

    model = UNet3D(in_channels=1, out_channels=1, base_channels=32, depth=3)
    model.eval()

    batch_sizes = [1, 2, 4]
    input_shapes = [(64, 64, 32), (128, 128, 64), (96, 96, 48)]

    for batch_size in batch_sizes:
        for d, h, w in input_shapes:
            input_tensor = torch.randn(batch_size, 1, d, h, w)

            try:
                with torch.no_grad():
                    output = model(input_tensor)

                assert output is not None, "Output should not be None"
                assert not torch.isnan(output).any(), "Output should not contain NaN"
                assert not torch.isinf(output).any(), "Output should not contain Inf"
                assert output.shape[0] == batch_size, \
                    f"Batch size mismatch: expected {batch_size}, got {output.shape[0]}"

                print(f"[OK] Batch size {batch_size}, shape ({d}, {h}, {w})")
            except Exception as e:
                print(f"[FAILED] Batch size {batch_size}, shape ({d}, {h}, {w}): {e}")
                raise

    print("[OK] All batch forward passes successful")


def verify_output_shape():
    """Verify that model returns tensor [B, 1, D, H, W]."""
    print("\n=== Testing Output Shape ===")

    model = UNet3D(in_channels=1, out_channels=1, base_channels=32, depth=3)
    model.eval()

    test_cases = [
        (2, 64, 64, 32),
        (4, 128, 128, 64),
        (1, 96, 96, 48),
    ]

    for batch_size, d, h, w in test_cases:
        input_tensor = torch.randn(batch_size, 1, d, h, w)

        with torch.no_grad():
            output = model(input_tensor)

        expected_shape = (batch_size, 1, d, h, w)
        assert output.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {output.shape}"

        assert output.ndim == 5, f"Output should be 5D, got {output.ndim}D"
        assert output.shape[1] == 1, f"Output channels should be 1, got {output.shape[1]}"

        print(f"[OK] Input shape ({batch_size}, 1, {d}, {h}, {w}) -> Output shape {output.shape}")

    print("[OK] All output shapes are correct")


def verify_gpu_forward():
    """Verify that forward pass works on GPU if available."""
    print("\n=== Testing GPU Forward Pass ===")

    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available, skipping GPU test")
        return

    device = torch.device("cuda:0")
    model = UNet3D(in_channels=1, out_channels=1, base_channels=32, depth=3)
    model = model.to(device)
    model.eval()

    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 64, 64, 32, device=device)

    with torch.no_grad():
        output = model(input_tensor)

    assert output.device.type == "cuda", f"Output should be on CUDA, got {output.device.type}"
    assert output.shape[0] == batch_size, "Batch size mismatch"
    assert output.shape[1] == 1, "Output channels should be 1"
    assert not torch.isnan(output).any(), "Output should not contain NaN"
    assert not torch.isinf(output).any(), "Output should not contain Inf"

    print(f"[OK] GPU forward pass successful")
    print(f"  Device: {output.device}")
    print(f"  Output shape: {output.shape}")


def verify_model_configuration():
    """Verify model configuration and features."""
    print("\n=== Testing Model Configuration ===")

    model = UNet3D(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        depth=4,
        bilinear=False,
    )

    info = model.get_model_info()
    print(f"Model configuration:")
    print(f"  Input channels: {info['in_channels']}")
    print(f"  Output channels: {info['out_channels']}")
    print(f"  Base channels: {info['base_channels']}")
    print(f"  Depth: {info['depth']}")
    print(f"  Parameters: {info['parameters']:,}")
    print(f"  Device: {info['device']}")

    assert info["in_channels"] == 1
    assert info["out_channels"] == 1
    assert info["base_channels"] == 64
    assert info["depth"] == 4
    assert info["parameters"] > 0

    print("[OK] Model configuration verified")


def verify_save_load():
    """Verify model saving and loading."""
    print("\n=== Testing Model Save/Load ===")

    model = UNet3D(in_channels=1, out_channels=1, base_channels=32, depth=3)
    model.eval()

    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    filepath = checkpoint_dir / "test_model.pth"

    model.save(filepath)
    assert filepath.exists(), "Model file should be created"

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

    assert torch.allclose(output_original, output_loaded, atol=1e-3), \
        "Loaded model should produce same output as original"

    print(f"[OK] Model save/load successful")
    print(f"  Saved to: {filepath}")


def verify_weight_initialization():
    """Verify that weights are properly initialized."""
    print("\n=== Testing Weight Initialization ===")

    model = UNet3D(in_channels=1, out_channels=1, base_channels=32, depth=3)

    conv_count = 0
    bn_count = 0

    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d):
            conv_count += 1
            assert m.weight is not None, "Conv3d weight should not be None"
            assert not torch.isnan(m.weight).any(), "Conv3d weight should not contain NaN"
            assert m.weight.std() > 0, "Conv3d weight should have non-zero std"
        elif isinstance(m, torch.nn.BatchNorm3d):
            bn_count += 1
            assert m.weight is not None, "BatchNorm3d weight should not be None"
            assert m.bias is not None, "BatchNorm3d bias should not be None"

    print(f"[OK] Weight initialization verified")
    print(f"  Conv3d layers: {conv_count}")
    print(f"  BatchNorm3d layers: {bn_count}")


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("3D U-Net Model Verification")
    print("=" * 60)

    try:
        verify_model_configuration()
        verify_weight_initialization()
        verify_batch_forward()
        verify_output_shape()
        verify_gpu_forward()
        verify_save_load()

        print("\n" + "=" * 60)
        print("[OK] All verifications passed!")
        print("=" * 60)
        print("\nModel is ready for training!")

    except Exception as e:
        print(f"\n[FAILED] Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

