"""Optional numerical check against the real UNet export and TensorRT engine."""

import pytest
import torch
from torch.nn import functional as F


def test_torch_unet_matches_tensorrt_across_temporal_frames(monkeypatch):
    from jasna import accelerator
    from jasna.engine_paths import UNET4X_ONNX_PATH, get_unet4x_engine_path
    from jasna.restorer.unet4x_secondary_restorer import Unet4xSecondaryRestorer

    if not torch.cuda.is_available() or not accelerator.is_nvidia_device():
        pytest.skip("TensorRT reference needs an NVIDIA GPU")
    if not UNET4X_ONNX_PATH.is_file() or not get_unet4x_engine_path().is_file():
        pytest.skip("needs the UNet ONNX model and its TensorRT engine")
    pytest.importorskip("tensorrt")
    device = torch.device("cuda:0")
    reference = Unet4xSecondaryRestorer(device=device)
    # Select the PyTorch backend on the same GPU to compare numerical behavior.
    monkeypatch.setattr(accelerator, "is_amd_device", lambda _device: True)
    monkeypatch.setattr(accelerator, "has_triton", lambda: False)
    candidate = Unet4xSecondaryRestorer(device=device)
    generator = torch.Generator().manual_seed(83)
    base = F.interpolate(
        torch.rand(1, 3, 64, 64, generator=generator),
        size=(256, 256), mode="bicubic", align_corners=False,
    )
    frames = (base + torch.randn(5, 3, 256, 256, generator=generator) * 0.02).clamp(0, 1)
    try:
        expected = reference.restore(frames, keep_start=0, keep_end=5)
        actual = candidate.restore(frames, keep_start=0, keep_end=5)
        assert len(actual) == len(expected) == 5
        for left, right in zip(actual, expected):
            mse = ((left.float() - right.float()) / 255).square().mean()
            assert float(-10 * mse.log10()) > 50
    finally:
        candidate.close()
        reference.close()
