"""The eager RGB↔YUV conversions must not allocate per frame.

Issue #252: on ROCm the per-frame conversion temporaries were allocated on the
encoder's private stream from its worker thread, and the caching allocator
handed those blocks to the restorer's fp16 tensors while the conversion was
still in flight, which destroyed whole frames.
"""
import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace

from jasna.media import rgb_to_yuv as rgb_to_yuv_module
from jasna.media import yuv_to_rgb as yuv_to_rgb_module
from jasna.media.rgb_to_yuv import RgbToYuvConverter
from jasna.media.yuv_to_rgb import YuvToRgbConverter

CPU = torch.device("cpu")
VARIANTS = ["nv12_bt709_limited", "nv12_bt601_full", "p010_bt709_limited", "p010_bt2020_full"]

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _frame(height: int, width: int, device: torch.device, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        0, 256, (3, height, width), generator=generator, dtype=torch.uint8
    ).to(device)


def _planes(height: int, width: int, device: torch.device, seed: int):
    generator = torch.Generator().manual_seed(seed)
    y = torch.randint(0, 256, (height, width), generator=generator, dtype=torch.uint8)
    uv = torch.randint(
        0, 256, (height // 2, width // 2, 2), generator=generator, dtype=torch.uint8
    )
    return y.to(device), uv.to(device)


@pytest.mark.parametrize("variant", VARIANTS)
def test_rgb_to_yuv_reuses_one_working_set(variant):
    converter = RgbToYuvConverter(variant, device=CPU)
    frames = [_frame(8, 12, CPU, seed) for seed in range(3)]

    first = [converter.convert(frame) for frame in frames]
    second = [converter.convert(frame) for frame in frames]

    for one, two in zip(first, second):
        assert torch.equal(one, two)
    # Different content must not survive in the reused scratch.
    assert not torch.equal(first[0], first[1])


def test_yuv_to_rgb_reuses_one_working_set():
    converter = YuvToRgbConverter(8, 12, AvColorspace.ITU709, False, False, CPU)
    planes = [_planes(8, 12, CPU, seed) for seed in range(3)]

    first = [converter.convert(y, uv) for y, uv in planes]
    second = [converter.convert(y, uv) for y, uv in planes]

    for one, two in zip(first, second):
        assert torch.equal(one, two)
    assert not torch.equal(first[0], first[1])


@pytest.fixture
def eager_on_cuda(monkeypatch):
    """Force the converters onto the eager path the AMD build takes."""
    monkeypatch.setattr(rgb_to_yuv_module, "is_nvidia_device", lambda device: False)
    monkeypatch.setattr(yuv_to_rgb_module, "is_nvidia_device", lambda device: False)
    return torch.device("cuda:0")


@cuda_only
@pytest.mark.parametrize("variant", VARIANTS)
def test_rgb_to_yuv_allocates_nothing_per_frame(variant, eager_on_cuda):
    device = eager_on_cuda
    converter = RgbToYuvConverter(variant, device=device)
    assert not converter.uses_kernel
    frame = _frame(64, 96, device, 0)
    luma = torch.empty((64, 96), dtype=converter.sample_dtype, device=device)
    chroma = torch.empty((32, 96), dtype=converter.sample_dtype, device=device)

    converter.convert_into(frame, luma, chroma)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated(device)

    for _ in range(10):
        converter.convert_into(frame, luma, chroma)
    torch.cuda.synchronize()

    assert torch.cuda.memory_allocated(device) == baseline


@cuda_only
def test_yuv_to_rgb_allocates_nothing_per_frame(eager_on_cuda):
    device = eager_on_cuda
    converter = YuvToRgbConverter(64, 96, AvColorspace.ITU709, False, False, device)
    y, uv = _planes(64, 96, device, 0)
    out = torch.empty((3, 64, 96), dtype=torch.uint8, device=device)

    converter.convert_into(y, uv, out)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated(device)

    for _ in range(10):
        converter.convert_into(y, uv, out)
    torch.cuda.synchronize()

    assert torch.cuda.memory_allocated(device) == baseline
