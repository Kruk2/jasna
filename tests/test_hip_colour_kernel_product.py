"""Real GPU parity for the AMD AOT colour route on supported platforms."""
import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace

from jasna.accelerator import is_amd_device
from jasna.media.hip_kernel import AMD_HIP_COLOR_KERNELS_ENV
from jasna.media.yuv_to_rgb import YuvToRgbConverter


def _supported_amd_gpu() -> bool:
    if not torch.cuda.is_available() or not is_amd_device(torch.device("cuda:0")):
        return False
    architecture = str(
        getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    ).split(":", 1)[0]
    return architecture == "gfx1100"


pytestmark = pytest.mark.skipif(
    not _supported_amd_gpu(), reason="needs an AMD gfx1100 GPU"
)

_SPACES = {
    "bt601": AvColorspace.ITU601,
    "bt709": AvColorspace.ITU709,
    "bt2020": AvColorspace.BT2020,
}


@pytest.mark.parametrize("bits", [8, 10])
@pytest.mark.parametrize("matrix", sorted(_SPACES))
@pytest.mark.parametrize("full_range", [False, True])
def test_product_yuv_kernel_matches_eager_within_one_code(
    monkeypatch, bits, matrix, full_range
):
    device = torch.device("cuda:0")
    height, width = 64, 96
    generator = torch.Generator(device=device).manual_seed(20260904)
    packed = torch.randint(
        0,
        1024 if bits == 10 else 256,
        (height + height // 2, width),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    if bits == 10:
        packed = packed.bitwise_left_shift(6)
    packed = packed.to(torch.uint16 if bits == 10 else torch.uint8)
    y = packed[:height]
    uv = packed[height:].view(height // 2, width // 2, 2)

    monkeypatch.setenv(AMD_HIP_COLOR_KERNELS_ENV, "0")
    eager = YuvToRgbConverter(
        height, width, _SPACES[matrix], full_range, bits == 10, device
    )
    expected = eager.convert(y, uv)

    monkeypatch.setenv(AMD_HIP_COLOR_KERNELS_ENV, "1")
    fused = YuvToRgbConverter(
        height, width, _SPACES[matrix], full_range, bits == 10, device
    )
    actual = fused.convert(y, uv)
    torch.cuda.synchronize(device)

    assert fused.uses_kernel
    assert (actual.to(torch.int16) - expected.to(torch.int16)).abs().max().item() <= 1


@pytest.mark.parametrize("bits", [8, 10])
def test_product_yuv_kernel_handles_pitch_boundary_repeats_and_streams(
    monkeypatch, bits
):
    device = torch.device("cuda:0")
    height, width, pitch = 66, 98, 112
    generator = torch.Generator(device=device).manual_seed(20260905 + bits)
    storage = torch.randint(
        0,
        1024 if bits == 10 else 256,
        (height + height // 2, pitch),
        dtype=torch.int32,
        device=device,
        generator=generator,
    )
    if bits == 10:
        storage = storage.bitwise_left_shift(6)
    storage = storage.to(torch.uint16 if bits == 10 else torch.uint8)
    y = storage[:height, :width]
    uv = storage[height:, :width].view(height // 2, width // 2, 2)

    monkeypatch.setenv(AMD_HIP_COLOR_KERNELS_ENV, "0")
    eager = YuvToRgbConverter(
        height, width, AvColorspace.ITU709, False, bits == 10, device
    ).convert(y, uv)

    monkeypatch.setenv(AMD_HIP_COLOR_KERNELS_ENV, "1")
    converter = YuvToRgbConverter(
        height, width, AvColorspace.ITU709, False, bits == 10, device
    )
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    outputs = []
    for _ in range(3):
        with torch.cuda.stream(stream):
            outputs.append(converter.convert(y, uv))
    stream.synchronize()

    assert y.stride(0) == pitch
    assert converter.uses_kernel
    assert all(torch.equal(outputs[0], output) for output in outputs[1:])
    assert (outputs[0].to(torch.int16) - eager.to(torch.int16)).abs().max().item() <= 1
