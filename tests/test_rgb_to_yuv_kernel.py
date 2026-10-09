import pytest
import torch
from unittest.mock import patch

from jasna.accelerator import is_amd_device
from jasna.media.hip_kernel import AMD_HIP_COLOR_KERNELS_ENV
from jasna.media.rgb_to_yuv import RgbToYuvConverter


def _supported_gpu_kernel() -> bool:
    if not torch.cuda.is_available():
        return False
    device = torch.device("cuda:0")
    if not is_amd_device(device):
        return True
    architecture = str(
        getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    ).split(":", 1)[0]
    return architecture == "gfx1100"


pytestmark = pytest.mark.skipif(
    not _supported_gpu_kernel(), reason="needs a supported NVIDIA or AMD GPU kernel"
)

VARIANTS = [
    f"{pixel_format}_{standard}_{value_range}"
    for pixel_format in ("nv12", "p010")
    for standard in ("bt601", "bt709", "bt2020")
    for value_range in ("limited", "full")
]


def _torch_reference(variant: str, frame: torch.Tensor) -> torch.Tensor:
    # Select the independent Torch oracle explicitly. The native side stays
    # forced on by the fixture; forcing HIP on a CPU-selector oracle should
    # fail rather than silently pick another backend. Math still runs on the
    # frame's real device, without a GPU->CPU copy.
    with patch.dict("os.environ", {AMD_HIP_COLOR_KERNELS_ENV: "0"}):
        return RgbToYuvConverter(variant, device=torch.device("cpu")).convert(frame)


def _device() -> torch.device:
    return torch.device("cuda:0")


@pytest.fixture(autouse=True)
def enable_amd_kernel(monkeypatch):
    if is_amd_device(_device()):
        monkeypatch.setenv(AMD_HIP_COLOR_KERNELS_ENV, "1")


def _random_frame(height: int, width: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(0)
    return torch.randint(
        0, 256, (3, height, width), generator=generator, device=_device(), dtype=torch.uint8
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_matches_the_torch_reference_within_one_code(variant):
    frame = _random_frame(64, 96)
    converter = RgbToYuvConverter(variant, device=_device())
    assert converter.uses_kernel

    ours = converter.convert(frame)
    reference = _torch_reference(variant, frame)

    assert ours.shape == reference.shape
    assert ours.dtype == reference.dtype
    # P010 stores codes shifted left by 6, so one code of disagreement is 64.
    tolerance = 64 if converter.ten_bit else 1
    if converter.ten_bit:
        ours = ours.view(torch.uint16)
        reference = reference.view(torch.uint16)
    assert (ours.to(torch.int32) - reference.to(torch.int32)).abs().max().item() <= tolerance


@pytest.mark.parametrize("variant", VARIANTS)
def test_flat_colours_match_the_torch_reference_exactly(variant):
    converter = RgbToYuvConverter(variant, device=_device())
    for colour in ((0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)):
        frame = torch.empty((3, 8, 8), device=_device(), dtype=torch.uint8)
        for channel, value in enumerate(colour):
            frame[channel] = value
        assert torch.equal(converter.convert(frame), _torch_reference(variant, frame))


def test_output_is_a_contiguous_packed_frame():
    frame = _random_frame(16, 24)
    packed = RgbToYuvConverter("nv12_bt709_limited", device=_device()).convert(frame)

    assert packed.shape == (16 + 8, 24)
    assert packed.dtype == torch.uint8
    assert packed.is_contiguous()


def test_writes_into_separate_luma_and_chroma_buffers():
    frame = _random_frame(16, 24)
    converter = RgbToYuvConverter("nv12_bt709_limited", device=_device())

    luma = torch.empty((16, 24), device=_device(), dtype=torch.uint8)
    chroma = torch.empty((8, 24), device=_device(), dtype=torch.uint8)
    converter.convert_into(frame, luma, chroma)

    packed = converter.convert(frame)
    assert torch.equal(luma, packed[:16])
    assert torch.equal(chroma, packed[16:])


def test_writes_into_a_pitched_destination():
    frame = _random_frame(16, 24)
    converter = RgbToYuvConverter("nv12_bt709_limited", device=_device())
    storage = torch.zeros((24, 32), device=_device(), dtype=torch.uint8)
    view = storage[:, :24]

    converter.convert_into(frame, view[:16], view[16:])

    assert torch.equal(view, converter.convert(frame))
    assert storage[:, 24:].eq(0).all()


@pytest.mark.parametrize("variant", ["nv12_bt709_limited", "p010_bt709_limited"])
def test_boundary_pitched_rgb_repeats_across_a_non_default_stream(variant):
    height, width, pitch = 66, 98, 112
    generator = torch.Generator(device="cuda").manual_seed(20260905)
    storage = torch.empty((3, height, pitch), device=_device(), dtype=torch.uint8)
    frame = storage[:, :, :width]
    frame.copy_(
        torch.randint(
            0,
            256,
            frame.shape,
            generator=generator,
            device=_device(),
            dtype=torch.uint8,
        )
    )
    converter = RgbToYuvConverter(variant, device=_device())
    reference = _torch_reference(variant, frame)
    stream = torch.cuda.Stream(device=_device())
    stream.wait_stream(torch.cuda.current_stream(_device()))
    with torch.cuda.stream(stream):
        first = converter.convert(frame)
        second = converter.convert(frame)
    stream.synchronize()

    tolerance = 64 if converter.ten_bit else 1
    if converter.ten_bit:
        first = first.view(torch.uint16)
        second = second.view(torch.uint16)
        reference = reference.view(torch.uint16)
    assert frame.stride(1) == pitch
    assert torch.equal(first, second)
    assert (first.to(torch.int32) - reference.to(torch.int32)).abs().max().item() <= tolerance


def test_rejects_odd_dimensions():
    converter = RgbToYuvConverter("nv12_bt709_limited", device=_device())
    frame = _random_frame(15, 24)
    packed = torch.empty((22, 24), device=_device(), dtype=torch.uint8)

    with pytest.raises(ValueError, match="even dimensions"):
        converter.convert_into(frame, packed[:15], packed[15:])


def test_rejects_a_non_uint8_frame():
    converter = RgbToYuvConverter("nv12_bt709_limited", device=_device())
    frame = torch.zeros((3, 16, 24), device=_device(), dtype=torch.float32)
    packed = torch.empty((24, 24), device=_device(), dtype=torch.uint8)

    with pytest.raises(ValueError, match="uint8"):
        converter.convert_into(frame, packed[:16], packed[16:])


def test_rejects_an_unknown_variant():
    with pytest.raises(ValueError, match="Unknown RGB to YUV variant"):
        RgbToYuvConverter("nv12_bt709_studio", device=_device())
