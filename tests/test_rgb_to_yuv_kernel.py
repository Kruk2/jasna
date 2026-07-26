import pytest
import torch

from jasna.media.rgb_to_yuv import _TORCH_CONVERTERS, RgbToYuvConverter

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

VARIANTS = sorted(_TORCH_CONVERTERS)


def _device() -> torch.device:
    return torch.device("cuda:0")


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
    reference = _TORCH_CONVERTERS[variant](frame)

    assert ours.shape == reference.shape
    assert ours.dtype == reference.dtype
    # P010 stores codes shifted left by 6, so one code of disagreement is 64.
    tolerance = 64 if converter.ten_bit else 1
    assert (ours.int() - reference.int()).abs().max().item() <= tolerance


@pytest.mark.parametrize("variant", VARIANTS)
def test_flat_colours_match_the_torch_reference_exactly(variant):
    converter = RgbToYuvConverter(variant, device=_device())
    for colour in ((0, 0, 0), (255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255)):
        frame = torch.empty((3, 8, 8), device=_device(), dtype=torch.uint8)
        for channel, value in enumerate(colour):
            frame[channel] = value
        assert torch.equal(converter.convert(frame), _TORCH_CONVERTERS[variant](frame))


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
