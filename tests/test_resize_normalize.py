import pytest
import torch
import torch.nn.functional as F

from jasna.media.resize_normalize import ResizeNormalizer
from jasna.mosaic.rfdetr import _IMAGENET_MEAN, _IMAGENET_STD
from jasna.mosaic.yolo import _YOLO_LETTERBOX_PAD_VALUE, _letterbox_geometry

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

IDENTITY = (0.0, 0.0, 0.0)
UNIT = (1.0, 1.0, 1.0)


def _device() -> torch.device:
    return torch.device("cuda:0")


def _frames(batch: int, height: int, width: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(0)
    return torch.randint(
        0, 256, (batch, 3, height, width), generator=generator,
        device=_device(), dtype=torch.uint8,
    )


def _rfdetr_reference(frames: torch.Tensor, resolution: int, dtype: torch.dtype):
    x = frames.to(device=_device(), dtype=dtype).div_(255.0)
    x = F.interpolate(x, size=(resolution, resolution), mode="bilinear", align_corners=False)
    mean = x.new_tensor(_IMAGENET_MEAN)[:, None, None]
    std = x.new_tensor(_IMAGENET_STD)[:, None, None]
    return (x - mean) / std


def _yolo_reference(frames: torch.Tensor, imgsz: int, dtype: torch.dtype):
    x = frames.to(device=_device(), dtype=dtype) / 255.0
    _, _, h, w = x.shape
    gain, left, top, unpad_w, unpad_h = _letterbox_geometry(h, w, (imgsz, imgsz))
    if (unpad_h, unpad_w) != (h, w):
        x = F.interpolate(x, size=(unpad_h, unpad_w), mode="bilinear", align_corners=False)
    right = imgsz - unpad_w - left
    bottom = imgsz - unpad_h - top
    if right or bottom or left or top:
        x = F.pad(x, (left, right, top, bottom), value=_YOLO_LETTERBOX_PAD_VALUE)
    return x, (left, top, unpad_w, unpad_h)


SHAPES = [(4, 1080, 1920), (2, 2160, 3840), (1, 512, 512), (3, 300, 401)]


@pytest.mark.parametrize("shape", SHAPES)
def test_rfdetr_preprocess_is_bit_identical_in_fp16(shape):
    frames = _frames(*shape)
    normalizer = ResizeNormalizer(
        device=_device(), dtype=torch.float16,
        mean=_IMAGENET_MEAN, std=_IMAGENET_STD, fill=IDENTITY,
    )

    got = normalizer.run(frames, out_hw=(576, 576), content=(0, 0, 576, 576))

    assert torch.equal(got, _rfdetr_reference(frames, 576, torch.float16))


@pytest.mark.parametrize("shape", SHAPES)
def test_yolo_letterbox_is_bit_identical_in_fp16(shape):
    frames = _frames(*shape)
    normalizer = ResizeNormalizer(
        device=_device(), dtype=torch.float16,
        mean=IDENTITY, std=UNIT, fill=(_YOLO_LETTERBOX_PAD_VALUE,) * 3,
    )

    reference, content = _yolo_reference(frames, 640, torch.float16)
    got = normalizer.run(frames, out_hw=(640, 640), content=content)

    assert torch.equal(got, reference)


def test_fp32_matches_the_torch_path_closely():
    frames = _frames(2, 1080, 1920)
    normalizer = ResizeNormalizer(
        device=_device(), dtype=torch.float32,
        mean=_IMAGENET_MEAN, std=_IMAGENET_STD, fill=IDENTITY,
    )

    got = normalizer.run(frames, out_hw=(576, 576), content=(0, 0, 576, 576))
    reference = _rfdetr_reference(frames, 576, torch.float32)

    # fp32 keeps Torch's accumulation order only approximately; the gap is a
    # couple of ulp and fp32 is not the dtype either detector runs in.
    assert (got - reference).abs().max().item() < 1e-5


def test_letterbox_padding_uses_the_fill_value():
    frames = _frames(1, 200, 400)
    normalizer = ResizeNormalizer(
        device=_device(), dtype=torch.float32,
        mean=IDENTITY, std=UNIT, fill=(_YOLO_LETTERBOX_PAD_VALUE,) * 3,
    )
    _, left, top, unpad_w, unpad_h = _letterbox_geometry(200, 400, (640, 640))

    out = normalizer.run(frames, out_hw=(640, 640), content=(left, top, unpad_w, unpad_h))

    assert torch.allclose(out[:, :, :top], torch.full_like(out[:, :, :top], _YOLO_LETTERBOX_PAD_VALUE))
    assert torch.allclose(
        out[:, :, top + unpad_h :],
        torch.full_like(out[:, :, top + unpad_h :], _YOLO_LETTERBOX_PAD_VALUE),
    )


def test_output_shape_and_dtype():
    frames = _frames(3, 480, 640)
    normalizer = ResizeNormalizer(
        device=_device(), dtype=torch.float16, mean=IDENTITY, std=UNIT, fill=IDENTITY
    )

    out = normalizer.run(frames, out_hw=(576, 576), content=(0, 0, 576, 576))

    assert out.shape == (3, 3, 576, 576)
    assert out.dtype is torch.float16
    assert out.is_contiguous()


def test_rejects_a_non_uint8_batch():
    normalizer = ResizeNormalizer(
        device=_device(), dtype=torch.float16, mean=IDENTITY, std=UNIT, fill=IDENTITY
    )
    frames = torch.zeros((1, 3, 64, 64), device=_device(), dtype=torch.float32)

    with pytest.raises(ValueError, match="uint8"):
        normalizer.run(frames, out_hw=(576, 576), content=(0, 0, 576, 576))


def test_rejects_a_non_bchw_batch():
    normalizer = ResizeNormalizer(
        device=_device(), dtype=torch.float16, mean=IDENTITY, std=UNIT, fill=IDENTITY
    )
    frames = torch.zeros((3, 64, 64), device=_device(), dtype=torch.uint8)

    with pytest.raises(ValueError, match=r"\(B, 3, H, W\)"):
        normalizer.run(frames, out_hw=(576, 576), content=(0, 0, 576, 576))
