import numpy as np
import pytest
import torch
import torch.nn.functional as F

from factories import restore_synchronously
from jasna.blend_buffer import BlendBuffer
from jasna.crop_buffer import RawCrop, extract_crop, scale_offsets, _torch_pad_reflect
from jasna.pipeline_items import SecondaryRestoreResult
from jasna.restorer.denoise import DenoiseStrength
from jasna.restorer.restoration_pipeline import RestorationPipeline
from jasna.tracking.clip_tracker import TrackedClip

pytestmark = pytest.mark.usefixtures("exact_crops")


def _make_raw_crops(frames: list[torch.Tensor], clip: TrackedClip) -> list[RawCrop]:
    frame_h, frame_w = frames[0].shape[1], frames[0].shape[2]
    return [extract_crop(frame, bbox, frame_h, frame_w) for frame, bbox in zip(frames, clip.bboxes)]


def _clip(bbox, frames: int = 1, mask_hw=(2, 2), mask_value: bool = False) -> TrackedClip:
    return TrackedClip(
        track_id=0,
        start_frame=0,
        mask_resolution=mask_hw,
        bboxes=[np.array(bbox, dtype=np.float32)] * frames,
        masks=[torch.full(mask_hw, mask_value, dtype=torch.bool)] * frames,
    )


def _restore(pipeline: RestorationPipeline, frames: list[torch.Tensor], clip: TrackedClip) -> SecondaryRestoreResult:
    frame_shape = (frames[0].shape[1], frames[0].shape[2])
    return restore_synchronously(pipeline, clip, _make_raw_crops(frames, clip), frame_shape, 0, len(frames))


def _scaled_offsets(sr: SecondaryRestoreResult) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Pad offsets and resize shapes in the restored frames' own pixels."""
    scaled = [scale_offsets(frame, po, rs) for frame, po, rs in zip(sr.restored_frames, sr.pad_offsets, sr.resize_shapes)]
    return [pad for pad, _ in scaled], [shape for _, shape in scaled]


class _IdentityRestorer:
    input_dtype = torch.float32
    device = torch.device("cpu")

    def raw_process(self, crops: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack([crop.to(dtype=torch.float32).div(255.0) for crop in crops], dim=0)


class _CaptureRestorer(_IdentityRestorer):
    def __init__(self) -> None:
        self.captured: list[torch.Tensor] | None = None

    def raw_process(self, crops: list[torch.Tensor]) -> torch.Tensor:
        self.captured = crops
        return super().raw_process(crops)


class _ConstantRestorer(_IdentityRestorer):
    """Primary restorer that turns every crop into the flat ``value`` (0..1)."""

    def __init__(self, value: float) -> None:
        self.value = value

    def raw_process(self, crops: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack([torch.full(crop.shape, self.value, dtype=torch.float32) for crop in crops])


class _Upscale2xSecondary:
    name = "upscale2x"
    num_workers = 1

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        del keep_start, keep_end
        x = frames_256.to(dtype=torch.float32)
        y = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False).clamp(0, 1)
        return list(y.mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8).unbind(0))


def test_restore_clip_uses_floor_ceil_xyxy_rounding() -> None:
    pipeline = RestorationPipeline(restorer=_IdentityRestorer())  # type: ignore[arg-type]

    restored = _restore(pipeline, [torch.zeros((3, 10, 10), dtype=torch.uint8)], _clip([2.1, 2.1, 6.2, 6.2], mask_hw=(4, 4)))

    # floor(x1/y1)=2, ceil(x2/y2)=7; xyxy are exclusive for slicing.
    assert restored.enlarged_bboxes == [(2, 2, 7, 7)]
    assert restored.crop_shapes == [(5, 5)]


def test_restore_clip_clamps_bbox_to_frame() -> None:
    pipeline = RestorationPipeline(restorer=_IdentityRestorer())  # type: ignore[arg-type]

    restored = _restore(pipeline, [torch.zeros((3, 10, 10), dtype=torch.uint8)], _clip([-1.2, -0.1, 12.3, 9.9]))

    assert restored.enlarged_bboxes == [(0, 0, 10, 10)]
    assert restored.crop_shapes == [(10, 10)]


def test_restore_clip_does_not_upscale_small_crops() -> None:
    restorer = _CaptureRestorer()
    pipeline = RestorationPipeline(restorer=restorer)  # type: ignore[arg-type]
    frame = torch.arange(3 * 30 * 40, dtype=torch.uint8).reshape(3, 30, 40)

    restored = _restore(pipeline, [frame], _clip([5.0, 7.0, 25.0, 17.0]))  # crop: (10, 20)

    pad_offsets, resize_shapes = _scaled_offsets(restored)
    assert restored.crop_shapes == [(10, 20)]
    assert resize_shapes == [(10, 20)]
    assert pad_offsets == [((256 - 20) // 2, (256 - 10) // 2)]
    assert restored.restored_frames[0].shape == (3, 256, 256)

    assert restorer.captured is not None
    assert len(restorer.captured) == 1
    assert restorer.captured[0].shape == (3, 256, 256)
    assert restorer.captured[0].dtype == torch.float32

    crop = frame[:, 7:17, 5:25]
    resized = crop.unsqueeze(0).to(dtype=torch.float32)
    resized = F.interpolate(resized, size=(10, 20), mode="bilinear", align_corners=False).squeeze(0)

    pad_left, pad_top = pad_offsets[0]
    pad_bottom = 256 - 10 - pad_top
    pad_right = 256 - 20 - pad_left
    expected = _torch_pad_reflect(resized, (pad_left, pad_right, pad_top, pad_bottom))
    assert torch.equal(restorer.captured[0], expected)


def test_restore_clip_secondary_output_can_be_larger_and_unpad_metadata_scales() -> None:
    pipeline = RestorationPipeline(  # type: ignore[arg-type]
        restorer=_CaptureRestorer(),
        secondary_restorer=_Upscale2xSecondary(),
    )
    frame = torch.arange(3 * 30 * 40, dtype=torch.uint8).reshape(3, 30, 40)

    restored = _restore(pipeline, [frame], _clip([5.0, 7.0, 25.0, 17.0]))  # crop: (10, 20)

    pad_offsets, resize_shapes = _scaled_offsets(restored)
    assert restored.restored_frames[0].shape == (3, 512, 512)
    assert pad_offsets == [(((256 - 20) // 2) * 2, ((256 - 10) // 2) * 2)]
    assert resize_shapes == [(10 * 2, 20 * 2)]


def test_restore_clip_downscales_when_crop_is_larger_than_restoration_size() -> None:
    pipeline = RestorationPipeline(restorer=_IdentityRestorer())  # type: ignore[arg-type]

    restored = _restore(pipeline, [torch.zeros((3, 400, 300), dtype=torch.uint8)], _clip([0.0, 0.0, 300.0, 400.0], mask_hw=(1, 1)))

    pad_offsets, resize_shapes = _scaled_offsets(restored)
    assert restored.crop_shapes == [(400, 300)]
    assert resize_shapes == [(256, 256)]
    assert pad_offsets == [(0, 0)]
    assert restored.restored_frames[0].shape == (3, 256, 256)


def test_torch_pad_reflect_reflects_again_when_padding_exceeds_the_image() -> None:
    image = torch.arange(3 * 3 * 2, dtype=torch.float32).reshape(3, 3, 2)

    out = _torch_pad_reflect(image, (10, 1, 2, 7))

    assert np.array_equal(out.numpy(), np.pad(image.numpy(), ((0, 0), (2, 7), (10, 1)), mode="reflect"))


def test_restore_clip_with_denoise_strength() -> None:
    torch.manual_seed(99)
    frames = [torch.randint(0, 256, (3, 30, 40), dtype=torch.uint8) for _ in range(5)]
    clip = _clip([5.0, 7.0, 25.0, 17.0], frames=5)

    restored_none = _restore(RestorationPipeline(restorer=_IdentityRestorer(), denoise_strength=DenoiseStrength.NONE), frames, clip)  # type: ignore[arg-type]
    restored_med = _restore(RestorationPipeline(restorer=_IdentityRestorer(), denoise_strength=DenoiseStrength.MEDIUM), frames, clip)  # type: ignore[arg-type]

    assert len(restored_none.restored_frames) == len(restored_med.restored_frames) == 5
    diff = sum(
        (a.float() - b.float()).abs().sum().item()
        for a, b in zip(restored_none.restored_frames, restored_med.restored_frames)
    )
    assert diff > 0, "Denoise MEDIUM should produce different output from NONE"


@pytest.mark.parametrize(
    ("keep_start", "keep_end", "crossfade_weights", "box_values"),
    [
        (0, 3, None, [255, 255, 255]),
        (1, 2, None, [0, 255, 0]),
        (0, 3, {0: 0.5, 1: 1.0, 2: 0.5}, [128, 255, 128]),
        (2, 1, None, [0, 0, 0]),
    ],
    ids=["whole-clip", "kept-middle", "crossfade", "empty-keep-range"],
)
def test_blending_a_restored_clip_paints_only_its_kept_frames(
    keep_start, keep_end, crossfade_weights, box_values, whole_box_blend_mask
) -> None:
    frames = [torch.zeros((3, 30, 40), dtype=torch.uint8) for _ in range(3)]
    clip = _clip([5.0, 7.0, 25.0, 17.0], frames=3, mask_value=True)
    pipeline = RestorationPipeline(restorer=_ConstantRestorer(1.0))  # type: ignore[arg-type]
    blend_buffer = BlendBuffer(vr_projector=None)
    for frame_idx in range(3):
        blend_buffer.register_frame(frame_idx, {0})

    blend_buffer.add_result(
        restore_synchronously(pipeline, clip, _make_raw_crops(frames, clip), (30, 40), keep_start, keep_end, crossfade_weights)
    )

    assert all(blend_buffer.is_frame_ready(frame_idx) for frame_idx in range(3))
    blended = [blend_buffer.blend_frame(frame_idx, frames[frame_idx]) for frame_idx in range(3)]
    assert [int(frame[0, 10, 10]) for frame in blended] == box_values
    assert all(int(frame[:, 0, 0].max()) == 0 for frame in blended)
