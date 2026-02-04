import numpy as np
import pytest
import torch
import torch.nn.functional as F

from jasna.restorer.restoration_pipeline import RestorationPipeline
from jasna.tracking.clip_tracker import TrackedClip


class _IdentityRestorer:
    dtype = torch.float32

    def restore(self, crops: list[torch.Tensor]) -> list[torch.Tensor]:
        return crops


class _CaptureRestorer:
    dtype = torch.float32

    def __init__(self) -> None:
        self.captured: list[torch.Tensor] | None = None

    def restore(self, crops: list[torch.Tensor]) -> list[torch.Tensor]:
        self.captured = crops
        return crops


def test_restore_clip_uses_floor_ceil_xyxy_rounding(monkeypatch) -> None:
    import jasna.restorer.restoration_pipeline as rp

    # Disable expansion so we can assert pure xyxy rounding + slicing.
    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    pipeline = RestorationPipeline(restorer=_IdentityRestorer())  # type: ignore[arg-type]

    frame = torch.zeros((3, 10, 10), dtype=torch.uint8)
    bbox = np.array([2.1, 2.1, 6.2, 6.2], dtype=np.float32)  # xyxy floats
    mask = torch.zeros((4, 4), dtype=torch.bool)

    clip = TrackedClip(
        track_id=0,
        start_frame=0,
        mask_resolution=(4, 4),
        bboxes=[bbox],
        masks=[mask],
    )

    restored = pipeline.restore_clip(clip, [frame], keep_start=0, keep_end=1)

    # floor(x1/y1)=2, ceil(x2/y2)=7; xyxy are exclusive for slicing.
    assert restored.enlarged_bboxes == [(2, 2, 7, 7)]
    assert restored.crop_shapes == [(5, 5)]


def test_restore_clip_clamps_bbox_to_frame(monkeypatch) -> None:
    import jasna.restorer.restoration_pipeline as rp

    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    pipeline = RestorationPipeline(restorer=_IdentityRestorer())  # type: ignore[arg-type]

    frame = torch.zeros((3, 10, 10), dtype=torch.uint8)
    bbox = np.array([-1.2, -0.1, 12.3, 9.9], dtype=np.float32)  # out of bounds
    mask = torch.zeros((2, 2), dtype=torch.bool)

    clip = TrackedClip(track_id=0, start_frame=0, mask_resolution=(2, 2), bboxes=[bbox], masks=[mask])
    restored = pipeline.restore_clip(clip, [frame], keep_start=0, keep_end=1)

    assert restored.enlarged_bboxes == [(0, 0, 10, 10)]
    assert restored.crop_shapes == [(10, 10)]


def test_restore_clip_does_not_upscale_small_crops(monkeypatch) -> None:
    import jasna.restorer.restoration_pipeline as rp

    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    restorer = _CaptureRestorer()
    pipeline = RestorationPipeline(restorer=restorer)  # type: ignore[arg-type]

    frame = torch.arange(3 * 30 * 40, dtype=torch.uint8).reshape(3, 30, 40)
    bbox = np.array([5.0, 7.0, 25.0, 17.0], dtype=np.float32)  # crop: (10, 20)
    mask = torch.zeros((2, 2), dtype=torch.bool)

    clip = TrackedClip(track_id=0, start_frame=0, mask_resolution=(2, 2), bboxes=[bbox], masks=[mask])
    restored = pipeline.restore_clip(clip, [frame], keep_start=0, keep_end=1)

    assert restored.crop_shapes == [(10, 20)]
    assert restored.resize_shapes == [(10, 20)]
    assert restored.pad_offsets == [((256 - 20) // 2, (256 - 10) // 2)]
    assert restored.restored_frames[0].shape == (3, 256, 256)

    assert restorer.captured is not None
    assert len(restorer.captured) == 1
    assert restorer.captured[0].shape == (256, 256, 3)
    assert restorer.captured[0].dtype == torch.uint8

    crop = frame[:, 7:17, 5:25]
    resized = crop.unsqueeze(0).to(dtype=torch.float32)
    resized = F.interpolate(resized, size=(10, 20), mode="bilinear", align_corners=False).squeeze(0)

    pad_left, pad_top = restored.pad_offsets[0]
    pad_bottom = 256 - 10 - pad_top
    pad_right = 256 - 20 - pad_left
    expected = rp._torch_pad_reflect(resized, (pad_left, pad_right, pad_top, pad_bottom)).to(torch.uint8).permute(1, 2, 0)
    assert torch.equal(restorer.captured[0], expected)


def test_restore_clip_downscales_when_crop_is_larger_than_restoration_size(monkeypatch) -> None:
    import jasna.restorer.restoration_pipeline as rp

    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    pipeline = RestorationPipeline(restorer=_IdentityRestorer())  # type: ignore[arg-type]

    frame = torch.zeros((3, 400, 300), dtype=torch.uint8)
    bbox = np.array([0.0, 0.0, 300.0, 400.0], dtype=np.float32)  # full frame crop
    mask = torch.zeros((1, 1), dtype=torch.bool)
    clip = TrackedClip(track_id=0, start_frame=0, mask_resolution=(1, 1), bboxes=[bbox], masks=[mask])

    restored = pipeline.restore_clip(clip, [frame], keep_start=0, keep_end=1)
    assert restored.crop_shapes == [(400, 300)]
    assert restored.resize_shapes == [(256, 256)]
    assert restored.pad_offsets == [(0, 0)]
    assert restored.restored_frames[0].shape == (3, 256, 256)


def test_torch_pad_reflect_supports_large_padding() -> None:
    import jasna.restorer.restoration_pipeline as rp

    image = torch.arange(3 * 2 * 2, dtype=torch.float32).reshape(3, 2, 2)
    out = rp._torch_pad_reflect(image, (10, 10, 10, 10))
    assert out.shape == (3, 22, 22)


def test_restore_clip_raises_on_bbox_with_zero_area(monkeypatch) -> None:
    import jasna.restorer.restoration_pipeline as rp

    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    pipeline = RestorationPipeline(restorer=_IdentityRestorer())  # type: ignore[arg-type]

    frame = torch.zeros((3, 10, 10), dtype=torch.uint8)
    bbox = np.array([2.0, 2.0, 2.0, 6.0], dtype=np.float32)  # x1 == x2 -> empty crop
    mask = torch.zeros((1, 1), dtype=torch.bool)
    clip = TrackedClip(track_id=0, start_frame=0, mask_resolution=(1, 1), bboxes=[bbox], masks=[mask])

    with pytest.raises(ZeroDivisionError):
        pipeline.restore_clip(clip, [frame], keep_start=0, keep_end=1)


def test_restore_clip_raises_on_mismatched_frame_and_clip_lengths(monkeypatch) -> None:
    import jasna.restorer.restoration_pipeline as rp

    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    pipeline = RestorationPipeline(restorer=_IdentityRestorer())  # type: ignore[arg-type]

    frames = [torch.zeros((3, 10, 10), dtype=torch.uint8), torch.zeros((3, 10, 10), dtype=torch.uint8)]
    bbox = np.array([2.0, 2.0, 6.0, 6.0], dtype=np.float32)
    mask = torch.zeros((1, 1), dtype=torch.bool)
    clip = TrackedClip(track_id=0, start_frame=0, mask_resolution=(1, 1), bboxes=[bbox], masks=[mask])

    with pytest.raises(IndexError):
        pipeline.restore_clip(clip, frames, keep_start=0, keep_end=1)

