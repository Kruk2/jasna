import numpy as np
import pytest
import torch
import torch.nn.functional as F

from jasna.restorer.denoise import DenoiseStep, DenoiseStrength
from jasna.restorer.restoration_pipeline import RestorationPipeline
from jasna.tracking.clip_tracker import TrackedClip


class _IdentityRestorer:
    dtype = torch.float32

    def restore(self, crops: list[torch.Tensor]) -> list[torch.Tensor]:
        return crops

    def raw_process(self, crops: list[torch.Tensor]) -> torch.Tensor:
        stacked = []
        for f in crops:
            stacked.append(f.permute(2, 0, 1).to(dtype=torch.float32).div(255.0))
        return torch.stack(stacked, dim=0)


class _CaptureRestorer:
    dtype = torch.float32

    def __init__(self) -> None:
        self.captured: list[torch.Tensor] | None = None

    def restore(self, crops: list[torch.Tensor]) -> list[torch.Tensor]:
        self.captured = crops
        return crops

    def raw_process(self, crops: list[torch.Tensor]) -> torch.Tensor:
        self.captured = crops
        stacked = []
        for f in crops:
            stacked.append(f.permute(2, 0, 1).to(dtype=torch.float32).div(255.0))
        return torch.stack(stacked, dim=0)


class _Upscale2xSecondary:
    name = "upscale2x"
    num_workers = 1

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> torch.Tensor:
        del keep_start, keep_end
        x = frames_256.to(dtype=torch.float32)
        y = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False).clamp(0, 1)
        return y.mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8)


class _Upscale2xSecondaryList:
    name = "upscale2x_list"
    num_workers = 2

    def restore(self, frames_256: torch.Tensor, *, keep_start: int, keep_end: int) -> list[torch.Tensor]:
        del keep_start, keep_end
        x = frames_256.to(dtype=torch.float32)
        y = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False).clamp(0, 1)
        y_u8 = y.mul(255.0).round().clamp(0, 255).to(dtype=torch.uint8)
        return list(torch.unbind(y_u8, 0))


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


def test_restore_clip_secondary_output_can_be_larger_and_unpad_metadata_scales(monkeypatch) -> None:
    import jasna.restorer.restoration_pipeline as rp

    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    restorer = _CaptureRestorer()
    pipeline = RestorationPipeline(  # type: ignore[arg-type]
        restorer=restorer,
        secondary_restorer=_Upscale2xSecondary(),
    )

    frame = torch.arange(3 * 30 * 40, dtype=torch.uint8).reshape(3, 30, 40)
    bbox = np.array([5.0, 7.0, 25.0, 17.0], dtype=np.float32)  # crop: (10, 20)
    mask = torch.zeros((2, 2), dtype=torch.bool)
    clip = TrackedClip(track_id=0, start_frame=0, mask_resolution=(2, 2), bboxes=[bbox], masks=[mask])

    restored = pipeline.restore_clip(clip, [frame], keep_start=0, keep_end=1)

    assert restored.restored_frames[0].shape == (3, 512, 512)
    assert restored.pad_offsets == [(((256 - 20) // 2) * 2, ((256 - 10) // 2) * 2)]
    assert restored.resize_shapes == [(10 * 2, 20 * 2)]


def test_restore_clip_secondary_output_as_list_is_supported(monkeypatch) -> None:
    import jasna.restorer.restoration_pipeline as rp

    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    restorer = _CaptureRestorer()
    pipeline = RestorationPipeline(  # type: ignore[arg-type]
        restorer=restorer,
        secondary_restorer=_Upscale2xSecondaryList(),
    )

    frame = torch.arange(3 * 30 * 40, dtype=torch.uint8).reshape(3, 30, 40)
    bbox = np.array([5.0, 7.0, 25.0, 17.0], dtype=np.float32)  # crop: (10, 20)
    mask = torch.zeros((2, 2), dtype=torch.bool)
    clip = TrackedClip(track_id=0, start_frame=0, mask_resolution=(2, 2), bboxes=[bbox], masks=[mask])

    restored = pipeline.restore_clip(clip, [frame], keep_start=0, keep_end=1)
    assert restored.restored_frames[0].shape == (3, 512, 512)
    assert restored.pad_offsets == [(((256 - 20) // 2) * 2, ((256 - 10) // 2) * 2)]
    assert restored.resize_shapes == [(10 * 2, 20 * 2)]


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


def test_restore_clip_with_denoise_strength(monkeypatch) -> None:
    import jasna.restorer.restoration_pipeline as rp

    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    pipeline_none = RestorationPipeline(restorer=_IdentityRestorer(), denoise_strength=DenoiseStrength.NONE)  # type: ignore[arg-type]
    pipeline_med = RestorationPipeline(restorer=_IdentityRestorer(), denoise_strength=DenoiseStrength.MEDIUM)  # type: ignore[arg-type]

    torch.manual_seed(99)
    T = 5
    frames = [torch.randint(0, 256, (3, 30, 40), dtype=torch.uint8) for _ in range(T)]
    bbox = np.array([5.0, 7.0, 25.0, 17.0], dtype=np.float32)
    mask = torch.zeros((2, 2), dtype=torch.bool)
    bboxes = [bbox] * T
    masks = [mask] * T

    clip = TrackedClip(track_id=0, start_frame=0, mask_resolution=(2, 2), bboxes=bboxes, masks=masks)

    restored_none = pipeline_none.restore_clip(clip, frames, keep_start=0, keep_end=T)
    restored_med = pipeline_med.restore_clip(clip, frames, keep_start=0, keep_end=T)

    assert len(restored_none.restored_frames) == T
    assert len(restored_med.restored_frames) == T

    diff = sum(
        (a.float() - b.float()).abs().sum().item()
        for a, b in zip(restored_none.restored_frames, restored_med.restored_frames)
    )
    assert diff > 0, "Denoise MEDIUM should produce different output from NONE"


def test_restore_clip_denoise_step_after_primary_with_secondary_produces_valid_output(monkeypatch) -> None:
    import jasna.restorer.restoration_pipeline as rp

    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    pipeline = RestorationPipeline(  # type: ignore[arg-type]
        restorer=_IdentityRestorer(),
        secondary_restorer=_Upscale2xSecondary(),
        denoise_strength=DenoiseStrength.MEDIUM,
        denoise_step=DenoiseStep.AFTER_PRIMARY,
    )

    torch.manual_seed(42)
    T = 3
    frames = [torch.randint(0, 256, (3, 30, 40), dtype=torch.uint8) for _ in range(T)]
    bbox = np.array([5.0, 7.0, 25.0, 17.0], dtype=np.float32)
    clip = TrackedClip(
        track_id=0,
        start_frame=0,
        mask_resolution=(2, 2),
        bboxes=[bbox] * T,
        masks=[torch.zeros((2, 2), dtype=torch.bool)] * T,
    )

    restored = pipeline.restore_clip(clip, frames, keep_start=0, keep_end=T)

    assert len(restored.restored_frames) == T
    assert restored.restored_frames[0].shape == (3, 512, 512)


def test_restore_clip_denoise_step_after_secondary_denoises_secondary_output(monkeypatch) -> None:
    import jasna.restorer.restoration_pipeline as rp

    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    pipeline_none = RestorationPipeline(  # type: ignore[arg-type]
        restorer=_IdentityRestorer(),
        secondary_restorer=_Upscale2xSecondary(),
        denoise_strength=DenoiseStrength.NONE,
        denoise_step=DenoiseStep.AFTER_SECONDARY,
    )
    pipeline_med = RestorationPipeline(  # type: ignore[arg-type]
        restorer=_IdentityRestorer(),
        secondary_restorer=_Upscale2xSecondary(),
        denoise_strength=DenoiseStrength.MEDIUM,
        denoise_step=DenoiseStep.AFTER_SECONDARY,
    )

    torch.manual_seed(123)
    T = 3
    frames = [torch.randint(0, 256, (3, 30, 40), dtype=torch.uint8) for _ in range(T)]
    bbox = np.array([5.0, 7.0, 25.0, 17.0], dtype=np.float32)
    clip = TrackedClip(
        track_id=0,
        start_frame=0,
        mask_resolution=(2, 2),
        bboxes=[bbox] * T,
        masks=[torch.zeros((2, 2), dtype=torch.bool)] * T,
    )

    restored_none = pipeline_none.restore_clip(clip, frames, keep_start=0, keep_end=T)
    restored_med = pipeline_med.restore_clip(clip, frames, keep_start=0, keep_end=T)

    diff = sum(
        (a.float() - b.float()).abs().sum().item()
        for a, b in zip(restored_none.restored_frames, restored_med.restored_frames)
    )
    assert diff > 0, "AFTER_SECONDARY with MEDIUM should differ from NONE"


def test_restore_clip_denoise_step_after_primary_vs_after_secondary_differ_with_secondary(monkeypatch) -> None:
    import jasna.restorer.restoration_pipeline as rp

    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    pipeline_after_primary = RestorationPipeline(  # type: ignore[arg-type]
        restorer=_IdentityRestorer(),
        secondary_restorer=_Upscale2xSecondary(),
        denoise_strength=DenoiseStrength.MEDIUM,
        denoise_step=DenoiseStep.AFTER_PRIMARY,
    )
    pipeline_after_secondary = RestorationPipeline(  # type: ignore[arg-type]
        restorer=_IdentityRestorer(),
        secondary_restorer=_Upscale2xSecondary(),
        denoise_strength=DenoiseStrength.MEDIUM,
        denoise_step=DenoiseStep.AFTER_SECONDARY,
    )

    torch.manual_seed(456)
    T = 3
    frames = [torch.randint(0, 256, (3, 30, 40), dtype=torch.uint8) for _ in range(T)]
    bbox = np.array([5.0, 7.0, 25.0, 17.0], dtype=np.float32)
    clip = TrackedClip(
        track_id=0,
        start_frame=0,
        mask_resolution=(2, 2),
        bboxes=[bbox] * T,
        masks=[torch.zeros((2, 2), dtype=torch.bool)] * T,
    )

    restored_ap = pipeline_after_primary.restore_clip(clip, frames, keep_start=0, keep_end=T)
    restored_as = pipeline_after_secondary.restore_clip(clip, frames, keep_start=0, keep_end=T)

    diff = sum(
        (a.float() - b.float()).abs().sum().item()
        for a, b in zip(restored_ap.restored_frames, restored_as.restored_frames)
    )
    assert diff > 0, "AFTER_PRIMARY vs AFTER_SECONDARY with secondary should produce different output"


def test_restore_clip_denoise_step_no_secondary_both_steps_denoise(monkeypatch) -> None:
    import jasna.restorer.restoration_pipeline as rp

    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    pipeline_after_primary = RestorationPipeline(  # type: ignore[arg-type]
        restorer=_IdentityRestorer(),
        denoise_strength=DenoiseStrength.MEDIUM,
        denoise_step=DenoiseStep.AFTER_PRIMARY,
    )
    pipeline_after_secondary = RestorationPipeline(  # type: ignore[arg-type]
        restorer=_IdentityRestorer(),
        denoise_strength=DenoiseStrength.MEDIUM,
        denoise_step=DenoiseStep.AFTER_SECONDARY,
    )

    torch.manual_seed(789)
    T = 3
    frames = [torch.randint(0, 256, (3, 30, 40), dtype=torch.uint8) for _ in range(T)]
    bbox = np.array([5.0, 7.0, 25.0, 17.0], dtype=np.float32)
    clip = TrackedClip(
        track_id=0,
        start_frame=0,
        mask_resolution=(2, 2),
        bboxes=[bbox] * T,
        masks=[torch.zeros((2, 2), dtype=torch.bool)] * T,
    )

    restored_ap = pipeline_after_primary.restore_clip(clip, frames, keep_start=0, keep_end=T)
    restored_as = pipeline_after_secondary.restore_clip(clip, frames, keep_start=0, keep_end=T)

    for a, b in zip(restored_ap.restored_frames, restored_as.restored_frames):
        assert torch.allclose(a.float(), b.float(), atol=1e-5), "No secondary: both steps denoise same output"


def test_secondary_num_workers_no_secondary() -> None:
    pipeline = RestorationPipeline(restorer=_IdentityRestorer())  # type: ignore[arg-type]
    assert pipeline.secondary_num_workers == 1


def test_secondary_num_workers_with_secondary() -> None:
    pipeline = RestorationPipeline(  # type: ignore[arg-type]
        restorer=_IdentityRestorer(),
        secondary_restorer=_Upscale2xSecondaryList(),
    )
    assert pipeline.secondary_num_workers == 2


def _make_clip_and_frames(monkeypatch, *, t=3, frame_hw=(30, 40)):
    import jasna.restorer.restoration_pipeline as rp
    monkeypatch.setattr(rp, "BORDER_RATIO", 0.0)
    monkeypatch.setattr(rp, "MIN_BORDER", 0)
    monkeypatch.setattr(rp, "MAX_EXPANSION_FACTOR", 0.0)

    torch.manual_seed(42)
    frames = [torch.randint(0, 256, (3, frame_hw[0], frame_hw[1]), dtype=torch.uint8) for _ in range(t)]
    bbox = np.array([5.0, 7.0, 25.0, 17.0], dtype=np.float32)
    mask = torch.zeros((2, 2), dtype=torch.bool)
    clip = TrackedClip(
        track_id=0,
        start_frame=0,
        mask_resolution=(2, 2),
        bboxes=[bbox] * t,
        masks=[mask] * t,
    )
    return clip, frames


def test_prepare_and_run_primary(monkeypatch) -> None:
    clip, frames = _make_clip_and_frames(monkeypatch)
    pipeline = RestorationPipeline(restorer=_IdentityRestorer())  # type: ignore[arg-type]

    result = pipeline.prepare_and_run_primary(clip, frames, 0, 3, None)

    assert result.clip is clip
    assert result.frames is frames
    assert result.keep_start == 0
    assert result.keep_end == 3
    assert result.crossfade_weights is None
    assert result.primary_raw.shape[0] == 3
    assert len(result.enlarged_bboxes) == 3
    assert len(result.crop_shapes) == 3
    assert len(result.pad_offsets) == 3
    assert len(result.resize_shapes) == 3


def test_prepare_and_run_primary_with_denoise(monkeypatch) -> None:
    clip, frames = _make_clip_and_frames(monkeypatch)
    pipeline = RestorationPipeline(
        restorer=_IdentityRestorer(),  # type: ignore[arg-type]
        denoise_strength=DenoiseStrength.MEDIUM,
        denoise_step=DenoiseStep.AFTER_PRIMARY,
    )

    result = pipeline.prepare_and_run_primary(clip, frames, 0, 3, None)
    assert result.primary_raw.shape[0] == 3


def test_run_secondary_from_primary(monkeypatch) -> None:
    clip, frames = _make_clip_and_frames(monkeypatch)
    pipeline = RestorationPipeline(restorer=_IdentityRestorer())  # type: ignore[arg-type]

    pr = pipeline.prepare_and_run_primary(clip, frames, 0, 3, None)
    sr = pipeline.run_secondary_from_primary(pr)

    assert sr.clip is clip
    assert sr.frames is frames
    assert sr.keep_start == 0
    assert sr.keep_end == 3
    assert len(sr.restored_frames) == 3
    assert sr.enlarged_bboxes == pr.enlarged_bboxes
    assert sr.crop_shapes == pr.crop_shapes
    assert sr.pad_offsets == pr.pad_offsets
    assert sr.resize_shapes == pr.resize_shapes
    assert not hasattr(pr, 'primary_raw') or pr.primary_raw is None  # deleted


def test_run_secondary_from_primary_with_secondary_restorer(monkeypatch) -> None:
    clip, frames = _make_clip_and_frames(monkeypatch)
    pipeline = RestorationPipeline(  # type: ignore[arg-type]
        restorer=_IdentityRestorer(),
        secondary_restorer=_Upscale2xSecondary(),
    )

    pr = pipeline.prepare_and_run_primary(clip, frames, 0, 3, None)
    sr = pipeline.run_secondary_from_primary(pr)

    assert len(sr.restored_frames) == 3
    assert sr.restored_frames[0].shape == (3, 512, 512)


def test_blend_secondary_result(monkeypatch) -> None:
    from jasna.tracking.frame_buffer import FrameBuffer

    clip, frames = _make_clip_and_frames(monkeypatch)
    pipeline = RestorationPipeline(restorer=_IdentityRestorer())  # type: ignore[arg-type]

    fb = FrameBuffer(device=torch.device("cpu"))
    for i in range(3):
        fb.add_frame(i, pts=i * 100, frame=frames[i], clip_track_ids={0})

    pr = pipeline.prepare_and_run_primary(clip, frames, 0, 3, None)
    sr = pipeline.run_secondary_from_primary(pr)
    pipeline.blend_secondary_result(sr, fb)

    for i in range(3):
        pending = fb.frames[i]
        assert 0 not in pending.pending_clips


def test_blend_secondary_result_with_crossfade(monkeypatch) -> None:
    from jasna.tracking.frame_buffer import FrameBuffer

    clip, frames = _make_clip_and_frames(monkeypatch)
    pipeline = RestorationPipeline(restorer=_IdentityRestorer())  # type: ignore[arg-type]

    fb = FrameBuffer(device=torch.device("cpu"))
    for i in range(3):
        fb.add_frame(i, pts=i * 100, frame=frames[i], clip_track_ids={0})

    crossfade_weights = {0: 0.5, 1: 1.0, 2: 0.5}
    pr = pipeline.prepare_and_run_primary(clip, frames, 0, 3, crossfade_weights)
    sr = pipeline.run_secondary_from_primary(pr)
    pipeline.blend_secondary_result(sr, fb)

    for i in range(3):
        pending = fb.frames[i]
        assert 0 not in pending.pending_clips


def test_blend_secondary_result_skips_out_of_range_frames(monkeypatch) -> None:
    from jasna.tracking.frame_buffer import FrameBuffer

    clip, frames = _make_clip_and_frames(monkeypatch)
    pipeline = RestorationPipeline(restorer=_IdentityRestorer())  # type: ignore[arg-type]

    fb = FrameBuffer(device=torch.device("cpu"))
    for i in range(3):
        fb.add_frame(i, pts=i * 100, frame=frames[i], clip_track_ids={0})

    pr = pipeline.prepare_and_run_primary(clip, frames, 1, 2, None)
    sr = pipeline.run_secondary_from_primary(pr)
    pipeline.blend_secondary_result(sr, fb)

    assert 0 not in fb.frames[0].pending_clips
    assert 0 not in fb.frames[1].pending_clips
    assert 0 not in fb.frames[2].pending_clips


def test_blend_secondary_result_empty_range(monkeypatch) -> None:
    """Cover line 284: ks >= ke early return in blend_secondary_result."""
    from jasna.tracking.frame_buffer import FrameBuffer

    clip, frames = _make_clip_and_frames(monkeypatch, t=3)
    pipeline = RestorationPipeline(restorer=_IdentityRestorer())  # type: ignore[arg-type]

    fb = FrameBuffer(device=torch.device("cpu"))
    for i in range(3):
        fb.add_frame(i, pts=i * 100, frame=frames[i], clip_track_ids={0})

    pr = pipeline.prepare_and_run_primary(clip, frames, 2, 1, None)
    sr = pipeline.run_secondary_from_primary(pr)
    pipeline.blend_secondary_result(sr, fb)

    for i in range(3):
        assert 0 not in fb.frames[i].pending_clips


def test_blend_secondary_result_needs_blend_false(monkeypatch) -> None:
    """Cover line 291: needs_blend returns False, continue."""
    from jasna.tracking.frame_buffer import FrameBuffer

    clip, frames = _make_clip_and_frames(monkeypatch, t=2)
    pipeline = RestorationPipeline(restorer=_IdentityRestorer())  # type: ignore[arg-type]

    fb = FrameBuffer(device=torch.device("cpu"))
    fb.add_frame(0, pts=0, frame=frames[0], clip_track_ids=set())
    fb.add_frame(1, pts=1, frame=frames[1], clip_track_ids=set())

    pr = pipeline.prepare_and_run_primary(clip, frames, 0, 2, None)
    sr = pipeline.run_secondary_from_primary(pr)
    pipeline.blend_secondary_result(sr, fb)

    assert fb.frames[0].blended_frame is fb.frames[0].frame
    assert fb.frames[1].blended_frame is fb.frames[1].frame


def test_build_secondary_result(monkeypatch) -> None:
    clip, frames = _make_clip_and_frames(monkeypatch)
    pipeline = RestorationPipeline(restorer=_IdentityRestorer())  # type: ignore[arg-type]

    pr = pipeline.prepare_and_run_primary(clip, frames, 0, 3, None)
    restored_frames = [torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8) for _ in range(3)]
    sr = pipeline.build_secondary_result(pr, restored_frames)

    assert sr.clip is clip
    assert sr.frames is frames
    assert sr.restored_frames is restored_frames
    assert sr.keep_start == 0
    assert sr.keep_end == 3
    assert sr.enlarged_bboxes == pr.enlarged_bboxes


def test_build_secondary_result_with_denoise(monkeypatch) -> None:
    clip, frames = _make_clip_and_frames(monkeypatch)
    pipeline = RestorationPipeline(
        restorer=_IdentityRestorer(),  # type: ignore[arg-type]
        denoise_strength=DenoiseStrength.MEDIUM,
        denoise_step=DenoiseStep.AFTER_SECONDARY,
    )

    pr = pipeline.prepare_and_run_primary(clip, frames, 0, 3, None)
    restored_frames = [torch.randint(0, 255, (3, 256, 256), dtype=torch.uint8) for _ in range(3)]
    sr = pipeline.build_secondary_result(pr, restored_frames)

    assert len(sr.restored_frames) == 3
    assert sr.restored_frames[0].dtype == torch.uint8


def test_restore_and_blend_clip_needs_blend_false(monkeypatch) -> None:
    """Cover line 200: needs_blend returns False in restore_and_blend_clip."""
    from jasna.tracking.frame_buffer import FrameBuffer

    clip, frames = _make_clip_and_frames(monkeypatch, t=2)
    pipeline = RestorationPipeline(restorer=_IdentityRestorer())  # type: ignore[arg-type]

    fb = FrameBuffer(device=torch.device("cpu"))
    fb.add_frame(0, pts=0, frame=frames[0], clip_track_ids=set())
    fb.add_frame(1, pts=1, frame=frames[1], clip_track_ids=set())

    pipeline.restore_and_blend_clip(
        clip, frames, keep_start=0, keep_end=2, frame_buffer=fb,
    )

    assert fb.frames[0].blended_frame is fb.frames[0].frame
    assert fb.frames[1].blended_frame is fb.frames[1].frame

