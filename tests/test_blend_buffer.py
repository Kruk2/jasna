from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
import torch

from jasna.vr180 import StereoLayout
from jasna.blend_buffer import BlendBuffer
from jasna.pipeline_items import SecondaryRestoreResult


RSIZE = 256

pytestmark = pytest.mark.usefixtures("whole_box_blend_mask")


def _make_sr(
    track_id: int,
    start_frame: int,
    frame_count: int,
    frame_shape: tuple[int, int] = (8, 8),
    keep_start: int = 0,
    keep_end: int | None = None,
    clip_keep_offset: int = 0,
    fill_value: int = 200,
    crossfade_weights: dict[int, float] | None = None,
) -> SecondaryRestoreResult:
    ke = keep_end if keep_end is not None else frame_count
    kept = ke - keep_start
    fh, fw = frame_shape
    return SecondaryRestoreResult(
        track_id=track_id,
        start_frame=start_frame,
        frame_count=frame_count,
        frame_shape=frame_shape,
        frame_device=torch.device("cpu"),
        masks=[torch.ones(fh, fw, dtype=torch.bool) for _ in range(kept)],
        restored_frames=[torch.full((3, RSIZE, RSIZE), fill_value, dtype=torch.uint8) for _ in range(kept)],
        keep_start=0,
        keep_end=kept,
        crossfade_weights=crossfade_weights,
        enlarged_bboxes=[(0, 0, fw, fh)] * kept,
        crop_shapes=[(fh, fw)] * kept,
        pad_offsets=[(0, 0)] * kept,
        resize_shapes=[(fh, fw)] * kept,
        clip_keep_offset=clip_keep_offset,
    )




class TestBlendBufferReadiness:
    def test_frame_with_no_pending_tracks_is_ready(self):
        bb = BlendBuffer(vr_projector=None)
        bb.register_frame(0, set())
        assert bb.is_frame_ready(0)

    def test_unregistered_frame_is_ready(self):
        bb = BlendBuffer(vr_projector=None)
        assert bb.is_frame_ready(99)

    def test_frame_not_ready_until_result_added(self):
        bb = BlendBuffer(vr_projector=None)
        bb.register_frame(0, {1})
        assert not bb.is_frame_ready(0)

        sr = _make_sr(track_id=1, start_frame=0, frame_count=1)
        bb.add_result(sr)
        assert bb.is_frame_ready(0)

    def test_frame_with_two_tracks_needs_both_results(self):
        bb = BlendBuffer(vr_projector=None)
        bb.register_frame(0, {1, 2})
        assert not bb.is_frame_ready(0)

        bb.add_result(_make_sr(track_id=1, start_frame=0, frame_count=1))
        assert not bb.is_frame_ready(0)

        bb.add_result(_make_sr(track_id=2, start_frame=0, frame_count=1))
        assert bb.is_frame_ready(0)


class TestBlendBufferBlending:
    def test_blend_replaces_region(self):
        bb = BlendBuffer(vr_projector=None)
        bb.register_frame(0, {1})
        sr = _make_sr(track_id=1, start_frame=0, frame_count=1, fill_value=200)
        bb.add_result(sr)

        original = torch.zeros(3, 8, 8, dtype=torch.uint8)
        blended = bb.blend_frame(0, original)
        assert blended.shape == original.shape
        assert torch.all(blended == 200)

    @pytest.mark.parametrize("pending_tracks", [set(), {1}], ids=["no-pending-track", "no-result-arrived"])
    def test_blend_returns_the_original_frame_without_a_result(self, pending_tracks):
        bb = BlendBuffer(vr_projector=None)
        bb.register_frame(0, pending_tracks)
        original = torch.zeros(3, 8, 8, dtype=torch.uint8)

        assert bb.blend_frame(0, original) is original

    def test_blend_copies_when_only_some_tracks_have_results(self):
        bb = BlendBuffer(vr_projector=None)
        bb.register_frame(0, {1, 2})
        bb.add_result(_make_sr(track_id=1, start_frame=0, frame_count=1, fill_value=200))

        original = torch.zeros(3, 8, 8, dtype=torch.uint8)
        result = bb.blend_frame(0, original)

        assert result is not original
        assert torch.all(result == 200)
        assert torch.all(original == 0)

    def test_result_cleaned_up_after_last_frame(self):
        bb = BlendBuffer(vr_projector=None)
        bb.register_frame(0, {1})
        bb.register_frame(1, {1})
        sr = _make_sr(track_id=1, start_frame=0, frame_count=2)
        bb.add_result(sr)

        bb.blend_frame(0, torch.zeros(3, 8, 8, dtype=torch.uint8))
        assert bb.offloadable_results() == [sr]

        bb.blend_frame(1, torch.zeros(3, 8, 8, dtype=torch.uint8))
        assert bb.offloadable_results() == []

    def test_crossfade_weights_applied(self):
        bb = BlendBuffer(vr_projector=None)
        bb.register_frame(0, {1})
        sr = _make_sr(track_id=1, start_frame=0, frame_count=1, fill_value=100, crossfade_weights={0: 0.5})
        bb.add_result(sr)

        original = torch.full((3, 8, 8), 200, dtype=torch.uint8)
        blended = bb.blend_frame(0, original)
        expected = 200 + int(round((100 - 200) * 0.5))
        assert torch.all(blended == expected)

    def test_two_tracks_both_applied_on_same_frame(self):
        bb = BlendBuffer(vr_projector=None)
        bb.register_frame(0, {1, 2})

        left = _make_sr(track_id=1, start_frame=0, frame_count=1, fill_value=100)
        right = _make_sr(track_id=2, start_frame=0, frame_count=1, fill_value=200)
        bb.add_result(replace(left, enlarged_bboxes=[(0, 0, 4, 8)], crop_shapes=[(8, 4)]))
        bb.add_result(replace(right, enlarged_bboxes=[(4, 0, 8, 8)], crop_shapes=[(8, 4)]))

        blended = bb.blend_frame(0, torch.zeros(3, 8, 8, dtype=torch.uint8))
        assert torch.all(blended[:, :, :4] == 100)
        assert torch.all(blended[:, :, 4:] == 200)


class TestBlendBufferDiscardedFrames:
    def test_discarded_frames_outside_keep_range_cleared(self):
        bb = BlendBuffer(vr_projector=None)
        bb.register_frame(0, {1})
        bb.register_frame(1, {1})
        bb.register_frame(2, {1})

        sr = _make_sr(track_id=1, start_frame=0, frame_count=3, keep_start=0, keep_end=1, clip_keep_offset=1)
        bb.add_result(sr)

        assert all(bb.is_frame_ready(i) for i in range(3))
        original = torch.zeros(3, 8, 8, dtype=torch.uint8)
        assert bb.blend_frame(0, original) is original
        assert torch.all(bb.blend_frame(1, original) == 200)
        assert bb.blend_frame(2, original) is original


class TestBlendBufferPendingClip:
    def test_add_pending_clip_adds_track_to_existing_frames(self):
        bb = BlendBuffer(vr_projector=None)
        bb.register_frame(0, {1})
        bb.add_pending_clip([0], 2)
        assert bb.pending_map[0] == {1, 2}

    def test_remove_pending_clip_removes_track(self):
        bb = BlendBuffer(vr_projector=None)
        bb.register_frame(0, {1, 2})
        bb.remove_pending_clip([0], 2)
        assert bb.pending_map[0] == {1}

    def test_add_pending_clip_ignores_unregistered_frames(self):
        bb = BlendBuffer(vr_projector=None)
        bb.add_pending_clip([99], 1)
        assert 99 not in bb.pending_map


class TestBlendBufferThreadSafety:
    def test_concurrent_add_pending_and_blend(self):
        bb = BlendBuffer(vr_projector=None)
        for i in range(100):
            bb.register_frame(i, {1})
        sr = _make_sr(track_id=1, start_frame=0, frame_count=100, fill_value=200)
        bb.add_result(sr)

        def add_pending():
            for i in range(100):
                bb.add_pending_clip([i], 2)

        def blend():
            return [bb.blend_frame(i, torch.zeros(3, 8, 8, dtype=torch.uint8)) for i in range(100)]

        with ThreadPoolExecutor(2) as pool:
            adding, blending = pool.submit(add_pending), pool.submit(blend)
            adding.result()
            blended = blending.result()

        assert all(torch.all(frame == 200) for frame in blended)
        assert bb.pending_map == {}
        assert bb.offloadable_results() == []


def test_remove_pending_clip_skips_unregistered_frames() -> None:
    bb = BlendBuffer(vr_projector=None)
    bb.register_frame(0, {1})
    bb.remove_pending_clip([0, 99], 1)
    assert bb.is_frame_ready(0)


def _vr_sr(enlarged_bbox, frame_shape, fill_value):
    x1, y1, x2, y2 = enlarged_bbox
    ch, cw = y2 - y1, x2 - x1
    patch_size = max(ch, cw)
    return SecondaryRestoreResult(
        track_id=1,
        start_frame=0,
        frame_count=1,
        frame_shape=frame_shape,
        frame_device=torch.device("cpu"),
        masks=[torch.ones(ch, cw, dtype=torch.bool)],
        restored_frames=[torch.full((3, RSIZE, RSIZE), fill_value, dtype=torch.uint8)],
        keep_start=0,
        keep_end=1,
        crossfade_weights=None,
        enlarged_bboxes=[enlarged_bbox],
        crop_shapes=[(patch_size, patch_size)],
        pad_offsets=[(0, 0)],
        resize_shapes=[(patch_size, patch_size)],
        clip_keep_offset=0,
    )


def test_vr_delta_composite_preserves_pixels_outside_mask(monkeypatch) -> None:
    # With a real region projector and a nontrivial restoration, source pixels
    # where the blend mask is 0 must survive bit-for-bit — the delta composite
    # inverse-projects only the restoration change, never the whole patch.
    from jasna.vr_projection import GnomonicProjector

    enlarged = (4, 4, 28, 24)
    frame_shape = (32, 64)
    x1, y1, x2, y2 = enlarged
    rh, rw = y2 - y1, x2 - x1

    def half_mask_fn(mask_lr, bbox_xyxy, fshape):
        m = torch.zeros((rh, rw), dtype=torch.float32)
        m[rh // 2 :, :] = 1.0  # bottom half restored, top half untouched
        return m

    projector = GnomonicProjector(StereoLayout("sbs", 64, 32), device=torch.device("cpu"))
    monkeypatch.setattr("jasna.blend_buffer.create_bbox_blend_mask", half_mask_fn)
    bb = BlendBuffer(vr_projector=projector)
    bb.register_frame(0, {1})
    bb.add_result(_vr_sr(enlarged, frame_shape, fill_value=255))

    torch.manual_seed(3)
    original = torch.randint(0, 256, (3, 32, 64), dtype=torch.uint8)
    blended = bb.blend_frame(0, original)

    top = slice(y1, y1 + rh // 2)
    bottom = slice(y1 + rh // 2, y2)
    assert torch.equal(blended[:, top, x1:x2], original[:, top, x1:x2])
    assert not torch.equal(blended[:, bottom, x1:x2], original[:, bottom, x1:x2])
    # Everything outside the region is untouched too.
    outside = original.clone()
    outside[:, y1:y2, x1:x2] = blended[:, y1:y2, x1:x2]
    assert torch.equal(blended, outside)


def test_apply_blend_skips_out_of_range_frame() -> None:
    bb = BlendBuffer(vr_projector=None)
    bb.register_frame(0, {1})
    sr = _make_sr(track_id=1, start_frame=5, frame_count=3, frame_shape=(8, 8))
    bb.add_result(sr)
    original = torch.zeros((3, 8, 8), dtype=torch.uint8)
    blended = bb.blend_frame(0, original)
    assert torch.equal(blended, original)


def test_debug_lines_name_frames_waiting_for_missing_tracks() -> None:
    bb = BlendBuffer(vr_projector=None)
    bb.register_frame(4, {1, 2})
    bb.register_frame(5, {2})

    lines = bb.debug_lines()

    assert lines[0].startswith("blend_buffer: pending_frames=2 results=0")
    assert "frame_range=[4..5]" in lines[1]
    assert lines[2:] == [
        "blend_buffer: frame 4 waiting for tracks [1, 2]",
        "blend_buffer: frame 5 waiting for tracks [2]",
    ]
