from __future__ import annotations

import threading

import torch

from jasna.blend_buffer import BlendBuffer
from jasna.pipeline_items import SecondaryRestoreResult


RSIZE = 256


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


def _identity_blend_mask(crop_mask: torch.Tensor) -> torch.Tensor:
    return crop_mask


class TestBlendBufferReadiness:
    def test_frame_with_no_pending_tracks_is_ready(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        bb.register_frame(0, set())
        assert bb.is_frame_ready(0)

    def test_unregistered_frame_is_ready(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        assert bb.is_frame_ready(99)

    def test_frame_not_ready_until_result_added(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        bb.register_frame(0, {1})
        assert not bb.is_frame_ready(0)

        sr = _make_sr(track_id=1, start_frame=0, frame_count=1)
        bb.add_result(sr)
        assert bb.is_frame_ready(0)

    def test_frame_with_two_tracks_needs_both_results(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        bb.register_frame(0, {1, 2})
        assert not bb.is_frame_ready(0)

        bb.add_result(_make_sr(track_id=1, start_frame=0, frame_count=1))
        assert not bb.is_frame_ready(0)

        bb.add_result(_make_sr(track_id=2, start_frame=0, frame_count=1))
        assert bb.is_frame_ready(0)


class TestBlendBufferBlending:
    def test_blend_replaces_region(self):
        bb = BlendBuffer(device=torch.device("cpu"), blend_mask_fn=_identity_blend_mask)
        bb.register_frame(0, {1})
        sr = _make_sr(track_id=1, start_frame=0, frame_count=1, fill_value=200)
        bb.add_result(sr)

        original = torch.zeros(3, 8, 8, dtype=torch.uint8)
        blended = bb.blend_frame(0, original)
        assert blended.shape == original.shape
        assert torch.all(blended == 200)

    def test_blend_no_pending_returns_original(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        original = torch.zeros(3, 8, 8, dtype=torch.uint8)
        result = bb.blend_frame(0, original)
        assert result is original

    def test_result_cleaned_up_after_last_frame(self):
        bb = BlendBuffer(device=torch.device("cpu"), blend_mask_fn=_identity_blend_mask)
        bb.register_frame(0, {1})
        bb.register_frame(1, {1})
        sr = _make_sr(track_id=1, start_frame=0, frame_count=2)
        bb.add_result(sr)

        bb.blend_frame(0, torch.zeros(3, 8, 8, dtype=torch.uint8))
        assert 1 in bb._results

        bb.blend_frame(1, torch.zeros(3, 8, 8, dtype=torch.uint8))
        assert 1 not in bb._results
        assert 1 not in bb._result_last_frame

    def test_crossfade_weights_applied(self):
        bb = BlendBuffer(device=torch.device("cpu"), blend_mask_fn=_identity_blend_mask)
        bb.register_frame(0, {1})
        sr = _make_sr(track_id=1, start_frame=0, frame_count=1, fill_value=100, crossfade_weights={0: 0.5})
        bb.add_result(sr)

        original = torch.full((3, 8, 8), 200, dtype=torch.uint8)
        blended = bb.blend_frame(0, original)
        expected = 200 + int(round((100 - 200) * 0.5))
        assert torch.all(blended == expected)

    def test_two_tracks_both_applied_on_same_frame(self):
        bb = BlendBuffer(device=torch.device("cpu"), blend_mask_fn=_identity_blend_mask)
        bb.register_frame(0, {1, 2})

        bb.add_result(_make_sr(track_id=1, start_frame=0, frame_count=1, fill_value=100))
        bb.add_result(_make_sr(track_id=2, start_frame=0, frame_count=1, fill_value=200))

        original = torch.zeros(3, 8, 8, dtype=torch.uint8)
        blended = bb.blend_frame(0, original)
        assert not torch.all(blended == 0), "at least one track must have been blended"


class TestBlendBufferDiscardedFrames:
    def test_discarded_frames_outside_keep_range_cleared(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        bb.register_frame(0, {1})
        bb.register_frame(1, {1})
        bb.register_frame(2, {1})

        sr = _make_sr(track_id=1, start_frame=0, frame_count=3, keep_start=0, keep_end=1, clip_keep_offset=1)
        bb.add_result(sr)

        assert bb.is_frame_ready(0)
        assert bb.is_frame_ready(2)
        assert not bb.is_frame_ready(1) or bb.is_frame_ready(1)


class TestBlendBufferPendingClip:
    def test_add_pending_clip_adds_track_to_existing_frames(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        bb.register_frame(0, {1})
        bb.add_pending_clip([0], 2)
        assert bb.pending_map[0] == {1, 2}

    def test_remove_pending_clip_removes_track(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        bb.register_frame(0, {1, 2})
        bb.remove_pending_clip([0], 2)
        assert bb.pending_map[0] == {1}

    def test_add_pending_clip_ignores_unregistered_frames(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        bb.add_pending_clip([99], 1)
        assert 99 not in bb.pending_map


class TestBlendBufferThreadSafety:
    def test_concurrent_register_and_ready_check(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        errors = []

        def writer():
            try:
                for i in range(1000):
                    bb.register_frame(i, {1})
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(1000):
                    bb.is_frame_ready(i)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors

    def test_concurrent_add_pending_and_blend(self):
        bb = BlendBuffer(device=torch.device("cpu"), blend_mask_fn=_identity_blend_mask)
        for i in range(100):
            bb.register_frame(i, {1})
        sr = _make_sr(track_id=1, start_frame=0, frame_count=100, fill_value=200)
        bb.add_result(sr)

        errors = []

        def adder():
            try:
                for i in range(100):
                    bb.add_pending_clip([i], 2)
            except Exception as e:
                errors.append(e)

        def blender():
            try:
                for i in range(100):
                    bb.blend_frame(i, torch.zeros(3, 8, 8, dtype=torch.uint8))
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=adder)
        t2 = threading.Thread(target=blender)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
        assert not errors


def test_remove_pending_clip_skips_unregistered_frames() -> None:
    bb = BlendBuffer(device=torch.device("cpu"))
    bb.register_frame(0, {1})
    bb.remove_pending_clip([0, 99], 1)
    assert bb.is_frame_ready(0)


def test_apply_blend_skips_out_of_range_frame() -> None:
    bb = BlendBuffer(device=torch.device("cpu"))
    bb.register_frame(0, {1})
    sr = _make_sr(track_id=1, start_frame=5, frame_count=3, frame_shape=(8, 8))
    bb.add_result(sr)
    original = torch.zeros((3, 8, 8), dtype=torch.uint8)
    blended = bb.blend_frame(0, original)
    assert torch.equal(blended, original)
