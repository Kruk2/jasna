from __future__ import annotations

import threading
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

from jasna.blend_buffer import BlendBuffer
from jasna.crop_buffer import CropBuffer, RawCrop
from jasna.pipeline_items import SecondaryRestoreResult
from jasna.vram_offloader import STALL_WARN_SECONDS, VRAM_SAFETYNET, VramOffloader, VramStats, plan_offload, stall_warning_due


def _make_sr(
    track_id: int,
    start_frame: int,
    frame_count: int,
    frame_shape: tuple[int, int] = (8, 8),
    device: str = "cpu",
) -> SecondaryRestoreResult:
    fh, fw = frame_shape
    dev = torch.device(device)
    return SecondaryRestoreResult(
        track_id=track_id,
        start_frame=start_frame,
        frame_count=frame_count,
        frame_shape=frame_shape,
        frame_device=dev,
        masks=[torch.ones(fh, fw, dtype=torch.bool, device=dev) for _ in range(frame_count)],
        restored_frames=[torch.full((3, 256, 256), 200, dtype=torch.uint8, device=dev) for _ in range(frame_count)],
        keep_start=0,
        keep_end=frame_count,
        crossfade_weights=None,
        enlarged_bboxes=[(0, 0, fw, fh)] * frame_count,
        crop_shapes=[(fh, fw)] * frame_count,
        pad_offsets=[(0, 0)] * frame_count,
        resize_shapes=[(fh, fw)] * frame_count,
        clip_keep_offset=0,
    )


class TestVramStats:
    def test_initial_state(self):
        stats = VramStats()
        assert stats.sample_count == 0
        assert stats.avg_bytes == 0.0

    def test_single_update(self):
        stats = VramStats()
        stats.update(1000)
        assert stats.min_bytes == 1000
        assert stats.max_bytes == 1000
        assert stats.avg_bytes == 1000.0
        assert stats.sample_count == 1

    def test_multiple_updates(self):
        stats = VramStats()
        stats.update(100)
        stats.update(300)
        stats.update(200)
        assert stats.min_bytes == 100
        assert stats.max_bytes == 300
        assert stats.avg_bytes == 200.0
        assert stats.sample_count == 3

    def test_summary_no_samples(self):
        stats = VramStats()
        assert "no samples" in stats.summary()

    def test_summary_with_data(self):
        stats = VramStats()
        stats.update(1024 * 1024)
        stats.offload_count = 3
        stats.total_offloaded_bytes = 512 * 1024 * 1024
        summary = stats.summary()
        assert "1 MiB" in summary
        assert "offloads: 3" in summary


def _crop_buffer(track_id: int, frames: int) -> CropBuffer:
    buffer = CropBuffer(track_id=track_id, start_frame=0)
    for _ in range(frames):
        buffer.add(RawCrop(crop=torch.zeros((3, 40, 40), dtype=torch.uint8), enlarged_bbox=(0, 0, 40, 40), crop_shape=(40, 40)))
    return buffer


def _frame_bytes(result: SecondaryRestoreResult) -> int:
    return result.restored_frames[0].nelement() * result.restored_frames[0].element_size()


class TestOffloadPlan:
    def test_the_latest_clip_goes_first_with_its_masks(self):
        early = _make_sr(track_id=1, start_frame=0, frame_count=2)
        late = _make_sr(track_id=2, start_frame=100, frame_count=2)

        steps = plan_offload([early, late], [], 2 * _frame_bytes(late) + 1, "cpu")

        assert [step.tensor for step in steps] == [*late.restored_frames, *late.masks, early.restored_frames[0]]

    def test_it_stops_once_enough_is_planned(self):
        late = _make_sr(track_id=2, start_frame=100, frame_count=3)

        steps = plan_offload([late], [], _frame_bytes(late), "cpu")

        assert [step.tensor for step in steps] == [late.restored_frames[0]]

    def test_crops_come_after_restored_clips_largest_buffer_first(self):
        result = _make_sr(track_id=1, start_frame=0, frame_count=1)
        small, large = _crop_buffer(1, 1), _crop_buffer(2, 2)

        steps = plan_offload([result], [small, large], 10**12, "cpu")

        assert [step.tensor for step in steps] == [
            *result.restored_frames, *result.masks, *(raw.crop for raw in large.crops), small.crops[0].crop,
        ]
        assert sum(step.nbytes for step in steps) == sum(step.tensor.nelement() for step in steps)

    def test_tensors_already_off_the_device_are_left_alone(self):
        assert plan_offload([_make_sr(track_id=1, start_frame=0, frame_count=2)], [_crop_buffer(1, 1)], 1, "cuda") == []

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_planned_steps_move_the_tensors_to_the_cpu(self):
        result = _make_sr(track_id=1, start_frame=0, frame_count=1, device="cuda")
        crops = _crop_buffer(1, 1)
        crops.crops[0].crop = crops.crops[0].crop.cuda()

        for step in plan_offload([result], [crops], 10**12, "cuda"):
            step.move()

        assert {result.restored_frames[0].device.type, result.masks[0].device.type, crops.crops[0].crop.device.type} == {"cpu"}


class TestVramOffloaderLoop:
    def test_running_over_the_threshold_offloads_and_empties_the_cache(self, monkeypatch):
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: SimpleNamespace(total_memory=VRAM_SAFETYNET + 1000))
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _device: (0, VRAM_SAFETYNET + 2000))
        emptied = threading.Event()
        monkeypatch.setattr(torch.cuda, "empty_cache", emptied.set)
        blend_buffer = BlendBuffer(vr_projector=None)
        blend_buffer.register_frame(0, {1})
        blend_buffer.add_result(_make_sr(track_id=1, start_frame=0, frame_count=1))

        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=blend_buffer,
            crop_buffers={},
            frame_queues={},
            metadata_queue=Queue(),
            encode_heartbeat=None,
        )
        offloader.start()
        assert emptied.wait(timeout=30)
        offloader.stop()

        assert offloader.stats.offload_count >= 1
        assert offloader.stats.total_offloaded_bytes > 0


@pytest.mark.parametrize(
    ("since_encode", "since_warning", "warn", "keeps_warning_time"),
    [
        (STALL_WARN_SECONDS - 1, 1000.0, False, False),
        (STALL_WARN_SECONDS + 1, 1000.0, True, False),
        (STALL_WARN_SECONDS + 1, STALL_WARN_SECONDS - 1, False, True),
        (STALL_WARN_SECONDS + 1, STALL_WARN_SECONDS, True, False),
    ],
    ids=["encoding", "stalled", "warned-recently", "warned-a-while-ago"],
)
def test_an_encode_stall_warns_once_per_interval(since_encode, since_warning, warn, keeps_warning_time):
    now = 5000.0
    last_warning = now - since_warning
    expected_warning_time = last_warning if keeps_warning_time else (now if warn else 0.0)

    assert stall_warning_due(now, now - since_encode, last_warning) == (warn, expected_warning_time)
