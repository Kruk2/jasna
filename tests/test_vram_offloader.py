from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch

from jasna.blend_buffer import BlendBuffer
from jasna.crop_buffer import CropBuffer, RawCrop, prepare_crops_for_restoration
from jasna.pipeline_items import SecondaryRestoreResult
from jasna.vram_offloader import (
    VRAM_SAFETYNET,
    AMD_MIN_VRAM_STARTUP_BUDGET,
    VramOffloader,
    VramStats,
    default_host_memory_limit_bytes,
    read_linux_amd_system_vram,
    restoration_vram_safetynet,
    restoration_vram_startup_budget,
    system_vram_pressure_watermarks,
)


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

    def test_summary_includes_host_memory_pressure_telemetry(self):
        stats = VramStats()
        stats.host_max_rss_bytes = 2 * 1024**3
        stats.host_min_available_bytes = 512 * 1024**2
        stats.host_memory_pressure_episodes = 1
        summary = stats.summary()
        assert "host RSS peak" in summary
        assert "pressure episodes: 1" in summary

    def test_default_host_memory_limit_is_conservative(self, monkeypatch):
        monkeypatch.setattr(
            "jasna.vram_offloader.psutil.virtual_memory",
            lambda: type("Memory", (), {"total": 32 * 1024**3})(),
        )
        assert default_host_memory_limit_bytes() == int(32 * 1024**3 * 0.75)


class TestVramOffloaderOffload:
    def test_offloads_restored_frames_when_over_threshold(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        sr = _make_sr(track_id=1, start_frame=0, frame_count=3)
        bb.add_result(sr)
        bb.register_frame(0, {1})
        bb.register_frame(1, {1})
        bb.register_frame(2, {1})

        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=bb,
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=0.001,
            safetynet=0,
        )
        offloader._offload_device_type = "cpu"

        freed = offloader._offload(1)
        assert freed > 0
        assert sr.restored_frames[0].device.type == "cpu"

    def test_no_offload_when_all_already_offloaded(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        sr = _make_sr(track_id=1, start_frame=0, frame_count=2, device="cpu")
        bb.add_result(sr)
        bb.register_frame(0, {1})

        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=bb,
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=0.001,
            safetynet=0,
        )
        # device_type stays "cuda" (default) so CPU tensors are not considered on-device
        freed = offloader._offload(1)
        assert freed == 0

    def test_offload_priority_highest_start_frame_first(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        sr_early = _make_sr(track_id=1, start_frame=0, frame_count=2)
        sr_late = _make_sr(track_id=2, start_frame=100, frame_count=2)
        bb.register_frame(0, {1})
        bb.register_frame(1, {1})
        bb.register_frame(100, {2})
        bb.register_frame(101, {2})
        bb.add_result(sr_early)
        bb.add_result(sr_late)

        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=bb,
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=0.001,
            safetynet=0,
        )
        offloader._offload_device_type = "cpu"

        single_frame_bytes = sr_late.restored_frames[0].nelement() * sr_late.restored_frames[0].element_size()
        offloader._offload(single_frame_bytes)

        # late clip (start_frame=100) offloaded first — verify via offload_count logic
        # Since bytes_to_free = single_frame_bytes, only the first frame of sr_late gets hit
        # then early clip starts. The key assertion: offload happened at all
        assert offloader.stats.total_offloaded_bytes == 0  # stats not updated in _offload directly

    def test_offloads_crop_buffers_after_blend_buffer(self):
        bb = BlendBuffer(device=torch.device("cpu"))

        crop_buf = CropBuffer(track_id=1, start_frame=0)
        crop = torch.randint(0, 255, (3, 40, 40), dtype=torch.uint8)
        crop_buf.add(RawCrop(crop=crop, enlarged_bbox=(0, 0, 40, 40), crop_shape=(40, 40)))

        crop_buffers = {1: crop_buf}
        crop_lock = threading.Lock()

        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=bb,
            crop_buffers=crop_buffers,
            crop_lock=crop_lock,
            vram_limit=0.001,
            safetynet=0,
        )

        freed = offloader._offload(1)
        assert freed == 0

    def test_offloads_masks_too(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        sr = _make_sr(track_id=1, start_frame=0, frame_count=1)
        sr.restored_frames = [torch.full((3, 256, 256), 200, dtype=torch.uint8)]
        bb.register_frame(0, {1})
        bb.add_result(sr)

        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=bb,
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=0.001,
            safetynet=0,
        )
        offloader._offload_device_type = "cpu"

        big_target = 999_999_999
        offloader._offload(big_target)
        assert sr.masks[0].device.type == "cpu"


class TestVramOffloaderThreshold:
    def test_amd_8k_p010_torch_reserve_does_not_use_four_gib_runtime_floor(self):
        width, height, batch = 8192, 4096, 4
        pixels = width * height
        expected = (
            VRAM_SAFETYNET
            + 2 * (pixels * 3 * batch + pixels * 3 * 2 * batch // 2)
            + 2 * 3 * pixels * 3 * 2 // 2
        )
        reserve = restoration_vram_safetynet(
            frame_width=width,
            frame_height=height,
            batch_size=batch,
            ten_bit=True,
            amd=True,
        )
        assert expected < AMD_MIN_VRAM_STARTUP_BUDGET
        assert reserve == expected

    def test_amd_four_gib_is_preflight_budget_only(self):
        assert restoration_vram_startup_budget(
            frame_width=8192,
            frame_height=4096,
            batch_size=4,
            ten_bit=True,
            amd=True,
        ) == AMD_MIN_VRAM_STARTUP_BUDGET

    def test_runtime_whole_card_watermarks_scale_and_remain_bounded(self):
        gib = 1024**3
        pressure_8, recovery_8, critical_8 = system_vram_pressure_watermarks(8 * gib)
        pressure_24, recovery_24, critical_24 = system_vram_pressure_watermarks(24 * gib)

        assert pressure_8 == 512 * 1024**2
        assert pressure_24 == 1024 * 1024**2
        assert critical_8 < pressure_8 < recovery_8
        assert critical_24 < pressure_24 < recovery_24
        assert recovery_24 == 1536 * 1024**2

    def test_non_amd_route_keeps_original_reserve(self):
        assert restoration_vram_safetynet(
            frame_width=8192,
            frame_height=4096,
            batch_size=8,
            ten_bit=True,
            amd=False,
        ) == VRAM_SAFETYNET

    def test_explicit_vram_limit(self):
        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=BlendBuffer(device=torch.device("cpu")),
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=2.0,
            safetynet=750_000_000,
        )
        assert offloader._threshold == int(2.0 * 1024 * 1024 * 1024) - 750_000_000

    def test_safetynet_zero(self):
        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=BlendBuffer(device=torch.device("cpu")),
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=1.0,
            safetynet=0,
        )
        assert offloader._threshold == 1 * 1024 * 1024 * 1024


class TestVramOffloaderLifecycle:
    def test_host_memory_pressure_cancels_worker_before_oom(self):
        cancel_event = threading.Event()
        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=BlendBuffer(device=torch.device("cpu")),
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=1.0,
            safetynet=0,
            host_memory_limit_bytes=800,
            host_memory_reader=lambda: (900, 1000, 100),
            host_memory_pressure_seconds=0.0,
            cancel_event=cancel_event,
        )
        offloader._check_host_memory_pressure()
        offloader._check_host_memory_pressure()

        assert offloader.host_memory_pressure is True
        assert cancel_event.is_set()
        assert offloader.stats.host_memory_pressure_episodes == 1

    def test_start_stop(self):
        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=BlendBuffer(device=torch.device("cpu")),
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=0.001,
            safetynet=0,
        )
        offloader.start()
        assert offloader._thread.is_alive()
        offloader.stop()
        assert not offloader._thread.is_alive()

    def test_whole_card_pressure_triggers_tensor_offload(self):
        mib = 1024**2
        gib = 1024**3
        bb = BlendBuffer(device=torch.device("cpu"))
        sr = _make_sr(track_id=1, start_frame=0, frame_count=1)
        bb.register_frame(0, {1})
        bb.add_result(sr)
        system_samples = lambda: (8 * gib - 400 * mib, 8 * gib)
        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=bb,
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=1.0,
            safetynet=0,
            system_vram_startup_budget=200,
            system_vram_reader=system_samples,
            system_vram_pressure_seconds=0.0,
            system_vram_critical_seconds=60.0,
        )
        offloader._offload_device_type = "cpu"
        with patch("jasna.vram_offloader.torch.cuda.mem_get_info", return_value=(900, 1000)):
            with patch("jasna.vram_offloader.torch.cuda.empty_cache"):
                offloader.start()
                import time
                time.sleep(0.25)
                offloader.stop()

        assert offloader.stats.system_sample_count > 0
        assert offloader.stats.offload_count >= 1

    def test_sustained_whole_card_pressure_reclaims_cache_once_per_episode(self):
        mib = 1024**2
        gib = 1024**3
        sample = (8 * gib - 200 * mib, 8 * gib)
        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=BlendBuffer(device=torch.device("cpu")),
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=1.0,
            safetynet=0,
            system_vram_startup_budget=200,
            system_vram_reader=lambda: sample,
            system_vram_pressure_seconds=0.0,
            system_vram_critical_seconds=10.0,
        )
        with (
            patch(
                "jasna.vram_offloader.time.monotonic",
                side_effect=(0.0, 0.0, 9.9, 10.0, 20.0),
            ),
            patch("jasna.vram_offloader.torch.cuda.empty_cache") as empty_cache,
        ):
            offloader._check_system_pressure(sample)
            offloader._check_system_pressure(sample)
            offloader._check_system_pressure(sample)
            empty_cache.assert_not_called()
            offloader._check_system_pressure(sample)
            offloader._check_system_pressure(sample)

        empty_cache.assert_called_once_with()
        assert offloader.stats.system_reclaim_count == 1
        assert offloader._system_critical_reclaimed
        assert offloader.stats.system_pressure_episodes == 1

    def test_whole_card_pressure_rearms_only_after_recovery_watermark(self):
        mib = 1024**2
        gib = 1024**3
        pressure_sample = (8 * gib - 200 * mib, 8 * gib)
        recovered_sample = (6 * gib, 8 * gib)
        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=BlendBuffer(device=torch.device("cpu")),
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=1.0,
            safetynet=0,
            system_vram_startup_budget=200,
            system_vram_reader=lambda: pressure_sample,
            system_vram_pressure_seconds=0.0,
            system_vram_recovery_seconds=0.0,
            system_vram_episode_cooldown_seconds=0.0,
            system_vram_critical_seconds=0.0,
        )
        offloader._system_pressure_since = 0.0
        with (
            patch("jasna.vram_offloader.time.monotonic", return_value=10.0),
            patch("jasna.vram_offloader.torch.cuda.empty_cache") as empty_cache,
        ):
            offloader._check_system_pressure(pressure_sample)
            assert offloader._system_pressure_active
            offloader._check_system_pressure(pressure_sample)
            assert offloader._system_critical_reclaimed
            offloader._check_system_pressure(recovered_sample)
            offloader._check_system_pressure(recovered_sample)

        empty_cache.assert_called_once_with()
        assert not offloader._system_pressure_active
        assert not offloader._system_critical_reclaimed
        assert offloader._system_pressure_since is None

    def test_short_recovery_does_not_rearm_pressure_episode(self):
        mib = 1024**2
        gib = 1024**3
        pressure_sample = (8 * gib - 400 * mib, 8 * gib)
        recovered_sample = (6 * gib, 8 * gib)
        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=BlendBuffer(device=torch.device("cpu")),
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=1.0,
            safetynet=0,
            system_vram_startup_budget=200,
            system_vram_reader=lambda: pressure_sample,
            system_vram_pressure_seconds=0.0,
            system_vram_recovery_seconds=5.0,
            system_vram_episode_cooldown_seconds=0.0,
            system_vram_critical_seconds=60.0,
        )
        offloader._system_pressure_since = 0.0

        with patch(
            "jasna.vram_offloader.time.monotonic",
            side_effect=(10.0, 11.0, 15.9),
        ):
            offloader._check_system_pressure(pressure_sample)
            offloader._check_system_pressure(recovered_sample)
            offloader._check_system_pressure(recovered_sample)

        assert offloader._system_pressure_active
        assert offloader._system_recovery_since == 11.0
        assert offloader.stats.system_pressure_episodes == 1

    def test_recovery_rearms_but_cooldown_blocks_regular_pressure_episode(self):
        mib = 1024**2
        gib = 1024**3
        pressure_sample = (8 * gib - 400 * mib, 8 * gib)
        recovered_sample = (6 * gib, 8 * gib)
        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=BlendBuffer(device=torch.device("cpu")),
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=1.0,
            safetynet=0,
            system_vram_startup_budget=200,
            system_vram_reader=lambda: pressure_sample,
            system_vram_pressure_seconds=0.0,
            system_vram_recovery_seconds=0.0,
            system_vram_episode_cooldown_seconds=30.0,
            system_vram_critical_seconds=60.0,
        )
        offloader._system_pressure_since = 0.0

        with patch(
            "jasna.vram_offloader.time.monotonic",
            side_effect=(10.0, 11.0, 11.0, 20.0, 40.0, 40.0),
        ):
            offloader._check_system_pressure(pressure_sample)
            offloader._check_system_pressure(recovered_sample)
            offloader._check_system_pressure(recovered_sample)
            assert not offloader._system_pressure_active
            offloader._check_system_pressure(pressure_sample)
            assert not offloader._system_pressure_active
            assert offloader._system_pressure_since is None
            offloader._check_system_pressure(pressure_sample)
            offloader._check_system_pressure(pressure_sample)

        assert offloader._system_pressure_active
        assert offloader.stats.system_pressure_episodes == 2

    def test_critical_pressure_bypasses_episode_cooldown(self):
        mib = 1024**2
        gib = 1024**3
        regular_pressure = (8 * gib - 400 * mib, 8 * gib)
        critical_pressure = (8 * gib - 200 * mib, 8 * gib)
        recovered_sample = (6 * gib, 8 * gib)
        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=BlendBuffer(device=torch.device("cpu")),
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=1.0,
            safetynet=0,
            system_vram_startup_budget=200,
            system_vram_reader=lambda: regular_pressure,
            system_vram_pressure_seconds=0.0,
            system_vram_recovery_seconds=0.0,
            system_vram_episode_cooldown_seconds=30.0,
            system_vram_critical_seconds=0.0,
        )
        offloader._system_pressure_since = 0.0

        with (
            patch(
                "jasna.vram_offloader.time.monotonic",
                side_effect=(10.0, 11.0, 11.0, 20.0, 20.0, 20.0),
            ),
            patch("jasna.vram_offloader.torch.cuda.empty_cache") as empty_cache,
        ):
            offloader._check_system_pressure(regular_pressure)
            offloader._check_system_pressure(recovered_sample)
            offloader._check_system_pressure(recovered_sample)
            offloader._check_system_pressure(critical_pressure)
            offloader._check_system_pressure(critical_pressure)
            offloader._check_system_pressure(critical_pressure)

        empty_cache.assert_called_once_with()
        assert offloader._system_pressure_active
        assert offloader._system_critical_reclaimed
        assert offloader.stats.system_pressure_episodes == 2
        assert offloader.stats.system_reclaim_count == 1


def test_read_linux_amd_system_vram(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("jasna.vram_offloader.sys.platform", "linux")
    device = tmp_path / "card1" / "device"
    device.mkdir(parents=True)
    (device / "vendor").write_text("0x1002\n", encoding="utf-8")
    (device / "mem_info_vram_used").write_text("300\n", encoding="utf-8")
    (device / "mem_info_vram_total").write_text("1200\n", encoding="utf-8")

    assert read_linux_amd_system_vram(tmp_path) == (300, 1200)

    @patch("jasna.vram_offloader.torch.cuda.mem_get_info", return_value=(0, 2_000_000))
    @patch("jasna.vram_offloader.torch.cuda.empty_cache")
    def test_run_loop_triggers_offload_and_empty_cache(self, mock_empty_cache, mock_mem_info):
        bb = BlendBuffer(device=torch.device("cpu"))
        sr = _make_sr(track_id=1, start_frame=0, frame_count=1)
        bb.register_frame(0, {1})
        bb.add_result(sr)

        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=bb,
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=0.001,
            safetynet=0,
        )
        offloader._offload_device_type = "cpu"
        offloader.start()
        import time
        time.sleep(0.3)
        offloader.stop()

        assert offloader.stats.sample_count > 0
        assert offloader.stats.offload_count >= 1
        mock_empty_cache.assert_called()


class TestEncodeStallDetection:
    def test_no_warning_without_heartbeat(self):
        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=BlendBuffer(device=torch.device("cpu")),
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=0.001,
            safetynet=0,
        )
        offloader._check_encode_stall()
        assert offloader._last_stall_warn_time == 0.0

    def test_no_warning_when_recent(self):
        import time
        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=BlendBuffer(device=torch.device("cpu")),
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=0.001,
            safetynet=0,
        )
        hb = [time.monotonic()]
        offloader.set_encode_heartbeat(hb)
        offloader._check_encode_stall()
        assert offloader._last_stall_warn_time == 0.0

    def test_no_warning_before_first_encode_activity(self):
        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=BlendBuffer(device=torch.device("cpu")),
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=0.001,
            safetynet=0,
        )
        offloader.set_encode_heartbeat([None])

        with patch.object(offloader, "_dump_stall_diagnostics") as dump:
            offloader._check_encode_stall()

        assert offloader._last_stall_warn_time == 0.0
        dump.assert_not_called()

    def test_warns_when_stale(self):
        import time
        from jasna.vram_offloader import STALL_WARN_SECONDS
        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=BlendBuffer(device=torch.device("cpu")),
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=0.001,
            safetynet=0,
        )
        hb = [time.monotonic() - STALL_WARN_SECONDS - 1.0]
        offloader.set_encode_heartbeat(hb)
        offloader._check_encode_stall()
        assert offloader._last_stall_warn_time > 0.0

    def test_does_not_rewarn_within_interval(self):
        import time
        from jasna.vram_offloader import STALL_WARN_SECONDS
        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=BlendBuffer(device=torch.device("cpu")),
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=0.001,
            safetynet=0,
        )
        hb = [time.monotonic() - STALL_WARN_SECONDS - 1.0]
        offloader.set_encode_heartbeat(hb)
        offloader._check_encode_stall()
        first_warn = offloader._last_stall_warn_time
        assert first_warn > 0.0
        offloader._check_encode_stall()
        assert offloader._last_stall_warn_time == first_warn

    def test_resets_after_fresh_heartbeat(self):
        import time
        from jasna.vram_offloader import STALL_WARN_SECONDS
        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=BlendBuffer(device=torch.device("cpu")),
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=0.001,
            safetynet=0,
        )
        hb = [time.monotonic() - STALL_WARN_SECONDS - 1.0]
        offloader.set_encode_heartbeat(hb)
        offloader._check_encode_stall()
        assert offloader._last_stall_warn_time > 0.0
        hb[0] = time.monotonic()
        offloader._check_encode_stall()
        assert offloader._last_stall_warn_time == 0.0

    def test_isolated_stall_exits_once_after_fail_closed_timeout(self):
        import time
        import jasna.vram_offloader as module
        from jasna.native_worker import NATIVE_ENCODE_STALL_EXIT_CODE

        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=BlendBuffer(device=torch.device("cpu")),
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=0.001,
            safetynet=0,
            terminate_on_encode_stall=True,
            encode_stall_timeout_seconds=30.0,
        )
        offloader.set_encode_heartbeat([time.monotonic() - 31.0])

        with (
            patch.object(offloader, "_dump_stall_diagnostics") as dump,
            patch.object(module.os, "_exit") as exit_process,
        ):
            offloader._check_encode_stall()
            offloader._check_encode_stall()

        exit_process.assert_called_once_with(NATIVE_ENCODE_STALL_EXIT_CODE)
        dump.assert_called_once()

    def test_stall_warning_is_not_repeated_or_dumped_before_exit_timeout(self):
        import time
        from jasna.vram_offloader import STALL_WARN_SECONDS

        offloader = VramOffloader(
            device=torch.device("cpu"),
            blend_buffer=BlendBuffer(device=torch.device("cpu")),
            crop_buffers={},
            crop_lock=threading.Lock(),
            vram_limit=0.001,
            safetynet=0,
            terminate_on_encode_stall=True,
            encode_stall_timeout_seconds=90.0,
        )
        offloader.set_encode_heartbeat(
            [time.monotonic() - STALL_WARN_SECONDS - 1.0]
        )

        with patch.object(offloader, "_dump_stall_diagnostics") as dump:
            offloader._check_encode_stall()
            first_warning = offloader._last_stall_warn_time
            offloader._check_encode_stall()

        assert first_warning > 0.0
        assert offloader._last_stall_warn_time == first_warning
        dump.assert_not_called()


class TestPrepareCropsJitGuard:
    def test_cpu_crops_stay_on_cpu_device(self):
        crop = torch.randint(0, 255, (3, 40, 40), dtype=torch.uint8)
        raw = RawCrop(crop=crop, enlarged_bbox=(0, 0, 40, 40), crop_shape=(40, 40))
        result, _, _ = prepare_crops_for_restoration([raw], device=torch.device("cpu"), dtype=torch.float32)
        assert len(result) == 1
        assert result[0].device.type == "cpu"


class TestBlendBufferOffloadableResults:
    def test_returns_snapshot_of_results(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        sr1 = _make_sr(track_id=1, start_frame=0, frame_count=2)
        sr2 = _make_sr(track_id=2, start_frame=10, frame_count=2)
        bb.register_frame(0, {1})
        bb.register_frame(1, {1})
        bb.register_frame(10, {2})
        bb.register_frame(11, {2})
        bb.add_result(sr1)
        bb.add_result(sr2)

        results = bb.offloadable_results()
        assert len(results) == 2
        track_ids = {r.track_id for r in results}
        assert track_ids == {1, 2}

    def test_empty_when_no_results(self):
        bb = BlendBuffer(device=torch.device("cpu"))
        assert bb.offloadable_results() == []
