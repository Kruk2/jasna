"""CPU checks for the shared pass boundary, including pre-loop native failures."""
from pathlib import Path
from types import SimpleNamespace
import threading
from unittest.mock import MagicMock

import pytest
import torch

from factories import make_pipeline
from jasna import pipeline_threads as threads
from jasna.accelerator import AcceleratorVendor
from jasna.pipeline import _OfflineFrameWriter
from jasna.vram_offloader import VramStats


@pytest.fixture
def pass_boundary(monkeypatch):
    monitor = MagicMock(host_memory_pressure=False, stats=VramStats())
    monkeypatch.setattr(threads, "VramOffloader", lambda **kwargs: monitor)
    monkeypatch.setattr(threads, "vendor_for_device", lambda device: AcceleratorVendor.NVIDIA)
    monkeypatch.setattr(threads, "empty_cache", MagicMock())
    monkeypatch.setattr(threads, "ipc_collect", MagicMock())
    for name in ("decode_detect_loop", "primary_restore_loop", "secondary_restore_loop", "blend_encode_loop"):
        monkeypatch.setattr(threads, name, lambda **kwargs: None)
    pipeline = make_pipeline(device=torch.device("cpu"))
    pipeline.vr_resolution = SimpleNamespace(resolved="off")
    metadata = SimpleNamespace(video_width=16, video_height=16, is_10bit=False)
    cancel = threading.Event()
    return pipeline, metadata, cancel, monitor


def _run(boundary, **kwargs):
    pipeline, metadata, cancel, _monitor = boundary
    return threads.run_restoration_pass(
        pipeline, metadata, MagicMock(), cancel,
        seek_ts=None, use_async_secondary=False, **kwargs,
    )


def test_pre_loop_failure_cancels_peers_and_stops_monitor(pass_boundary, monkeypatch):
    root_error = TypeError("native worker argument binding")

    def fail(**kwargs):
        raise root_error

    monkeypatch.setattr(threads, "decode_detect_loop", fail)
    assert _run(pass_boundary) is root_error
    pipeline, _, cancel, monitor = pass_boundary
    assert cancel.is_set()
    monitor.stop.assert_called_once_with()
    assert pipeline._last_pass_vram_stats is monitor.stats
    threads.empty_cache.assert_called_once_with(pipeline.device)
    threads.ipc_collect.assert_called_once_with(pipeline.device)


def test_poll_failure_still_joins_workers_and_releases_monitor(pass_boundary, monkeypatch):
    root_error = RuntimeError("player poll failed")
    stopped = threading.Event()

    def worker(**kwargs):
        kwargs["cancel_event"].wait(2)
        stopped.set()

    def poll():
        raise root_error

    monkeypatch.setattr(threads, "decode_detect_loop", worker)
    assert _run(pass_boundary, poll=poll) is root_error
    assert stopped.is_set()
    pass_boundary[3].stop.assert_called_once_with()


def test_telemetry_and_cache_failures_do_not_replace_worker_root_cause(pass_boundary, monkeypatch):
    root_error = RuntimeError("HIP execution failed")

    def fail(**kwargs):
        raise root_error

    resident = MagicMock()
    resident.decoder_backend.validate_telemetry.side_effect = RuntimeError("telemetry cleanup failed")
    monkeypatch.setattr(threads, "decode_detect_loop", fail)
    threads.empty_cache.side_effect = RuntimeError("cache cleanup failed")
    assert _run(pass_boundary, resident_coordinator=resident) is root_error
    threads.ipc_collect.assert_called_once()


def test_telemetry_failure_without_worker_error_is_not_silently_accepted(pass_boundary):
    root_error = RuntimeError("resident contract violated")
    resident = MagicMock()
    resident.validate_encoder_telemetry.side_effect = root_error
    assert _run(pass_boundary, resident_coordinator=resident) is root_error
    threads.empty_cache.assert_called_once()


def test_failed_pass_aborts_dual_writer_and_preserves_original_error(monkeypatch):
    import jasna.pipeline as module

    pipeline = make_pipeline()
    root_error = RuntimeError("restore failed")
    writer = MagicMock()
    writer.close.side_effect = RuntimeError("encoder cleanup failed")
    monkeypatch.setattr(module, "_OfflineFrameWriter", lambda *args, **kwargs: writer)
    monkeypatch.setattr(module, "run_restoration_pass", lambda *args, **kwargs: root_error)
    metadata = SimpleNamespace(video_fps_exact=30, average_fps=30, num_frames=3)
    with pytest.raises(RuntimeError) as failure:
        pipeline._run_pass(metadata=metadata, encoder_ctx=MagicMock(), progress=MagicMock())
    assert failure.value is root_error
    writer.close.assert_called_once_with(abort=True)


def test_dual_writer_abort_does_not_flush_or_assemble_partial_output(monkeypatch):
    monkeypatch.setattr("jasna.media.dual_gop_encoder.use_dual_gop_writer", lambda *args, **kwargs: False)
    encoder = MagicMock()
    writer = _OfflineFrameWriter(encoder, [None])
    dual = MagicMock()
    writer._dual_gop = dual
    writer.close(abort=True)
    dual.abort.assert_called_once_with()
    dual.close.assert_not_called()
    encoder.__enter__.assert_not_called()
