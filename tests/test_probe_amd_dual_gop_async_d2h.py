from __future__ import annotations

from contextlib import nullcontext
import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch

from jasna.media import dual_gop_encoder as production
from scripts import probe_amd_dual_gop_async_d2h as probe


def _bare_writer() -> probe.AsyncD2HProbeWriter:
    writer = object.__new__(probe.AsyncD2HProbeWriter)
    writer.failed = threading.Event()
    writer._dispatch_queue = queue.Queue()
    writer._dispatch_state_lock = threading.Lock()
    writer._dispatch_error = None
    writer._dispatch_exited = False
    writer._retained_failed_owners = []
    writer._copy_event_wait_seconds = 0.0
    writer._prepared_frames = 0
    writer._slot_wait_seconds = 0.0
    writer.host_pool = MagicMock()
    return writer


def test_dispatch_waits_for_copy_before_handing_frame_to_worker() -> None:
    writer = _bare_writer()
    order = []
    writer._put = MagicMock(side_effect=lambda *_args: order.append("put"))
    worker = object()
    event = MagicMock()
    event.synchronize.side_effect = lambda: order.append("ready")
    item = probe._DeferredFrame(worker, 17, "host", event, 4, "nv12")
    writer._dispatch_queue.put(item)
    writer._dispatch_queue.put(probe._StopDispatcher())

    with patch.object(probe.av.VideoFrame, "from_dlpack", return_value="frame"):
        writer._dispatch_loop()

    assert order == ["ready", "put"]
    writer._put.assert_called_once()
    routed_worker, routed_frame = writer._put.call_args.args
    assert routed_worker is worker
    assert routed_frame == production._Frame("frame", 17, "host")
    writer.host_pool.release.assert_not_called()


def test_dispatch_failure_releases_ready_host_owner() -> None:
    writer = _bare_writer()
    writer._put = MagicMock(side_effect=RuntimeError("worker failed"))
    event = MagicMock()
    item = probe._DeferredFrame(object(), 23, "host", event, 4, "nv12")
    writer._dispatch_queue.put(item)

    with patch.object(probe.av.VideoFrame, "from_dlpack", return_value="frame"):
        writer._dispatch_loop()

    event.synchronize.assert_has_calls([call(), call()])
    writer.host_pool.release.assert_called_once_with("host")
    assert writer.failed.is_set()
    assert isinstance(writer._dispatch_error, RuntimeError)


def test_dispatch_transfers_owner_once_when_error_races_after_queue_put() -> None:
    writer = _bare_writer()
    writer._dispatch_thread = MagicMock()
    writer._dispatch_thread.is_alive.return_value = True
    item = probe._DeferredFrame(
        object(),
        29,
        "host",
        MagicMock(),
        4,
        "nv12",
    )
    real_put = writer._dispatch_queue.put

    def publish_error_after_put(queued_item) -> None:
        real_put(queued_item)
        writer._dispatch_error = RuntimeError("raced failure")

    writer._dispatch_queue.put = MagicMock(side_effect=publish_error_after_put)

    writer._dispatch(item)
    writer.host_pool.release.assert_not_called()

    writer._drain_dispatch_queue()

    writer.host_pool.release.assert_called_once_with("host")


def test_write_preserves_start_frame_end_order(monkeypatch, tmp_path) -> None:
    writer = _bare_writer()
    writer.closed = False
    writer.frame_count = 0
    writer.gop_frames = 1
    writer.group_index = -1
    writer.group_worker = None
    writer.workers = ["worker-0", "worker-1"]
    writer.work_dir = tmp_path
    writer.template = SimpleNamespace(
        metadata=SimpleNamespace(video_height=4),
        spec=SimpleNamespace(frame_format="nv12"),
    )
    ready = object()
    writer._prepare_async = MagicMock(return_value=("host", ready))
    writer._dispatch = MagicMock()

    writer.write("torch-frame", 101, apply_lut=False)

    assert writer.frame_count == 1
    assert writer.group_worker is None
    assert writer._dispatch.call_args_list == [
        call(
            probe._DispatchControl(
                "worker-0",
                production._StartGroup(0, tmp_path / "gop-000000.ts", 101),
            )
        ),
        call(probe._DeferredFrame("worker-0", 101, "host", ready, 4, "nv12")),
        call(probe._DispatchControl("worker-0", production._EndGroup())),
    ]
    writer._prepare_async.assert_called_once_with(
        "torch-frame",
        apply_lut=False,
    )


class _FakeTensor:
    def __init__(self, name: str) -> None:
        self.name = name
        self.copy_calls = []

    def __getitem__(self, item):
        return (self.name, item)

    def view(self, _dtype):
        return self

    def copy_(self, source, *, non_blocking):
        self.copy_calls.append((source, non_blocking))
        return self


def test_prepare_async_queues_nonblocking_copy_without_producer_sync(
    monkeypatch,
) -> None:
    packed = _FakeTensor("packed")
    host = _FakeTensor("host")
    compute_stream = MagicMock()
    copy_stream = MagicMock()
    ready = MagicMock()
    template = SimpleNamespace(
        device=torch.device("cpu"),
        metadata=SimpleNamespace(video_height=4),
        spec=SimpleNamespace(ten_bit=False, frame_format="nv12"),
        stream=compute_stream,
        _packed=packed,
        _cas_luma=None,
        _lut_applier=None,
        _to_yuv=MagicMock(return_value=packed),
    )
    writer = _bare_writer()
    writer.template = template
    writer._device_slots = [probe._DevicePackSlot(packed, None)]
    writer._slot_cursor = 0
    writer._copy_stream = copy_stream
    writer.host_pool.acquire.return_value = host
    monkeypatch.setattr(probe, "stream_context", lambda _stream: nullcontext())
    monkeypatch.setattr(probe, "new_event", lambda _device: ready)
    from_dlpack = MagicMock(return_value="av-frame")
    monkeypatch.setattr(probe.av.VideoFrame, "from_dlpack", from_dlpack)

    result = writer._prepare_async("input", apply_lut=False)

    assert result == (host, ready)
    copy_stream.wait_stream.assert_called_once_with(compute_stream)
    assert host.copy_calls == [(packed, True)]
    ready.record.assert_called_once_with(copy_stream)
    from_dlpack.assert_not_called()
    compute_stream.synchronize.assert_not_called()
    assert writer._device_slots[0].copy_done is ready


def test_next_slot_waits_before_reuse() -> None:
    writer = _bare_writer()
    event = MagicMock()
    slot = probe._DevicePackSlot(_FakeTensor("packed"), None, event)
    writer._device_slots = [slot]
    writer._slot_cursor = 0

    assert writer._next_slot() is slot

    event.synchronize.assert_called_once_with()
    assert writer._slot_wait_seconds >= 0


def test_install_probe_is_process_local_and_validates_slot_count(monkeypatch) -> None:
    original = production.AmdDualGopFrameWriter
    monkeypatch.setattr(production, "AmdDualGopFrameWriter", original)

    probe.install_async_d2h_probe(slots=3)

    assert production.AmdDualGopFrameWriter is probe.AsyncD2HProbeWriter
    assert probe.AsyncD2HProbeWriter.probe_slots == 3
