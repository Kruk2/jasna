"""Run an opt-in dual-GOP asynchronous D2H experiment.

This module deliberately lives outside the product import graph.  It patches
the dual-GOP writer only inside this probe process, leaving GUI/CLI defaults
unchanged until a same-source real-video A/B proves the route is both correct
and faster.

Usage::

    python scripts/probe_amd_dual_gop_async_d2h.py --mode baseline -- \
        --input input.mp4 --output baseline.mp4 --amd-dual-gop-encode ...

    python scripts/probe_amd_dual_gop_async_d2h.py --mode async --slots 3 -- \
        --input input.mp4 --output async.mp4 --amd-dual-gop-encode ...
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import queue
import sys
import threading
import time
from typing import Any

import av
import torch

from jasna.accelerator import new_event, new_stream, stream_context
from jasna.media import dual_gop_encoder as production


log = logging.getLogger(__name__)


@dataclass
class _DevicePackSlot:
    packed: torch.Tensor
    cas_luma: torch.Tensor | None
    copy_done: Any | None = None


@dataclass(frozen=True)
class _DeferredFrame:
    worker: production._EncoderWorker
    pts: int
    host_yuv: torch.Tensor
    ready_event: Any
    height: int
    frame_format: str


@dataclass(frozen=True)
class _DispatchControl:
    worker: production._EncoderWorker
    payload: object


@dataclass(frozen=True)
class _StopDispatcher:
    pass


class AsyncD2HProbeWriter(production.AmdDualGopFrameWriter):
    """Experimental writer that overlaps YUV conversion with pinned D2H.

    The product writer synchronizes the producer stream and performs a
    blocking D2H for every frame.  This probe keeps the already-validated AMF
    host-native ownership contract, but places the D2H on a dedicated stream.
    A small device-buffer ring prevents the packed YUV storage from being
    overwritten before its copy completes.  A dispatcher waits for each copy
    event before exposing the host frame to the unchanged encoder workers.
    """

    probe_slots = 3

    def __init__(self, template) -> None:
        super().__init__(template)
        slot_count = int(type(self).probe_slots)
        if slot_count < 2 or slot_count > 8:
            raise ValueError("async D2H probe slots must be between 2 and 8")
        if template._packed is None:
            raise RuntimeError("dual-GOP probe requires the packed device buffer")

        first_packed = template._packed
        first_cas_luma = template._cas_luma
        self._device_slots = [
            _DevicePackSlot(first_packed, first_cas_luma),
            *[
                _DevicePackSlot(
                    torch.empty_like(first_packed),
                    (
                        torch.empty_like(first_cas_luma)
                        if first_cas_luma is not None
                        else None
                    ),
                )
                for _ in range(slot_count - 1)
            ],
        ]
        self._slot_cursor = 0
        self._copy_stream = new_stream(template.device)
        self._dispatch_queue: queue.Queue[object] = queue.Queue()
        self._dispatch_state_lock = threading.Lock()
        self._dispatch_error: BaseException | None = None
        self._dispatch_exited = False
        self._retained_failed_owners: list[torch.Tensor] = []
        self._prepared_frames = 0
        self._slot_wait_seconds = 0.0
        self._copy_event_wait_seconds = 0.0
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            name="AmdDualGopAsyncD2HDispatch",
            daemon=True,
        )
        self._dispatch_thread.start()
        frame_mib = first_packed.numel() * first_packed.element_size() / (1024**2)
        log.info(
            "Experimental dual-GOP async D2H enabled: slots=%d, "
            "packed device ring=%.1f MiB",
            slot_count,
            frame_mib * slot_count,
        )

    def _next_slot(self) -> _DevicePackSlot:
        slot = self._device_slots[self._slot_cursor]
        self._slot_cursor = (self._slot_cursor + 1) % len(self._device_slots)
        if slot.copy_done is not None:
            started = time.monotonic()
            slot.copy_done.synchronize()
            self._slot_wait_seconds += time.monotonic() - started
        return slot

    def _prepare_async(
        self,
        frame: torch.Tensor,
        *,
        apply_lut: bool,
    ) -> tuple[torch.Tensor, Any]:
        template = self.template
        height = template.metadata.video_height
        slot = self._next_slot()
        host_yuv = self.host_pool.acquire(self.failed)
        ready_event = None
        try:
            template._packed = slot.packed
            template._cas_luma = slot.cas_luma
            with stream_context(template.stream):
                if apply_lut and template._lut_applier is not None:
                    frame = template._lut_applier.apply(frame)
                packed = template._to_yuv(frame, height)

            ready_event = new_event(template.device)
            with stream_context(self._copy_stream):
                self._copy_stream.wait_stream(template.stream)
                host_yuv.copy_(
                    packed.view(torch.uint16) if template.spec.ten_bit else packed,
                    non_blocking=True,
                )
                ready_event.record(self._copy_stream)
            slot.copy_done = ready_event
        except BaseException:
            if ready_event is not None:
                ready_event.synchronize()
            self.host_pool.release(host_yuv)
            raise
        self._prepared_frames += 1
        return host_yuv, ready_event

    def _release_deferred(self, item: _DeferredFrame) -> None:
        try:
            item.ready_event.synchronize()
        except BaseException:
            # A failed HIP event does not prove that the device has stopped
            # writing the destination.  Retain the owner until process teardown
            # instead of returning potentially live storage to the pool.
            self._retained_failed_owners.append(item.host_yuv)
            log.exception("Async D2H cleanup could not prove host-frame readiness")
            return
        self.host_pool.release(item.host_yuv)

    def _dispatch_loop(self) -> None:
        try:
            while True:
                item = self._dispatch_queue.get()
                handed_off = False
                try:
                    if isinstance(item, _StopDispatcher):
                        return
                    if isinstance(item, _DeferredFrame):
                        started = time.monotonic()
                        item.ready_event.synchronize()
                        self._copy_event_wait_seconds += time.monotonic() - started
                        # Keep the production ordering guarantee: expose the
                        # host-native frame only after its pinned planes are
                        # complete. Creating the DLPack frame before this event
                        # can make partially-written storage visible to FFmpeg.
                        frame = av.VideoFrame.from_dlpack(
                            [
                                item.host_yuv[: item.height],
                                item.host_yuv[item.height :],
                            ],
                            format=item.frame_format,
                        )
                        self._put(
                            item.worker,
                            production._Frame(frame, item.pts, item.host_yuv),
                        )
                        handed_off = True
                    elif isinstance(item, _DispatchControl):
                        self._put(item.worker, item.payload)
                    else:  # pragma: no cover - invariant guard
                        raise RuntimeError(
                            f"unknown async D2H dispatch item: {type(item)!r}"
                        )
                except BaseException:
                    if isinstance(item, _DeferredFrame) and not handed_off:
                        self._release_deferred(item)
                    raise
                finally:
                    self._dispatch_queue.task_done()
        except BaseException as exc:
            with self._dispatch_state_lock:
                self._dispatch_error = exc
                self.failed.set()
            log.exception("[dual-gop-async-d2h-dispatch] crashed")
        finally:
            with self._dispatch_state_lock:
                self._drain_dispatch_queue()
                self._dispatch_exited = True

    def _drain_dispatch_queue(self) -> None:
        while True:
            try:
                item = self._dispatch_queue.get_nowait()
            except queue.Empty:
                break
            try:
                if isinstance(item, _DeferredFrame):
                    self._release_deferred(item)
            finally:
                self._dispatch_queue.task_done()

    def _raise_dispatch_error(self) -> None:
        if self._dispatch_error is not None:
            raise RuntimeError(
                f"async D2H dispatcher failed: {self._dispatch_error!r}"
            ) from self._dispatch_error
        if self._dispatch_exited or not self._dispatch_thread.is_alive():
            raise RuntimeError("async D2H dispatcher exited unexpectedly")

    def _dispatch(self, item: object) -> None:
        # Serialize insertion with the dispatcher's final failure drain. An
        # item is either rejected before ownership transfer or guaranteed to
        # be visible to that drain.
        with self._dispatch_state_lock:
            self._raise_dispatch_error()
            self._dispatch_queue.put(item)

    def write(self, frame: torch.Tensor, pts: int, *, apply_lut: bool = True) -> None:
        if self.closed:
            raise RuntimeError("dual GOP writer is already closed")
        within_group = self.frame_count % self.gop_frames
        if within_group == 0:
            self.group_index += 1
            self.group_worker = self.workers[self.group_index % len(self.workers)]
            normalized = self.work_dir / f"gop-{self.group_index:06d}.ts"
            self._dispatch(
                _DispatchControl(
                    self.group_worker,
                    production._StartGroup(
                        self.group_index,
                        normalized,
                        int(pts),
                    ),
                )
            )
        host_yuv, ready_event = self._prepare_async(
            frame,
            apply_lut=apply_lut,
        )
        deferred = _DeferredFrame(
            self.group_worker,
            int(pts),
            host_yuv,
            ready_event,
            int(self.template.metadata.video_height),
            str(self.template.spec.frame_format),
        )
        try:
            self._dispatch(deferred)
        except BaseException:
            self._release_deferred(deferred)
            raise
        self.frame_count += 1
        if self.frame_count % self.gop_frames == 0:
            self._dispatch(
                _DispatchControl(self.group_worker, production._EndGroup())
            )
            self.group_worker = None

    def _stop_dispatcher(self, *, normal: bool) -> None:
        if normal:
            self._raise_dispatch_error()
        if self._dispatch_thread.is_alive():
            self._dispatch(_StopDispatcher())
            self._dispatch_thread.join(timeout=1800 if normal else 30)
        if self._dispatch_thread.is_alive():
            raise RuntimeError("async D2H dispatcher did not stop")
        with self._dispatch_state_lock:
            self._drain_dispatch_queue()
            dispatch_error = self._dispatch_error
        if normal and dispatch_error is not None:
            raise RuntimeError(
                f"async D2H dispatcher failed: {dispatch_error!r}"
            ) from dispatch_error
        log.info(
            "Experimental async D2H stats: frames=%d, slots=%d, "
            "producer-slot-wait=%.3fs, dispatcher-copy-wait=%.3fs",
            self._prepared_frames,
            len(self._device_slots),
            self._slot_wait_seconds,
            self._copy_event_wait_seconds,
        )

    def close(self) -> None:
        if self.closed:
            return
        try:
            if self.group_worker is not None:
                self._dispatch(
                    _DispatchControl(self.group_worker, production._EndGroup())
                )
                self.group_worker = None
            self._stop_dispatcher(normal=True)
        except BaseException:
            self.abort()
            raise
        super().close()

    def abort(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.failed.set()
        try:
            try:
                self._stop_dispatcher(normal=False)
            finally:
                self._abort_workers()
        finally:
            self._release_template_runtime()
            log.warning(
                "Preserving aborted async D2H probe workspace for diagnosis: %s",
                self.work_dir,
            )

    def _release_template_runtime(self) -> None:
        copy_stream = getattr(self, "_copy_stream", None)
        if copy_stream is not None:
            copy_stream.synchronize()
        self._device_slots = []
        super()._release_template_runtime()


def install_async_d2h_probe(*, slots: int) -> None:
    """Patch the writer in this process only."""

    if slots < 2 or slots > 8:
        raise ValueError("--slots must be between 2 and 8")
    AsyncD2HProbeWriter.probe_slots = int(slots)
    production.AmdDualGopFrameWriter = AsyncD2HProbeWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run baseline or experimental asynchronous D2H Jasna CLI",
    )
    parser.add_argument("--mode", choices=("baseline", "async"), required=True)
    parser.add_argument("--slots", type=int, default=3)
    parser.add_argument(
        "jasna_args",
        nargs=argparse.REMAINDER,
        help="arguments after -- are passed to jasna.main",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    jasna_args = list(args.jasna_args)
    if jasna_args[:1] == ["--"]:
        jasna_args.pop(0)
    if not jasna_args:
        raise SystemExit("missing Jasna CLI arguments after --")
    if args.mode == "async":
        install_async_d2h_probe(slots=int(args.slots))

    sys.argv = ["jasna", *jasna_args]
    from jasna.main import main as jasna_main

    started = time.monotonic()
    try:
        jasna_main()
    finally:
        print(
            f"ASYNC_D2H_PROBE mode={args.mode} slots={int(args.slots)} "
            f"wall_seconds={time.monotonic() - started:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
