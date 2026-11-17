"""The encode pass: every planned window's reference crops onto its canvas, VAE-encoded on a thread of
its own while the frame loop decodes and crops the next windows."""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor

import torch

from jasna.accelerator import device_free_bytes
from jasna.ltx.camera import WINDOW_FRAMES
from jasna.ltx.compose import crop_to_canvas
from jasna.ltx.inference import (
    PendingReply,
    WorkerEncoder,
    vae_pixels,
)
from jasna.ltx.plan import TrackPlan
from jasna.ltx.windows import Window
from jasna.ltx.progress import StageBar
from jasna.ltx.resume_files import LatentStore
from jasna.ltx.spans import FrameSource, LtxSegment, check_cancel, segment_batches
from jasna.vr_projection import RegionProjector

ENCODE_AHEAD = 2
"""Filled windows an encode pass keeps encoding or queued while its frame loop fills the next
ones; each keeps its pixels (~95 MB at 512 px) until it is encoded."""


ENCODE_GPU_RESERVE_BYTES = 10 << 30
"""VRAM an encode pass leaves free beside the window pixels it keeps on the GPU: the encoder's
activations (up to ~5.7 GiB for a bf16 768 px window, an eager AMD one reserves up to ~8.3 GiB)."""


def encode_gpu_window_bytes(device: torch.device) -> int:
    """Bytes of window pixels an encode pass keeps on ``device``: what the GPU has free beyond
    ``ENCODE_GPU_RESERVE_BYTES``, none on the CPU."""
    if device.type != "cuda":
        return 0
    return max(0, device_free_bytes(device) - ENCODE_GPU_RESERVE_BYTES)


class _WindowBuffers:
    """``[3, WINDOW_FRAMES, canvas, canvas]`` uint8 pixels for the open windows, reused once a
    window is encoded, so no window stacks its frames or faults in fresh pages. The first ones up
    to ``gpu_bytes`` live on ``device``, where the frame loop queues its crops without waiting for
    each to reach host memory behind the encoder's kernels; the rest live in host memory."""

    def __init__(self, device: torch.device, gpu_bytes: int) -> None:
        self._device = device
        self._gpu_bytes_left = gpu_bytes
        self._free: dict[int, list[torch.Tensor]] = {}

    def take(self, canvas: int) -> torch.Tensor:
        free = self._free.setdefault(canvas, [])
        if free:
            return free.pop()
        shape = (3, WINDOW_FRAMES, canvas, canvas)
        size = math.prod(shape)
        if size <= self._gpu_bytes_left:
            self._gpu_bytes_left -= size
            return torch.empty(shape, dtype=torch.uint8, device=self._device)
        return torch.empty(shape, dtype=torch.uint8)

    def give_back(self, canvas: int, pixels: torch.Tensor) -> None:
        self._free[canvas].append(pixels)


def _copy_canvas(window_pixels: torch.Tensor, position: int, real_frames: int, canvas: torch.Tensor) -> None:
    """``canvas`` into frame ``position``; the last real frame also fills the padding frames."""
    last = position == real_frames - 1
    window_pixels[:, position : WINDOW_FRAMES if last else position + 1].copy_(canvas.unsqueeze(1))


class _WindowEncodes:
    """Encodes filled windows on a thread of its own (on CUDA on a stream of its own), so the frame
    loop decodes and crops the next windows while the encoder runs; once ``ENCODE_AHEAD`` windows
    wait, ``submit`` first finishes the oldest. Pixels on the GPU are read once the frame loop's
    stream has cropped them. A window's buffer goes back once its latent is in host memory; the
    latent then goes to its store on a thread of its own, so the encoder never waits for the disk
    (each write ends with an fsync, ~30 ms on a hard disk). The worker's encoder gets each window's
    host copy from the encode thread while its latent comes back on a thread of its own, so the
    next window crosses the pipe while the worker encodes the last one."""

    def __init__(self, encoder, buffers: _WindowBuffers, device: torch.device, bar: StageBar) -> None:
        self._encoder = encoder
        self._buffers = buffers
        self._device = device
        self._bar = bar
        self._thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LtxEncode")
        self._receives = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LtxEncodeReceive")
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LtxLatentWrite")
        self._stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self._queued: list[tuple[Future[None], int, torch.Tensor]] = []
        self._writes: list[Future[None]] = []

    def submit(self, store: LatentStore, index: int, canvas: int, window_pixels: torch.Tensor) -> None:
        while len(self._queued) >= ENCODE_AHEAD:
            self._finish_oldest()
        cropped = None
        if window_pixels.is_cuda:
            cropped = torch.cuda.Event()
            cropped.record()
        if isinstance(self._encoder, WorkerEncoder):
            sent = self._thread.submit(self._send, window_pixels, cropped)
            encode = self._receives.submit(self._receive, sent, store, index, canvas)
        else:
            encode = self._thread.submit(self._encode, store, index, canvas, window_pixels, cropped)
        self._queued.append((encode, canvas, window_pixels))

    def _cropped_pixels(self, window_pixels: torch.Tensor, cropped: torch.cuda.Event | None) -> torch.Tensor:
        if cropped is not None:
            self._stream.wait_event(cropped)
            window_pixels.record_stream(self._stream)
        return window_pixels

    def _encode(
        self, store: LatentStore, index: int, canvas: int, window_pixels: torch.Tensor, cropped: torch.cuda.Event | None
    ) -> None:
        with torch.cuda.stream(self._stream), self._bar.window(canvas), torch.inference_mode():
            pixels = vae_pixels(self._cropped_pixels(window_pixels, cropped), self._device).unsqueeze(0)
            latent = self._encoder(pixels).cpu()
        self._writes.append(self._writer.submit(store.put, index, latent))

    def _send(self, window_pixels: torch.Tensor, cropped: torch.cuda.Event | None) -> PendingReply:
        with torch.cuda.stream(self._stream):
            pixels = self._cropped_pixels(window_pixels, cropped).unsqueeze(0).cpu()
        return self._encoder.send(pixels)

    def _receive(self, sent: Future[PendingReply], store: LatentStore, index: int, canvas: int) -> None:
        reply = sent.result()
        with self._bar.window(canvas):
            (latent,) = reply.tensors()
        self._writes.append(self._writer.submit(store.put, index, latent))

    def _finish_oldest(self) -> None:
        future, canvas, window_pixels = self._queued.pop(0)
        future.result()
        self._bar.windows_done(canvas, 1)
        self._buffers.give_back(canvas, window_pixels)

    def finish(self) -> None:
        while self._queued:
            self._finish_oldest()
        for write in self._writes:
            write.result()

    def close(self) -> None:
        self._thread.shutdown(cancel_futures=True)
        # every window sent to the worker is read to its end: the replies behind it wait for that
        self._receives.shutdown()
        self._writer.shutdown()


class _SegmentEncoder:
    def __init__(
        self,
        plans: list[TrackPlan],
        store: LatentStore,
        buffers: _WindowBuffers,
        encodes: _WindowEncodes,
        projector: RegionProjector | None,
    ) -> None:
        self._pending = iter(sorted((w for plan in plans for w in plan.windows), key=lambda w: w.start))
        self._upcoming = next(self._pending, None)
        self._active: dict[int, tuple[Window, torch.Tensor]] = {}
        self._projector = projector
        self._store = store
        self._buffers = buffers
        self._encodes = encodes
        self._frame_idx = 0

    def frame(self, frame: torch.Tensor) -> None:
        while self._upcoming is not None and self._upcoming.start == self._frame_idx:
            self._active[self._upcoming.index] = (self._upcoming, self._buffers.take(self._upcoming.canvas))
            self._upcoming = next(self._pending, None)
        for index, (window, window_pixels) in list(self._active.items()):
            position = self._frame_idx - window.start
            _copy_canvas(window_pixels, position, window.real_frames, crop_to_canvas(frame, window.crops[position], self._projector))
            if position == window.real_frames - 1:
                self._encodes.submit(self._store, index, window.canvas, window_pixels)
                del self._active[index]
        self._frame_idx += 1

    def finish(self) -> None:
        if self._active or self._upcoming is not None:
            raise RuntimeError("the video ended before every planned window was filled")


def encode_span(
    frames: FrameSource,
    segments: Sequence[LtxSegment],
    plans: Sequence[list[TrackPlan]],
    encoder,
    stores: Sequence[LatentStore],
    *,
    batch_size: int,
    device: torch.device,
    bar: StageBar,
    cancel: threading.Event,
    projector: RegionProjector | None,
) -> None:
    """VAE-encode every planned window's reference crops into its segment's store."""
    buffers = _WindowBuffers(device, encode_gpu_window_bytes(device))
    encodes = _WindowEncodes(encoder, buffers, device, bar)
    try:
        encoders = [_SegmentEncoder(p, store, buffers, encodes, projector) for p, store in zip(plans, stores)]
        for owner, batch, _pts in segment_batches(frames(), segments, batch_size):
            check_cancel(cancel)
            bar.update(len(batch))
            if owner is not None:
                for frame in batch:
                    encoders[owner].frame(frame)
        encodes.finish()
        for segment_encoder in encoders:
            segment_encoder.finish()
    finally:
        encodes.close()
