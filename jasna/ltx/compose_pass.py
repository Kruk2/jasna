"""The compose pass: windows decoded ahead on a thread of their own while each frame composites the
decoded windows covering it and goes to the span's writer."""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing

import torch

from jasna.ltx.compose import Candidate, composite_frame
from jasna.ltx.inference import (
    GPU_WINDOWS_MIN_FREE_BYTES,
    PendingReply,
    WorkerDecoder,
    decoded_frames,
)
from jasna.ltx.camera import feather_pixels
from jasna.ltx.plan import TrackPlan
from jasna.ltx.windows import Window, crossfade_weights
from jasna.ltx.progress import StageBar
from jasna.ltx.resume_files import LatentStore
from jasna.ltx.spans import LtxSpan, check_cancel, segment_batches
from jasna.ltx.super_res import LtxSuperRes, enlarged_windows, upscaled_crop
from jasna.ltx.vram_budget import DecodeBudget
from jasna.vr_projection import RegionProjector

DECODE_AHEAD = 2
"""Windows a compose pass keeps decoding or decoded ahead of the frames while they wait in host
memory (~95 MB each at 512 px)."""


GPU_DECODE_AHEAD = 16
"""Windows kept decoding or decoded ahead while they wait on the GPU (from
``GPU_WINDOWS_MIN_FREE_BYTES`` of budget on): enough for the decoder to work through stretches
with few windows, where the frames run at video decode and encode speed, instead of waiting."""


def _decode_free_bytes(budget: DecodeBudget, canvas: int, empty_caches: Callable[[], None]) -> int:
    """The budget a window decodes with. Under ``GPU_WINDOWS_MIN_FREE_BYTES`` the allocators' cached
    blocks go back to the driver first, so they count as free; from it on, freeing them (which
    waits for the whole GPU) would change neither the tile layout nor where the window waits, so
    they stay cached."""
    free = budget(canvas)
    if free < GPU_WINDOWS_MIN_FREE_BYTES:
        empty_caches()
        free = budget(canvas)
    return free


def _held_device(device: torch.device, free: int) -> torch.device:
    """Where a decoded window waits: in host memory while its budget is under
    ``GPU_WINDOWS_MIN_FREE_BYTES``, since a frame can need every open window of its tracks and
    holding them on the GPU would shrink the next decode's budget; from there on on the GPU."""
    return device if device.type == "cuda" and free >= GPU_WINDOWS_MIN_FREE_BYTES else torch.device("cpu")


def _decode_window(
    decoder, latent: torch.Tensor, seed: int, canvas: int, budget: DecodeBudget, device: torch.device
) -> torch.Tensor:
    """Decoded window as uint8 ``[F, 3, S, S]``, held where ``_held_device`` says."""
    free = _decode_free_bytes(budget, canvas, torch.cuda.empty_cache)
    with torch.inference_mode():
        return decoded_frames(decoder, latent, seed, free, device).to(_held_device(device, free))


def _send_window_decode(
    decoder: WorkerDecoder, latent: torch.Tensor, seed: int, canvas: int, budget: DecodeBudget
) -> tuple[PendingReply, int]:
    """Send one window to the worker's decoder and return, with its budget, once its result
    header is in: the worker has finished it, so the next window's budget is read while no
    decode runs there, as in this process, and goes out while this window's frames still cross
    the pipe. A budget under ``GPU_WINDOWS_MIN_FREE_BYTES`` is read again once both processes'
    allocators have handed their caches back, as in this process."""

    def empty_caches() -> None:
        torch.cuda.empty_cache()
        decoder.release_cache()

    free = _decode_free_bytes(budget, canvas, empty_caches)
    reply = decoder.send_window(latent, seed, free)
    reply.answered()
    return reply, free


def _decode_shares(windows: list[Window]) -> dict[int, list[tuple[int, float]]]:
    """Per segment frame, the ``(canvas, share)`` of each window decode it counts: a window
    counts as decoding from the start of the window ``DECODE_AHEAD`` before it (the first ones
    from frame 0) to its own start (on the GPU it may decode earlier), so its decode counts
    evenly over those frames, or all at its start when there are none."""
    shares: dict[int, list[tuple[int, float]]] = {}
    submitted = [0] * DECODE_AHEAD + [window.start for window in windows]
    for window, start in zip(windows, submitted):
        frames = range(start, window.start) or range(window.start, window.start + 1)
        for frame_idx in frames:
            shares.setdefault(frame_idx, []).append((window.canvas, 1 / len(frames)))
    return shares


class _SegmentComposer:
    def __init__(
        self,
        plans: list[TrackPlan],
        decoder,
        finals: LatentStore,
        *,
        seed: int,
        feather: int,
        budget: DecodeBudget,
        device: torch.device,
        bar: StageBar,
        decodes: ThreadPoolExecutor,
        receives: ThreadPoolExecutor,
        latent_reads: ThreadPoolExecutor,
        stream: torch.cuda.Stream | None,
        super_res: LtxSuperRes | None,
        projector: RegionProjector | None,
    ) -> None:
        self._weights: dict[int, torch.Tensor] = {}
        self._tracks: dict[int, TrackPlan] = {}
        for plan in plans:
            for window, weight in zip(plan.windows, crossfade_weights(plan.windows)):
                self._weights[window.index] = weight
                self._tracks[window.index] = plan
        windows = sorted((w for plan in plans for w in plan.windows), key=lambda w: w.start)
        self._super_res = super_res
        self._projector = projector
        self._upscaled = enlarged_windows(windows) if super_res is not None else set()
        self._pending = iter(windows)
        self._upcoming = next(self._pending, None)
        self._covering: list[Window] = []
        self._decoded: dict[int, torch.Tensor] = {}
        self._decoder = decoder
        self._finals = finals
        self._seed = seed
        self._feather = feather
        self._budget = budget
        self._device = device
        self._bar = bar
        self._frame_idx = 0
        self._decodes = decodes
        self._receives = receives
        self._latent_reads = latent_reads
        self._stream = stream
        self._decode_order = iter(windows)
        self._decode_shares = _decode_shares(windows)
        self._decoding: dict[int, Future[tuple[torch.Tensor, torch.cuda.Event | None]]] = {}
        self._ahead = DECODE_AHEAD

    def _decode(
        self, window: Window, latent: Future[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        """The decoded window, and on CUDA an event its decode stream records after it: the decode
        thread starts the next window while the GPU still finishes this one."""
        latent = latent.result()
        with torch.cuda.stream(self._stream), self._bar.window(window.canvas):
            video = _decode_window(
                self._decoder,
                latent,
                self._seed + window.noise,
                window.canvas,
                self._budget,
                self._device,
            )
        return video, self._decoded_mark()

    def _decoded_mark(self) -> torch.cuda.Event | None:
        if self._stream is None:
            return None
        decoded = torch.cuda.Event()
        decoded.record(self._stream)
        return decoded

    def _send_decode(self, window: Window, latent: Future[torch.Tensor]) -> tuple[PendingReply, int]:
        """The worker's decode of ``window`` up to its result header (``_send_window_decode``), on
        the decode thread, so windows go out in start order."""
        latent = latent.result()
        with self._bar.window(window.canvas):
            return _send_window_decode(self._decoder, latent, self._seed + window.noise, window.canvas, self._budget)

    def _receive_decode(
        self, sent: Future[tuple[PendingReply, int]]
    ) -> tuple[torch.Tensor, torch.cuda.Event | None]:
        """The frames of a window the worker decoded, read on a thread of their own while the next
        window decodes there."""
        reply, free = sent.result()
        with torch.cuda.stream(self._stream):
            video = reply.tensors()[0].to(_held_device(self._device, free))
        return video, self._decoded_mark()

    def _decode_ahead(self) -> None:
        """Queue the next windows in start order until ``DECODE_AHEAD`` are decoding or wait
        unused, so they decode back to back while the frames before their starts are
        composited and written; ``GPU_DECODE_AHEAD`` while the last window taken waits on the GPU.
        Each latent is read from disk on a thread of its own while the windows before it decode,
        so the decoder never waits for the disk. The worker's decodes go out on the decode thread
        and their frames come back on the receive thread, so the next window decodes in the worker
        while the last one's frames cross the pipe."""
        while len(self._decoding) < self._ahead and (window := next(self._decode_order, None)) is not None:
            latent = self._latent_reads.submit(self._finals.get, window.index)
            if isinstance(self._decoder, WorkerDecoder):
                sent = self._decodes.submit(self._send_decode, window, latent)
                self._decoding[window.index] = self._receives.submit(self._receive_decode, sent)
            else:
                self._decoding[window.index] = self._decodes.submit(self._decode, window, latent)

    def _decoded_window(self, window: Window) -> torch.Tensor:
        with self._bar.waiting():
            video, decoded = self._decoding.pop(window.index).result()
        if decoded is not None:
            torch.cuda.current_stream(self._device).wait_event(decoded)
        if video.is_cuda:
            # decoded on the decode stream, read by this stream's frames: keep its memory from the
            # next decode until they ran
            video.record_stream(torch.cuda.current_stream(video.device))
        self._ahead = GPU_DECODE_AHEAD if video.is_cuda else DECODE_AHEAD
        self._decode_ahead()
        return video

    def frame(self, frame: torch.Tensor) -> torch.Tensor:
        frame_idx = self._frame_idx
        if frame_idx == 0:
            self._decode_ahead()
        while self._upcoming is not None and self._upcoming.start == frame_idx:
            self._covering.append(self._upcoming)
            self._upcoming = next(self._pending, None)
        for window in [w for w in self._covering if w.stop <= frame_idx]:
            self._covering.remove(window)
            self._decoded.pop(window.index, None)
        candidates = []
        for window in self._covering:
            if window.index not in self._decoded:
                self._decoded[window.index] = self._decoded_window(window)
            local = frame_idx - window.start
            track = self._tracks[window.index]
            canvas, crop = self._decoded[window.index][local].to(self._device, non_blocking=True), window.crops[local]
            if window.index in self._upscaled:
                canvas, crop = self._super_res.upscale(canvas), upscaled_crop(crop)
            candidates.append(
                Candidate(
                    canvas=canvas,
                    crop=crop,
                    weight=float(self._weights[window.index][local]),
                    track=track.track_id,
                    polygons=track.polygons[frame_idx - track.start],
                    presence=track.presence[frame_idx - track.start],
                )
            )
        for canvas, share in self._decode_shares.get(frame_idx, ()):
            self._bar.windows_done(canvas, share)
        self._frame_idx += 1
        return composite_frame(frame, candidates, feather=self._feather, projector=self._projector)


def compose_span(
    span: LtxSpan,
    plans: Sequence[list[TrackPlan]],
    decoder,
    finals: Sequence[LatentStore],
    *,
    frame_h: int,
    batch_size: int,
    budget: DecodeBudget,
    device: torch.device,
    bar: StageBar,
    cancel: threading.Event,
    super_res: LtxSuperRes | None,
    projector: RegionProjector | None,
) -> None:
    """Write every frame of ``span``: segment frames composited, the rest unchanged. Windows
    decode ``DECODE_AHEAD`` ahead on a thread; on CUDA on a stream of their own, below the priority of the
    frames' stream, whose short kernels would otherwise queue behind the decode's. The video is read
    on a thread of its own too: the reader waits for each batch's conversion on the host, and on the
    frame loop that wait also took every composite queued before the batch."""
    from jasna.media.video_decoder import BatchPrefetcher

    feather = feather_pixels(frame_h)
    decodes = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LtxDecode")
    receives = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LtxDecodeReceive")
    latent_reads = ThreadPoolExecutor(max_workers=1, thread_name_prefix="LtxLatentRead")
    cuda = device.type == "cuda"
    decode_stream = torch.cuda.Stream(device) if cuda else None
    try:
        composers = [
            _SegmentComposer(
                p, decoder, store, seed=segment.seed, feather=feather, budget=budget, device=device, bar=bar,
                decodes=decodes, receives=receives, latent_reads=latent_reads, stream=decode_stream,
                super_res=super_res, projector=projector,
            )
            for p, store, segment in zip(plans, finals, span.segments)
        ]
        with torch.cuda.stream(torch.cuda.Stream(device, priority=-1) if cuda else None):
            writer = span.open_writer()
            try:
                with closing(BatchPrefetcher(span.frames(), device)) as frames:
                    for owner, batch, pts in segment_batches(iter(frames), span.segments, batch_size):
                        check_cancel(cancel)
                        bar.update(len(batch))
                        for frame, frame_pts in zip(batch, pts):
                            if owner is None:
                                writer.write(frame, frame_pts)
                            else:
                                writer.write(composers[owner].frame(frame), frame_pts)
                writer.finish()
            finally:
                writer.close()
    finally:
        decodes.shutdown(cancel_futures=True)
        # every window sent to the worker is read to its end: the replies behind it wait for that
        receives.shutdown()
        latent_reads.shutdown(cancel_futures=True)
