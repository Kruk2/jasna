"""LTX seed preview for the segment editor.

Restores one range with one seed the way a job restores it (the same render span,
detection batches, canvas decision, decode tiles and seed rule, through the same
``Pipeline`` and ``ltx.restore`` code), so what the user picks is what the job makes.
The plan and reference latents of a range are kept, so another seed only reruns
denoise and decode; the transformer stays loaded between seeds while it leaves room
to decode. Restored frames are saved as JPEG files, one per source frame.
"""

from __future__ import annotations

import queue
import shutil
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from jasna.gui.models import AppSettings
from jasna.gui.queues import replace_pending
from jasna.gui.video_session import build_video_session, release_session_memory, video_session_key
from jasna.media.probe import VideoMetadata
from jasna.media.splice import KeyframeIndex, segment_render_span
from jasna.segments import SegmentRange, SegmentRestoration
from jasna.accelerator import preferred_device

DECODE_HEADROOM_BYTES = 2 << 30
JPEG_QUALITY = 95


@dataclass(frozen=True)
class SeedFrame:
    seconds: float
    path: Path


@dataclass(frozen=True)
class SeedProgress:
    seed: int
    stage: str
    fraction: float
    eta_seconds: float
    generation: int


@dataclass(frozen=True)
class SeedReady:
    seed: int
    frames: tuple[SeedFrame, ...]
    generation: int


@dataclass(frozen=True)
class SeedFailed:
    message: str
    generation: int


SeedEvent = SeedProgress | SeedReady | SeedFailed


class SeedFrameWriter:
    """FrameWriter that saves the frames of one effect range as JPEG files."""

    def __init__(self, directory: Path, effect_range: tuple[int, int], metadata: VideoMetadata, lut_applier) -> None:
        self._directory = directory
        self._start_pts, self._end_pts = effect_range
        self._metadata = metadata
        self._lut_applier = lut_applier
        self.frames: list[SeedFrame] = []

    def write(self, frame: torch.Tensor, pts: int, *, apply_lut: bool = True) -> None:
        if not self._start_pts <= int(pts) < self._end_pts:
            return
        from jasna.gui.restoration_preview import frame_image

        size = (int(self._metadata.video_width), int(self._metadata.video_height))
        path = self._directory / f"{int(pts)}.jpg"
        frame_image(frame, size, self._lut_applier, apply_lut=apply_lut).save(
            path, quality=JPEG_QUALITY, subsampling=0
        )
        seconds = (int(pts) - self._metadata.start_pts) * float(self._metadata.time_base)
        self.frames.append(SeedFrame(max(0.0, seconds), path))

    def close(self) -> None:
        pass


def prepared_key(segment: SegmentRange, settings: AppSettings) -> tuple:
    """What a range's plan and reference latents depend on (not the seed)."""
    return (segment.start, segment.end, video_session_key(settings), settings.ltx_large_canvas)


class SeedRenderer:
    """Renders LTX ranges seed by seed on the calling thread, keeping the session, the
    transformer and every range's prepared references between calls."""

    def __init__(self, path: Path, metadata: VideoMetadata, index: KeyframeIndex, work_dir: Path) -> None:
        self.path = Path(path)
        self.metadata = metadata
        self.index = index
        self._temp = Path(tempfile.mkdtemp(prefix=".jasna-seeds-", dir=work_dir))
        self._session = None
        self._session_key: tuple | None = None
        self._prepared: dict[tuple, object] = {}
        self._transformer = None
        self._transformer_path: Path | None = None
        self._runs = 0

    def render(
        self,
        segment: SegmentRange,
        seed: int,
        settings: AppSettings,
        *,
        writer_for: Callable[[Path, tuple[int, int]], object],
        report: Callable[[str, float, float], None],
        cancel: threading.Event,
    ):
        """Restore ``segment`` with ``seed``; returns the writer ``writer_for(directory,
        effect range)`` made, after every frame of the range went through it."""
        from jasna.gui.video_session import video_session_config
        from jasna.ltx.plan import LARGE_CANVAS
        from jasna.ltx.camera import PLAN_CANVAS
        from jasna.ltx.restore import (
            PreparedSpan,
            Progress,
            compose_spans,
            denoise_spans,
            final_stores,
            prepare_spans,
        )
        from jasna.ltx.sampler import LtxTransformer
        from jasna.models.ltx_vae import load_video_decoder
        from jasna.session_factory import build_pipeline

        settings = replace(settings, restoration_model="ltx")
        seeded = replace(segment, restoration=SegmentRestoration("ltx", int(seed)))
        self._ensure_session(settings)
        config = video_session_config(settings, codec=settings.codec, encoder_settings={})
        pipeline = build_pipeline(config, self._session, self.path, self.path, segments=(seeded,))
        pipeline.configure_vr(self.metadata)
        render = pipeline.ltx_segment_render(self.metadata)
        span = segment_render_span(seeded, self.index)
        self._runs += 1
        run_dir = self._temp / f"run-{self._runs}"
        run_dir.mkdir()
        writer = writer_for(run_dir, span.effect_ranges[0])
        ltx_span = pipeline.ltx_span(self.metadata, self.index, span, (seeded,), lambda: writer)
        frames = round((span.end_pts - span.start_pts) * self.index.time_base * self.metadata.video_fps)
        progress = Progress(frames, disable=True, report=report)

        key = prepared_key(segment, settings)
        prepared = self._prepared.get(key)
        if prepared is None:
            directory = self._temp / f"prepared-{len(self._prepared)}"
            directory.mkdir()
            (prepared,) = prepare_spans([ltx_span], render, directory, progress=progress, cancel=cancel)
            self._prepared[key] = prepared
        else:
            progress.skip("scan", "encode")
        prepared = PreparedSpan(ltx_span, prepared.plans, prepared.references)
        finals = final_stores([prepared], run_dir)
        decoder = None
        if prepared.windows:
            if self._transformer is None or self._transformer_path != render.files.transformer:
                self._drop_transformer()
                self._transformer = LtxTransformer(render.files.transformer, render.device)
                self._transformer_path = render.files.transformer
            denoise_spans([prepared], finals, self._transformer, keep_references=True, progress=progress, cancel=cancel)
            decode_bytes = render.budget(LARGE_CANVAS if render.large_canvas else PLAN_CANVAS) + DECODE_HEADROOM_BYTES
            if torch.cuda.mem_get_info(render.device)[0] < decode_bytes:
                self._drop_transformer()
            decoder = load_video_decoder(render.files.vae, render.files.tuned_decoder, render.device)
        try:
            compose_spans([prepared], finals, decoder, render, progress=progress, cancel=cancel)
        finally:
            del decoder
            torch.cuda.empty_cache()
        return writer

    def _ensure_session(self, settings: AppSettings) -> None:
        key = video_session_key(settings)
        if self._session is not None and key == self._session_key:
            return
        self._drop_session()
        self._session = build_video_session(settings, log=lambda _msg: None)
        self._session_key = key

    def _drop_transformer(self) -> None:
        if self._transformer is not None:
            self._transformer.close()
            self._transformer = None
            self._transformer_path = None
            torch.cuda.empty_cache()

    def _drop_session(self) -> None:
        self._drop_transformer()
        if self._session is not None:
            device = self._session.device
            self._session.close()
            self._session = None
            self._session_key = None
            release_session_memory(device)

    def release(self) -> None:
        """Free the GPU; the prepared references stay on disk for the next seed."""
        self._drop_session()

    def close(self) -> None:
        self._drop_session()
        self._prepared.clear()
        shutil.rmtree(self._temp)


@dataclass(frozen=True)
class _TrySeed:
    segment: SegmentRange
    seed: int
    settings: AppSettings
    generation: int


@dataclass(frozen=True)
class _Release:
    pass


@dataclass(frozen=True)
class _Stop:
    pass


class LtxSeedPreviewWorker:
    """Background thread around ``SeedRenderer``: one seed at a time, a new request
    cancels the running one. Posts ``SeedEvent``s to ``events``; never touches Tk."""

    def __init__(
        self,
        path: str | Path,
        metadata: VideoMetadata,
        index: KeyframeIndex,
        work_dir: Path,
        *,
        on_stopped: Callable[[], None],
    ) -> None:
        self.path = Path(path)
        self.metadata = metadata
        self.index = index
        self.work_dir = Path(work_dir)
        self._on_stopped = on_stopped
        self.events: queue.Queue[SeedEvent] = queue.Queue()
        self._commands: queue.Queue[_TrySeed | _Release | _Stop] = queue.Queue(maxsize=1)
        self._closed = threading.Event()
        self._generation = 0
        self._active_cancel: threading.Event | None = None
        self._cancel_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name=f"ltx-seed-preview-{self.path.name}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def try_seed(self, segment: SegmentRange, seed: int, settings: AppSettings) -> int:
        self._generation += 1
        self._cancel_active()
        replace_pending(self._commands, _TrySeed(segment, int(seed), settings, self._generation))
        return self._generation

    def cancel(self) -> None:
        self._generation += 1
        self._cancel_active()

    def release(self) -> None:
        """Cancel any run and free the GPU for another preview or a scan."""
        self.cancel()
        replace_pending(self._commands, _Release())

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._cancel_active()
        replace_pending(self._commands, _Stop())

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def _cancel_active(self) -> None:
        with self._cancel_lock:
            if self._active_cancel is not None:
                self._active_cancel.set()

    def _make_renderer(self) -> SeedRenderer:
        return SeedRenderer(self.path, self.metadata, self.index, self.work_dir)

    def _run(self) -> None:
        from jasna.ltx.restore import Cancelled

        renderer: SeedRenderer | None = None
        try:
            while True:
                command = self._commands.get()
                if isinstance(command, _Stop):
                    break
                if isinstance(command, _Release):
                    if renderer is not None:
                        renderer.release()
                    continue
                cancel = threading.Event()
                with self._cancel_lock:
                    self._active_cancel = cancel
                generation = command.generation
                try:
                    if renderer is None:
                        renderer = self._make_renderer()
                    writer = renderer.render(
                        command.segment,
                        command.seed,
                        command.settings,
                        writer_for=lambda directory, effect: SeedFrameWriter(
                            directory, effect, self.metadata, _lut_applier(command.settings)
                        ),
                        report=lambda stage, fraction, eta: self.events.put(
                            SeedProgress(command.seed, stage, fraction, eta, generation)
                        ),
                        cancel=cancel,
                    )
                    self.events.put(SeedReady(command.seed, tuple(writer.frames), generation))
                except Cancelled:
                    pass
                except Exception as exc:
                    if not self._closed.is_set():
                        self.events.put(SeedFailed(str(exc), generation))
                finally:
                    with self._cancel_lock:
                        self._active_cancel = None
        finally:
            try:
                if renderer is not None:
                    renderer.close()
            finally:
                self._on_stopped()


def _lut_applier(settings: AppSettings):
    lut_path = (settings.lut_path or "").strip()
    if not lut_path:
        return None
    from jasna.media.lut import GpuLutApplier, parse_cube_file

    return GpuLutApplier(parse_cube_file(lut_path), preferred_device())
