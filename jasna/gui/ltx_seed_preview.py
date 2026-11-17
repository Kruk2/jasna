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
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from jasna.accelerator import device_free_bytes
from jasna.gui.models import AppSettings
from jasna.gui.preview_frames import preview_image
from jasna.gui.queues import CommandWorker, drain
from jasna.gui.video_session import SessionCache, build_preview_pipeline, build_video_session, video_session_key
from jasna.media.probe import VideoMetadata
from jasna.media.splice import KeyframeIndex, segment_render_span
from jasna.os_utils import make_scratch_folder
from jasna.segments import SegmentRange, SegmentRestoration

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
    """FrameWriter that saves the frames of one effect range as JPEG files, the left eye of VR
    like the other previews."""

    def __init__(
        self, directory: Path, effect_range: tuple[int, int], metadata: VideoMetadata, lut_applier, stereo: str
    ) -> None:
        self._directory = directory
        self._start_pts, self._end_pts = effect_range
        self._metadata = metadata
        self._lut_applier = lut_applier
        self._stereo = stereo
        self.frames: list[SeedFrame] = []

    def write(self, frame: torch.Tensor, pts: int) -> None:
        if not self._start_pts <= int(pts) < self._end_pts:
            return
        size = (int(self._metadata.video_width), int(self._metadata.video_height))
        path = self._directory / f"{int(pts)}.jpg"
        preview_image(frame, size, lut_applier=self._lut_applier, stereo=self._stereo).save(
            path, quality=JPEG_QUALITY, subsampling=0
        )
        seconds = (int(pts) - self._metadata.start_pts) * float(self._metadata.time_base)
        self.frames.append(SeedFrame(max(0.0, seconds), path))

    def finish(self) -> None:
        pass

    def close(self) -> None:
        pass


def prepared_key(segment: SegmentRange, settings: AppSettings) -> tuple:
    """What a range's plan and reference latents depend on (not the seed)."""
    return (
        segment.start,
        segment.end,
        video_session_key(settings),
        settings.vr_mode,
        settings.vr_projection,
        settings.ltx_large_canvas,
        settings.ltx_overlap,
    )


class SeedRenderer:
    """Renders LTX ranges seed by seed on the calling thread, keeping the session, the
    transformer and every range's prepared references between calls."""

    def __init__(self, path: Path, metadata: VideoMetadata, index: KeyframeIndex, work_dir: Path) -> None:
        self.path = Path(path)
        self.metadata = metadata
        self.index = index
        self._temp = make_scratch_folder(Path(work_dir), ".jasna-seeds-")
        self._sessions = SessionCache(on_close=self._drop_runtime)
        self._prepared: dict[tuple, object] = {}
        self._transformer = None
        self._transformer_path: Path | None = None
        self._runtime = None
        self._runs = 0

    def render(
        self,
        segment: SegmentRange,
        seed: int,
        settings: AppSettings,
        *,
        writer_for: Callable[[Path, tuple[int, int], str], object],
        report: Callable[[str, float, float], None],
        cancel: threading.Event,
    ):
        """Restore ``segment`` with ``seed``; returns the writer ``writer_for(directory,
        effect range, stereo layout)`` made, after every frame of the range went through it."""
        from jasna.ltx.progress import Progress
        from jasna.ltx.restore import PreparedSpan, compose_spans, denoise_spans, prepare_spans
        from jasna.ltx.resume_files import final_stores
        from jasna.ltx.inference import ModelRuntime

        settings = replace(settings, restoration_model="ltx")
        seeded = replace(segment, restoration=SegmentRestoration("ltx", int(seed)))
        session = self._sessions.get(settings, lambda: build_video_session(settings, log=lambda _msg: None))
        pipeline = build_preview_pipeline(settings, session, self.path, self.metadata, segments=(seeded,))
        render = pipeline.ltx_segment_render(self.metadata)
        if self._runtime is None or self._runtime.files != render.files:
            self._drop_runtime()
            self._runtime = ModelRuntime(render.files, render.device, cancel)
        self._runtime.cancel = cancel
        span = segment_render_span(seeded, self.index)
        self._runs += 1
        run_dir = self._temp / f"run-{self._runs}"
        run_dir.mkdir()
        writer = writer_for(run_dir, span.effect_ranges[0], pipeline.vr_resolution.resolved)
        ltx_span = pipeline.ltx_span(self.metadata, self.index, span, (seeded,), lambda: writer)
        frames = round((span.end_pts - span.start_pts) * self.index.time_base * self.metadata.video_fps)
        progress = Progress(frames, disable=True, writer_adds_frame_work=False, report=report)

        key = prepared_key(segment, settings)
        prepared = self._prepared.get(key)
        if prepared is None:
            directory = self._temp / f"prepared-{self._runs}"
            directory.mkdir()
            (prepared,) = prepare_spans(
                [ltx_span], render, directory, progress=progress, cancel=cancel, runtime=self._runtime
            )
            self._prepared[key] = prepared
        else:
            progress.skip("scan", "encode")
        prepared = PreparedSpan(ltx_span, prepared.plans, prepared.references)
        finals = final_stores([prepared], run_dir)
        decoder = None
        if prepared.windows:
            progress.begin("denoise")
            if self._transformer is None or self._transformer_path != render.files.transformer:
                self._drop_transformer()
                self._transformer = self._runtime.load_transformer()
                self._transformer_path = render.files.transformer
            denoise_spans([prepared], finals, self._transformer, keep_references=True, progress=progress, cancel=cancel)
            decode_bytes = render.budget(prepared.largest_canvas) + DECODE_HEADROOM_BYTES
            if device_free_bytes(render.device) < decode_bytes:
                self._drop_transformer()
            progress.begin("compose")
            decoder = self._runtime.load_decoder()
        try:
            compose_spans([prepared], finals, decoder, render, progress=progress, cancel=cancel)
        finally:
            del decoder
            self._runtime.unload_decoder()
            torch.cuda.empty_cache()
        return writer

    def _drop_transformer(self) -> None:
        if self._transformer is not None:
            self._transformer = None
            self._transformer_path = None
            self._runtime.unload_transformer()
            torch.cuda.empty_cache()

    def _drop_runtime(self) -> None:
        self._transformer = None
        self._transformer_path = None
        if self._runtime is not None:
            self._runtime.close()
            self._runtime = None

    def release(self) -> None:
        """Free the GPU; the prepared references stay on disk for the next seed."""
        self._sessions.close()

    def close(self) -> None:
        self._sessions.close()
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


class LtxSeedPreviewWorker(CommandWorker):
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
        super().__init__(f"ltx-seed-preview-{Path(path).name}")
        self.path = Path(path)
        self.metadata = metadata
        self.index = index
        self.work_dir = Path(work_dir)
        self._on_stopped = on_stopped
        self.events: queue.Queue[SeedEvent] = queue.Queue()

    def try_seed(self, segment: SegmentRange, seed: int, settings: AppSettings) -> int:
        self._generation += 1
        self._cancel_active_pass()
        self._send(_TrySeed(segment, int(seed), settings, self._generation))
        return self._generation

    def cancel(self) -> None:
        self._generation += 1
        drain(self._commands)
        self._cancel_active_pass()

    def release(self) -> None:
        """Cancel any run and free the GPU for another preview or a scan."""
        self.cancel()
        self._send(_Release())

    def _make_renderer(self) -> SeedRenderer:
        return SeedRenderer(self.path, self.metadata, self.index, self.work_dir)

    def _run(self) -> None:
        from jasna.ltx.inference import Cancelled
        from jasna.media.lut import lut_applier

        renderer: SeedRenderer | None = None
        try:
            while (command := self._next_command()) is not None:
                if isinstance(command, _Release):
                    if renderer is not None:
                        renderer.release()
                    continue
                generation = command.generation
                with self._cancellable_pass() as cancel:
                    try:
                        if renderer is None:
                            renderer = self._make_renderer()
                        writer = renderer.render(
                            command.segment,
                            command.seed,
                            command.settings,
                            writer_for=lambda directory, effect, stereo: SeedFrameWriter(
                                directory,
                                effect,
                                self.metadata,
                                lut_applier(command.settings.lut_path, torch.device("cuda:0")),
                                stereo,
                            ),
                            report=lambda fraction, eta, stage: self.events.put(
                                SeedProgress(command.seed, stage.name, fraction, eta, generation)
                            ),
                            cancel=cancel,
                        )
                        self.events.put(SeedReady(command.seed, tuple(writer.frames), generation))
                    except Cancelled:
                        if renderer is not None:
                            renderer.release()
                    except Exception as exc:
                        if renderer is not None:
                            renderer.release()
                        if not self._closed.is_set():
                            self.events.put(SeedFailed(str(exc), generation))
        finally:
            try:
                if renderer is not None:
                    renderer.close()
            finally:
                self._on_stopped()
