"""Restoration preview for the segment editor.

Runs the real restoration pipeline (decode/detect -> primary -> secondary ->
blend) over bounded still or playback windows through
``jasna.pipeline_threads.run_restoration_pass``.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from jasna.gui.models import AppSettings
from jasna.gui.video_session import build_video_session, release_session_memory, video_session_key
from jasna.session_factory import RestorationSession
from jasna.media.probe import VideoMetadata


@dataclass(frozen=True)
class RestorationStatus:
    message: str
    generation: int


@dataclass(frozen=True)
class RestorationFrame:
    seconds: float
    image: Image.Image
    generation: int


@dataclass(frozen=True)
class RestoredClipFrame:
    seconds: float
    image: Image.Image


@dataclass(frozen=True)
class RestorationClip:
    frames: tuple[RestoredClipFrame, ...]
    generation: int


@dataclass(frozen=True)
class RestorationFailed:
    message: str
    generation: int


RestorationEvent = RestorationStatus | RestorationFrame | RestorationClip | RestorationFailed


@dataclass(frozen=True)
class _Request:
    center_seconds: float
    settings: AppSettings
    projection: str
    generation: int
    playback: bool


@dataclass(frozen=True)
class _Stop:
    pass


@dataclass(frozen=True)
class _Cancel:
    pass


@dataclass(frozen=True)
class PreviewWindow:
    seek_ts: float
    end_pts: int
    center_pts: int


def preview_window(metadata: VideoMetadata, center_seconds: float, max_clip_size: int) -> PreviewWindow:
    window_duration = max_clip_size / float(metadata.video_fps)
    start_seconds = min(
        max(0.0, center_seconds - window_duration / 2),
        max(0.0, float(metadata.duration) - window_duration),
    )
    time_base = float(metadata.time_base)
    end_pts = metadata.start_pts + round((start_seconds + window_duration) / time_base)
    center_pts = metadata.start_pts + round(max(start_seconds, center_seconds) / time_base)
    return PreviewWindow(seek_ts=start_seconds, end_pts=end_pts, center_pts=center_pts)


def playback_window(metadata: VideoMetadata, start_seconds: float, max_clip_size: int) -> PreviewWindow:
    start_seconds = min(max(0.0, start_seconds), float(metadata.duration))
    end_seconds = min(
        float(metadata.duration),
        start_seconds + max_clip_size / float(metadata.video_fps),
    )
    time_base = float(metadata.time_base)
    start_pts = metadata.start_pts + round(start_seconds / time_base)
    end_pts = metadata.start_pts + round(end_seconds / time_base)
    return PreviewWindow(seek_ts=start_seconds, end_pts=end_pts, center_pts=start_pts)


def _frame_image(
    frame,
    max_size: tuple[int, int],
    lut_applier=None,
    *,
    apply_lut: bool = True,
    left_eye_only: bool = False,
) -> Image.Image:
    if left_eye_only:
        frame = frame[:, :, : frame.shape[-1] // 2]
    if apply_lut and lut_applier is not None:
        frame = lut_applier.apply(frame)
    image = Image.fromarray(frame.cpu().permute(1, 2, 0).numpy()).copy()
    max_width, max_height = max_size
    scale = min(1.0, max_width / image.width, max_height / image.height)
    if scale < 1.0:
        image = image.resize(
            (max(2, round(image.width * scale)), max(2, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return image


class _CenterFrameCollector:
    """FrameWriter that keeps the restored frame closest to ``center_pts`` and
    cancels the pass once the center has been written."""

    def __init__(
        self,
        center_pts: int,
        cancel_event: threading.Event,
        lut_applier=None,
        *,
        left_eye_only: bool = False,
    ) -> None:
        self._center_pts = center_pts
        self._cancel_event = cancel_event
        self._lut_applier = lut_applier
        self._left_eye_only = bool(left_eye_only)
        self._best_pts: int | None = None
        self._best_frame = None
        self.done = False

    def write(self, frame, pts: int, *, apply_lut: bool = True) -> None:
        if self.done:
            return
        if self._best_pts is None or abs(pts - self._center_pts) < abs(self._best_pts - self._center_pts):
            self._best_pts = pts
            if self._left_eye_only:
                frame = frame[:, :, : frame.shape[-1] // 2]
            if apply_lut and self._lut_applier is not None:
                frame = self._lut_applier.apply(frame)
            self._best_frame = frame.cpu()
        if pts >= self._center_pts:
            self.done = True
            self._cancel_event.set()

    def after_write(self, frames_written: int) -> None:
        pass

    @property
    def has_result(self) -> bool:
        return self._best_frame is not None

    def result_image(self, max_size: tuple[int, int]) -> Image.Image:
        return _frame_image(self._best_frame, max_size, apply_lut=False)


class _PlaybackFrameCollector:
    def __init__(
        self,
        metadata: VideoMetadata,
        max_size: tuple[int, int],
        lut_applier=None,
        *,
        left_eye_only: bool = False,
    ) -> None:
        self._metadata = metadata
        self._max_size = max_size
        self._lut_applier = lut_applier
        self._left_eye_only = bool(left_eye_only)
        self._frames: list[tuple[int, Image.Image]] = []

    @property
    def done(self) -> bool:
        return bool(self._frames)

    def write(self, frame, pts: int, *, apply_lut: bool = True) -> None:
        self._frames.append(
            (
                pts,
                _frame_image(
                    frame,
                    self._max_size,
                    self._lut_applier,
                    apply_lut=apply_lut,
                    left_eye_only=self._left_eye_only,
                ),
            )
        )

    def after_write(self, frames_written: int) -> None:
        pass

    def result_frames(self) -> tuple[RestoredClipFrame, ...]:
        time_base = float(self._metadata.time_base)
        return tuple(
            RestoredClipFrame(
                seconds=max(0.0, (pts - self._metadata.start_pts) * time_base),
                image=image,
            )
            for pts, image in self._frames
        )


class RestorationPreviewWorker:
    """Background worker that restores one bounded window per request.

    Requests are coalesced (queue of one); a new request cancels the in-flight
    pass. The heavy session is cached and rebuilt only when the relevant
    settings (``video_session_key``) change.
    """

    def __init__(
        self,
        path: str | Path,
        metadata: VideoMetadata,
        *,
        max_size: tuple[int, int] = (640, 360),
        on_stopped: Callable[[], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self.metadata = metadata
        self.max_size = max_size
        self._on_stopped = on_stopped
        self.events: queue.Queue[RestorationEvent] = queue.Queue()
        self._commands: queue.Queue[_Request | _Cancel | _Stop] = queue.Queue(maxsize=1)
        self._closed = threading.Event()
        self._generation = 0
        self._active_cancel: threading.Event | None = None
        self._cancel_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run,
            name=f"restoration-preview-{self.path.name}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def request(
        self,
        center_seconds: float,
        settings: AppSettings,
        *,
        projection: str,
        playback: bool = False,
    ) -> int:
        self._generation += 1
        self._cancel_active_pass()
        self._replace_command(
            _Request(
                center_seconds=max(0.0, float(center_seconds)),
                settings=settings,
                projection=projection,
                generation=self._generation,
                playback=bool(playback),
            )
        )
        return self._generation

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._replace_command(_Stop(), allow_closed=True)
        self._cancel_active_pass()

    def cancel(self) -> None:
        if self._closed.is_set():
            return
        self._generation += 1
        self._cancel_active_pass()
        self._replace_command(_Cancel())

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout=timeout)

    def _cancel_active_pass(self) -> None:
        with self._cancel_lock:
            if self._active_cancel is not None:
                self._active_cancel.set()

    def _replace_command(
        self,
        command: _Request | _Cancel | _Stop,
        *,
        allow_closed: bool = False,
    ) -> None:
        if self._closed.is_set() and not allow_closed:
            return
        try:
            while True:
                self._commands.get_nowait()
        except queue.Empty:
            pass
        try:
            self._commands.put_nowait(command)
        except queue.Full:
            pass

    def _run(self) -> None:
        session: RestorationSession | None = None
        session_key: tuple | None = None
        pipeline = None
        pipeline_settings: AppSettings | None = None
        try:
            while not self._closed.is_set():
                try:
                    command = self._commands.get(timeout=0.1)
                except queue.Empty:
                    continue
                if isinstance(command, _Stop):
                    break
                if isinstance(command, _Cancel):
                    continue
                try:
                    key = video_session_key(command.settings)
                    if session is None or key != session_key:
                        pipeline = None
                        if session is not None:
                            session.close()
                            release_session_memory(session.device)
                            session = None
                            session_key = None
                        self.events.put(RestorationStatus("loading_models", command.generation))
                        session = build_video_session(
                            command.settings,
                            disable_basicvsrpp_tensorrt=False,
                            log=lambda msg: self.events.put(RestorationStatus(msg, command.generation)),
                        )
                        session_key = key
                    if pipeline is None or command.settings != pipeline_settings:
                        pipeline = self._build_pipeline(command.settings, session)
                        pipeline_settings = command.settings
                    if not self._commands.empty() or self._closed.is_set():
                        continue
                    self.events.put(RestorationStatus("restoring", command.generation))
                    result = self._run_preview_pass(command, pipeline)
                    if result is not None:
                        self.events.put(result)
                except Exception as exc:
                    if not self._closed.is_set():
                        self.events.put(RestorationFailed(str(exc), command.generation))
        finally:
            if session is not None:
                session.close()
                release_session_memory(session.device)
            if self._on_stopped is not None:
                self._on_stopped()

    def _build_pipeline(self, settings: AppSettings, session: RestorationSession):
        from jasna.gui.video_session import video_session_config
        from jasna.session_factory import build_pipeline

        config = video_session_config(settings, codec=settings.codec, encoder_settings={})
        pipeline = build_pipeline(config, session, self.path, self.path)
        pipeline.validate_metadata(self.metadata)
        return pipeline

    def _run_preview_pass(self, command: _Request, pipeline) -> RestorationFrame | RestorationClip | None:
        from jasna.pipeline_threads import run_restoration_pass

        settings = command.settings
        pipeline.vr_projection = command.projection
        pipeline.configure_vr(self.metadata)
        left_eye_only = pipeline.vr_resolution.is_sbs
        window = (
            playback_window(self.metadata, command.center_seconds, settings.max_clip_size)
            if command.playback
            else preview_window(self.metadata, command.center_seconds, settings.max_clip_size)
        )
        cancel_event = threading.Event()
        with self._cancel_lock:
            self._active_cancel = cancel_event

        lut_applier = None
        lut_path = (settings.lut_path or "").strip()
        if lut_path:
            from jasna.media.lut import GpuLutApplier, parse_cube_file

            lut_applier = GpuLutApplier(parse_cube_file(lut_path), pipeline.device)

        collector = (
            _PlaybackFrameCollector(self.metadata, self.max_size, lut_applier, left_eye_only=left_eye_only)
            if command.playback
            else _CenterFrameCollector(window.center_pts, cancel_event, lut_applier, left_eye_only=left_eye_only)
        )
        try:
            error = run_restoration_pass(
                pipeline,
                self.metadata,
                collector,
                cancel_event,
                seek_ts=window.seek_ts if window.seek_ts > 0 else None,
                use_async_secondary=False,
                end_pts=window.end_pts,
                poll=lambda: not self._commands.empty() or self._closed.is_set(),
            )
        finally:
            with self._cancel_lock:
                self._active_cancel = None

        superseded = not self._commands.empty() or self._closed.is_set()
        if error is not None and not collector.done and not superseded:
            raise error
        if not superseded:
            if isinstance(collector, _PlaybackFrameCollector) and collector.done:
                return RestorationClip(collector.result_frames(), command.generation)
            if isinstance(collector, _CenterFrameCollector) and collector.has_result:
                return RestorationFrame(
                    max(
                        0.0,
                        (collector._best_pts - self.metadata.start_pts)
                        * float(self.metadata.time_base),
                    ),
                    collector.result_image(self.max_size),
                    command.generation,
                )
        return None
