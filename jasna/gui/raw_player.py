from __future__ import annotations

import importlib
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from jasna._frozen import is_frozen
from jasna.gui.models import AppSettings
from jasna.media import VideoMetadata
from jasna.os_utils import resolve_executable, subprocess_no_window_kwargs

logger = logging.getLogger(__name__)

DISPLAY_FPS = 30.0
BUFFER_SECONDS = 3.0
BUFFER_FRAMES = 90
PREROLL_SECONDS = 1.5


@dataclass(frozen=True)
class PlayerFrame:
    seconds: float
    image: Image.Image
    generation: int


@dataclass(frozen=True)
class PlayerStatus:
    message: str
    generation: int


@dataclass(frozen=True)
class PlayerEnded:
    generation: int


@dataclass(frozen=True)
class PlayerFailed:
    message: str
    generation: int


PlayerEvent = PlayerStatus | PlayerEnded | PlayerFailed


class TimestampFrameBuffer:
    def __init__(
        self,
        *,
        max_seconds: float = BUFFER_SECONDS,
        max_frames: int = BUFFER_FRAMES,
    ) -> None:
        self._max_seconds = float(max_seconds)
        self._max_frames = int(max_frames)
        self._frames: deque[PlayerFrame] = deque()
        self._condition = threading.Condition()
        self._generation = 0
        self._eof = False

    @property
    def generation(self) -> int:
        with self._condition:
            return self._generation

    @property
    def eof(self) -> bool:
        with self._condition:
            return self._eof

    def reset(self, generation: int) -> None:
        with self._condition:
            self._generation = int(generation)
            self._frames.clear()
            self._eof = False
            self._condition.notify_all()

    def mark_eof(self, generation: int) -> None:
        with self._condition:
            if generation == self._generation:
                self._eof = True
                self._condition.notify_all()

    def put(self, frame: PlayerFrame, cancel_event: threading.Event) -> bool:
        with self._condition:
            while (
                frame.generation == self._generation
                and self._would_overflow_locked(frame)
            ):
                if cancel_event.is_set():
                    return False
                self._condition.wait(timeout=0.05)
            if cancel_event.is_set() or frame.generation != self._generation:
                return False
            self._frames.append(frame)
            self._condition.notify_all()
            return True

    def peek(self) -> PlayerFrame | None:
        with self._condition:
            return self._frames[0] if self._frames else None

    def pop_due(self, seconds: float, tolerance: float = 0.0) -> PlayerFrame | None:
        due: PlayerFrame | None = None
        with self._condition:
            limit = float(seconds) + float(tolerance)
            while self._frames and self._frames[0].seconds <= limit:
                due = self._frames.popleft()
            if due is not None:
                self._condition.notify_all()
        return due

    def ready(self, minimum_seconds: float = PREROLL_SECONDS) -> bool:
        with self._condition:
            if not self._frames:
                return False
            if self._eof:
                return True
            return self._span_locked() >= float(minimum_seconds)

    def empty(self) -> bool:
        with self._condition:
            return not self._frames

    def __len__(self) -> int:
        with self._condition:
            return len(self._frames)

    def _span_locked(self) -> float:
        if len(self._frames) < 2:
            return 0.0
        return self._frames[-1].seconds - self._frames[0].seconds

    def _would_overflow_locked(self, frame: PlayerFrame) -> bool:
        return len(self._frames) >= self._max_frames or (
            bool(self._frames)
            and frame.seconds - self._frames[0].seconds >= self._max_seconds
        )


class RawFrameWriter:
    def __init__(
        self,
        metadata: VideoMetadata,
        frame_buffer: TimestampFrameBuffer,
        cancel_event: threading.Event,
        generation: int,
        max_size: tuple[int, int],
        lut_applier=None,
    ) -> None:
        self._metadata = metadata
        self._frame_buffer = frame_buffer
        self._cancel_event = cancel_event
        self._generation = int(generation)
        self._max_size = max_size
        self._size_lock = threading.Lock()
        self._lut_applier = lut_applier
        self._last_emitted_seconds: float | None = None
        self.frames_written = 0

    def set_max_size(self, max_size: tuple[int, int]) -> None:
        with self._size_lock:
            self._max_size = (
                max(2, int(max_size[0])),
                max(2, int(max_size[1])),
            )

    def write(self, frame, pts: int, *, apply_lut: bool = True) -> None:
        seconds = max(
            0.0,
            (int(pts) - int(self._metadata.start_pts)) * float(self._metadata.time_base),
        )
        minimum_interval = 1.0 / DISPLAY_FPS
        if (
            self._last_emitted_seconds is not None
            and seconds - self._last_emitted_seconds < minimum_interval - 1e-6
        ):
            return
        if apply_lut and self._lut_applier is not None:
            frame = self._lut_applier.apply(frame)
        with self._size_lock:
            max_size = self._max_size
        player_frame = PlayerFrame(
            seconds=seconds,
            image=_gpu_frame_image(frame, max_size, exact_size=True),
            generation=self._generation,
        )
        if self._frame_buffer.put(player_frame, self._cancel_event):
            self._last_emitted_seconds = seconds
            self.frames_written += 1

    def after_write(self, frames_written: int) -> None:
        pass


def _gpu_frame_image(
    frame,
    max_size: tuple[int, int],
    *,
    exact_size: bool = False,
) -> Image.Image:
    import torch.nn.functional as functional

    height, width = int(frame.shape[-2]), int(frame.shape[-1])
    max_width, max_height = max(2, int(max_size[0])), max(2, int(max_size[1]))
    if exact_size:
        target = (max_height, max_width)
    else:
        scale = min(max_width / width, max_height / height)
        target = (
            max(2, round(height * scale)),
            max(2, round(width * scale)),
        )
    if target != (height, width):
        resize_dtype = frame.dtype
        if not frame.dtype.is_floating_point:
            import torch

            resize_dtype = torch.float16 if frame.is_cuda else torch.float32
        frame = functional.interpolate(
            frame.unsqueeze(0).to(resize_dtype),
            size=target,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).round_().clamp_(0, 255).to(frame.dtype)
    array = frame.to("cpu").permute(1, 2, 0).contiguous().numpy()
    return Image.fromarray(array).copy()


@dataclass(frozen=True)
class _Play:
    seconds: float
    generation: int
    settings: AppSettings


@dataclass(frozen=True)
class _Stop:
    pass


class RawPlayerWorker:
    def __init__(
        self,
        path: str | Path,
        metadata: VideoMetadata,
        settings: AppSettings,
        frame_buffer: TimestampFrameBuffer,
        *,
        max_size: tuple[int, int],
        on_stopped: Callable[[], None] | None = None,
    ) -> None:
        self.path = Path(path)
        self.metadata = metadata
        self.settings = settings
        self.frame_buffer = frame_buffer
        self.events: queue.Queue[PlayerEvent] = queue.Queue()
        self._max_size = max_size
        self._commands: queue.Queue[_Play | _Stop] = queue.Queue(maxsize=1)
        self._closed = threading.Event()
        self._generation = 0
        self._active_cancel: threading.Event | None = None
        self._active_writer: RawFrameWriter | None = None
        self._cancel_lock = threading.Lock()
        self._on_stopped = on_stopped
        self._thread = threading.Thread(
            target=self._run,
            name=f"raw-player-{self.path.name}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def play_from(self, seconds: float) -> int:
        self._generation += 1
        generation = self._generation
        self.frame_buffer.reset(generation)
        self._cancel_active_pass()
        self._replace_command(
            _Play(
                max(0.0, float(seconds)),
                generation,
                self.settings,
            )
        )
        return generation

    def reload_from(self, settings: AppSettings, seconds: float) -> int:
        self.settings = settings
        return self.play_from(seconds)

    def set_max_size(self, max_size: tuple[int, int]) -> None:
        self._max_size = max_size
        with self._cancel_lock:
            if self._active_writer is not None:
                self._active_writer.set_max_size(max_size)

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._cancel_active_pass()
        self._replace_command(_Stop(), allow_closed=True)

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _cancel_active_pass(self) -> None:
        with self._cancel_lock:
            if self._active_cancel is not None:
                self._active_cancel.set()

    def _replace_command(self, command: _Play | _Stop, *, allow_closed: bool = False) -> None:
        if self._closed.is_set() and not allow_closed:
            return
        try:
            while True:
                self._commands.get_nowait()
        except queue.Empty:
            pass
        self._commands.put_nowait(command)

    def _run(self) -> None:
        session = None
        pipeline = None
        session_key = None
        try:
            while not self._closed.is_set():
                try:
                    command = self._commands.get(timeout=0.1)
                except queue.Empty:
                    continue
                if isinstance(command, _Stop):
                    break
                try:
                    from jasna.gui.video_session import video_session_key

                    command_session_key = video_session_key(command.settings)
                    if session is None or command_session_key != session_key:
                        if pipeline is not None or session is not None:
                            self._release_pipeline(pipeline, session)
                            pipeline = None
                            session = None
                            session_key = None
                        self.events.put(PlayerStatus("loading_models", command.generation))
                        session, pipeline = self._build_pipeline(command.settings)
                        session_key = command_session_key
                    if not self._commands.empty() or self._closed.is_set():
                        continue
                    self.events.put(PlayerStatus("restoring", command.generation))
                    completed = self._run_pass(command, pipeline, session)
                    superseded = not self._commands.empty() or self._closed.is_set()
                    if completed and not superseded:
                        self.frame_buffer.mark_eof(command.generation)
                        self.events.put(PlayerEnded(command.generation))
                except Exception as exc:
                    if not self._closed.is_set() and self._commands.empty():
                        self.events.put(PlayerFailed(str(exc), command.generation))
        finally:
            self._release_pipeline(pipeline, session)
            if self._on_stopped is not None:
                self._on_stopped()

    @staticmethod
    def _release_pipeline(pipeline, session) -> None:
        try:
            if pipeline is not None:
                pipeline.close()
        finally:
            if session is not None:
                from jasna.gui.video_session import release_session_memory

                try:
                    session.close()
                finally:
                    release_session_memory(session.device)

    def _build_pipeline(self, settings: AppSettings | None = None):
        from jasna.gui.video_session import (
            build_video_session,
            video_session_config,
        )
        from jasna.session_factory import build_pipeline

        settings = settings or self.settings
        session = build_video_session(
            settings,
            disable_basicvsrpp_tensorrt=False,
            log=logger.info,
        )
        pipeline = None
        try:
            config = video_session_config(
                settings,
                codec=settings.codec,
                encoder_settings={},
            )
            pipeline = build_pipeline(
                config,
                session,
                self.path,
                self.path,
            )
            pipeline._validate_metadata(self.metadata)
            pipeline.configure_vr(self.metadata)
            return session, pipeline
        except Exception:
            if pipeline is not None:
                pipeline.close()
            session.close()
            from jasna.gui.video_session import release_session_memory

            release_session_memory(session.device)
            raise

    def _run_pass(self, command: _Play, pipeline, session) -> bool:
        cancel_event = threading.Event()
        lut_applier = None
        lut_path = (self.settings.lut_path or "").strip()
        if lut_path:
            from jasna.media.lut import GpuLutApplier, parse_cube_file

            lut_applier = GpuLutApplier(parse_cube_file(lut_path), session.device)
        writer = RawFrameWriter(
            self.metadata,
            self.frame_buffer,
            cancel_event,
            command.generation,
            self._max_size,
            lut_applier,
        )
        with self._cancel_lock:
            self._active_cancel = cancel_event
            self._active_writer = writer
        try:
            run_raw_restoration_pass(
                pipeline,
                self.metadata,
                writer,
                cancel_event,
                seek_seconds=command.seconds,
            )
            return not cancel_event.is_set()
        finally:
            with self._cancel_lock:
                self._active_cancel = None
                self._active_writer = None


def run_raw_restoration_pass(
    pipeline,
    metadata: VideoMetadata,
    frame_writer: RawFrameWriter,
    cancel_event: threading.Event,
    *,
    seek_seconds: float,
) -> None:
    from queue import Empty, Queue

    from jasna.blend_buffer import BlendBuffer
    from jasna.crop_buffer import CropBuffer
    from jasna.frame_queue import FrameQueue
    from jasna.pipeline_threads import (
        blend_encode_loop,
        decode_detect_loop,
        primary_restore_loop,
        secondary_restore_loop,
    )
    from jasna.restorer.secondary_restorer import AsyncSecondaryRestorer
    from jasna.vram_offloader import VramOffloader

    if isinstance(pipeline.restoration_pipeline.secondary_restorer, AsyncSecondaryRestorer):
        raise ValueError("Topaz Video AI is not available in the video player")

    device = pipeline.device
    secondary_workers = max(1, int(pipeline.restoration_pipeline.secondary_num_workers))
    clip_queue = FrameQueue(max_frames=pipeline.max_clip_size)
    secondary_queue = FrameQueue(max_frames=pipeline.max_clip_size * secondary_workers)
    encode_queue = FrameQueue(max_frames=pipeline.max_clip_size)
    metadata_queue = Queue(maxsize=pipeline.max_clip_size * 5)
    error_holder: list[BaseException] = []
    blend_buffer = BlendBuffer(device=device, vr_projector=pipeline._vr_projector)
    crop_buffers: dict[int, CropBuffer] = {}
    crop_lock = threading.Lock()
    primary_idle_event = threading.Event()
    frame_shape: list[tuple[int, int]] = []
    vram_offloader = VramOffloader(
        device=device,
        blend_buffer=blend_buffer,
        crop_buffers=crop_buffers,
        crop_lock=crop_lock,
    )
    vram_offloader.set_pipeline_queues(
        clip_queue,
        secondary_queue,
        encode_queue,
        metadata_queue,
    )
    seek_ts = float(seek_seconds) if seek_seconds > 0 else None

    threads = [
        threading.Thread(
            target=lambda: decode_detect_loop(
                input_video=str(pipeline.input_video),
                batch_size=pipeline.batch_size,
                device=device,
                metadata=metadata,
                detection_model=pipeline._job_detection_model,
                max_clip_size=pipeline.max_clip_size,
                temporal_overlap=pipeline.temporal_overlap,
                max_detection_gap=pipeline.max_detection_gap,
                min_detection_duration=pipeline.min_detection_duration,
                enable_crossfade=pipeline.enable_crossfade,
                scene_detection=pipeline.scene_detection,
                blend_buffer=blend_buffer,
                crop_buffers=crop_buffers,
                clip_queue=clip_queue,
                metadata_queue=metadata_queue,
                error_holder=error_holder,
                frame_shape=frame_shape,
                cancel_event=cancel_event,
                seek_ts=seek_ts,
                vr_mode=pipeline._vr_resolution.resolved,
                vr_projector=pipeline._vr_projector,
            ),
            name="PlayerDecodeDetect",
            daemon=True,
        ),
        threading.Thread(
            target=lambda: primary_restore_loop(
                device=device,
                restoration_pipeline=pipeline.restoration_pipeline,
                clip_queue=clip_queue,
                secondary_queue=secondary_queue,
                error_holder=error_holder,
                primary_idle_event=primary_idle_event,
                cancel_event=cancel_event,
            ),
            name="PlayerPrimaryRestore",
            daemon=True,
        ),
        threading.Thread(
            target=lambda: secondary_restore_loop(
                device=device,
                restoration_pipeline=pipeline.restoration_pipeline,
                secondary_queue=secondary_queue,
                encode_queue=encode_queue,
                error_holder=error_holder,
                cancel_event=cancel_event,
            ),
            name="PlayerSecondaryRestore",
            daemon=True,
        ),
        threading.Thread(
            target=lambda: blend_encode_loop(
                input_video=str(pipeline.input_video),
                batch_size=pipeline.batch_size,
                device=device,
                metadata=metadata,
                blend_buffer=blend_buffer,
                encode_queue=encode_queue,
                metadata_queue=metadata_queue,
                error_holder=error_holder,
                frame_writer=frame_writer,
                cancel_event=cancel_event,
                seek_ts=seek_ts,
                vram_offloader=vram_offloader,
            ),
            name="PlayerBlend",
            daemon=True,
        ),
    ]

    vram_offloader.start()
    for thread in threads:
        thread.start()
    while any(thread.is_alive() for thread in threads) and not cancel_event.wait(0.05):
        pass

    queues = (clip_queue, secondary_queue, encode_queue, metadata_queue)
    for thread in threads:
        while thread.is_alive():
            for pipeline_queue in queues:
                try:
                    while True:
                        pipeline_queue.get_nowait()
                except Empty:
                    pass
            thread.join(timeout=0.02)
    vram_offloader.stop()

    error = error_holder[0] if error_holder else None
    del clip_queue, secondary_queue, encode_queue, metadata_queue
    del blend_buffer, crop_buffers
    import gc

    gc.collect()
    if getattr(device, "type", None) != "cpu":
        from jasna.accelerator import empty_cache, ipc_collect

        empty_cache(device)
        ipc_collect(device)
    if error is not None and not cancel_event.is_set():
        raise error


class PlaybackClock(Protocol):
    def play(self) -> None: ...
    def pause(self) -> None: ...
    def seek(self, seconds: float) -> None: ...
    def seconds(self) -> float: ...
    def close(self) -> None: ...


class SoftwareClock:
    def __init__(self) -> None:
        self._position = 0.0
        self._anchor = time.monotonic()
        self._playing = False

    def play(self) -> None:
        if not self._playing:
            self._anchor = time.monotonic()
            self._playing = True

    def pause(self) -> None:
        if self._playing:
            self._position = self.seconds()
            self._playing = False

    def seek(self, seconds: float) -> None:
        self._position = max(0.0, float(seconds))
        self._anchor = time.monotonic()

    def seconds(self) -> float:
        if not self._playing:
            return self._position
        return self._position + time.monotonic() - self._anchor

    def close(self) -> None:
        self.pause()


class VlcUnavailableError(RuntimeError):
    pass


class VlcAudioClock:
    def __init__(self, path: str | Path) -> None:
        self._dll_directory = _configure_bundled_vlc()
        try:
            vlc = importlib.import_module("vlc")
            self._instance = vlc.Instance("--no-video", "--no-spu", "--quiet")
        except (ImportError, OSError) as exc:
            raise VlcUnavailableError(
                "python-vlc and libVLC 3 are required for video player audio"
            ) from exc
        if self._instance is None:
            raise VlcUnavailableError("libVLC could not be initialized")
        self._media = self._instance.media_new(str(Path(path).resolve()))
        self._media.add_option(":no-video")
        self._player = self._instance.media_player_new()
        self._player.set_media(self._media)
        self._clock = SoftwareClock()
        self._pending_seconds: float | None = 0.0
        self._started = False
        self._volume = 100
        self._muted = False
        self._audio_settings_applied = False

    def play(self) -> None:
        if self._player.play() == -1:
            raise VlcUnavailableError("libVLC could not start audio playback")
        self._started = True
        if self._pending_seconds is not None:
            self._player.set_time(round(self._pending_seconds * 1000))
            self._clock.seek(self._pending_seconds)
            self._pending_seconds = None
        self._clock.play()

    def pause(self) -> None:
        if self._started:
            self._clock.pause()
            self._player.set_pause(1)

    def seek(self, seconds: float) -> None:
        target = max(0.0, float(seconds))
        self._clock.seek(target)
        if self._started:
            self._player.set_time(round(target * 1000))
            self._pending_seconds = None
        else:
            self._pending_seconds = target

    def seconds(self) -> float:
        if not self._started:
            return self._clock.seconds()
        current = self._player.get_time()
        if (
            not self._audio_settings_applied
            and (current > 0 or self._player.is_playing())
        ):
            self._apply_audio_settings()
        return self._clock.seconds()

    def set_volume(self, volume: int) -> None:
        self._volume = max(0, min(100, int(volume)))
        self._audio_settings_applied = False
        if self._audio_output_ready():
            self._apply_audio_settings()

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)
        self._audio_settings_applied = False
        if self._audio_output_ready():
            self._apply_audio_settings()

    def _audio_output_ready(self) -> bool:
        return self._started and bool(self._player.is_playing())

    def _apply_audio_settings(self) -> None:
        if self._player.audio_set_volume(self._volume) == -1:
            raise VlcUnavailableError("libVLC could not change the audio volume")
        self._player.audio_set_mute(self._muted)
        self._audio_settings_applied = True

    def close(self) -> None:
        self._clock.close()
        self._player.stop()
        self._player.release()
        self._media.release()
        self._instance.release()
        if self._dll_directory is not None:
            self._dll_directory.close()


def source_has_audio(path: str | Path) -> bool:
    result = subprocess.run(
        [
            resolve_executable("ffprobe"),
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **subprocess_no_window_kwargs(),
    )
    if result.returncode != 0:
        raise OSError(f"ffprobe audio probe failed: {result.stderr.strip()}")
    return bool(result.stdout.strip())


def _configure_bundled_vlc():
    if not is_frozen():
        return None
    vlc_dir = Path(sys.executable).parent / "vlc"
    if not vlc_dir.is_dir():
        return None
    plugins = vlc_dir / "plugins"
    os.environ["VLC_PLUGIN_PATH"] = str(plugins)
    if os.name == "nt":
        os.environ["PYTHON_VLC_LIB_PATH"] = str(vlc_dir / "libvlc.dll")
        return os.add_dll_directory(str(vlc_dir))
    candidates = (vlc_dir / "libvlc.so.5", vlc_dir / "libvlc.so")
    library = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    os.environ["PYTHON_VLC_LIB_PATH"] = str(library)
    return None
