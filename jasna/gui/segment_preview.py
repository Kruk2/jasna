from __future__ import annotations

import queue
import threading
from contextlib import ExitStack
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Iterator

import av
from PIL import Image

from jasna.gui.queues import CommandWorker, Stop
from jasna.media.container_utils import demux_video, idle_decoder_threads, keyframe_seek_pts
from jasna.media.probe import VideoMetadata, get_video_meta_data, resolve_video_start_pts
from jasna.media.splice import KeyframeIndex, probe_keyframes
from jasna.vr180 import VrModeResolution, resolve_vr_mode


def _decode_video(container, stream) -> Iterator[av.VideoFrame]:
    return (frame for packet in demux_video(container, stream) for frame in packet.decode())


@dataclass(frozen=True)
class PreviewLoaded:
    metadata: VideoMetadata
    vr_resolution: VrModeResolution


@dataclass(frozen=True)
class PreviewKeyframes:
    index: KeyframeIndex | None
    error: str | None


@dataclass(frozen=True)
class PreviewFrame:
    seconds: float
    image: Image.Image
    generation: int


@dataclass(frozen=True)
class PreviewFullFrame:
    seconds: float
    image: Image.Image


@dataclass(frozen=True)
class PreviewEnded:
    generation: int


@dataclass(frozen=True)
class PreviewFailed:
    message: str


@dataclass(frozen=True)
class _Seek:
    seconds: float
    generation: int


@dataclass(frozen=True)
class _Next:
    generation: int


@dataclass(frozen=True)
class _Previous:
    seconds: float
    generation: int


PreviewEvent = PreviewLoaded | PreviewKeyframes | PreviewFrame | PreviewFullFrame | PreviewEnded | PreviewFailed


class SegmentPreviewWorker(CommandWorker):
    """Single-threaded PyAV preview decoder controlled by coalesced commands."""

    def __init__(
        self,
        path: str | Path,
        *,
        max_size: tuple[int, int] = (960, 540),
        vr_mode: str = "auto",
    ) -> None:
        super().__init__(f"segment-preview-{Path(path).name}")
        self.path = Path(path)
        self.max_size = max_size
        self.vr_mode = str(vr_mode)
        self._stereo = "off"
        self.events: queue.Queue[PreviewEvent] = queue.Queue()
        self._grab_requested = threading.Event()

    def seek(self, seconds: float) -> int:
        self._generation += 1
        self._send(_Seek(max(0.0, float(seconds)), self._generation))
        return self._generation

    def next_frame(self) -> None:
        if self._closed.is_set():
            return
        if self._commands.empty():
            self._commands.put_nowait(_Next(self._generation))

    def previous_frame(self, seconds: float) -> int:
        self._generation += 1
        self._send(_Previous(max(0.0, float(seconds)), self._generation))
        return self._generation

    def grab_full(self) -> None:
        """Emit the shown frame at native resolution, once there is one; seeks asked meanwhile
        still run first."""
        self._grab_requested.set()

    def _run(self) -> None:
        try:
            metadata = get_video_meta_data(str(self.path))
            if float(metadata.duration) <= 0:
                raise ValueError(f"Could not determine duration of {self.path.name}")
            vr_resolution = resolve_vr_mode(self.vr_mode, metadata, self.path)
            self._stereo = vr_resolution.resolved
            self.events.put(PreviewLoaded(metadata, vr_resolution))
            threading.Thread(
                target=self._probe_keyframes,
                args=(metadata,),
                name=f"segment-keyframes-{self.path.name}",
                daemon=True,
            ).start()
            with av.open(str(self.path), metadata_errors="replace") as container, ExitStack() as before_close:
                stream = container.streams.video[0]
                before_close.callback(idle_decoder_threads, stream.codec_context)
                start_pts = resolve_video_start_pts(stream.start_time, metadata.start_pts)
                if not stream.codec_context.is_hwaccel:
                    stream.codec_context.thread_type = "AUTO"
                decoded: Iterator[av.VideoFrame] | None = None
                last_frame: av.VideoFrame | None = None
                while not self._closed.is_set():
                    if last_frame is not None and self._commands.empty() and self._grab_requested.is_set():
                        self._grab_requested.clear()
                        self.events.put(
                            PreviewFullFrame(
                                self._frame_seconds(last_frame, stream, start_pts),
                                self._to_full_image(last_frame),
                            )
                        )
                    command = self._poll_command()
                    if command is None or isinstance(command, Stop):
                        continue
                    if isinstance(command, _Seek):
                        frame, decoded = self._seek(
                            container,
                            stream,
                            command.seconds,
                            start_pts,
                        )
                    elif isinstance(command, _Previous):
                        frame, decoded = self._previous(
                            container,
                            stream,
                            command.seconds,
                            start_pts,
                        )
                    else:
                        frame = next(decoded, None) if decoded is not None else None
                    if frame is None:
                        self.events.put(PreviewEnded(command.generation))
                        continue
                    last_frame = frame
                    self.events.put(
                        PreviewFrame(
                            self._frame_seconds(frame, stream, start_pts),
                            self._to_image(frame),
                            command.generation,
                        )
                    )
        except Exception as exc:
            if not self._closed.is_set():
                self.events.put(PreviewFailed(str(exc)))

    def _probe_keyframes(self, metadata: VideoMetadata) -> None:
        try:
            event = PreviewKeyframes(probe_keyframes(self.path, metadata), None)
        except Exception as exc:
            event = PreviewKeyframes(None, str(exc))
        if not self._closed.is_set():
            self.events.put(event)

    def _seek(self, container, stream, seconds: float, start_pts: int):
        target_pts = start_pts + round(float(seconds) / stream.time_base)
        container.seek(keyframe_seek_pts(container, stream, target_pts), stream=stream, backward=True)
        decoded = _decode_video(container, stream)
        closest = None
        for frame in decoded:
            closest = frame
            if frame.pts is None or frame.pts >= target_pts:
                return frame, decoded
        return closest, decoded

    def _previous(self, container, stream, seconds: float, start_pts: int):
        target_pts = start_pts + round(float(seconds) / stream.time_base)
        one_second_pts = max(1, round(1.0 / stream.time_base))
        seek_pts = keyframe_seek_pts(container, stream, max(start_pts, target_pts - one_second_pts))
        container.seek(seek_pts, stream=stream, backward=True)
        decoded = _decode_video(container, stream)
        previous = None
        for frame in decoded:
            if frame.pts is None or frame.pts >= target_pts:
                if previous is None:
                    return frame, decoded
                return previous, chain((frame,), decoded)
            previous = frame
        return previous, decoded

    def _frame_seconds(self, frame, stream, start_pts: int) -> float:
        if frame.pts is None:
            return 0.0
        return max(0.0, float((frame.pts - start_pts) * stream.time_base))

    def _to_full_image(self, frame) -> Image.Image:
        return frame.reformat(format="rgb24").to_image()

    def _to_image(self, frame) -> Image.Image:
        eyes_x = 2 if self._stereo == "sbs" else 1
        eyes_y = 2 if self._stereo == "tb" else 1
        content_width = int(frame.width) // eyes_x
        content_height = int(frame.height) // eyes_y
        max_width, max_height = self.max_size
        scale = min(1.0, max_width / content_width, max_height / content_height)
        target_width = max(2, round(content_width * scale))
        target_height = max(2, round(content_height * scale))
        if target_width % 2:
            target_width -= 1
        if target_height % 2:
            target_height -= 1
        preview = frame.reformat(
            width=target_width * eyes_x,
            height=target_height * eyes_y,
            format="rgb24",
        )
        return preview.to_image().crop((0, 0, target_width, target_height))
