from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import av
from PIL import Image

from jasna.media import VideoMetadata, get_video_meta_data, resolve_video_start_pts


@dataclass(frozen=True)
class PreviewLoaded:
    metadata: VideoMetadata


@dataclass(frozen=True)
class PreviewFrame:
    seconds: float
    image: Image.Image
    generation: int


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
class _Stop:
    pass


PreviewEvent = PreviewLoaded | PreviewFrame | PreviewEnded | PreviewFailed
_Command = _Seek | _Next | _Stop


class SegmentPreviewWorker:
    """Single-threaded PyAV preview decoder controlled by coalesced commands."""

    def __init__(self, path: str | Path, *, max_size: tuple[int, int] = (960, 540)) -> None:
        self.path = Path(path)
        self.max_size = max_size
        self.events: queue.Queue[PreviewEvent] = queue.Queue()
        self._commands: queue.Queue[_Command] = queue.Queue(maxsize=1)
        self._closed = threading.Event()
        self._generation = 0
        self._thread = threading.Thread(
            target=self._run,
            name=f"segment-preview-{self.path.name}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def seek(self, seconds: float) -> int:
        self._generation += 1
        self._replace_command(_Seek(max(0.0, float(seconds)), self._generation))
        return self._generation

    def next_frame(self) -> None:
        if self._closed.is_set():
            return
        try:
            self._commands.put_nowait(_Next(self._generation))
        except queue.Full:
            pass

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._replace_command(_Stop(), allow_closed=True)

    def _replace_command(self, command: _Command, *, allow_closed: bool = False) -> None:
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
        try:
            metadata = get_video_meta_data(str(self.path))
            if float(metadata.duration) <= 0:
                raise ValueError(f"Could not determine duration of {self.path.name}")
            self.events.put(PreviewLoaded(metadata))
            with av.open(str(self.path)) as container:
                stream = container.streams.video[0]
                start_pts = resolve_video_start_pts(stream.start_time, metadata.start_pts)
                if not stream.codec_context.is_hwaccel:
                    stream.codec_context.thread_type = "AUTO"
                decoded: Iterator[av.VideoFrame] | None = None
                while not self._closed.is_set():
                    try:
                        command = self._commands.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    if isinstance(command, _Stop):
                        break
                    if isinstance(command, _Seek):
                        frame, decoded = self._seek(
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

    def _seek(self, container, stream, seconds: float, start_pts: int):
        target_pts = start_pts + round(float(seconds) / stream.time_base)
        container.seek(target_pts, stream=stream, backward=True)
        decoded = container.decode(stream)
        closest = None
        for frame in decoded:
            closest = frame
            if frame.pts is None or frame.pts >= target_pts:
                return frame, decoded
        return closest, decoded

    def _frame_seconds(self, frame, stream, start_pts: int) -> float:
        if frame.pts is None:
            return 0.0
        return max(0.0, float((frame.pts - start_pts) * stream.time_base))

    def _to_image(self, frame) -> Image.Image:
        width, height = int(frame.width), int(frame.height)
        max_width, max_height = self.max_size
        scale = min(1.0, max_width / width, max_height / height)
        target_width = max(2, round(width * scale))
        target_height = max(2, round(height * scale))
        if target_width % 2:
            target_width -= 1
        if target_height % 2:
            target_height -= 1
        preview = frame.reformat(
            width=target_width,
            height=target_height,
            format="rgb24",
        )
        return preview.to_image()
