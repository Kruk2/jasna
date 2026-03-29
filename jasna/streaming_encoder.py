from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
from pathlib import Path

import torch

from jasna.media import VideoMetadata
from jasna.os_utils import get_subprocess_startup_info, resolve_executable

log = logging.getLogger(__name__)


class StreamingEncoder:
    """Encodes RGB frames to HLS MPEGTS segments via ffmpeg (h264_nvenc) + audio copy."""

    def __init__(
        self,
        segments_dir: Path,
        segment_duration: float,
        metadata: VideoMetadata,
        source_video: str,
        device: torch.device | None = None,
    ):
        self.segments_dir = Path(segments_dir)
        self.segment_duration = float(segment_duration)
        self.metadata = metadata
        self.source_video = source_video

        self._width = metadata.video_width
        self._height = metadata.video_height
        self._fps = metadata.video_fps_exact
        self._gop_size = max(1, round(float(metadata.video_fps_exact) * segment_duration))
        self._frame_bytes = self._width * self._height * 3

        gpu_idx = 0
        if device is not None and device.type == 'cuda':
            gpu_idx = device.index if device.index is not None else 0
        self._gpu_index = gpu_idx

        self._ffmpeg = resolve_executable('ffmpeg')
        self._process: subprocess.Popen | None = None
        self._stderr_thread: threading.Thread | None = None
        self._write_queue: queue.Queue = queue.Queue(maxsize=16)
        self._writer_thread: threading.Thread | None = None
        self._stop_sentinel = object()
        self._started = False

    def start(self, start_number: int = 0) -> None:
        self._launch_ffmpeg(start_number)
        self._started = True
        self._writer_thread = threading.Thread(
            target=self._writer_loop, daemon=True, name="StreamingWriterThread",
        )
        self._writer_thread.start()
        log.debug("[stream-enc] started at segment %d", start_number)

    def write_frame(self, frame: torch.Tensor, pts: int) -> None:
        if not self._started:
            return
        if frame.dim() == 3 and frame.shape[0] == 3:
            raw = frame.permute(1, 2, 0).contiguous().cpu().numpy().tobytes()
        else:
            raw = frame.cpu().numpy().tobytes()
        while self._started:
            try:
                self._write_queue.put(raw, timeout=0.1)
                return
            except queue.Full:
                continue

    def flush_and_restart(self, start_number: int) -> None:
        self._started = False
        self._stop_writer(drain=True)
        self._kill_ffmpeg()
        self._cleanup_segments()
        self.start(start_number=start_number)

    def _cleanup_segments(self) -> None:
        for f in self.segments_dir.glob('*.ts'):
            f.unlink(missing_ok=True)
        for f in self.segments_dir.glob('*.m3u8'):
            f.unlink(missing_ok=True)

    def stop(self) -> None:
        self._started = False
        self._stop_writer(drain=False)
        self._close_ffmpeg()
        log.debug("[stream-enc] stopped")

    def _launch_ffmpeg(self, start_number: int) -> None:
        seek_time = start_number * self.segment_duration
        fps_str = f"{self._fps.numerator}/{self._fps.denominator}" if hasattr(self._fps, 'numerator') else str(float(self._fps))

        cmd: list[str] = [self._ffmpeg, '-y', '-hide_banner', '-loglevel', 'warning']

        has_source = self.source_video and os.path.isfile(self.source_video)
        if has_source:
            if seek_time > 0:
                cmd += ['-ss', f'{seek_time:.3f}']
            cmd += ['-i', self.source_video]

        cmd += [
            '-f', 'rawvideo',
            '-pix_fmt', 'rgb24',
            '-s', f'{self._width}x{self._height}',
            '-r', fps_str,
            '-i', 'pipe:0',
        ]

        if has_source:
            cmd += ['-map', '1:v:0', '-map', '0:a:0?']
        else:
            cmd += ['-map', '0:v:0']

        cmd += [
            '-c:v', 'h264_nvenc',
            '-preset', 'p1',
            '-tune', 'ull',
            '-rc', 'vbr',
            '-cq', '28',
            '-gpu', str(self._gpu_index),
            '-g', str(self._gop_size),
            '-pix_fmt', 'yuv420p',
        ]

        if has_source:
            cmd += ['-c:a', 'copy']

        if seek_time > 0:
            cmd += ['-output_ts_offset', f'{seek_time:.3f}']

        seg_pattern = str(self.segments_dir / 'seg_%05d.ts')
        playlist_path = str(self.segments_dir / '_hls_internal.m3u8')
        cmd += [
            '-f', 'hls',
            '-hls_time', str(self.segment_duration),
            '-hls_segment_type', 'mpegts',
            '-hls_segment_filename', seg_pattern,
            '-start_number', str(start_number),
            '-hls_list_size', '0',
            playlist_path,
        ]

        log.debug("[stream-enc] cmd: %s", ' '.join(cmd))

        creationflags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            startupinfo=get_subprocess_startup_info(),
            creationflags=creationflags,
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True, name="StreamingStderrThread",
        )
        self._stderr_thread.start()

    def _drain_stderr(self) -> None:
        proc = self._process
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            text = line.decode('utf-8', errors='replace').rstrip()
            if text:
                log.warning("[stream-enc ffmpeg] %s", text)

    def _writer_loop(self) -> None:
        while True:
            item = self._write_queue.get()
            if item is self._stop_sentinel:
                return
            if self._process is None or self._process.stdin is None:
                continue
            try:
                self._process.stdin.write(item)
            except (BrokenPipeError, OSError):
                log.warning("[stream-enc] pipe broken, writer exiting")
                return

    def _stop_writer(self, drain: bool) -> None:
        if self._writer_thread is None or not self._writer_thread.is_alive():
            self._writer_thread = None
            return
        if drain:
            while True:
                try:
                    self._write_queue.get_nowait()
                except queue.Empty:
                    break
        self._write_queue.put(self._stop_sentinel)
        self._writer_thread.join(timeout=5.0)
        self._writer_thread = None

    def _close_ffmpeg(self) -> None:
        proc = self._process
        if proc is None:
            return
        if proc.stdin and not proc.stdin.closed:
            try:
                proc.stdin.close()
            except OSError:
                pass
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            log.warning("[stream-enc] ffmpeg did not exit in time, killing")
            proc.kill()
            proc.wait()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)
            self._stderr_thread = None
        rc = proc.returncode
        if rc and rc != 0:
            log.warning("[stream-enc] ffmpeg exited with code %d", rc)
        self._process = None

    def _kill_ffmpeg(self) -> None:
        proc = self._process
        if proc is None:
            return
        if proc.stdin and not proc.stdin.closed:
            try:
                proc.stdin.close()
            except OSError:
                pass
        proc.kill()
        proc.wait()
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)
            self._stderr_thread = None
        self._process = None
