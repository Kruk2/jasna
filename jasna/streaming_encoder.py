from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
from pathlib import Path

import torch

from jasna.accelerator import AcceleratorVendor, vendor_for_device
from jasna.media import VideoMetadata
from jasna.os_utils import find_executable, get_subprocess_startup_info

log = logging.getLogger(__name__)

_WRITE_QUEUE_SIZE = 16
_WRITER_STOP_TIMEOUT = 5.0


class StreamingEncodeError(RuntimeError):
    pass


class StreamingEncoder:
    """Encodes RGB frames to HLS MPEGTS segments with the active GPU backend."""

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
        self._vendor = vendor_for_device(device)

        self._width = metadata.video_width
        self._height = metadata.video_height
        self._fps = metadata.video_fps_exact
        self._gop_size = max(1, round(float(metadata.video_fps_exact) * segment_duration))
        self._frame_bytes = self._width * self._height * 3

        gpu_idx = 0
        if device is not None and device.type == 'cuda':
            gpu_idx = device.index if device.index is not None else 0
        self._gpu_index = gpu_idx

        self._ffmpeg = find_executable('ffmpeg')
        if self._ffmpeg is None:
            raise RuntimeError(
                "ffmpeg not found (bundled tools/ or PATH); required for HLS streaming"
            )
        self._process: subprocess.Popen | None = None
        self._stderr_thread: threading.Thread | None = None
        self._write_queue: queue.Queue = queue.Queue(maxsize=_WRITE_QUEUE_SIZE)
        self._writer_thread: threading.Thread | None = None
        self._stop_sentinel = object()
        self._started = False
        self.failed = False
        self._writer_error: BaseException | None = None

    def start(self, start_number: int = 0) -> None:
        if self._process is not None or (
            self._writer_thread is not None and self._writer_thread.is_alive()
        ):
            raise RuntimeError("Streaming encoder is already started")
        self._write_queue = queue.Queue(maxsize=_WRITE_QUEUE_SIZE)
        self._stop_sentinel = object()
        self._writer_error = None
        self.failed = False
        self._launch_ffmpeg(start_number)
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("FFmpeg streaming process has no stdin")
        self._started = True
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            args=(process, self._write_queue, self._stop_sentinel),
            daemon=True,
            name="StreamingWriterThread",
        )
        self._writer_thread.start()
        log.debug("[stream-enc] started at segment %d", start_number)

    def write_frame(self, frame: torch.Tensor, pts: int) -> None:
        self.raise_if_failed()
        if not self._started:
            return
        if frame.dim() == 3 and frame.shape[0] == 3:
            raw = frame.permute(1, 2, 0).contiguous().cpu().numpy().tobytes()
        else:
            raw = frame.cpu().numpy().tobytes()
        write_queue = self._write_queue
        while self._started and write_queue is self._write_queue:
            self.raise_if_failed()
            try:
                write_queue.put(raw, timeout=0.1)
                return
            except queue.Full:
                continue
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        error = self._writer_error
        if error is not None:
            raise StreamingEncodeError("Streaming writer failed") from error

    def flush_and_restart(self, start_number: int) -> None:
        self._started = False
        self._stop_writer(discard_pending=True)
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
        self._stop_writer(discard_pending=False)
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

        if self._vendor is AcceleratorVendor.AMD:
            cmd += [
                '-c:v', 'h264_amf',
                '-usage', 'lowlatency_high_quality',
                '-quality', 'balanced',
                '-rc', 'qvbr',
                '-qvbr_quality_level', '19',
                '-bf', '0',
                '-profile:v', 'high',
                '-g', str(self._gop_size),
                '-pix_fmt', 'yuv420p',
            ]
        else:
            cmd += [
                '-c:v', 'h264_nvenc',
                '-preset', 'p4',
                '-tune', 'll',
                '-rc', 'vbr',
                '-cq', '19',
                '-bf', '0',
                '-profile:v', 'high',
                '-spatial-aq', '1',
                '-temporal-aq', '1',
                '-rc-lookahead', '8',
                '-gpu', str(self._gpu_index),
                '-g', str(self._gop_size),
                '-pix_fmt', 'yuv420p',
            ]

        sar = self.metadata.sample_aspect_ratio
        if sar != 1:
            cmd += ['-vf', f'setsar={sar.numerator}/{sar.denominator}']

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

    def _writer_loop(
        self,
        process: subprocess.Popen,
        write_queue: queue.Queue,
        stop_sentinel: object,
    ) -> None:
        while True:
            item = write_queue.get()
            if item is stop_sentinel:
                return
            try:
                process.stdin.write(item)
            except (BrokenPipeError, OSError, ValueError) as exc:
                log.warning("[stream-enc] pipe broken, writer exiting")
                if self._write_queue is write_queue:
                    self._writer_error = exc
                    self._started = False
                    self.failed = True
                return

    def _stop_writer(self, discard_pending: bool) -> None:
        writer_thread = self._writer_thread
        if writer_thread is None or not writer_thread.is_alive():
            self._writer_thread = None
            return
        write_queue = self._write_queue
        if discard_pending:
            while True:
                try:
                    write_queue.get_nowait()
                except queue.Empty:
                    break
        try:
            write_queue.put(self._stop_sentinel, timeout=_WRITER_STOP_TIMEOUT)
        except queue.Full:
            log.warning("[stream-enc] writer queue did not drain, killing ffmpeg")
            self._kill_ffmpeg()
        writer_thread.join(timeout=_WRITER_STOP_TIMEOUT)
        if writer_thread.is_alive():
            log.warning("[stream-enc] writer did not stop, killing ffmpeg")
            self._kill_ffmpeg()
            writer_thread.join(timeout=_WRITER_STOP_TIMEOUT)
        if writer_thread.is_alive():
            raise RuntimeError("Streaming writer thread did not stop")
        self._writer_thread = None

    def _close_ffmpeg(self) -> None:
        proc = self._process
        if proc is None:
            return
        if proc.stdin and not proc.stdin.closed:
            try:
                proc.stdin.close()
            except (OSError, ValueError):
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
        proc.kill()
        proc.wait()
        if proc.stdin and not proc.stdin.closed:
            try:
                proc.stdin.close()
            except (OSError, ValueError):
                pass
        if self._stderr_thread is not None:
            self._stderr_thread.join(timeout=2.0)
            self._stderr_thread = None
        self._process = None
