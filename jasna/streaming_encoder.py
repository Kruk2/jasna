from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path

from jasna.os_utils import get_subprocess_startup_info, resolve_executable

log = logging.getLogger(__name__)


class HlsEncoder:
    def __init__(
        self,
        segments_dir: Path,
        segment_duration: float,
        fps: float,
        start_number: int = 0,
    ):
        self.segments_dir = Path(segments_dir)
        self.segment_duration = float(segment_duration)
        self.fps = float(fps)
        self._proc: subprocess.Popen | None = None
        self._start_number = int(start_number)
        self._lock = threading.Lock()
        self._failed = False

    def start(self, start_number: int | None = None) -> None:
        if start_number is not None:
            self._start_number = int(start_number)
        ts_offset = self._start_number * self.segment_duration
        cmd = [
            resolve_executable("ffmpeg"),
            "-hide_banner",
            "-loglevel", "error",
            "-framerate", str(self.fps),
            "-f", "hevc",
            "-i", "pipe:0",
            "-c:v", "copy",
            "-output_ts_offset", str(ts_offset),
            "-f", "hls",
            "-hls_time", str(self.segment_duration),
            "-hls_list_size", "0",
            "-hls_flags", "independent_segments",
            "-hls_segment_type", "mpegts",
            "-start_number", str(self._start_number),
            "-hls_segment_filename", str(self.segments_dir / "seg_%05d.ts"),
            str(self.segments_dir / "live.m3u8"),
        ]
        log.debug("[hls-encoder] starting ffmpeg: %s", " ".join(cmd))
        self._failed = False
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            startupinfo=get_subprocess_startup_info(),
        )

    def write(self, data: bytes) -> None:
        with self._lock:
            if self._failed or self._proc is None or self._proc.stdin is None:
                return
            try:
                self._proc.stdin.write(data)
            except (BrokenPipeError, OSError) as e:
                log.warning("[hls-encoder] pipe write failed: %s", e)
                self._failed = True

    def flush_and_restart(self, start_number: int) -> None:
        self._kill_process()
        m3u8 = self.segments_dir / "live.m3u8"
        if m3u8.exists():
            m3u8.unlink()
        self.start(start_number=start_number)

    def stop(self) -> None:
        self._close_process()

    def _close_process(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None

        if proc is None:
            return

        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)

        if proc.returncode and proc.returncode != 0:
            stderr = b""
            if proc.stderr:
                try:
                    stderr = proc.stderr.read()
                except Exception:
                    pass
            log.warning(
                "[hls-encoder] ffmpeg exited with code %d: %s",
                proc.returncode,
                stderr.decode(errors="replace")[:500],
            )

    def _kill_process(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None

        if proc is None:
            return

        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
        proc.kill()
        proc.wait(timeout=5.0)
