from __future__ import annotations

import logging
import math
import os
import tempfile
import threading
import time
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path

from jasna.media import VideoMetadata

log = logging.getLogger(__name__)

_FORWARD_SEEK_THRESHOLD = 5

_HLS_PLAYER_HTML = """\
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Jasna Stream</title>
<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<style>
  body { margin:0; background:#111; display:flex; justify-content:center; align-items:center; height:100vh; }
  video { max-width:100%; max-height:100vh; }
</style></head><body>
<video id="v" controls autoplay></video>
<script>
  var video = document.getElementById('v');
  if (Hls.isSupported()) {
    var hls = new Hls({maxBufferLength: 10, maxMaxBufferLength: 30});
    hls.loadSource('/stream.m3u8');
    hls.attachMedia(video);
  } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
    video.src = '/stream.m3u8';
  }
</script></body></html>
"""


def _generate_vod_playlist(
    total_duration: float,
    segment_duration: float,
) -> tuple[str, int]:
    segment_count = max(1, math.ceil(total_duration / segment_duration))
    last_seg_duration = total_duration - (segment_count - 1) * segment_duration
    if last_seg_duration <= 0:
        last_seg_duration = segment_duration

    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        f"#EXT-X-TARGETDURATION:{math.ceil(segment_duration)}",
        "#EXT-X-PLAYLIST-TYPE:VOD",
        "#EXT-X-MEDIA-SEQUENCE:0",
    ]
    for i in range(segment_count):
        dur = segment_duration if i < segment_count - 1 else last_seg_duration
        lines.append(f"#EXTINF:{dur:.3f},")
        lines.append(f"seg_{i:05d}.ts")
    lines.append("#EXT-X-ENDLIST")
    lines.append("")
    return "\n".join(lines), segment_count


class _StreamRequestHandler(SimpleHTTPRequestHandler):

    def __init__(self, *args, server_state: HlsStreamingServer, **kwargs):
        self._state = server_state
        super().__init__(*args, **kwargs)

    def do_GET(self):
        try:
            self._handle_get()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def _handle_get(self):
        path = self.path.split("?")[0]

        if path == "/":
            data = _HLS_PLAYER_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/stream.m3u8":
            data = self._state.playlist_text.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/vnd.apple.mpegurl")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
            return

        if path.startswith("/seg_") and path.endswith(".ts"):
            seg_name = path.lstrip("/")
            seg_path = self._state.segments_dir / seg_name
            seg_num = self._parse_segment_number(seg_name)
            if seg_num is not None:
                self._state.notify_segment_requested(seg_num)
            if seg_path.exists():
                self._serve_file(seg_path)
                return

            if seg_num is None or seg_num >= self._state.segment_count:
                self.send_error(404)
                return

            if self._state.needs_seek(seg_num):
                self._state.request_seek(seg_num)

            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                if seg_path.exists():
                    time.sleep(0.1)
                    self._serve_file(seg_path)
                    return
                if self._state.needs_seek(seg_num):
                    break
                time.sleep(0.2)

            self.send_error(404)
            return

        self.send_error(404)

    def _serve_file(self, path: Path):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "video/mp2t")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _parse_segment_number(name: str) -> int | None:
        try:
            return int(name.replace("seg_", "").replace(".ts", ""))
        except ValueError:
            return None

    def log_message(self, format, *args):
        pass


class HlsStreamingServer:
    def __init__(
        self,
        metadata: VideoMetadata,
        segment_duration: float = 4.0,
        port: int = 8765,
    ):
        self.metadata = metadata
        self.segment_duration = float(segment_duration)
        self.port = int(port)

        self.segments_dir = Path(tempfile.mkdtemp(prefix="jasna_hls_"))
        self.playlist_text, self.segment_count = _generate_vod_playlist(
            total_duration=metadata.duration,
            segment_duration=self.segment_duration,
        )
        (self.segments_dir / "stream.m3u8").write_text(self.playlist_text)

        self._seek_lock = threading.Lock()
        self.seek_requested = threading.Event()
        self.seek_target_segment: int = -1
        self._last_seek_time: float = 0.0

        self._demand_lock = threading.Condition()
        self._highest_requested_segment: int = -1
        self._current_pass_start: int = 0
        self._produced_segment: int = -1

        self._httpd: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}/stream.m3u8"

    def segment_start_frame(self, segment_index: int) -> int:
        fps = self.metadata.video_fps
        return int(segment_index * self.segment_duration * fps)

    def frames_per_segment(self) -> int:
        return max(1, int(self.metadata.video_fps * self.segment_duration))

    def request_seek(self, segment_index: int) -> None:
        with self._seek_lock:
            now = time.monotonic()
            self.seek_target_segment = segment_index
            self._last_seek_time = now
            self.seek_requested.set()

    def consume_seek(self) -> int | None:
        with self._seek_lock:
            if not self.seek_requested.is_set():
                return None
            target = self.seek_target_segment
            self.seek_requested.clear()
            return target

    def notify_segment_requested(self, segment_index: int) -> None:
        with self._demand_lock:
            if segment_index > self._highest_requested_segment:
                self._highest_requested_segment = segment_index
                log.debug("[stream-server] player requested segment %d", segment_index)
            self._demand_lock.notify_all()

    def wait_for_demand(
        self,
        current_segment: int,
        max_ahead: int,
        cancel_event: threading.Event,
    ) -> None:
        with self._demand_lock:
            if current_segment > self._highest_requested_segment + max_ahead:
                log.debug(
                    "[stream-server] pausing pipeline at segment %d (player at %d, max_ahead=%d)",
                    current_segment, self._highest_requested_segment, max_ahead,
                )
            while (
                current_segment > self._highest_requested_segment + max_ahead
                and not cancel_event.is_set()
            ):
                self._demand_lock.wait(timeout=0.5)

    def update_production(self, segment: int) -> None:
        with self._demand_lock:
            if segment > self._produced_segment:
                self._produced_segment = segment

    def needs_seek(self, segment: int) -> bool:
        with self._demand_lock:
            if segment < self._current_pass_start:
                return True
            if segment > self._produced_segment + _FORWARD_SEEK_THRESHOLD:
                return True
            return False

    def reset_demand(self, start_segment: int = 0) -> None:
        with self._demand_lock:
            self._highest_requested_segment = start_segment - 1
            self._current_pass_start = start_segment
            self._produced_segment = start_segment - 1
            self._demand_lock.notify_all()

    def start(self) -> str:
        class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True

        handler = partial(_StreamRequestHandler, server_state=self)
        self._httpd = _ThreadingHTTPServer(("0.0.0.0", self.port), handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="HlsHttpServer",
            daemon=True,
        )
        self._thread.start()
        log.info("HLS streaming at %s", self.url)
        log.info("Open in browser: http://localhost:%d/", self.port)
        return self.url

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._cleanup_segments()

    def _cleanup_segments(self) -> None:
        if self.segments_dir and self.segments_dir.exists():
            for f in self.segments_dir.iterdir():
                try:
                    f.unlink()
                except OSError:
                    pass
            try:
                self.segments_dir.rmdir()
            except OSError:
                pass
