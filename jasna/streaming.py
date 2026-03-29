from __future__ import annotations

import atexit
import json
import logging
import math
import os
import shutil
import tempfile
import threading
import time
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn
from urllib.parse import parse_qs, urlparse

from jasna.media import VideoMetadata

log = logging.getLogger(__name__)

_FORWARD_SEEK_THRESHOLD = 5

_PAGE_HTML = """\
<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Jasna Stream</title>
<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:system-ui,-apple-system,sans-serif;background:#0a0a0a;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
  #picker{display:flex;flex-direction:column;align-items:center;justify-content:center;flex:1;gap:20px}
  #picker h1{font-size:28px;font-weight:300;letter-spacing:2px;color:#fff}
  .irow{display:flex;gap:8px;width:min(600px,90vw)}
  .irow input{flex:1;padding:10px 14px;border-radius:8px;border:1px solid #333;background:#181818;color:#e0e0e0;font-size:14px;outline:none}
  .irow input:focus{border-color:#4a8eff}
  .btn{padding:10px 20px;border-radius:8px;border:none;cursor:pointer;font-size:14px;font-weight:500;transition:opacity .15s}
  .btn:hover{opacity:.85}.btn:disabled{opacity:.4;cursor:default}
  .ba{background:#2563eb;color:#fff}.bd{background:#2a2a2a;color:#ccc}
  #status{font-size:13px;color:#888;min-height:20px}
  #player{display:none;flex-direction:column;flex:1;overflow:hidden}
  #player video{flex:1;width:100%;background:#000;min-height:0;object-fit:contain}
  .tb{display:flex;align-items:center;gap:12px;padding:10px 16px;background:#141414;border-bottom:1px solid #222}
  .tb .name{flex:1;font-size:13px;color:#999;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
</style></head><body>
<div id="picker">
  <h1>JASNA</h1>
  <div class="irow">
    <input id="pi" type="text" placeholder="Video file path..." autocomplete="off"/>
    <button class="btn ba" onclick="browse()">Browse</button>
  </div>
  <div id="status"></div>
</div>
<div id="player">
  <div class="tb">
    <span class="name" id="fname"></span>
    <button class="btn bd" onclick="changeVideo()">Change Video</button>
  </div>
  <video id="v" controls autoplay></video>
</div>
<script>
var hls=null,loaded=false,pi=document.getElementById('pi');
pi.addEventListener('keydown',function(e){if(e.key==='Enter'&&pi.value.trim())loadVideo(pi.value.trim())});
async function browse(){
  try{var r=await fetch('/api/browse',{method:'POST'});var d=await r.json();
  if(d.path){pi.value=d.path;loadVideo(d.path)}}
  catch(e){document.getElementById('status').textContent='Browse failed: '+e.message}}
async function loadVideo(p){
  if(!p)return;
  document.getElementById('status').textContent='Preparing stream...';
  try{var r=await fetch('/open',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({path:p})});
  var d=await r.json();
  if(d.error){document.getElementById('status').textContent=d.error;return}
  document.getElementById('fname').textContent=p.split(/[\\/]/).pop();
  showPlayer()}
  catch(e){document.getElementById('status').textContent='Error: '+e.message}}
function showPlayer(){
  loaded=true;
  document.getElementById('picker').style.display='none';
  document.getElementById('player').style.display='flex';
  document.getElementById('status').textContent='';
  startHls()}
function startHls(){
  var v=document.getElementById('v');
  if(hls){hls.destroy();hls=null}
  if(Hls.isSupported()){hls=new Hls({maxBufferLength:10,maxMaxBufferLength:30});hls.loadSource('/stream.m3u8');hls.attachMedia(v)}
  else if(v.canPlayType('application/vnd.apple.mpegurl')){v.src='/stream.m3u8'}}
async function changeVideo(){
  await fetch('/stop',{method:'POST'});
  if(hls){hls.destroy();hls=null}
  document.getElementById('v').removeAttribute('src');
  document.getElementById('player').style.display='none';
  document.getElementById('picker').style.display='flex';
  loaded=false}
</script></body></html>
"""


def _generate_vod_playlist(
    total_duration: float,
    segment_duration: float,
    start_segment: int = 0,
) -> tuple[str, int]:
    segment_count = max(1, math.ceil(total_duration / segment_duration))
    last_seg_duration = total_duration - (segment_count - 1) * segment_duration
    if last_seg_duration <= 0:
        last_seg_duration = segment_duration

    first = max(0, min(start_segment, segment_count - 1))
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        f"#EXT-X-TARGETDURATION:{math.ceil(segment_duration)}",
        "#EXT-X-PLAYLIST-TYPE:VOD",
        f"#EXT-X-MEDIA-SEQUENCE:{first}",
    ]
    for i in range(first, segment_count):
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

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        try:
            self._handle_get()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def do_POST(self):
        try:
            self._handle_post()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass

    def _handle_get(self):
        path = self.path.split("?")[0]

        if path == "/":
            data = _PAGE_HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
            return

        if path == "/status":
            streaming = self._state.is_loaded
            resp: dict = {"streaming": streaming}
            if streaming and self._state._current_video_path is not None:
                resp["path"] = str(self._state._current_video_path)
            self._send_json(resp)
            return

        if path == "/stream.m3u8":
            if not self._state.is_loaded:
                self.send_error(404)
                return
            start_seconds = self._parse_query_float("start")
            if start_seconds > 0:
                start_seg = int(start_seconds / self._state.segment_duration)
                playlist = self._state.playlist_text_from(start_seg)
            else:
                playlist = self._state.playlist_text
            data = playlist.encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/vnd.apple.mpegurl")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
            return

        if path.startswith("/seg_") and path.endswith(".ts"):
            if not self._state.is_loaded:
                self.send_error(404)
                return
            seg_name = path.lstrip("/")
            seg_path = self._state.segments_dir / seg_name
            seg_num = self._parse_segment_number(seg_name)
            if seg_num is not None:
                self._state.notify_segment_requested(seg_num)
            if seg_path.exists() and seg_path.stat().st_size > 0:
                self._serve_file(seg_path)
                return

            if seg_num is None or seg_num >= self._state.segment_count:
                self.send_error(404)
                return

            if self._state.needs_seek(seg_num):
                self._state.request_seek(seg_num)

            deadline = time.monotonic() + 30.0
            while time.monotonic() < deadline:
                if not self._state.is_loaded:
                    self.send_error(404)
                    return
                try:
                    sz = seg_path.stat().st_size
                except OSError:
                    sz = -1
                if sz > 0:
                    log.debug("[stream-server] serving %s (%d bytes, waited %.1fs)",
                              seg_name, sz, 30.0 - (deadline - time.monotonic()))
                    self._serve_file(seg_path)
                    return
                time.sleep(0.2)

            log.warning("[stream-server] timeout waiting for %s (exists=%s)",
                        seg_name, seg_path.exists())
            self.send_error(404)
            return

        self.send_error(404)

    def _handle_post(self):
        path = self.path.split("?")[0]

        if path == "/api/browse":
            selected = self._open_file_dialog()
            self._send_json({"path": selected})
            return

        if path in ("/api/stop", "/stop"):
            self._state.stop_current()
            self._send_json({"ok": True})
            return

        if path in ("/api/load", "/open"):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length)) if length > 0 else {}
            video_path = str(body.get("path", "")).strip()
            if not video_path or not Path(video_path).is_file():
                self._send_json({"error": "File not found"}, status=400)
                return
            start = float(body.get("start", 0))
            same_video = (
                self._state.is_loaded
                and self._state._current_video_path == Path(video_path)
            )
            if path == "/open":
                if same_video:
                    if start > 0:
                        target_seg = int(start / self._state.segment_duration)
                        self._state.request_seek(target_seg)
                    self._send_json({"status": "ready"})
                else:
                    self._state.select_video(Path(video_path))
                    ok = self._state.wait_until_first_segment(timeout=30.0)
                    if ok:
                        self._send_json({"status": "ready"})
                    else:
                        self._send_json({"error": "Timed out waiting for stream to be ready"}, status=500)
            else:
                self._state.select_video(Path(video_path))
                self._send_json({"ok": True, "filename": Path(video_path).name})
            return

        self.send_error(404)

    def _send_json(self, obj: dict, status: int = 200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _open_file_dialog() -> str | None:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            path = filedialog.askopenfilename(
                title="Select Video",
                filetypes=[
                    ("Video files", "*.mp4 *.mkv *.avi *.mov *.wmv *.webm *.flv *.ts"),
                    ("All files", "*.*"),
                ],
            )
            root.destroy()
            return path if path else None
        except Exception:
            return None

    def _serve_file(self, path: Path):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "video/mp2t")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _parse_query_float(self, key: str) -> float:
        qs = parse_qs(urlparse(self.path).query)
        try:
            return float(qs[key][0])
        except (KeyError, IndexError, ValueError):
            return 0.0

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
        segment_duration: float = 4.0,
        port: int = 8765,
        max_segments_ahead: int = 3,
    ):
        self.segment_duration = float(segment_duration)
        self.port = int(port)

        self.metadata: VideoMetadata | None = None
        self.segments_dir: Path | None = None
        self.segment_count: int = 0
        self._vod_playlist: str = ""
        self._finished = False

        self._pending_video: Path | None = None
        self._current_video_path: Path | None = None
        self._video_selected = threading.Event()
        self.video_change = threading.Event()
        self._first_segment_ready = threading.Event()

        self._seek_lock = threading.Lock()
        self.seek_requested = threading.Event()
        self.seek_target_segment: int = -1
        self._last_seek_time: float = 0.0

        self._max_segments_kept: int = 2 * max_segments_ahead

        self._demand_lock = threading.Condition()
        self._highest_requested_segment: int = -1
        self._current_pass_start: int = 0
        self._produced_segment: int = -1

        self._httpd: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def is_loaded(self) -> bool:
        return self.metadata is not None

    def stop_current(self) -> None:
        self.metadata = None
        self._vod_playlist = ""
        self.video_change.set()

    def select_video(self, path: Path) -> None:
        self._pending_video = path
        self._current_video_path = path
        self._first_segment_ready.clear()
        if self.metadata is not None:
            self.stop_current()
        self._video_selected.set()

    def wait_for_video(self) -> Path:
        while not self._video_selected.wait(timeout=0.5):
            pass
        self._video_selected.clear()
        return self._pending_video

    def load_video(self, metadata: VideoMetadata) -> None:
        self.metadata = metadata
        if self.segments_dir is None:
            self.segments_dir = Path(tempfile.mkdtemp(prefix="jasna_hls_"))
            atexit.register(shutil.rmtree, str(self.segments_dir), True)
        self._vod_playlist, self.segment_count = _generate_vod_playlist(
            total_duration=metadata.duration,
            segment_duration=self.segment_duration,
        )
        self._finished = False
        self.video_change.clear()
        self.seek_requested.clear()
        self.seek_target_segment = -1
        with self._demand_lock:
            self._highest_requested_segment = -1
            self._current_pass_start = 0
            self._produced_segment = -1

    def unload_video(self) -> None:
        self.metadata = None
        self._vod_playlist = ""
        self.segment_count = 0
        self._finished = False
        self.video_change.clear()
        if self.segments_dir:
            for f in self.segments_dir.glob("*.ts"):
                f.unlink(missing_ok=True)
            for f in self.segments_dir.glob("*.m3u8"):
                f.unlink(missing_ok=True)

    @property
    def playlist_text(self) -> str:
        return self._vod_playlist

    def playlist_text_from(self, start_segment: int) -> str:
        if start_segment <= 0 or self.metadata is None:
            return self._vod_playlist
        text, _ = _generate_vod_playlist(self.metadata.duration, self.segment_duration, start_segment)
        return text

    def mark_finished(self) -> None:
        self._finished = True

    def wait_until_first_segment(self, timeout: float = 30.0) -> bool:
        return self._first_segment_ready.wait(timeout=timeout)

    @property
    def url(self) -> str:
        return f"http://localhost:{self.port}/stream.m3u8"

    def segment_start_time(self, segment_index: int) -> float:
        return segment_index * self.segment_duration

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
        self._evict_old_segments(segment_index)

    def _evict_old_segments(self, player_segment: int) -> None:
        if self.segments_dir is None:
            return
        min_keep = player_segment - self._max_segments_kept
        if min_keep <= 0:
            return
        for f in self.segments_dir.glob("seg_*.ts"):
            try:
                num = int(f.stem.split("_", 1)[1])
            except (ValueError, IndexError):
                continue
            if num < min_keep:
                f.unlink(missing_ok=True)

    def wait_for_demand(
        self,
        current_segment: int,
        max_ahead: int,
        cancel_event: threading.Event,
    ) -> None:
        with self._demand_lock:
            if (
                self._highest_requested_segment >= 0
                and current_segment > self._highest_requested_segment + max_ahead
            ):
                log.debug(
                    "[stream-server] pausing pipeline at segment %d (player at %d, max_ahead=%d)",
                    current_segment, self._highest_requested_segment, max_ahead,
                )
            while (
                self._highest_requested_segment >= 0
                and current_segment > self._highest_requested_segment + max_ahead
                and not cancel_event.is_set()
            ):
                self._demand_lock.wait(timeout=0.5)

    def update_production(self, segment: int) -> None:
        with self._demand_lock:
            if segment > self._produced_segment:
                self._produced_segment = segment
        if not self._first_segment_ready.is_set():
            self._first_segment_ready.set()

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
