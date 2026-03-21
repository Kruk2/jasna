import json
import logging
import subprocess
from pathlib import Path

from jasna.os_utils import get_subprocess_startup_info, resolve_executable

logger = logging.getLogger(__name__)

_INCOMPATIBLE_AUDIO: dict[str, frozenset[str]] = {
    '.mp4': frozenset({'wmav1', 'wmav2', 'wmapro', 'vorbis'}),
    '.mov': frozenset({'wmav1', 'wmav2', 'wmapro', 'vorbis', 'opus'}),
    '.avi': frozenset({'opus', 'vorbis', 'flac'}),
    '.webm': frozenset({'aac', 'mp3', 'wmav1', 'wmav2', 'wmapro', 'dts', 'ac3', 'eac3', 'pcm_s16le', 'pcm_s24le', 'pcm_s32le', 'pcm_f32le'}),
}


def probe_audio_codec(video_file: str) -> str | None:
    cmd = [
        resolve_executable("ffprobe"),
        "-v", "quiet",
        "-print_format", "json",
        "-select_streams", "a:0",
        "-show_streams",
        video_file,
    ]
    result = subprocess.run(cmd, capture_output=True, startupinfo=get_subprocess_startup_info())
    if result.returncode != 0:
        return None
    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        return None
    return streams[0].get("codec_name")


def needs_audio_reencode(audio_codec: str | None, output_suffix: str) -> bool:
    if audio_codec is None:
        return False
    blocked = _INCOMPATIBLE_AUDIO.get(output_suffix.lower(), frozenset())
    return audio_codec.lower() in blocked


def audio_codec_args(video_file: str, output_path: Path) -> list[str]:
    audio_codec = probe_audio_codec(video_file)
    if needs_audio_reencode(audio_codec, output_path.suffix):
        logger.info("[remux] re-encoding audio %s -> aac for %s", audio_codec, output_path.suffix)
        return ["aac", "-b:a", "256k"]
    return ["copy"]
