import subprocess
from pathlib import Path

import pytest

from jasna.media.audio_utils import source_audio_codec
from jasna.os_utils import resolve_executable


def _write_source(path: Path, audio_args: list[str]) -> Path:
    subprocess.run(
        [
            resolve_executable("ffmpeg"), "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.2",
            *audio_args,
            "-c:v", "mpeg4", "-shortest", str(path),
        ],
        check=True,
    )
    return path


@pytest.mark.parametrize(("encoder", "expected"), [("libmp3lame", "mp3"), ("pcm_s16le", "pcm_s16le")])
def test_source_audio_codec_names_the_codec_not_the_decoder(tmp_path, encoder, expected):
    source = _write_source(
        tmp_path / "source.mkv",
        ["-f", "lavfi", "-i", "sine=d=0.2", "-c:a", encoder],
    )
    assert source_audio_codec(str(source)) == expected


def test_source_audio_codec_is_none_without_audio(tmp_path):
    source = _write_source(tmp_path / "source.mkv", [])
    assert source_audio_codec(str(source)) is None
