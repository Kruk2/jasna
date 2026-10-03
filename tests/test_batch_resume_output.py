import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from jasna.gui.models import AppSettings, JobItem, JobStatus
from jasna.gui.processor import Processor
from jasna.gui.resume_validation import (
    ResumeOutputValidationError,
    validate_resume_video_output,
)
from jasna.media.media_files import folder_output_path
from jasna.os_utils import resolve_executable


def _make_video(path: Path, *, duration: float = 1.25, codec: str = "libx264") -> None:
    subprocess.run(
        [
            resolve_executable("ffmpeg"),
            "-hide_banner",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=96x64:rate=12:duration={duration}",
            "-an",
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
    )


def test_folder_output_path_preserves_relative_subfolders(tmp_path: Path) -> None:
    root = tmp_path / "input"
    source = root / "season" / "clip.mp4"

    output = folder_output_path(
        tmp_path / "output",
        source,
        "{original}_restored.mp4",
        input_root=root,
        preserve_structure=True,
    )

    assert output == tmp_path / "output" / "season" / "clip_restored.mp4"


def test_folder_output_path_never_preserves_parent_traversal(tmp_path: Path) -> None:
    root = tmp_path / "input"
    source = root / ".." / "outside" / "clip.mp4"

    output = folder_output_path(
        tmp_path / "output",
        source,
        "{original}_restored.mp4",
        input_root=root,
        preserve_structure=True,
    )

    assert output == tmp_path / "output" / "clip_restored.mp4"










def test_resume_video_validation_rejects_invalid_media_contracts(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    _make_video(source, duration=3.0)
    valid = tmp_path / "valid.mp4"
    shutil.copy2(source, valid)
    validate_resume_video_output(source, valid, configured_codec="h264")

    empty = tmp_path / "empty.mp4"
    empty.touch()
    invalid = tmp_path / "invalid.mp4"
    invalid.write_bytes(b"not a media file")
    truncated = tmp_path / "truncated.mp4"
    truncated.write_bytes(valid.read_bytes()[: valid.stat().st_size // 2])
    wrong_duration = tmp_path / "wrong-duration.mp4"
    _make_video(wrong_duration, duration=0.25)
    wrong_codec = tmp_path / "wrong-codec.mp4"
    _make_video(wrong_codec, duration=3.0, codec="mpeg4")

    for candidate in (
        empty,
        invalid,
        truncated,
        wrong_duration,
        wrong_codec,
    ):
        with pytest.raises(ResumeOutputValidationError):
            validate_resume_video_output(
                source,
                candidate,
                configured_codec="h264",
            )

    validate_resume_video_output(
        source,
        wrong_codec,
        configured_codec="mpeg4",
    )
