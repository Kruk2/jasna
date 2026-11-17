import logging
import subprocess
import sys
from pathlib import Path

import av
import av.logging
import pytest
import torch

from jasna.media.ffmpeg_log import log_ffmpeg_errors


@pytest.fixture
def restore_ffmpeg_log_level():
    level = av.logging.get_level()
    yield
    av.logging.set_level(level)


def test_ffmpeg_error_reaches_python_log(restore_ffmpeg_log_level, caplog):
    log_ffmpeg_errors()
    encoder = av.CodecContext.create("mpeg4", "w")

    with caplog.at_level(logging.ERROR), pytest.raises(av.error.ArgumentError):
        encoder.open()

    assert [(r.name, r.levelno, r.getMessage()) for r in caplog.records] == [
        ("libav.mpeg4", logging.ERROR, "Invalid video pixel format: -1")
    ]


_TEST_CLIP = Path("assets/test_clip1_1080p.mp4")

_CLOSE_A_READER = f"""
from contextlib import closing
import torch
from jasna.media.ffmpeg_log import log_ffmpeg_errors
from jasna.media.probe import get_video_meta_data
from jasna.media.video_decoder import VideoReader

log_ffmpeg_errors()
metadata = get_video_meta_data({str(_TEST_CLIP)!r})
for _ in range(3):
    with VideoReader(
        {str(_TEST_CLIP)!r}, batch_size=4, device=torch.device("cuda", 0), metadata=metadata, decode_backend="pyav-sw"
    ) as reader, closing(reader.frames()) as frames:
        next(frames)
"""

_CLOSE_A_PREVIEW = f"""
from jasna.gui.segment_preview import PreviewFrame, SegmentPreviewWorker
from jasna.media.ffmpeg_log import log_ffmpeg_errors

log_ffmpeg_errors()
for _ in range(3):
    worker = SegmentPreviewWorker({str(_TEST_CLIP)!r}, max_size=(80, 80))
    worker.start()
    worker.seek(0.5)
    while not isinstance(worker.events.get(timeout=30), PreviewFrame):
        pass
    worker.close()
    worker.join(10)
    del worker
"""


@pytest.mark.skipif(not _TEST_CLIP.exists(), reason="needs the test clip")
@pytest.mark.parametrize(
    "script",
    [
        pytest.param(
            _CLOSE_A_READER,
            id="video reader",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
        ),
        pytest.param(_CLOSE_A_PREVIEW, id="segment preview"),
    ],
)
def test_closing_a_decoder_mid_video_returns_with_ffmpeg_logging_on(script):
    # A decoder thread that logs while its decoder is freed stops the whole process, so the
    # close runs in a process of its own and has to end.
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)

    assert result.returncode == 0, result.stderr
