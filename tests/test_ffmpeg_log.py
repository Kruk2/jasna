import logging

import av
import av.logging
import pytest

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
