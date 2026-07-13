import logging
from unittest.mock import MagicMock

import av
import pytest

from jasna.media.video_decoder import (
    CORRUPT_PACKET_TOLERANCE,
    NvidiaVideoReader,
    VideoDecodeError,
)


def _reader():
    reader = NvidiaVideoReader.__new__(NvidiaVideoReader)
    reader.file = "broken.mp4"
    return reader


def _corrupt_packet():
    packet = MagicMock()
    packet.decode.side_effect = av.error.InvalidDataError(
        1094995529, "Invalid data found when processing input"
    )
    return packet


def _good_packet():
    packet = MagicMock()
    packet.decode.return_value = [MagicMock()]
    return packet


def test_corrupt_packet_is_skipped_and_logged(caplog):
    reader = _reader()

    with caplog.at_level(logging.WARNING):
        frames, errors = reader._decode_packet(_corrupt_packet(), 0)

    assert frames == []
    assert errors == 1
    assert "Recovered video corruption in broken.mp4" in caplog.text
    assert "Invalid data found when processing input" in caplog.text


def test_error_counter_resets_after_good_packet():
    reader = _reader()

    _, errors = reader._decode_packet(_corrupt_packet(), 0)
    assert errors == 1
    frames, errors = reader._decode_packet(_good_packet(), errors)
    assert len(frames) == 1
    assert errors == 0


def test_too_many_consecutive_corrupt_packets_raise():
    reader = _reader()

    errors = 0
    with pytest.raises(
        VideoDecodeError,
        match=r"broken\.mp4.*too many consecutive corrupt packets",
    ):
        for _ in range(CORRUPT_PACKET_TOLERANCE + 1):
            _, errors = reader._decode_packet(_corrupt_packet(), errors)


def test_other_ffmpeg_errors_raise_video_decode_error():
    reader = _reader()
    packet = MagicMock()
    packet.decode.side_effect = av.error.MemoryError(12, "Cannot allocate memory")

    with pytest.raises(VideoDecodeError, match=r"broken\.mp4"):
        reader._decode_packet(packet, 0)
