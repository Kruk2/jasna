import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import python_vali as vali

from jasna.media.video_decoder import NvidiaVideoReader, VideoDecodeError


def _reader(details, success):
    reader = NvidiaVideoReader.__new__(NvidiaVideoReader)
    reader.file = "broken.mp4"
    reader.decoder = MagicMock()
    reader.decoder.DecodeSingleSurfaceAsyncDetailed.return_value = (success, details)
    reader.decode_surface = MagicMock()
    return reader


def test_decode_surface_logs_recovered_corruption(caplog):
    details = SimpleNamespace(
        info=vali.TaskExecInfo.SUCCESS,
        message="recovered after 1 corrupt packet: Invalid data found when processing input",
    )
    reader = _reader(details, success=True)

    with caplog.at_level(logging.WARNING):
        assert reader._decode_surface(vali.PacketData(), None)

    assert "Recovered video corruption in broken.mp4" in caplog.text
    assert "Invalid data found when processing input" in caplog.text


def test_decode_surface_returns_false_at_end_of_stream():
    details = SimpleNamespace(info=vali.TaskExecInfo.END_OF_STREAM, message="end of stream")
    reader = _reader(details, success=False)

    assert not reader._decode_surface(vali.PacketData(), None)


def test_decode_surface_raises_clean_error_with_native_details():
    details = SimpleNamespace(
        info=SimpleNamespace(name="CORRUPT_DATA"),
        message="too many consecutive corrupt packets (10)",
    )
    reader = _reader(details, success=False)

    with pytest.raises(
        VideoDecodeError,
        match=r"broken\.mp4.*CORRUPT_DATA.*too many consecutive corrupt packets",
    ):
        reader._decode_surface(vali.PacketData(), None)
