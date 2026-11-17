import gc
import inspect
import logging
from unittest.mock import MagicMock

import av
import pytest
import torch

from factories import write_dated_source
from jasna.accelerator import AcceleratorVendor
from jasna.media import video_decoder
from jasna.media.probe import get_video_meta_data
from jasna.media.video_decoder import (
    CORRUPT_PACKET_TOLERANCE,
    VideoReader,
    VideoDecodeError,
)


def _source(tmp_path) -> video_decoder._PyAvFrameSource:
    path = str(write_dated_source(tmp_path / "broken.mp4"))
    return video_decoder._PyAvFrameSource(
        path, 1, torch.device("cuda:0"), get_video_meta_data(path), 1, AcceleratorVendor.NVIDIA, True
    )


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


def test_corrupt_packet_is_skipped_and_logged(caplog, tmp_path):
    source = _source(tmp_path)

    with caplog.at_level(logging.WARNING):
        frames, errors = source._decode_packet(_corrupt_packet(), 0)

    assert frames == []
    assert errors == 1
    assert "Recovered video corruption in" in caplog.text
    assert "broken.mp4" in caplog.text
    assert "Invalid data found when processing input" in caplog.text


def test_error_counter_resets_after_good_packet(tmp_path):
    source = _source(tmp_path)

    _, errors = source._decode_packet(_corrupt_packet(), 0)
    assert errors == 1
    frames, errors = source._decode_packet(_good_packet(), errors)
    assert len(frames) == 1
    assert errors == 0


def test_too_many_consecutive_corrupt_packets_raise(tmp_path):
    source = _source(tmp_path)

    errors = 0
    with pytest.raises(
        VideoDecodeError,
        match=r"broken\.mp4.*too many consecutive corrupt packets",
    ):
        for _ in range(CORRUPT_PACKET_TOLERANCE + 1):
            _, errors = source._decode_packet(_corrupt_packet(), errors)


def test_other_ffmpeg_errors_raise_video_decode_error(tmp_path):
    source = _source(tmp_path)
    packet = MagicMock()
    packet.decode.side_effect = av.error.MemoryError(12, "Cannot allocate memory")

    with pytest.raises(VideoDecodeError, match=r"broken\.mp4"):
        source._decode_packet(packet, 0)


def test_backend_dispatch_releases_outer_first_group_reference(monkeypatch, tmp_path):
    source = write_dated_source(tmp_path / "source.mp4")
    monkeypatch.setattr(video_decoder, "vendor_for_device", lambda _device: AcceleratorVendor.NVIDIA)
    monkeypatch.setattr(video_decoder, "current_stream", lambda _device: None)
    reader = VideoReader(str(source), 4, torch.device("cuda:0"), get_video_meta_data(str(source)), decode_backend="pyav-sw")

    def software_backend(_source, decoded, group):
        dispatchers = [
            holder for holder in gc.get_referrers(group)
            if inspect.isgenerator(holder) and holder.gi_code is video_decoder._PyAvFrameSource.frames.__code__
        ]
        yield not dispatchers, []

    monkeypatch.setattr(video_decoder._PyAvFrameSource, "_frames_software", software_backend)
    with reader:
        released, _ = next(reader.frames())

    assert released, "outer dispatch retained the first decoded group"
