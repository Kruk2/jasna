"""Unit tests for NvidiaVideoEncoder internals (encode, _process_buffer, _encode_frame, __exit__)."""
from __future__ import annotations

import heapq
import io
import queue
import threading
from collections import deque
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.media import VideoMetadata
from jasna.media.video_encoder import NvidiaVideoEncoder


def _fake_metadata(**overrides) -> VideoMetadata:
    defaults = dict(
        video_file="fake_input.mkv",
        num_frames=100,
        video_fps=24.0,
        average_fps=24.0,
        video_fps_exact=Fraction(24, 1),
        codec_name="hevc",
        duration=100.0 / 24.0,
        video_width=1920,
        video_height=1080,
        time_base=Fraction(1, 24),
        start_pts=0,
        color_space=AvColorspace.ITU709,
        color_range=AvColorRange.MPEG,
        is_10bit=True,
    )
    defaults.update(overrides)
    return VideoMetadata(**defaults)


class _FakeThread:
    def __init__(self, *args, **kwargs):
        pass
    def start(self):
        pass
    def join(self, timeout=None):
        pass


class _FakeCudaStream:
    def __init__(self, *args, **kwargs):
        self.cuda_stream = 0
        self.wait_event = MagicMock()
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def _make_encoder(tmp_path, encoder_settings=None):
    output_path = tmp_path / "output" / "result.mkv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    working_dir = tmp_path / "work"
    working_dir.mkdir(exist_ok=True)

    mock_nvc_encoder = MagicMock()
    mock_nvc_encoder.EndEncode.return_value = []

    with (
        patch("jasna.media.video_encoder.nvc") as mock_nvc,
        patch("jasna.media.video_encoder.threading.Thread", _FakeThread),
        patch("jasna.media.video_encoder.torch.cuda.Stream", _FakeCudaStream),
    ):
        mock_nvc.CreateEncoder.return_value = mock_nvc_encoder
        import torch
        enc = NvidiaVideoEncoder(
            file=str(output_path),
            device=torch.device("cuda:0"),
            metadata=_fake_metadata(),
            codec="hevc",
            encoder_settings=encoder_settings or {},
            stream_mode=False,
            working_directory=working_dir,
        )
    return enc, mock_nvc_encoder


class TestEncoderSettings:
    def test_empty_settings_no_update(self, tmp_path):
        enc, nvc_enc = _make_encoder(tmp_path, encoder_settings={})
        enc.raw_hevc.close()

    def test_nonempty_settings_passed_through(self, tmp_path):
        with (
            patch("jasna.media.video_encoder.nvc") as mock_nvc,
            patch("jasna.media.video_encoder.threading.Thread", _FakeThread),
            patch("jasna.media.video_encoder.torch.cuda.Stream", _FakeCudaStream),
        ):
            mock_nvc.CreateEncoder.return_value = MagicMock(EndEncode=MagicMock(return_value=[]))
            output_path = tmp_path / "output" / "result.mkv"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            working_dir = tmp_path / "work"
            working_dir.mkdir(exist_ok=True)

            import torch
            enc = NvidiaVideoEncoder(
                file=str(output_path),
                device=torch.device("cuda:0"),
                metadata=_fake_metadata(),
                codec="hevc",
                encoder_settings={"cq": 18, "custom_key": "custom_val"},
                stream_mode=False,
                working_directory=working_dir,
            )

            create_kwargs = mock_nvc.CreateEncoder.call_args[1]
            assert create_kwargs["cq"] == 18
            assert create_kwargs["custom_key"] == "custom_val"
            enc.raw_hevc.close()


class TestColorValidation:
    def test_unsupported_color_raises(self, tmp_path):
        with (
            patch("jasna.media.video_encoder.nvc") as mock_nvc,
            patch("jasna.media.video_encoder.threading.Thread", _FakeThread),
            patch("jasna.media.video_encoder.torch.cuda.Stream", _FakeCudaStream),
        ):
            mock_nvc.CreateEncoder.return_value = MagicMock(EndEncode=MagicMock(return_value=[]))

            output_path = tmp_path / "output" / "result.mkv"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            import torch
            with pytest.raises(ValueError, match="Unsupported color"):
                NvidiaVideoEncoder(
                    file=str(output_path),
                    device=torch.device("cuda:0"),
                    metadata=_fake_metadata(
                        color_space=AvColorspace.ITU601,
                        color_range=AvColorRange.JPEG,
                    ),
                    codec="hevc",
                    encoder_settings={},
                    stream_mode=False,
                )


class TestEncode:
    def test_encode_pushes_to_buffer_and_heap(self, tmp_path):
        enc, _ = _make_encoder(tmp_path)
        import torch
        frame = torch.zeros(3, 8, 8)
        enc.encode(frame, pts=100)

        assert len(enc.frame_buffer) == 1
        assert 100 in enc.pts_set
        assert len(enc.pts_heap) == 1
        enc.raw_hevc.close()

    def test_encode_dedup_pts(self, tmp_path):
        enc, _ = _make_encoder(tmp_path)
        import torch
        frame = torch.zeros(3, 8, 8)
        enc.encode(frame, pts=50)
        enc.encode(frame, pts=50)

        assert len(enc.pts_set) == 2
        assert 50 in enc.pts_set
        assert 51 in enc.pts_set
        enc.raw_hevc.close()

    def test_encode_triggers_process_buffer_at_threshold(self, tmp_path):
        enc, _ = _make_encoder(tmp_path)
        enc._encode_queue = MagicMock()
        enc._encode_queue.put = MagicMock()
        enc._build_encode_item = MagicMock(return_value=("frame", 0, "event"))

        import torch
        frame = torch.zeros(3, 8, 8)
        for i in range(enc.BUFFER_MAX_SIZE // 2 + 1):
            enc.encode(frame, pts=i)

        enc._encode_queue.put.assert_called_once()
        enc.raw_hevc.close()


class TestProcessBuffer:
    def test_does_not_flush_below_threshold(self, tmp_path):
        enc, _ = _make_encoder(tmp_path)
        import torch
        enc.frame_buffer.append(torch.zeros(3, 8, 8))
        enc.pts_heap = [10]
        enc.pts_set = {10}
        enc._encode_queue = MagicMock()
        enc._build_encode_item = MagicMock(return_value=("f", 10, "e"))

        enc._process_buffer(flush_all=False)
        enc._encode_queue.put.assert_not_called()
        enc.raw_hevc.close()

    def test_flush_all_sends_to_queue(self, tmp_path):
        enc, _ = _make_encoder(tmp_path)
        import torch
        enc.frame_buffer.append(torch.zeros(3, 8, 8))
        heapq.heappush(enc.pts_heap, 10)
        enc.pts_set.add(10)
        enc._encode_queue = MagicMock()
        enc._build_encode_item = MagicMock(return_value=("f", 10, "e"))

        enc._process_buffer(flush_all=True)
        enc._encode_queue.put.assert_called_once()
        assert len(enc.frame_buffer) == 0
        enc.raw_hevc.close()


class TestEncodeFrame:
    def test_writes_bitstream_to_raw_hevc(self, tmp_path):
        enc, nvc_enc = _make_encoder(tmp_path)
        nvc_enc.Encode.return_value = b'\x00\x00\x01\x40'

        import torch
        buf = io.BytesIO()
        enc.raw_hevc = buf
        with patch("jasna.media.video_encoder.torch.cuda.stream", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))):
            enc._encode_frame(torch.zeros(3, 8, 8), pts=42)

        assert 42 in list(enc.reordered_pts_queue)
        assert buf.getvalue() == bytearray(b'\x00\x00\x01\x40')

    def test_empty_bitstream_no_write(self, tmp_path):
        enc, nvc_enc = _make_encoder(tmp_path)
        nvc_enc.Encode.return_value = b''

        import torch
        buf = io.BytesIO()
        enc.raw_hevc = buf
        with patch("jasna.media.video_encoder.torch.cuda.stream", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))):
            enc._encode_frame(torch.zeros(3, 8, 8), pts=10)

        assert 10 in list(enc.reordered_pts_queue)
        assert buf.getvalue() == b''


class TestExit:
    def test_exit_flushes_buffer_and_endencodes(self, tmp_path):
        enc, nvc_enc = _make_encoder(tmp_path)
        import torch

        enc._encode_queue = MagicMock()
        enc._encode_queue.join = MagicMock()
        enc._encode_queue.put = MagicMock()
        enc._encode_thread = MagicMock()

        enc.frame_buffer.append(torch.zeros(3, 8, 8))
        heapq.heappush(enc.pts_heap, 5)
        enc.pts_set.add(5)
        enc._build_encode_item = MagicMock(return_value=("f", 5, "e"))

        nvc_enc.EndEncode.side_effect = [b'\x00\x01\x02', b'']

        written_data = []
        raw_hevc_mock = MagicMock()
        raw_hevc_mock.write = lambda data: written_data.append(bytes(data))
        raw_hevc_mock.close = MagicMock()
        enc.raw_hevc = raw_hevc_mock

        with (
            patch("jasna.media.video_encoder.torch.cuda.stream", return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock(return_value=False))),
            patch("jasna.media.video_encoder.mux_hevc_to_mkv") as mock_mux,
            patch("jasna.media.video_encoder.remux_with_audio_and_metadata") as mock_remux,
            patch.object(Path, "unlink", MagicMock()),
        ):
            mock_mux.side_effect = lambda *a, **kw: enc.output_path.touch()
            enc.__exit__(None, None, None)

        assert enc._encode_queue.put.call_count >= 1
        assert b'\x00\x01\x02' in written_data


class TestEncodeWorkerCrash:
    def test_worker_logs_and_reraises(self, tmp_path):
        enc, nvc_enc = _make_encoder(tmp_path)
        import torch

        enc._encode_queue = queue.Queue()
        enc._handle_encode_item = MagicMock(side_effect=RuntimeError("encode boom"))

        enc._encode_queue.put((torch.zeros(3, 8, 8), 0, MagicMock()))

        with patch("jasna.media.video_encoder.torch.cuda.set_device"):
            with pytest.raises(RuntimeError, match="encode boom"):
                enc._encode_worker()

        enc.raw_hevc.close()
