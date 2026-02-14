from pathlib import Path
from fractions import Fraction
from unittest.mock import MagicMock, patch

import pytest
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.media import VideoMetadata
from jasna.media.video_encoder import NvidiaVideoEncoder


def _fake_metadata() -> VideoMetadata:
    return VideoMetadata(
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


def test_encoder_uses_working_directory_for_temp_paths(tmp_path: Path) -> None:
    output_path = tmp_path / "output" / "result.mkv"
    output_path.parent.mkdir(parents=True)
    working_dir = tmp_path / "work"
    working_dir.mkdir()

    mock_encoder = MagicMock()
    mock_encoder.EndEncode.return_value = []

    mock_mux = MagicMock(side_effect=lambda hevc_path, output_path, *a, **kw: output_path.touch())

    with (
        patch("jasna.media.video_encoder.nvc") as mock_nvc,
        patch("jasna.media.video_encoder.remux_with_audio_and_metadata") as mock_remux,
        patch("jasna.media.video_encoder.mux_hevc_to_mkv", mock_mux),
    ):
        mock_nvc.CreateEncoder.return_value = mock_encoder

        import torch

        with (
            NvidiaVideoEncoder(
                file=str(output_path),
                device=torch.device("cuda:0"),
                stream=torch.cuda.Stream(),
                metadata=_fake_metadata(),
                codec="hevc",
                encoder_settings={},
                stream_mode=False,
                working_directory=working_dir,
            ) as enc
        ):
            assert enc.hevc_path.parent == working_dir
            assert enc.hevc_path.name == "result.hevc"
            assert enc.temp_video_path.parent == working_dir
            assert enc.temp_video_path.name == "result_temp_video.mkv"

    mock_mux.assert_called_once()
    mock_remux.assert_called_once()
    call_args = mock_mux.call_args[0]
    assert call_args[0].parent == working_dir
    assert call_args[0].name == "result.hevc"
    mock_nvc.CreateEncoder.assert_called_once()
    assert mock_nvc.CreateEncoder.call_args[1]["gpu_id"] == 0


def test_encoder_unlinks_hevc_before_remux(tmp_path: Path) -> None:
    output_path = tmp_path / "output" / "result.mkv"
    output_path.parent.mkdir(parents=True)
    working_dir = tmp_path / "work"
    working_dir.mkdir()

    mock_encoder = MagicMock()
    mock_encoder.EndEncode.return_value = []

    call_order = []
    hevc_path_from_mux = []

    def track_mux(hevc_path, output_path, *args, **kwargs):
        call_order.append("mux")
        hevc_path_from_mux.append(hevc_path)
        output_path.touch()

    def track_remux(*args, **kwargs):
        call_order.append("remux")
        if hevc_path_from_mux:
            assert not hevc_path_from_mux[0].exists()

    original_unlink = Path.unlink

    def track_unlink(self):
        call_order.append("unlink_hevc" if self.suffix == ".hevc" else "unlink_other")
        return original_unlink(self)

    with (
        patch("jasna.media.video_encoder.nvc") as mock_nvc,
        patch("jasna.media.video_encoder.remux_with_audio_and_metadata", side_effect=track_remux),
        patch("jasna.media.video_encoder.mux_hevc_to_mkv", side_effect=track_mux),
        patch.object(Path, "unlink", track_unlink),
    ):
        mock_nvc.CreateEncoder.return_value = mock_encoder

        import torch

        with NvidiaVideoEncoder(
            file=str(output_path),
            device=torch.device("cuda:0"),
            stream=torch.cuda.Stream(),
            metadata=_fake_metadata(),
            codec="hevc",
            encoder_settings={},
            stream_mode=False,
            working_directory=working_dir,
        ):
            pass

    mux_idx = call_order.index("mux")
    unlink_hevc_idx = call_order.index("unlink_hevc")
    remux_idx = call_order.index("remux")
    assert mux_idx < unlink_hevc_idx < remux_idx
