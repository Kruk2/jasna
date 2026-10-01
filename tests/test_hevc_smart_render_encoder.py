from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.accelerator import AcceleratorVendor
from jasna.media import (
    VideoMetadata,
    hevc_level_to_amf_option,
    parse_hevc_level_idc,
)


def _metadata(**overrides) -> VideoMetadata:
    metadata = VideoMetadata(
        video_file="input.mkv",
        video_height=1080,
        video_width=1920,
        video_fps=30.0,
        average_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        codec_name="hevc",
        duration=1.0,
        time_base=Fraction(1, 30),
        start_pts=0,
        color_range=AvColorRange.MPEG,
        color_space=AvColorspace.ITU709,
        num_frames=30,
        is_10bit=True,
        hevc_level=183,
    )
    return replace(metadata, **overrides)


@pytest.mark.parametrize(
    ("stream", "expected"),
    [
        ({"codec_name": "hevc", "level": 183}, 183),
        ({"codec_name": "HEVC", "level": "186"}, 186),
        ({"codec_name": "hevc", "level": 180.0}, 180),
        ({"codec_name": "hevc", "level": 180.5}, None),
        ({"codec_name": "hevc", "level": True}, None),
        ({"codec_name": "h264", "level": 183}, None),
    ],
)
def test_parse_hevc_level(stream, expected) -> None:
    assert parse_hevc_level_idc(stream) == expected


@pytest.mark.parametrize(
    ("level", "expected"),
    [(30, "1.0"), (183, "6.1"), ("186", "6.2"), (181, None), (None, None)],
)
def test_map_hevc_level_to_amf(level, expected) -> None:
    assert hevc_level_to_amf_option(level) == expected


def test_linux_amd_hevc_fragment_uses_source_rate_vbr_peak_and_source_level(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "part.nut"),
        torch.device("cuda:0"),
        _metadata(video_bitrate=1_000_000),
        codec="hevc",
        encoder_settings={"cq": 28},
        smart_fragment=True,
        mux_audio=False,
    )

    assert encoder.encoder_options["rc"] == "vbr_peak"
    assert encoder.encoder_options["maxrate"] == "1250000"
    assert encoder.encoder_options["bufsize"] == "2500000"
    assert encoder.encoder_options["preanalysis"] == "0"
    assert encoder.encoder_options["vbaq"] == "0"
    assert encoder.encoder_options["level"] == "6.1"
    assert encoder.encoder_options["forced_idr"] == "1"
    assert encoder._target_bit_rate == 1_000_000
    assert "qp_i" not in encoder.encoder_options
    assert "qp_p" not in encoder.encoder_options
    assert "qvbr_quality_level" not in encoder.encoder_options


def test_linux_amd_hevc_full_encode_uses_requested_gui_source_rate(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "full.mp4"),
        torch.device("cuda:0"),
        _metadata(video_bitrate=20_000_000),
        codec="hevc",
        encoder_settings={"cq": 25},
        auto_source_rate=True,
    )

    assert encoder.encoder_options["rc"] == "vbr_peak"
    assert encoder.encoder_options["maxrate"] == "25000000"
    assert encoder.encoder_options["bufsize"] == "50000000"
    assert encoder.encoder_options["preanalysis"] == "0"
    assert encoder.encoder_options["vbaq"] == "0"
    assert encoder._target_bit_rate == 20_000_000
    assert "qp_i" not in encoder.encoder_options
    assert "qp_p" not in encoder.encoder_options
    assert "qvbr_quality_level" not in encoder.encoder_options


def test_linux_amd_hevc_full_encode_keeps_cqp_without_gui_source_rate(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "full.mp4"),
        torch.device("cuda:0"),
        _metadata(video_bitrate=20_000_000),
        codec="hevc",
        encoder_settings={"cq": 25, "rc": "cqp"},
        auto_source_rate=True,
    )

    assert encoder.encoder_options["rc"] == "cqp"
    assert encoder._target_bit_rate is None


def test_linux_amd_hevc_fragment_vbr_peak_can_be_rolled_back_to_cqp(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setenv(module.AMF_HEVC_VBR_PEAK_ENV, "0")
    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "part.nut"),
        torch.device("cuda:0"),
        _metadata(),
        codec="hevc",
        encoder_settings={"cq": 28},
        smart_fragment=True,
        mux_audio=False,
    )

    assert encoder.encoder_options["rc"] == "cqp"
    assert encoder.encoder_options["qp_i"] == "30"
    assert encoder.encoder_options["qp_p"] == "30"
    assert encoder.encoder_options["preanalysis"] == "0"
    assert encoder.encoder_options["level"] == "6.1"
    assert encoder.encoder_options["forced_idr"] == "1"
    assert encoder._target_bit_rate is None


def test_linux_amd_hevc_fragment_with_explicit_rc_skips_automatic_vbr_peak(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "part.nut"),
        torch.device("cuda:0"),
        _metadata(),
        codec="hevc",
        encoder_settings={"rc": "cqp", "cq": 28},
        smart_fragment=True,
        mux_audio=False,
    )

    assert encoder.encoder_options["rc"] == "cqp"
    assert encoder.encoder_options["qp_i"] == "28"
    assert encoder.encoder_options["qp_p"] == "28"
    assert encoder._target_bit_rate is None


def test_linux_amd_hevc_fragment_without_source_bitrate_keeps_cqp(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "part.nut"),
        torch.device("cuda:0"),
        _metadata(video_bitrate=0),
        codec="hevc",
        encoder_settings={"cq": 28},
        smart_fragment=True,
        mux_audio=False,
    )

    assert encoder.encoder_options["rc"] == "cqp"
    assert encoder.encoder_options["qp_i"] == "30"
    assert encoder.encoder_options["qp_p"] == "30"
    assert encoder._target_bit_rate is None


@pytest.mark.parametrize("vbr_peak_env", [None, "auto"], ids=["unset", "auto"])
def test_windows_amd_hevc_fragment_auto_or_unset_vbr_peak_keeps_cqp(
    monkeypatch, tmp_path, vbr_peak_env
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "win32")
    if vbr_peak_env is None:
        monkeypatch.delenv(module.AMF_HEVC_VBR_PEAK_ENV, raising=False)
    else:
        monkeypatch.setenv(module.AMF_HEVC_VBR_PEAK_ENV, vbr_peak_env)
    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "part.nut"),
        torch.device("cuda:0"),
        _metadata(),
        codec="hevc",
        encoder_settings={"cq": 28, "preanalysis": 1},
        smart_fragment=True,
        mux_audio=False,
    )

    assert encoder.encoder_options["qp_i"] == "28"
    assert encoder.encoder_options["preanalysis"] == "1"
    assert "level" not in encoder.encoder_options


def test_windows_amd_hevc_fragment_vbr_peak_override_is_rejected(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setenv(module.AMF_HEVC_VBR_PEAK_ENV, "1")

    with pytest.raises(
        ValueError,
        match="pinned runtime cannot complete copy/render seam validation",
    ):
        module.NvidiaVideoEncoder(
            str(tmp_path / "part.nut"),
            torch.device("cuda:0"),
            _metadata(video_bitrate=20_000_000),
            codec="hevc",
            encoder_settings={"cq": 28},
            smart_fragment=True,
            mux_audio=False,
        )


def test_resolve_hevc_vui_uses_decoded_source_values(monkeypatch) -> None:
    import jasna.media.video_encoder as module

    original = _metadata(
        video_fps=19001 / 317,
        average_fps=19001 / 317,
        video_fps_exact=Fraction(19001, 317),
        color_primaries="",
        color_transfer="",
    )
    stream = SimpleNamespace(
        codec_context=SimpleNamespace(
            framerate=Fraction(60_000, 1_001),
            rate=Fraction(60_000, 1_001),
        )
    )
    frame = SimpleNamespace(
        color_range=1,
        colorspace=1,
        color_primaries=9,
        color_trc=16,
    )
    source = MagicMock()
    source.streams.video = [stream]
    source.decode.return_value = iter([frame])
    opened = MagicMock()
    opened.__enter__.return_value = source
    monkeypatch.setattr(module.av, "open", MagicMock(return_value=opened))

    resolved, output_fps = module.resolve_hevc_smart_render_vui(original)

    assert output_fps == Fraction(60_000, 1_001)
    assert resolved.video_fps_exact == output_fps
    assert resolved.color_range == AvColorRange.MPEG
    assert resolved.color_space == AvColorspace.ITU709
    assert resolved.color_primaries == "bt2020"
    assert resolved.color_transfer == "smpte2084"
    assert original.video_fps_exact == Fraction(19001, 317)


def test_resolve_hevc_vui_fails_closed_to_existing_metadata(monkeypatch) -> None:
    import jasna.media.video_encoder as module

    original = _metadata()
    monkeypatch.setattr(module.av, "open", MagicMock(side_effect=OSError("unreadable")))

    resolved, output_fps = module.resolve_hevc_smart_render_vui(original)

    assert output_fps == original.video_fps_exact
    assert resolved == original
    assert resolved is not original
