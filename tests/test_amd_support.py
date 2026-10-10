from __future__ import annotations

import sys
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.accelerator import (
    AcceleratorVendor,
    vendor_for_device,
)
from jasna.media.probe import VideoMetadata
from jasna.media.encoder_settings import validate_encoder_settings


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        video_file="input.mp4",
        video_height=16,
        video_width=16,
        video_fps=30.0,
        average_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        codec_name="h264",
        duration=1.0,
        time_base=Fraction(1, 30),
        start_pts=0,
        color_range=AvColorRange.MPEG,
        color_space=AvColorspace.ITU709,
        num_frames=30,
        is_10bit=False,
    )


def test_rocm_uses_cuda_device_api_but_reports_amd(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "7.2.1")
    assert vendor_for_device("cuda:0") is AcceleratorVendor.AMD


def test_amd_basicvsrpp_skips_tensorrt_compilation(monkeypatch) -> None:
    import jasna.accelerator as accelerator
    import jasna.engine_compiler as compiler

    monkeypatch.setattr(accelerator, "is_nvidia_device", lambda _device: False)
    monkeypatch.setattr(accelerator, "is_amd_device", lambda _device: True)
    monkeypatch.setattr(
        compiler,
        "all_basicvsrpp_sub_engines_exist",
        MagicMock(side_effect=AssertionError("TensorRT probe on AMD")),
    )
    result = compiler.ensure_engines_compiled(
        compiler.EngineCompilationRequest(
            device="cuda:0",
            fp16=True,
            basicvsrpp=True,
            basicvsrpp_model_path="model.pth",
        )
    )
    assert result.use_basicvsrpp_tensorrt is False


def test_amf_encoder_settings_are_vendor_specific() -> None:
    assert validate_encoder_settings(
        {"preanalysis": 1, "cq": 24},
        codec="h264",
        vendor=AcceleratorVendor.AMD,
    ) == {"preanalysis": 1, "cq": 24}
    with pytest.raises(ValueError, match="temporal-aq"):
        validate_encoder_settings(
            {"temporal-aq": 1},
            codec="h264",
            vendor=AcceleratorVendor.AMD,
        )


def test_windows_resident_encoder_does_not_create_independent_amf_hwaccel(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    coordinator = object()
    encoder = module.VideoEncoder(
        str(tmp_path / "resident.mkv"),
        torch.device("cuda:0"),
        replace(_metadata(), codec_name="hevc"),
        codec="hevc",
        encoder_settings={},
        match_input_bit_depth=True,
        resident_coordinator=coordinator,
    )

    assert encoder._resident_coordinator is coordinator
    assert "hwaccel" not in encoder._video_stream_kwargs()
    assert encoder._host_yuv is None if hasattr(encoder, "_host_yuv") else True
    assert encoder.encoder_options["g"] == "60"
    assert encoder.encoder_options["bf"] == "0"
    assert encoder.encoder_options["preanalysis"] == "0"
    assert encoder.encoder_options["async_depth"] == "4"


@pytest.mark.parametrize(
    "encoder_settings",
    (
        {"g": 250},
        {"bf": 1},
        {"preanalysis": 1},
    ),
)
def test_windows_resident_encoder_rejects_unvalidated_inflight_options(
    monkeypatch,
    tmp_path,
    encoder_settings,
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )

    with pytest.raises(
        ValueError,
        match="requires g=60, bf=0, preanalysis=0, and async_depth=4",
    ):
        module.VideoEncoder(
            str(tmp_path / "resident.mkv"),
            torch.device("cuda:0"),
            replace(_metadata(), codec_name="hevc"),
            codec="hevc",
            encoder_settings=encoder_settings,
            match_input_bit_depth=True,
            resident_coordinator=object(),
        )


def test_amf_hevc_uses_compatible_defaults() -> None:
    from jasna.media.video_encoder import AMF_ENCODER_SPECS

    options = AMF_ENCODER_SPECS["hevc"].default_options
    assert options["rc"] == "cqp"
    assert options["preanalysis"] == "0"
    assert options["vbaq"] == "0"
    assert options["qp_i"] == "25"
    assert options["qp_p"] == "25"
    assert "qvbr_quality_level" not in options


def test_linux_amd_h264_smart_uses_source_rate_without_preanalysis(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    encoder = module.VideoEncoder(
        str(tmp_path / "part.nut"),
        torch.device("cuda:0"),
        replace(_metadata(), video_bitrate=23_032_483),
        codec="h264",
        encoder_settings={
            "rc": "vbr_peak",
            "preanalysis": 0,
            "vbaq": 0,
            "bf": 3,
            "bf_ref": 1,
        },
        smart_fragment=True,
        mux_audio=False,
    )

    assert encoder.encoder_options["rc"] == "vbr_peak"
    assert encoder.encoder_options["maxrate"] == "23032483"
    assert encoder.encoder_options["bufsize"] == "46064966"
    assert encoder.encoder_options["preanalysis"] == "0"
    assert encoder.encoder_options["vbaq"] == "0"
    assert encoder.encoder_options["forced_idr"] == "1"
    assert encoder._target_bit_rate == 23_032_483
    assert "qvbr_quality_level" not in encoder.encoder_options


def test_linux_amd_h264_smart_requires_source_bitrate(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    with pytest.raises(ValueError, match="positive source video bitrate"):
        module.VideoEncoder(
            str(tmp_path / "part.nut"),
            torch.device("cuda:0"),
            replace(_metadata(), video_bitrate=0),
            codec="h264",
            encoder_settings={"rc": "vbr_peak", "preanalysis": 0},
            smart_fragment=True,
            mux_audio=False,
        )


def test_amf_av1_uses_codec_specific_adaptive_quantization() -> None:
    from jasna.media.video_encoder import AMF_ENCODER_SPECS

    options = AMF_ENCODER_SPECS["av1"].default_options
    assert options["aq_mode"] == "none"
    assert "vbaq" not in options
    assert validate_encoder_settings(
        {"aq_mode": "caq"},
        codec="av1",
        vendor=AcceleratorVendor.AMD,
    ) == {"aq_mode": "caq"}
    with pytest.raises(ValueError, match="vbaq"):
        validate_encoder_settings(
            {"vbaq": 1},
            codec="av1",
            vendor=AcceleratorVendor.AMD,
        )


def test_amf_av1_main10_uses_upstream_cqp_without_preanalysis(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    encoder = module.VideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        replace(_metadata(), is_10bit=True, video_bitrate=20_000_000),
        codec="av1",
        encoder_settings={},
    )

    assert encoder.encoder_name == "av1_amf"
    assert encoder.spec.frame_format == "p010le"
    assert encoder.encoder_options["rc"] == "cqp"
    assert encoder.encoder_options["qp_i"] == "160"
    assert encoder.encoder_options["preanalysis"] == "0"
    assert "qvbr_quality_level" not in encoder.encoder_options
    assert encoder._target_bit_rate is None


def test_amf_av1_cqp_is_defined_without_source_bitrate(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    encoder = module.VideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        replace(_metadata(), is_10bit=True, video_bitrate=0),
        codec="av1",
        encoder_settings={},
    )

    assert encoder._target_bit_rate is None
    assert encoder.encoder_options["rc"] == "cqp"
    assert "maxrate" not in encoder.encoder_options


def test_amf_av1_main10_policy_does_not_affect_eight_bit(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    encoder = module.VideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        replace(_metadata(), is_10bit=False, video_bitrate=20_000_000),
        codec="av1",
        encoder_settings={},
        match_input_bit_depth=True,
    )

    assert encoder.spec.frame_format == "nv12"
    assert encoder.encoder_options["rc"] == "cqp"
    assert encoder.encoder_options["preanalysis"] == "0"
    assert encoder._target_bit_rate is None


def test_source_bitrate_ceiling_omits_out_of_range_ffmpeg_values(caplog) -> None:
    from jasna.media.video_encoder import source_bitrate_cap_options

    with caplog.at_level("WARNING"):
        options = source_bitrate_cap_options(
            replace(
                _metadata(),
                codec_name="hevc",
                video_bitrate=2_000_000_000,
            ),
            output_codec="hevc",
            vendor=AcceleratorVendor.AMD,
        )

    assert options == {}
    assert "exceeds the encoder option range" in caplog.text


def test_video_encoder_selects_amf_and_normalizes_cq(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    encoder = module.VideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        _metadata(),
        codec="h264",
        encoder_settings={"cq": 21},
    )
    assert encoder.encoder_name == "h264_amf"
    assert encoder.spec.frame_format == "nv12"
    assert encoder.encoder_options["qvbr_quality_level"] == "21"
    assert "cq" not in encoder.encoder_options


def test_amf_hevc_maps_cq_to_constant_qp(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    encoder = module.VideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        _metadata(),
        codec="hevc",
        encoder_settings={"cq": 21},
    )
    assert encoder.spec.frame_format == "p010le"
    assert encoder.encoder_options["rc"] == "cqp"
    assert encoder.encoder_options["qp_i"] == "21"
    assert encoder.encoder_options["qp_p"] == "21"
    assert "cq" not in encoder.encoder_options
    assert "qvbr_quality_level" not in encoder.encoder_options


def test_amf_hevc_cqp_skips_source_bitrate_cap(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    encoder = module.VideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        replace(_metadata(), video_bitrate=20_000_000),
        codec="hevc",
        encoder_settings={"cq": 21},
    )
    assert "maxrate" not in encoder.encoder_options
    assert "bufsize" not in encoder.encoder_options


@pytest.mark.parametrize(
    ("settings", "smart_fragment", "frame_format", "expected_qindex"),
    [
        ({}, False, "p010le", "160"),
        ({"cq": 21}, False, "p010le", "105"),
        ({"qvbr_quality_level": 51}, False, "p010le", "255"),
        ({"cq": 1}, True, "nv12", "5"),
    ],
)
def test_amf_av1_maps_cq_to_constant_qindex(
    settings, smart_fragment, frame_format, expected_qindex
) -> None:
    from jasna.media.video_encoder import resolve_encoder_options

    spec, options = resolve_encoder_options(
        AcceleratorVendor.AMD,
        "av1",
        replace(_metadata(), video_bitrate=20_000_000),
        settings,
        smart_fragment=smart_fragment,
    )
    assert spec.frame_format == frame_format
    assert options["rc"] == "cqp"
    assert options["preanalysis"] == "0"
    assert options["aq_mode"] == "none"
    assert all(options[key] == expected_qindex for key in ("qp_i", "qp_p", "qp_b"))
    assert "qvbr_quality_level" not in options
    assert "maxrate" not in options
    assert "bufsize" not in options


def test_amf_hevc_vbr_peak_candidate_uses_source_rate_contract(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setenv(module.AMF_HEVC_VBR_PEAK_ENV, "1")

    encoder = module.VideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        replace(
            _metadata(),
            codec_name="hevc",
            is_10bit=True,
            video_bitrate=20_000_000,
        ),
        codec="hevc",
        encoder_settings={"cq": 18},
        smart_fragment=True,
    )

    assert encoder.encoder_options["rc"] == "vbr_peak"
    assert encoder.encoder_options["preanalysis"] == "0"
    assert encoder.encoder_options["vbaq"] == "0"
    assert encoder.encoder_options["maxrate"] == "25000000"
    assert encoder.encoder_options["bufsize"] == "50000000"
    assert encoder.encoder_options["forced_idr"] == "1"
    assert encoder._target_bit_rate == 20_000_000
    assert "qp_i" not in encoder.encoder_options
    assert "qp_p" not in encoder.encoder_options
    assert "qvbr_quality_level" not in encoder.encoder_options


def test_amf_hevc_vbr_peak_candidate_requires_source_bitrate(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setenv(module.AMF_HEVC_VBR_PEAK_ENV, "1")

    with pytest.raises(ValueError, match="positive source video bitrate"):
        module.VideoEncoder(
            str(tmp_path / "out.mp4"),
            torch.device("cuda:0"),
            replace(_metadata(), codec_name="hevc", video_bitrate=0),
            codec="hevc",
            encoder_settings={"cq": 18},
        )


def test_amf_hevc_vbr_peak_candidate_rejects_custom_rate_contract(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setenv(module.AMF_HEVC_VBR_PEAK_ENV, "1")

    with pytest.raises(ValueError, match="remove custom maxrate"):
        module.VideoEncoder(
            str(tmp_path / "out.mp4"),
            torch.device("cuda:0"),
            replace(
                _metadata(),
                codec_name="hevc",
                video_bitrate=20_000_000,
            ),
            codec="hevc",
            encoder_settings={"cq": 18, "maxrate": 30_000_000},
        )


@pytest.mark.parametrize(
    ("is_10bit", "match_input_bit_depth", "frame_format", "profile"),
    [
        (False, True, "nv12", "main"),
        (True, False, "p010le", "main10"),
    ],
    ids=["main_nv12", "main10_p010"],
)
def test_windows_amd_hevc_full_encode_vbr_peak_override_uses_source_rate_contract(
    monkeypatch,
    tmp_path,
    is_10bit,
    match_input_bit_depth,
    frame_format,
    profile,
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setenv(module.AMF_HEVC_VBR_PEAK_ENV, "1")

    encoder = module.VideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        replace(
            _metadata(),
            codec_name="hevc",
            is_10bit=is_10bit,
            video_bitrate=20_000_000,
        ),
        codec="hevc",
        encoder_settings={"cq": 18},
        match_input_bit_depth=match_input_bit_depth,
    )

    assert encoder.spec.frame_format == frame_format
    assert encoder.encoder_options["profile"] == profile
    assert encoder.encoder_options["rc"] == "vbr_peak"
    assert encoder.encoder_options["maxrate"] == "25000000"
    assert encoder.encoder_options["bufsize"] == "40000000"
    assert encoder.encoder_options["preanalysis"] == "0"
    assert encoder.encoder_options["vbaq"] == "0"
    assert encoder._target_bit_rate == 20_000_000
    assert "cq" not in encoder.encoder_options
    assert "qp_i" not in encoder.encoder_options
    assert "qp_p" not in encoder.encoder_options
    assert "qvbr_quality_level" not in encoder.encoder_options


@pytest.mark.parametrize("vbr_peak_env", [None, "auto"], ids=["unset", "auto"])
def test_windows_amd_hevc_full_encode_auto_or_unset_vbr_peak_keeps_cqp(
    monkeypatch,
    tmp_path,
    vbr_peak_env,
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

    encoder = module.VideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        replace(_metadata(), codec_name="hevc", video_bitrate=20_000_000),
        codec="hevc",
        encoder_settings={"cq": 18},
        auto_source_rate=True,
    )

    assert encoder.encoder_options["rc"] == "cqp"
    assert encoder.encoder_options["qp_i"] == "18"
    assert encoder.encoder_options["qp_p"] == "18"
    assert "maxrate" not in encoder.encoder_options
    assert "bufsize" not in encoder.encoder_options
    assert encoder._target_bit_rate is None


@pytest.mark.parametrize(
    "metadata",
    [
        replace(_metadata(), codec_name="hevc"),
        replace(_metadata(), codec_name="hevc", video_bitrate=0),
    ],
    ids=["missing", "zero"],
)
def test_windows_amd_hevc_full_encode_vbr_peak_override_requires_source_bitrate(
    monkeypatch,
    tmp_path,
    metadata,
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setenv(module.AMF_HEVC_VBR_PEAK_ENV, "1")

    with pytest.raises(ValueError, match="positive source video bitrate"):
        module.VideoEncoder(
            str(tmp_path / "out.mp4"),
            torch.device("cuda:0"),
            metadata,
            codec="hevc",
            encoder_settings={"cq": 18},
        )


def test_windows_amd_hevc_full_encode_vbr_peak_override_rejects_out_of_range_contract(
    monkeypatch,
    tmp_path,
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setenv(module.AMF_HEVC_VBR_PEAK_ENV, "1")

    with pytest.raises(ValueError, match="could not derive a safe peak/buffer"):
        module.VideoEncoder(
            str(tmp_path / "out.mp4"),
            torch.device("cuda:0"),
            replace(
                _metadata(),
                codec_name="hevc",
                video_bitrate=1_500_000_000,
            ),
            codec="hevc",
            encoder_settings={"cq": 18},
        )


@pytest.mark.parametrize(
    ("encoder_settings", "error"),
    [
        ({"cq": 18, "rc": "cqp"}, "conflicts with rc"),
        ({"cq": 18, "maxrate": 30_000_000}, "remove custom maxrate"),
        ({"cq": 18, "bufsize": 60_000_000}, "remove custom bufsize"),
    ],
)
def test_windows_amd_hevc_full_encode_vbr_peak_override_rejects_conflicting_rate_settings(
    monkeypatch,
    tmp_path,
    encoder_settings,
    error,
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setenv(module.AMF_HEVC_VBR_PEAK_ENV, "1")

    with pytest.raises(ValueError, match=error):
        module.VideoEncoder(
            str(tmp_path / "out.mp4"),
            torch.device("cuda:0"),
            replace(
                _metadata(),
                codec_name="hevc",
                video_bitrate=20_000_000,
            ),
            codec="hevc",
            encoder_settings=encoder_settings,
        )


def test_amf_hevc_vbr_peak_switch_rejects_invalid_value(
    monkeypatch,
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setenv(module.AMF_HEVC_VBR_PEAK_ENV, "sometimes")
    with pytest.raises(ValueError, match=module.AMF_HEVC_VBR_PEAK_ENV):
        module._amf_hevc_vbr_peak_override()


def test_amf_host_native_input_is_automatic_for_8k_main10(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")

    encoder = module.VideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        replace(
            _metadata(),
            codec_name="hevc",
            is_10bit=True,
            video_width=8192,
            video_height=4096,
        ),
        codec="hevc",
        encoder_settings={},
    )

    assert encoder._amf_host_zero_copy is True
    assert encoder.encoder_options["host_zero_copy"] == "1"
    assert encoder.encoder_options["async_depth"] == "4"
    assert "hwaccel" not in encoder._video_stream_kwargs()


def test_amf_host_native_input_is_automatic_for_5k_main10(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")

    encoder = module.VideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        replace(
            _metadata(),
            codec_name="hevc",
            is_10bit=True,
            video_width=5760,
            video_height=2880,
        ),
        codec="hevc",
        encoder_settings={},
    )

    assert encoder.spec.frame_format == "p010le"
    assert encoder._amf_host_zero_copy is True
    assert encoder.encoder_options["host_zero_copy"] == "1"
    assert encoder.encoder_options["async_depth"] == "4"
    assert "hwaccel" not in encoder._video_stream_kwargs()


def test_amf_host_native_input_is_automatic_for_4k_main10(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")

    encoder = module.VideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        replace(
            _metadata(),
            codec_name="hevc",
            is_10bit=True,
            video_width=3840,
            video_height=2160,
        ),
        codec="hevc",
        encoder_settings={},
    )

    assert encoder.spec.frame_format == "p010le"
    assert encoder._amf_host_zero_copy is True
    assert encoder.encoder_options["host_zero_copy"] == "1"
    assert encoder.encoder_options["async_depth"] == "4"
    assert "hwaccel" not in encoder._video_stream_kwargs()


def test_amf_host_native_input_is_selected_for_5k_main8_dual_gop(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")

    encoder = module.VideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        replace(
            _metadata(),
            codec_name="hevc",
            is_10bit=False,
            video_width=5760,
            video_height=2880,
        ),
        codec="hevc",
        encoder_settings={},
        match_input_bit_depth=True,
        prefer_amf_host_native=True,
    )

    assert encoder.spec.frame_format == "nv12"
    assert encoder._amf_host_zero_copy is True
    assert encoder.encoder_options["host_zero_copy"] == "1"
    assert encoder.encoder_options["async_depth"] == "4"
    assert "hwaccel" not in encoder._video_stream_kwargs()


def test_amf_host_native_input_stays_disabled_for_5k_main8_single_session(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")

    encoder = module.VideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        replace(
            _metadata(),
            codec_name="hevc",
            is_10bit=False,
            video_width=5760,
            video_height=2880,
        ),
        codec="hevc",
        encoder_settings={},
        match_input_bit_depth=True,
    )

    assert encoder.spec.frame_format == "nv12"
    assert encoder._amf_host_zero_copy is False
    assert "host_zero_copy" not in encoder.encoder_options


def test_amf_host_native_input_can_be_disabled(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setenv(module.AMF_HOST_ZERO_COPY_ENV, "0")

    encoder = module.VideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        replace(
            _metadata(),
            codec_name="hevc",
            is_10bit=True,
            video_width=8192,
            video_height=4096,
        ),
        codec="hevc",
        encoder_settings={},
    )

    assert encoder._amf_host_zero_copy is False
    assert "host_zero_copy" not in encoder.encoder_options
    assert isinstance(encoder._video_stream_kwargs()["hwaccel"], module.HWAccel)


@pytest.mark.parametrize(
    ("metadata_overrides", "encoder_kwargs"),
    [
        (
            {
                "codec_name": "hevc",
                "is_10bit": True,
                "video_width": 1920,
                "video_height": 1080,
            },
            {},
        ),
        (
            {
                "codec_name": "hevc",
                "is_10bit": False,
                "video_width": 3840,
                "video_height": 2160,
            },
            {"match_input_bit_depth": True},
        ),
        (
            {
                "codec_name": "hevc",
                "is_10bit": False,
                "video_width": 5760,
                "video_height": 2880,
            },
            {"match_input_bit_depth": True},
        ),
    ],
)
def test_amf_host_native_input_auto_keeps_unvalidated_formats_on_copy_path(
    monkeypatch,
    tmp_path,
    metadata_overrides: dict[str, object],
    encoder_kwargs: dict[str, object],
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")

    encoder = module.VideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        replace(_metadata(), **metadata_overrides),
        codec="hevc",
        encoder_settings={},
        **encoder_kwargs,
    )

    assert encoder._amf_host_zero_copy is False
    assert "host_zero_copy" not in encoder.encoder_options
    assert isinstance(encoder._video_stream_kwargs()["hwaccel"], module.HWAccel)


def test_amf_host_native_input_forced_route_accepts_linux_amd_hevc_nv12(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setenv(module.AMF_HOST_ZERO_COPY_ENV, "1")

    encoder = module.VideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        replace(
            _metadata(),
            codec_name="hevc",
            is_10bit=False,
            video_width=3840,
            video_height=2160,
        ),
        codec="hevc",
        encoder_settings={},
        match_input_bit_depth=True,
    )

    assert encoder.spec.frame_format == "nv12"
    assert encoder._amf_host_zero_copy is True
    assert encoder.encoder_options["host_zero_copy"] == "1"


def test_amf_host_native_input_forced_route_rejects_non_hevc(
    monkeypatch, tmp_path
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setenv(module.AMF_HOST_ZERO_COPY_ENV, "1")

    with pytest.raises(ValueError, match="only for Linux AMD HEVC"):
        module.VideoEncoder(
            str(tmp_path / "out.mp4"),
            torch.device("cuda:0"),
            _metadata(),
            codec="h264",
            encoder_settings={},
        )


def test_amf_host_native_switch_rejects_invalid_value(monkeypatch) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setenv(module.AMF_HOST_ZERO_COPY_ENV, "sometimes")
    with pytest.raises(ValueError, match=module.AMF_HOST_ZERO_COPY_ENV):
        module._amf_host_zero_copy_override()




@pytest.mark.parametrize("rc", ["qvbr", "hqvbr", 4, 5])
def test_amf_av1_p010_rejects_qvbr(monkeypatch, tmp_path, rc: str | int) -> None:
    import jasna.media.video_encoder as module
    monkeypatch.setattr(module, "vendor_for_device", lambda _device: AcceleratorVendor.AMD)
    with pytest.raises(ValueError, match="AMD AV1 Main10.*QVBR"):
        module.VideoEncoder(
            str(tmp_path / "out.mp4"),
            torch.device("cuda:0"),
            _metadata(),
            codec="av1",
            encoder_settings={"cq": 21, "rc": rc},
        )


@pytest.mark.parametrize("rc", ["qvbr", "hqvbr", 4, 5])
def test_amf_hevc_rejects_qvbr_for_main10(
    monkeypatch, tmp_path, rc: str | int
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    with pytest.raises(ValueError, match="AMD HEVC Main10.*QVBR"):
        module.VideoEncoder(
            str(tmp_path / "out.mp4"),
            torch.device("cuda:0"),
            _metadata(),
            codec="hevc",
            encoder_settings={"cq": 21, "rc": rc},
        )


def test_amf_hevc_8bit_allows_qvbr(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    encoder = module.VideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        _metadata(),
        codec="hevc",
        encoder_settings={"cq": 21, "rc": "qvbr"},
        smart_fragment=True,
    )
    assert encoder.spec.frame_format == "nv12"
    assert encoder.encoder_options["rc"] == "qvbr"
    assert encoder.encoder_options["qvbr_quality_level"] == "21"
    assert "qp_i" not in encoder.encoder_options
    assert "qp_p" not in encoder.encoder_options


@pytest.mark.parametrize("codec", ["h264", "hevc", "av1"])
def test_smart_render_uses_amf_fragment_options(
    monkeypatch,
    tmp_path,
    codec: str,
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    encoder = module.VideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        replace(_metadata(), video_bitrate=1_000_000),
        codec=codec,
        encoder_settings={},
        smart_fragment=True,
    )

    assert encoder.encoder_name == f"{codec}_amf"
    assert encoder.encoder_options["forced_idr"] == "1"
    assert "forced-idr" not in encoder.encoder_options


def test_amf_h264_smart_settings_are_accepted() -> None:
    settings = {"bf": 3, "bf_ref": 1, "pa_adaptive_mini_gop": 0}

    assert validate_encoder_settings(
        settings,
        codec="h264",
        vendor=AcceleratorVendor.AMD,
    ) == settings


def test_streaming_encoder_selects_amf(monkeypatch, tmp_path) -> None:
    import jasna.streaming_encoder as module

    monkeypatch.setattr(module, "find_executable", lambda _name: "/ffmpeg")
    popen = MagicMock()
    popen.stderr = []
    monkeypatch.setattr(module.subprocess, "Popen", MagicMock(return_value=popen))
    encoder = module.StreamingEncoder(
        tmp_path,
        4.0,
        _metadata(),
        "missing.mp4",
        torch.device("cuda:0"),
    )
    encoder._vendor = AcceleratorVendor.AMD
    encoder._launch_ffmpeg(0)
    cmd = module.subprocess.Popen.call_args.args[0]
    assert cmd[cmd.index("-c:v") + 1] == "h264_amf"
    assert "-qvbr_quality_level" in cmd
    assert "h264_nvenc" not in cmd


def test_amf_decoder_context_is_created(monkeypatch) -> None:
    import jasna.media.video_decoder as module

    decoder = MagicMock()
    monkeypatch.setattr(
        module.av,
        "CodecContext",
        SimpleNamespace(create=MagicMock(return_value=decoder)),
    )
    reader = module.VideoReader(
        "input.mp4",
        4,
        torch.device("cuda:0"),
        _metadata(),
    )
    source = SimpleNamespace(
        name="h264",
        extradata=b"header",
        width=16,
        height=16,
        time_base=Fraction(1, 30),
        framerate=Fraction(30, 1),
        sample_aspect_ratio=Fraction(1, 1),
        thread_type=None,
    )
    reader._setup_amf_decoder(source)
    create = module.av.CodecContext.create
    assert create.call_args.args[:2] == ("h264_amf", "r")
    decoder.open.assert_called_once_with(strict=False)
    assert reader._decoder_ctx is decoder


def test_amf_decoder_accepts_missing_source_rationals(monkeypatch) -> None:
    import jasna.media.video_decoder as module

    class FakeDecoder:
        def __setattr__(self, name, value):
            if name in {"framerate", "sample_aspect_ratio"} and value is None:
                raise AttributeError("'NoneType' object has no attribute 'numerator'")
            object.__setattr__(self, name, value)

        def open(self, strict=False):
            self.opened = True

    decoder = FakeDecoder()
    monkeypatch.setattr(
        module.av,
        "CodecContext",
        SimpleNamespace(create=MagicMock(return_value=decoder)),
    )
    reader = module.VideoReader(
        "input.mp4", 4, torch.device("cuda:0"), _metadata()
    )
    source = SimpleNamespace(
        name="h264",
        extradata=b"header",
        width=16,
        height=16,
        framerate=None,
        sample_aspect_ratio=None,
        thread_type=None,
    )

    reader._setup_amf_decoder(source)

    assert decoder.sample_aspect_ratio == Fraction(1, 1)
    assert decoder.opened is True


def test_amf_decoder_survives_pyav18_time_base_regression(monkeypatch) -> None:
    import jasna.media.video_decoder as module

    class FakeDecoder:
        def __init__(self):
            object.__setattr__(self, "opened", False)

        def __setattr__(self, name, value):
            if name == "time_base":
                raise RuntimeError("Cannot access 'time_base' as a decoder")
            object.__setattr__(self, name, value)

        def open(self, strict=False):
            object.__setattr__(self, "opened", True)

    decoder = FakeDecoder()
    monkeypatch.setattr(
        module.av,
        "CodecContext",
        SimpleNamespace(create=MagicMock(return_value=decoder)),
    )
    reader = module.VideoReader(
        "input.mp4",
        4,
        torch.device("cuda:0"),
        _metadata(),
    )
    source = SimpleNamespace(
        name="hevc",
        extradata=b"header",
        width=16,
        height=16,
        time_base=Fraction(1, 30),
        framerate=Fraction(30, 1),
        sample_aspect_ratio=Fraction(1, 1),
        thread_type=None,
    )
    reader._setup_amf_decoder(source)
    assert decoder.opened is True
    assert reader._decoder_ctx is decoder


def test_amf_decoder_ignores_missing_optional_timing_metadata(monkeypatch) -> None:
    import jasna.media.video_decoder as module

    class FakeDecoder:
        def __init__(self):
            object.__setattr__(self, "opened", False)

        def __setattr__(self, name, value):
            if name in {"framerate", "sample_aspect_ratio"} and value is None:
                raise AssertionError(f"optional metadata was assigned: {name}={value!r}")
            object.__setattr__(self, name, value)

        def open(self, strict=False):
            assert strict is False
            object.__setattr__(self, "opened", True)

    decoder = FakeDecoder()
    monkeypatch.setattr(
        module.av,
        "CodecContext",
        SimpleNamespace(create=MagicMock(return_value=decoder)),
    )
    reader = module.VideoReader(
        "input.mp4",
        4,
        torch.device("cuda:0"),
        _metadata(),
    )
    source = SimpleNamespace(
        name="h264",
        extradata=b"header",
        width=16,
        height=16,
        time_base=Fraction(1, 30),
        framerate=None,
        sample_aspect_ratio=None,
        thread_type=None,
    )
    reader._setup_amf_decoder(source)
    assert decoder.opened is True
    assert reader._decoder_ctx is decoder


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_yuv_eager_converter_runs_on_gpu_planes(monkeypatch) -> None:
    import jasna.media.yuv_to_rgb as module

    monkeypatch.setattr(module, "is_nvidia_device", lambda _device: False)
    H = W = 16
    generator = torch.Generator().manual_seed(0)
    y = torch.randint(16, 236, (H, W), dtype=torch.uint8, generator=generator)
    uv = torch.randint(16, 240, (H // 2, W // 2, 2), dtype=torch.uint8, generator=generator)

    cpu = module.YuvToRgbConverter(
        H, W, AvColorspace.ITU709, False, False, torch.device("cpu")
    )
    expected = torch.empty((3, H, W), dtype=torch.uint8)
    cpu.convert_into(y, uv, expected)

    gpu = module.YuvToRgbConverter(
        H, W, AvColorspace.ITU709, False, False, torch.device("cuda:0")
    )
    out = torch.empty((3, H, W), dtype=torch.uint8, device="cuda:0")
    gpu.convert_into(y.cuda(), uv.cuda(), out)

    assert (out.cpu().int() - expected.int()).abs().max() <= 1


def test_amf_8bit_downgrade_drops_bitdepth(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    encoder = module.VideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        _metadata(),
        codec="hevc",
        encoder_settings={},
        smart_fragment=True,
    )
    assert encoder.spec.frame_format == "nv12"
    assert encoder.encoder_options["profile"] == "main"
    assert encoder.encoder_options["forced_idr"] == "1"
    assert "bitdepth" not in encoder.encoder_options

def test_rfdetr_torch_runner_maps_outputs(monkeypatch, tmp_path) -> None:
    import jasna.mosaic.rfdetr_torch_runner as module

    weights = tmp_path / "rfdetr-v6.pt"
    weights.write_bytes(b"pt")

    class FakeCore:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, x):
            batch = x.shape[0]
            return {
                "pred_boxes": torch.zeros(batch, 5, 4),
                "pred_logits": torch.zeros(batch, 5, 3),
                "pred_masks": torch.zeros(batch, 5, 8, 8),
            }

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = SimpleNamespace(model=FakeCore())

    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(RFDETRSegMedium=FakeModel))
    monkeypatch.setattr(
        module.torch,
        "load",
        lambda *_a, **_k: {"model": {"class_embed.weight": torch.zeros(3, 256)}},
    )

    runner = module.RfDetrTorchRunner(
        weights,
        input_shapes=[(2, 3, 576, 576)],
        device=torch.device("cpu"),
        fp16=False,
        resolution=576,
        variant="medium",
    )

    assert runner.input_names == ["input"]
    assert runner.input_dtypes == {"input": torch.float32}
    assert runner.output_names == ["dets", "labels", "masks"]
    assert runner.outputs["dets"].ndim == 3
    assert runner.outputs["dets"].shape[-1] == 4
    assert runner.outputs["labels"].ndim == 3
    assert runner.outputs["masks"].ndim == 4
    # num_classes derived from class_embed rows - 1 (rfdetr adds a reserve slot).
    assert runner._wrapper.kwargs["num_classes"] == 2

    out = runner.infer({"input": torch.zeros(2, 3, 576, 576)})
    assert set(out) == {"dets", "labels", "masks"}
    assert out["dets"].shape == (2, 5, 4)
    assert out["labels"].shape == (2, 5, 3)
    assert out["masks"].shape == (2, 5, 8, 8)

    runner.close()
    with pytest.raises(RuntimeError, match="closed"):
        runner.infer({"input": torch.zeros(2, 3, 576, 576)})


def test_rfdetr_torch_runner_rejects_unknown_variant(monkeypatch, tmp_path) -> None:
    import jasna.mosaic.rfdetr_torch_runner as module

    weights = tmp_path / "rfdetr-v6.pt"
    weights.write_bytes(b"pt")
    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace())

    with pytest.raises(RuntimeError, match="unsupported variant"):
        module.RfDetrTorchRunner(
            weights,
            input_shapes=[(1, 3, 576, 576)],
            device=torch.device("cpu"),
            fp16=False,
            resolution=576,
            variant="mystery",
        )


def test_amd_rfdetr_needs_no_detection_engine(monkeypatch) -> None:
    import jasna.accelerator as accelerator
    import jasna.engine_compiler as compiler

    monkeypatch.setattr(accelerator, "is_amd_device", lambda _device=None: True)

    assert compiler._detection_engine_exists(
        "rfdetr-v6",
        "rfdetr-v6.pt",
        batch_size=4,
        fp16=True,
        device="cpu",
    )
