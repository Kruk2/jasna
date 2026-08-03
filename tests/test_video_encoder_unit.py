"""Unit tests for NvidiaVideoEncoder internals (options, color guard, buffer, worker, audio pump)."""
from __future__ import annotations

import queue
import threading
from contextlib import nullcontext
from collections import deque
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

import jasna.media.video_encoder as video_encoder_module
from jasna.media import VideoMetadata
from jasna.media.video_encoder import (
    DEFAULT_AV1_ENCODER_OPTIONS,
    DEFAULT_ENCODER_OPTIONS,
    DEFAULT_H264_ENCODER_OPTIONS,
    ENCODER_SPECS,
    _CODEC_MAP,
    _align_yuv_pitch,
    _mov_container_options,
    NvidiaVideoEncoder,
)


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


def _make_encoder(tmp_path, encoder_settings=None, codec="hevc", **meta_overrides) -> NvidiaVideoEncoder:
    return NvidiaVideoEncoder(
        file=str(tmp_path / "result.mkv"),
        device=torch.device("cuda:0"),
        metadata=_fake_metadata(**meta_overrides),
        codec=codec,
        encoder_settings=encoder_settings or {},
    )


# Measured hevc_nvenc configuration; the HEVC path must never drift from it.
_HEVC_OPTIONS_SNAPSHOT = {
    "preset": "p5",
    "tune": "hq",
    "profile": "main10",
    "rc": "vbr",
    "cq": "28",
    "qmin": "17",
    "qmax": "34",
    "nonref_p": "1",
    "g": "250",
    "temporal-aq": "1",
    "rc-lookahead": "32",
    "lookahead_level": "1",
    "spatial_aq": "1",
    "aq-strength": "8",
    "init_qpI": "17",
    "init_qpP": "17",
    "init_qpB": "17",
    "bf": "4",
    "b_ref_mode": "middle",
}


class TestCodecSpecs:
    def test_public_to_ffmpeg_codec_mapping(self):
        assert _CODEC_MAP == {"hevc": "hevc_nvenc", "h264": "h264_nvenc", "av1": "av1_nvenc"}

    def test_hevc_defaults_snapshot_unchanged(self):
        assert DEFAULT_ENCODER_OPTIONS == _HEVC_OPTIONS_SNAPSHOT
        assert dict(ENCODER_SPECS["hevc"].default_options) == _HEVC_OPTIONS_SNAPSHOT

    def test_h264_defaults_snapshot(self):
        expected = dict(_HEVC_OPTIONS_SNAPSHOT)
        expected["profile"] = "high"
        expected["cq"] = "27"
        del expected["lookahead_level"]
        assert DEFAULT_H264_ENCODER_OPTIONS == expected

    def test_av1_defaults_snapshot(self):
        expected = dict(_HEVC_OPTIONS_SNAPSHOT)
        del expected["profile"]
        expected["cq"] = "35"
        del expected["qmin"]
        del expected["qmax"]
        del expected["spatial_aq"]
        del expected["init_qpI"]
        del expected["init_qpP"]
        del expected["init_qpB"]
        expected["spatial-aq"] = "1"
        assert DEFAULT_AV1_ENCODER_OPTIONS == expected

    def test_av1_does_not_reuse_hevc_qp_scale(self):
        assert DEFAULT_AV1_ENCODER_OPTIONS["cq"] == "35"
        assert not {
            "qmin",
            "qmax",
            "init_qpI",
            "init_qpP",
            "init_qpB",
        } & DEFAULT_AV1_ENCODER_OPTIONS.keys()

    def test_frame_formats_and_bit_depth(self):
        assert ENCODER_SPECS["hevc"].frame_format == "p010le"
        assert ENCODER_SPECS["hevc"].ten_bit is True
        assert ENCODER_SPECS["h264"].frame_format == "nv12"
        assert ENCODER_SPECS["h264"].ten_bit is False
        assert ENCODER_SPECS["av1"].frame_format == "p010le"
        assert ENCODER_SPECS["av1"].ten_bit is True


class TestContainerOptions:
    @pytest.mark.parametrize("suffix", [".mp4", ".MP4", ".mov"])
    def test_faststart_by_default(self, suffix):
        assert _mov_container_options(suffix, fmp4=False) == {"movflags": "+faststart"}

    @pytest.mark.parametrize("suffix", [".mp4", ".MP4", ".mov"])
    def test_fragmented_replaces_faststart(self, suffix):
        assert _mov_container_options(suffix, fmp4=True) == {
            "movflags": "+frag_keyframe+empty_moov"
        }

    @pytest.mark.parametrize("suffix", [".mkv", ".nut", ".ts"])
    def test_other_containers_get_no_movflags(self, suffix):
        assert _mov_container_options(suffix, fmp4=False) == {}
        assert _mov_container_options(suffix, fmp4=True) == {}

    def test_fmp4_defaults_off_and_leaves_encoder_options_alone(self, tmp_path):
        default = _make_encoder(tmp_path)
        fragmented = NvidiaVideoEncoder(
            file=str(tmp_path / "result.mp4"),
            device=torch.device("cuda:0"),
            metadata=_fake_metadata(),
            codec="hevc",
            encoder_settings={},
            fmp4=True,
        )
        assert default.fmp4 is False
        assert fragmented.fmp4 is True
        assert fragmented.encoder_options == default.encoder_options


class TestEncoderOptions:
    def test_smart_fragment_preserves_normal_closed_gop_settings(self, tmp_path):
        enc = NvidiaVideoEncoder(
            file=str(tmp_path / "part.nut"),
            device=torch.device("cuda:0"),
            metadata=_fake_metadata(),
            codec="hevc",
            encoder_settings={},
            smart_fragment=True,
            mux_audio=False,
        )
        assert enc.encoder_options["g"] == DEFAULT_ENCODER_OPTIONS["g"]
        assert enc.encoder_options["bf"] == DEFAULT_ENCODER_OPTIONS["bf"]
        assert enc.encoder_options["forced-idr"] == "1"
        assert enc.encoder_options["b_ref_mode"] == DEFAULT_ENCODER_OPTIONS["b_ref_mode"]
        assert enc.mux_audio is False

    def test_smart_fragment_preserves_custom_gop_size(self, tmp_path):
        enc = NvidiaVideoEncoder(
            file=str(tmp_path / "part.nut"),
            device=torch.device("cuda:0"),
            metadata=_fake_metadata(),
            codec="hevc",
            encoder_settings={"g": "180"},
            smart_fragment=True,
            mux_audio=False,
        )

        assert enc.encoder_options["g"] == "180"

    @pytest.mark.parametrize("codec", ["hevc", "av1"])
    def test_smart_fragment_can_match_eight_bit_source(self, tmp_path, codec):
        enc = NvidiaVideoEncoder(
            file=str(tmp_path / "part.nut"),
            device=torch.device("cuda:0"),
            metadata=_fake_metadata(is_10bit=False),
            codec=codec,
            encoder_settings={},
            match_input_bit_depth=True,
        )
        assert enc.spec.frame_format == "nv12"
        assert enc.spec.ten_bit is False
        if codec == "hevc":
            assert enc.encoder_options["profile"] == "main"

    def test_output_fps_defaults_to_source_rate(self, tmp_path):
        enc = _make_encoder(tmp_path)
        assert enc.output_fps == Fraction(24, 1)

    def test_output_fps_can_override_source_rate(self, tmp_path):
        enc = NvidiaVideoEncoder(
            file=str(tmp_path / "result.mkv"),
            device=torch.device("cuda:0"),
            metadata=_fake_metadata(video_fps_exact=Fraction(60_000, 1_001)),
            codec="hevc",
            encoder_settings={},
            output_fps=Fraction(30_000, 1_001),
        )
        assert enc.output_fps == Fraction(30_000, 1_001)

    def test_defaults_used_when_no_settings(self, tmp_path):
        enc = _make_encoder(tmp_path)
        assert enc.encoder_options == DEFAULT_ENCODER_OPTIONS

    @pytest.mark.parametrize(
        ("codec", "defaults"),
        [
            ("hevc", DEFAULT_ENCODER_OPTIONS),
            ("h264", DEFAULT_H264_ENCODER_OPTIONS),
            ("av1", DEFAULT_AV1_ENCODER_OPTIONS),
        ],
    )
    def test_defaults_per_codec(self, tmp_path, codec, defaults):
        enc = _make_encoder(tmp_path, codec=codec)
        assert enc.encoder_options == defaults
        assert enc.encoder_name == _CODEC_MAP[codec]

    @pytest.mark.parametrize("codec", ["hevc", "h264", "av1"])
    def test_settings_override_and_stringify(self, tmp_path, codec):
        enc = _make_encoder(tmp_path, codec=codec, encoder_settings={"cq": 22, "temporal-aq": False, "maxrate": "10M"})
        assert enc.encoder_options["cq"] == "22"
        assert enc.encoder_options["temporal-aq"] == "0"
        assert enc.encoder_options["maxrate"] == "10M"
        assert enc.encoder_options["preset"] == "p5"

    def test_source_bitrate_adds_ceiling(self, tmp_path):
        enc = _make_encoder(tmp_path, codec_name="hevc", video_bitrate=20_000_000)
        assert enc.encoder_options["maxrate"] == "25000000"
        assert enc.encoder_options["bufsize"] == "50000000"

    def test_source_bitrate_ceiling_is_tighter_for_non_hevc(self, tmp_path):
        enc = _make_encoder(tmp_path, codec_name="h264", video_bitrate=20_000_000)
        assert enc.encoder_options["maxrate"] == "20000000"
        assert enc.encoder_options["bufsize"] == "40000000"

    def test_no_ceiling_without_source_bitrate(self, tmp_path):
        enc = _make_encoder(tmp_path, video_bitrate=0)
        assert "maxrate" not in enc.encoder_options
        assert "bufsize" not in enc.encoder_options

    def test_explicit_maxrate_replaces_derived_ceiling(self, tmp_path):
        enc = _make_encoder(
            tmp_path,
            encoder_settings={"maxrate": "3000000"},
            codec_name="hevc",
            video_bitrate=20_000_000,
        )
        assert enc.encoder_options["maxrate"] == "3000000"
        # The derived buffer must not survive a user-chosen ceiling.
        assert "bufsize" not in enc.encoder_options

    def test_unsupported_codec_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported codec"):
            NvidiaVideoEncoder(
                file=str(tmp_path / "o.mkv"),
                device=torch.device("cuda:0"),
                metadata=_fake_metadata(),
                codec="vp9",
                encoder_settings={},
            )

    def test_codec_specific_settings_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="for codec av1.*profile"):
            _make_encoder(tmp_path, codec="av1", encoder_settings={"profile": "main"})
        with pytest.raises(ValueError, match="for codec av1.*spatial_aq"):
            _make_encoder(tmp_path, codec="av1", encoder_settings={"spatial_aq": 1})
        with pytest.raises(ValueError, match="for codec h264.*tier"):
            _make_encoder(tmp_path, codec="h264", encoder_settings={"tier": "high"})

    def test_h264_lookahead_level_override_is_dropped(self, tmp_path):
        enc = _make_encoder(tmp_path, codec="h264", encoder_settings={"lookahead_level": 1})
        assert "lookahead_level" not in enc.encoder_options

    @pytest.mark.parametrize("codec", ["hevc", "h264"])
    def test_hyphenated_spatial_aq_replaces_underscore_default(self, tmp_path, codec):
        enc = _make_encoder(tmp_path, codec=codec, encoder_settings={"spatial-aq": 0})
        assert enc.encoder_options["spatial_aq"] == "0"
        assert "spatial-aq" not in enc.encoder_options

    def test_leftover_options_raise(self, tmp_path):
        enc = _make_encoder(tmp_path)
        enc.out_stream = MagicMock()
        enc.out_stream.codec_context.options = {"bogus": "1"}
        with pytest.raises(ValueError, match="did not accept encoder option.*bogus"):
            enc._validate_encoder_options()

    def test_no_leftover_options_pass(self, tmp_path):
        enc = _make_encoder(tmp_path)
        enc.out_stream = MagicMock()
        enc.out_stream.codec_context.options = {}
        enc._options_validated = False
        enc._validate_encoder_options()
        assert enc._options_validated


class _StopEncode(Exception):
    """Cuts _encode_frame short once the encoder input has been captured."""


class TestSharpening:
    def test_no_sharpener_by_default(self, tmp_path):
        assert _make_encoder(tmp_path)._cas is None

    def test_sharpener_built_when_requested(self, tmp_path):
        enc = NvidiaVideoEncoder(
            file=str(tmp_path / "result.mkv"),
            device=torch.device("cuda:0"),
            metadata=_fake_metadata(),
            codec="hevc",
            encoder_settings={},
            sharpen_strength=0.4,
        )
        assert enc._cas is not None
        assert enc._cas.weight_scale == pytest.approx(-1.0 / (16.0 - 12.0 * 0.4))

    def test_sharpener_follows_the_encoder_bit_depth(self, tmp_path):
        ten_bit = NvidiaVideoEncoder(
            file=str(tmp_path / "a.mkv"),
            device=torch.device("cuda:0"),
            metadata=_fake_metadata(),
            codec="hevc",
            encoder_settings={},
            sharpen_strength=0.4,
        )
        eight_bit = NvidiaVideoEncoder(
            file=str(tmp_path / "b.nut"),
            device=torch.device("cuda:0"),
            metadata=_fake_metadata(is_10bit=False),
            codec="hevc",
            encoder_settings={},
            sharpen_strength=0.4,
            match_input_bit_depth=True,
            smart_fragment=True,
            mux_audio=False,
        )
        assert ten_bit._cas.ten_bit is True
        assert ten_bit._cas.peak == 1023.0
        assert eight_bit._cas.ten_bit is False
        assert eight_bit._cas.peak == 255.0

    def test_zero_strength_leaves_the_converted_frame_untouched(self, tmp_path, monkeypatch):
        enc = _make_encoder(tmp_path, codec="h264")  # nv12, so planes stay uint8
        packed = torch.arange(24, dtype=torch.uint8, device=enc.device).reshape(6, 4)
        enc._converter = SimpleNamespace(
            sample_dtype=torch.uint8,
            uses_kernel=False,
            convert_into=lambda frame, luma, chroma: (
                luma.copy_(packed[:4]),
                chroma.copy_(packed[4:]),
            ),
        )
        enc.stream = SimpleNamespace(cuda_stream=1234, synchronize=lambda: None)
        enc.metadata = _fake_metadata(video_height=4, video_width=4)
        enc._cuda_ctx = None
        seen = []

        def capture(planes, **kwargs):
            seen.append(planes)
            raise _StopEncode

        monkeypatch.setattr(
            video_encoder_module, "stream_context", lambda _s: nullcontext()
        )
        monkeypatch.setattr(video_encoder_module, "_align_yuv_pitch", lambda p: p)
        monkeypatch.setattr(video_encoder_module.av.VideoFrame, "from_dlpack", capture)

        with pytest.raises(_StopEncode):
            enc._encode_frame(torch.zeros(3, 4, 4), pts=0)

        assert torch.equal(torch.cat(seen[0]), packed)

    def test_sharpening_runs_before_pitch_alignment(self, tmp_path, monkeypatch):
        enc = NvidiaVideoEncoder(
            file=str(tmp_path / "result.mkv"),
            device=torch.device("cuda:0"),
            metadata=_fake_metadata(),
            codec="hevc",
            encoder_settings={},
            sharpen_strength=0.4,
        )
        enc._converter = SimpleNamespace(
            sample_dtype=torch.uint8,
            uses_kernel=False,
            convert_into=lambda frame, luma, chroma: None,
        )
        enc.stream = SimpleNamespace(cuda_stream=1234, synchronize=lambda: None)
        enc.metadata = _fake_metadata(video_height=4, video_width=4)
        enc._cuda_ctx = None
        order = []
        enc._cas = SimpleNamespace(
            sharpen_into=lambda source, destination: order.append(
                ("sharpen", destination.is_contiguous())
            )
        )

        def align(packed):
            order.append(("align", True))
            return packed

        def stop(*args, **kwargs):
            raise _StopEncode

        monkeypatch.setattr(
            video_encoder_module, "stream_context", lambda _s: nullcontext()
        )
        monkeypatch.setattr(video_encoder_module, "_align_yuv_pitch", align)
        monkeypatch.setattr(video_encoder_module.av.VideoFrame, "from_dlpack", stop)

        with pytest.raises(_StopEncode):
            enc._encode_frame(torch.zeros(3, 4, 4), pts=0)

        assert [step for step, _ in order] == ["sharpen", "align"]
        assert order[0][1] is True  # sharpening sees a contiguous plane


class TestColorHandling:
    def test_unsupported_color_range_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported color space or color range"):
            _make_encoder(tmp_path, color_range=AvColorRange.UNSPECIFIED)

    @pytest.mark.parametrize(
        ("color_space", "color_range", "expected"),
        [
            (AvColorspace.ITU709, AvColorRange.MPEG, "p010_bt709_limited"),
            (AvColorspace.ITU709, AvColorRange.JPEG, "p010_bt709_full"),
            (AvColorspace.ITU601, AvColorRange.MPEG, "p010_bt601_limited"),
            (AvColorspace.ITU601, AvColorRange.JPEG, "p010_bt601_full"),
            (AvColorspace.BT2020, AvColorRange.MPEG, "p010_bt2020_limited"),
            (AvColorspace.BT2020, AvColorRange.JPEG, "p010_bt2020_full"),
        ],
    )
    @pytest.mark.parametrize("codec", ["hevc", "av1"])
    def test_selects_p010_converter_for_hevc_and_av1(self, tmp_path, codec, color_space, color_range, expected):
        enc = _make_encoder(tmp_path, codec=codec, color_space=color_space, color_range=color_range)
        assert enc._converter.variant == expected

    @pytest.mark.parametrize(
        ("color_space", "color_range", "expected"),
        [
            (AvColorspace.ITU709, AvColorRange.MPEG, "nv12_bt709_limited"),
            (AvColorspace.ITU709, AvColorRange.JPEG, "nv12_bt709_full"),
            (AvColorspace.ITU601, AvColorRange.MPEG, "nv12_bt601_limited"),
            (AvColorspace.ITU601, AvColorRange.JPEG, "nv12_bt601_full"),
            (AvColorspace.BT2020, AvColorRange.MPEG, "nv12_bt2020_limited"),
            (AvColorspace.BT2020, AvColorRange.JPEG, "nv12_bt2020_full"),
        ],
    )
    def test_selects_nv12_converter_for_h264(self, tmp_path, color_space, color_range, expected):
        enc = _make_encoder(tmp_path, codec="h264", color_space=color_space, color_range=color_range)
        assert enc._converter.variant == expected

    @pytest.mark.parametrize("codec", ["h264", "av1"])
    def test_unsupported_color_range_raises_for_new_codecs(self, tmp_path, codec):
        with pytest.raises(ValueError, match="Unsupported color space or color range"):
            _make_encoder(tmp_path, codec=codec, color_range=AvColorRange.UNSPECIFIED)


def _buffered_encoder(tmp_path) -> NvidiaVideoEncoder:
    enc = _make_encoder(tmp_path)
    enc.pts_heap = []
    enc.frame_buffer = deque()
    enc.pts_set = set()
    enc._last_emitted_pts = None
    enc._worker_error = None
    enc._encode_queue = MagicMock()
    enc._build_encode_item = MagicMock(side_effect=lambda frame, pts: (frame, pts, None))
    return enc


class TestEncodeBuffer:
    @pytest.mark.parametrize(
        ("dtype", "width", "expected_pitch"),
        [
            (torch.uint8, 852, 864),
            (torch.uint8, 854, 864),
            (torch.uint8, 856, 864),
            (torch.uint8, 860, 864),
            (torch.uint8, 864, 864),
            (torch.int16, 852, 856),
            (torch.int16, 854, 856),
            (torch.int16, 856, 856),
            (torch.int16, 860, 864),
            (torch.int16, 864, 864),
        ],
    )
    def test_yuv_pitch_is_aligned_without_changing_visible_data(
        self,
        dtype,
        width,
        expected_pitch,
    ):
        height = 4
        packed = torch.arange(height * 3 // 2 * width, dtype=dtype).reshape(
            height * 3 // 2,
            width,
        )

        aligned = _align_yuv_pitch(packed)

        assert aligned.shape == packed.shape
        assert torch.equal(aligned, packed)
        assert aligned.stride() == (expected_pitch, 1)
        assert aligned.stride(0) * aligned.element_size() % 16 == 0
        assert aligned[height:].data_ptr() - aligned.data_ptr() == (
            height * aligned.stride(0) * aligned.element_size()
        )
        if packed.stride(0) * packed.element_size() % 16 == 0:
            assert aligned.data_ptr() == packed.data_ptr()
        else:
            assert aligned.data_ptr() != packed.data_ptr()

    def test_from_dlpack_uses_shared_cuda_stream_without_host_sync(
        self, tmp_path, monkeypatch
    ):
        enc = _make_encoder(tmp_path, codec="hevc", video_width=2, video_height=2)
        # Force NVIDIA contracts even when the host GPU is AMD/ROCm.
        enc.vendor = video_encoder_module.AcceleratorVendor.NVIDIA
        enc.stream = MagicMock()
        enc.stream.cuda_stream = 1234
        enc._cuda_ctx = object()
        enc._lut_applier = None
        enc._converter = SimpleNamespace(
            sample_dtype=torch.int16,
            uses_kernel=False,
            convert_into=lambda frame, luma, chroma: None,
        )
        enc.out_stream = MagicMock()
        enc.out_stream.encode.return_value = []
        hw_frame = SimpleNamespace(pts=None, time_base=None)
        from_dlpack = MagicMock(return_value=hw_frame)
        monkeypatch.setattr(
            video_encoder_module.av,
            "VideoFrame",
            SimpleNamespace(from_dlpack=from_dlpack),
        )
        monkeypatch.setattr(
            video_encoder_module, "stream_context", lambda _stream: nullcontext()
        )
        monkeypatch.setattr(video_encoder_module, "_align_yuv_pitch", lambda p: p)

        enc._encode_frame(torch.zeros((3, 2, 2), dtype=torch.uint8), 7)

        _, kwargs = from_dlpack.call_args
        assert kwargs == {
            "format": "p010le",
            "stream": 1234,
            "cuda_context": enc._cuda_ctx,
        }
        enc.stream.synchronize.assert_not_called()

    def test_amd_host_transfer_still_synchronizes_before_from_dlpack(
        self, tmp_path, monkeypatch
    ):
        enc = _make_encoder(tmp_path, codec="h264", video_width=2, video_height=2)
        enc.vendor = video_encoder_module.AcceleratorVendor.AMD
        enc.stream = MagicMock()
        order: list = []
        enc.stream.synchronize = MagicMock(side_effect=lambda: order.append("sync"))
        enc._host_yuv = MagicMock(name="host_yuv")

        def _track_copy(*args, **kwargs):
            order.append(("copy", kwargs.get("non_blocking")))

        enc._host_yuv.copy_ = MagicMock(side_effect=_track_copy)
        enc._packed = torch.zeros((3, 2), dtype=torch.uint8)
        enc._lut_applier = None
        enc._converter = SimpleNamespace(
            sample_dtype=torch.uint8,
            uses_kernel=False,
            convert_into=lambda frame, luma, chroma: None,
        )
        enc.out_stream = MagicMock()
        enc.out_stream.encode.return_value = []
        hw_frame = SimpleNamespace(pts=None, time_base=None)
        from_dlpack = MagicMock(return_value=hw_frame)
        monkeypatch.setattr(
            video_encoder_module.av,
            "VideoFrame",
            SimpleNamespace(from_dlpack=from_dlpack),
        )
        monkeypatch.setattr(
            video_encoder_module, "stream_context", lambda _stream: nullcontext()
        )

        enc._encode_frame(torch.zeros((3, 2, 2), dtype=torch.uint8), 7)

        enc.stream.synchronize.assert_called_once_with()
        enc._host_yuv.copy_.assert_called_once()
        _, copy_kwargs = enc._host_yuv.copy_.call_args
        assert copy_kwargs.get("non_blocking") is False
        # Finish device work before filling host planes AMF will read (issue #252).
        assert order == ["sync", ("copy", False)]
        _, kwargs = from_dlpack.call_args
        assert kwargs == {"format": "nv12"}

    def test_pts_origin_is_removed_from_fragment_timestamps(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        enc.pts_origin = 100
        enc.encode("frame", 110)
        assert enc.pts_heap == [10]

    def test_bridge_frame_records_lut_bypass(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        enc.encode("frame", 10, apply_lut=False)
        assert list(enc.frame_buffer) == ["frame"]
        assert list(enc._lut_flags) == [False]

    def test_encode_pushes_to_buffer_and_heap(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        enc.encode("frame0", 10)
        assert list(enc.frame_buffer) == ["frame0"]
        assert enc.pts_heap == [10]
        assert enc.pts_set == {10}

    def test_encode_dedup_pts(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        enc.encode("a", 5)
        enc.encode("b", 5)
        assert sorted(enc.pts_set) == [5, 6]

    def test_flush_starts_above_half_buffer(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        for i in range(enc.BUFFER_MAX_SIZE // 2):
            enc.encode(f"f{i}", i)
        enc._encode_queue.put.assert_not_called()
        enc.encode("one-more", 99)
        enc._encode_queue.put.assert_called_once_with(("f0", 0, None))

    def test_smallest_pts_pairs_with_oldest_frame(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        for i, pts in enumerate([30, 10, 20, 40]):
            enc.encode(f"f{i}", pts)
        enc._process_buffer(flush_all=True)
        enc._encode_queue.put.assert_called_once_with(("f0", 10, None))

    def test_emitted_pts_stay_strictly_increasing_on_scrambled_source(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        scrambled = [
            1761760, 1801800, 1841840, 1881880, 1801801, 1801803, 1801804,
            1801805, 1801802, 1801807, 1801808, 1801809, 1801806, 1801811,
            1801812, 1801813, 1801810,
        ]
        for i, pts in enumerate(scrambled):
            enc.encode(f"f{i}", pts)
        while enc.frame_buffer:
            enc._process_buffer(flush_all=True)
        emitted = [call.args[0][1] for call in enc._encode_queue.put.call_args_list]
        assert len(emitted) == len(scrambled)
        assert all(b > a for a, b in zip(emitted, emitted[1:]))

    def test_in_order_pts_pass_through_unchanged(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        ordered = [1000 * i for i in range(10)]
        for i, pts in enumerate(ordered):
            enc.encode(f"f{i}", pts)
        while enc.frame_buffer:
            enc._process_buffer(flush_all=True)
        emitted = [call.args[0][1] for call in enc._encode_queue.put.call_args_list]
        assert emitted == ordered

    def test_encode_raises_pending_worker_error(self, tmp_path):
        enc = _buffered_encoder(tmp_path)
        enc._worker_error = RuntimeError("nvenc exploded")
        with pytest.raises(RuntimeError, match="nvenc exploded"):
            enc.encode("frame", 0)


class TestWorkerErrorChannel:
    def test_worker_records_error_and_keeps_consuming(self, tmp_path):
        enc = _make_encoder(tmp_path)
        enc.device = torch.device("cpu")
        enc._worker_error = None
        enc._encode_queue = queue.Queue()
        enc._stop_sentinel = object()
        enc._handle_encode_item = MagicMock(side_effect=RuntimeError("mux failed"))

        worker = threading.Thread(target=enc._encode_worker, daemon=True)
        worker.start()
        enc._encode_queue.put(("frame", 0, None))
        enc._encode_queue.put(("frame", 1, None))
        enc._encode_queue.join()
        enc._encode_queue.put(enc._stop_sentinel)
        worker.join(timeout=5)

        assert not worker.is_alive()
        assert isinstance(enc._worker_error, RuntimeError)
        assert enc._handle_encode_item.call_count == 1


def _packet(stream_index, dts, time_base=Fraction(1, 1000)):
    return SimpleNamespace(
        stream=SimpleNamespace(index=stream_index),
        dts=dts,
        pts=dts,
        time_base=time_base,
    )


class TestAudioPump:
    def _audio_encoder(self, tmp_path, packets):
        enc = _make_encoder(tmp_path)
        out_a = MagicMock()
        enc._audio_pipes = {1: ("copy", out_a, None)}
        enc._audio_backlog = deque()
        enc._audio_iter = iter(packets)
        enc.dst = MagicMock()
        return enc, out_a

    def test_copy_reassigns_stream_and_skips_flush_packet(self, tmp_path):
        enc, out_a = self._audio_encoder(tmp_path, [])
        pkt = _packet(1, dts=0)
        assert enc._produce_audio_packets(pkt) == [pkt]
        assert pkt.stream is out_a

        flush = SimpleNamespace(stream=SimpleNamespace(index=1), dts=None, pts=None, time_base=None)
        assert enc._produce_audio_packets(flush) == []

    def test_pump_respects_threshold(self, tmp_path):
        packets = [_packet(1, dts=0), _packet(1, dts=500), _packet(1, dts=1500)]
        enc, _ = self._audio_encoder(tmp_path, packets)

        enc._pump_audio(1.0)
        assert enc.dst.mux.call_count == 2
        assert len(enc._audio_backlog) == 1  # dts=1500 held back

        enc._pump_audio(None)
        assert enc.dst.mux.call_count == 3

    def test_pump_without_audio_is_noop(self, tmp_path):
        enc = _make_encoder(tmp_path)
        enc._audio_iter = None
        enc._pump_audio(1.0)


class TestDropUnsupportedNvencOverrides:
    def _drop(self, codec, overrides, defaults=None):
        video_encoder_module._drop_unsupported_nvenc_overrides(
            codec, overrides, defaults if defaults is not None else {"bf": "4"}
        )
        return overrides

    def test_h264_lookahead_level_dropped(self):
        assert self._drop("h264", {"lookahead_level": "1", "cq": "22"}) == {"cq": "22"}

    def test_hevc_and_av1_keep_lookahead_level(self):
        assert self._drop("hevc", {"lookahead_level": "2"}) == {"lookahead_level": "2"}
        assert self._drop("av1", {"lookahead_level": "2"}) == {"lookahead_level": "2"}

    def test_weighted_pred_dropped_with_default_b_frames(self):
        assert self._drop("h264", {"weighted_pred": "1"}) == {}
        assert self._drop("hevc", {"weighted_pred": "1"}) == {}

    def test_weighted_pred_kept_when_b_frames_disabled(self):
        assert self._drop("h264", {"weighted_pred": "1", "bf": "0"}) == {
            "weighted_pred": "1",
            "bf": "0",
        }

    def test_weighted_pred_zero_untouched(self):
        assert self._drop("h264", {"weighted_pred": "0"}) == {"weighted_pred": "0"}

    def test_av1_weighted_pred_always_dropped(self):
        assert self._drop("av1", {"weighted_pred": "1", "bf": "0"}) == {"bf": "0"}
