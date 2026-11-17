"""Unit tests for VideoEncoder internals (options, color guard, buffer, worker, audio pump)."""
from __future__ import annotations

import logging
import subprocess
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock

import av
import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

import jasna.media.encoder_options as encoder_options
from jasna.media.encoder_options import (
    DEFAULT_AV1_ENCODER_OPTIONS,
    DEFAULT_ENCODER_OPTIONS,
    DEFAULT_H264_ENCODER_OPTIONS,
    ENCODER_SPECS,
)
from jasna.media.probe import VideoMetadata
from jasna.os_utils import resolve_executable
from jasna.media.source_streams import SourceStreamCopier, _normalized_audio_layout
from jasna.media.video_encoder import (
    ENCODE_BUFFER_SIZE,
    EncodeOrder,
    _mov_container_options,
    VideoEncoder,
    nvenc_yuv_frame,
)

from factories import video_metadata, write_double_adts_aac_source, write_two_block_wma_source

pytestmark = pytest.mark.usefixtures("nvidia_vendor")

PROCESSED_AT = "2026-11-15T18:00:00.000000Z"


def _fake_metadata(**overrides) -> VideoMetadata:
    values = dict(video_file="fake_input.mkv", video_width=1920, video_height=1080, is_10bit=True)
    values.update(overrides)
    return video_metadata(fps=24, num_frames=100, **values)


def _make_encoder(tmp_path, encoder_settings=None, codec="hevc", **meta_overrides) -> VideoEncoder:
    return VideoEncoder(
        file=str(tmp_path / "result.mkv"),
        device=torch.device("cuda:0"),
        metadata=_fake_metadata(**meta_overrides),
        codec=codec,
        encoder_settings=encoder_settings or {},
    )


class TestContainerOptions:
    @pytest.mark.parametrize("suffix", [".mp4", ".MP4", ".mov"])
    def test_faststart_by_default(self, suffix):
        assert _mov_container_options(suffix, fmp4=False) == {"movflags": "+faststart"}

    @pytest.mark.parametrize("suffix", [".mp4", ".MP4", ".mov"])
    def test_fragmented_replaces_faststart(self, suffix):
        assert _mov_container_options(suffix, fmp4=True) == {
            "movflags": "+frag_keyframe+empty_moov+delay_moov"
        }

    @pytest.mark.parametrize("suffix", [".mkv", ".nut", ".ts"])
    def test_other_containers_get_no_movflags(self, suffix):
        assert _mov_container_options(suffix, fmp4=False) == {}
        assert _mov_container_options(suffix, fmp4=True) == {}

    def test_fmp4_leaves_encoder_options_alone(self, tmp_path):
        fragmented = VideoEncoder(
            file=str(tmp_path / "result.mp4"),
            device=torch.device("cuda:0"),
            metadata=_fake_metadata(),
            codec="hevc",
            encoder_settings={},
            fmp4=True,
        )
        assert fragmented.encoder_options == _make_encoder(tmp_path).encoder_options


@pytest.mark.parametrize(
    ("count_only", "expected"),
    [("1 channels", "mono"), ("2 channels", "stereo"), ("6 channels", "5.1"), ("8 channels", "7.1")],
)
def test_normalizes_count_only_layout_to_default_layout(count_only, expected):
    normalized = _normalized_audio_layout(av.AudioLayout(count_only))

    assert normalized.name == expected


@pytest.mark.parametrize("name", ["stereo", "5.1", "9 channels"])
def test_preserves_named_audio_layout(name):
    layout = av.AudioLayout(name)

    assert _normalized_audio_layout(layout) is layout


def _copier(src, dst, *, output_path=Path("result.mkv"), smart_fragment=False) -> SourceStreamCopier:
    return SourceStreamCopier(
        src, dst, output_path, video_index=0, creation_time=PROCESSED_AT, comment=None, smart_fragment=smart_fragment
    )


def _write_srt_subtitle_source(tmp_path: Path, cue_time: str) -> Path:
    """MKV with a short test video and one SRT cue shown at ``cue_time``."""
    subtitle = tmp_path / "cue.srt"
    subtitle.write_text(f"1\n{cue_time}\nCue\n", encoding="utf-8")
    source = tmp_path / "subtitled.mkv"
    subprocess.run(
        [
            resolve_executable("ffmpeg"), "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=12:duration=2",
            "-i", str(subtitle),
            "-c:v", "libx264", "-c:s", "srt", str(source),
        ],
        check=True,
    )
    return source


class TestEncoderOptions:
    def test_smart_fragment_preserves_normal_closed_gop_settings(self, tmp_path):
        enc = VideoEncoder(
            file=str(tmp_path / "part.nut"),
            device=torch.device("cuda:0"),
            metadata=_fake_metadata(),
            codec="hevc",
            encoder_settings={},
            smart_fragment=True,
        )
        assert enc.encoder_options["g"] == DEFAULT_ENCODER_OPTIONS["g"]
        assert enc.encoder_options["bf"] == DEFAULT_ENCODER_OPTIONS["bf"]
        assert enc.encoder_options["forced-idr"] == "1"
        assert enc.encoder_options["b_ref_mode"] == DEFAULT_ENCODER_OPTIONS["b_ref_mode"]

    def test_smart_fragment_preserves_custom_gop_size(self, tmp_path):
        enc = VideoEncoder(
            file=str(tmp_path / "part.nut"),
            device=torch.device("cuda:0"),
            metadata=_fake_metadata(),
            codec="hevc",
            encoder_settings={"g": "180"},
            smart_fragment=True,
        )

        assert enc.encoder_options["g"] == "180"

    def test_output_fps_defaults_to_source_rate(self, tmp_path):
        enc = _make_encoder(tmp_path)
        assert enc.output_fps == Fraction(24, 1)

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
        assert enc.encoder_name == ENCODER_SPECS[codec].encoder_name

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

    def test_nvenc_h264_output_gets_two_x_ceiling(self, tmp_path):
        enc = _make_encoder(
            tmp_path,
            codec="h264",
            codec_name="h264",
            video_bitrate=20_000_000,
        )
        assert enc.encoder_options["maxrate"] == "40000000"
        assert enc.encoder_options["bufsize"] == "80000000"

    def test_no_ceiling_without_source_bitrate(self, tmp_path):
        enc = _make_encoder(tmp_path, video_bitrate=0)
        assert "maxrate" not in enc.encoder_options
        assert "bufsize" not in enc.encoder_options

    @pytest.mark.parametrize("codec", ["hevc", "h264"])
    def test_explicit_maxrate_replaces_derived_ceiling(self, tmp_path, codec):
        enc = _make_encoder(
            tmp_path,
            codec=codec,
            encoder_settings={"maxrate": "3000000"},
            codec_name="hevc",
            video_bitrate=20_000_000,
        )
        assert enc.encoder_options["maxrate"] == "3000000"
        # The derived buffer must not survive a user-chosen ceiling.
        assert "bufsize" not in enc.encoder_options

    def test_unsupported_codec_raises(self, tmp_path):
        with pytest.raises(ValueError, match="Unsupported codec"):
            VideoEncoder(
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


class TestSharpening:
    def test_sharpener_follows_the_encoder_bit_depth(self, tmp_path):
        ten_bit = VideoEncoder(
            file=str(tmp_path / "a.mkv"),
            device=torch.device("cuda:0"),
            metadata=_fake_metadata(),
            codec="hevc",
            encoder_settings={},
            sharpen_strength=0.4,
        )
        eight_bit = VideoEncoder(
            file=str(tmp_path / "b.nut"),
            device=torch.device("cuda:0"),
            metadata=_fake_metadata(is_10bit=False),
            codec="hevc",
            encoder_settings={},
            sharpen_strength=0.4,
            smart_fragment=True,
        )
        assert ten_bit._cas.ten_bit is True
        assert ten_bit._cas.peak == 1023.0
        assert eight_bit._cas.ten_bit is False
        assert eight_bit._cas.peak == 255.0

class TestColorHandling:
    @pytest.mark.parametrize("codec", ["hevc", "h264", "av1"])
    def test_unsupported_color_range_raises(self, tmp_path, codec):
        with pytest.raises(ValueError, match="Unsupported color space or color range"):
            _make_encoder(tmp_path, codec=codec, color_range=AvColorRange.UNSPECIFIED)

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


def _emitted(order: EncodeOrder) -> list[tuple[object, int]]:
    emitted = []
    while order:
        emitted.append(order.pop(flush=True))
    return emitted


class TestEncodeOrder:
    @pytest.mark.parametrize(
        "source_pts",
        [[1000 * i for i in range(10)], [46, 79, 113, 179]],
        ids=["in-order", "vc1-wmv-skipped-frames"],
    )
    def test_in_order_pts_pass_through_unchanged(self, source_pts):
        order = EncodeOrder(hold=4, pts_origin=0)
        for i, pts in enumerate(source_pts):
            order.push(f"f{i}", pts)

        assert [pts for _frame, pts in _emitted(order)] == source_pts

    def test_pts_origin_is_removed_from_fragment_timestamps(self):
        order = EncodeOrder(hold=4, pts_origin=100)
        order.push("frame", 110)

        assert order.pop(flush=True) == ("frame", 10)

    def test_repeated_pts_moves_up_by_one(self):
        order = EncodeOrder(hold=4, pts_origin=0)
        order.push("a", 5)
        order.push("b", 5)

        assert _emitted(order) == [("a", 5), ("b", 6)]

    def test_frames_wait_until_more_than_hold_are_buffered(self):
        order = EncodeOrder(hold=ENCODE_BUFFER_SIZE // 2, pts_origin=0)
        for i in range(ENCODE_BUFFER_SIZE // 2):
            order.push(f"f{i}", i)
        assert order.pop(flush=False) is None

        order.push("one-more", 99)

        assert order.pop(flush=False) == ("f0", 0)

    def test_smallest_pts_pairs_with_oldest_frame(self):
        order = EncodeOrder(hold=4, pts_origin=0)
        for i, pts in enumerate([30, 10, 20, 40]):
            order.push(f"f{i}", pts)

        assert order.pop(flush=True) == ("f0", 10)

    def test_emitted_pts_stay_strictly_increasing_on_scrambled_source(self):
        order = EncodeOrder(hold=ENCODE_BUFFER_SIZE // 2, pts_origin=0)
        scrambled = [
            1761760, 1801800, 1841840, 1881880, 1801801, 1801803, 1801804,
            1801805, 1801802, 1801807, 1801808, 1801809, 1801806, 1801811,
            1801812, 1801813, 1801810,
        ]
        emitted = []
        for i, pts in enumerate(scrambled):
            order.push(f"f{i}", pts)
            ready = order.pop(flush=False)
            if ready is not None:
                emitted.append(ready[1])
        emitted += [pts for _frame, pts in _emitted(order)]

        assert len(emitted) == len(scrambled)
        assert all(b > a for a, b in zip(emitted, emitted[1:]))


class TestNvencFrame:
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
    def test_nvenc_frame_rows_start_on_an_aligned_pitch(
        self,
        dtype,
        width,
        expected_pitch,
    ):
        height = 4

        frame = nvenc_yuv_frame(height, width, dtype, torch.device("cpu"))

        assert frame.shape == (height * 3 // 2, width)
        assert frame.dtype == dtype
        assert frame.stride() == (expected_pitch, 1)
        assert frame.stride(0) * frame.element_size() % 16 == 0


def test_encode_worker_failure_is_raised_when_the_output_closes(tmp_path):
    source = write_double_adts_aac_source(tmp_path)
    encoder = VideoEncoder(
        file=str(tmp_path / "result.mkv"),
        device=torch.device("cuda:0"),
        metadata=video_metadata(fps=12, num_frames=12, video_file=str(source), video_width=64, video_height=64),
        codec="hevc",
        encoder_settings={},
    )
    host_frame = torch.zeros((3, 64, 64), dtype=torch.uint8)

    with pytest.raises(NotImplementedError, match="record_stream"):
        with encoder:
            for pts in range(ENCODE_BUFFER_SIZE):
                encoder.encode(host_frame, pts)


class TestSourceStreamPump:
    def test_double_adts_aac_is_filtered_for_mp4_copy(self, tmp_path):
        source = write_double_adts_aac_source(tmp_path)
        destination = tmp_path / "result.mp4"
        with av.open(str(source)) as src, av.open(str(destination), "w") as dst:
            _copier(src, dst, output_path=destination).drain()

        with av.open(str(destination)) as result:
            assert sum(1 for _ in result.decode(audio=0)) > 1

    def test_wma_frames_without_timestamps_transcode_to_mp4(self, tmp_path):
        source = write_two_block_wma_source(tmp_path)
        destination = tmp_path / "result.mp4"
        with av.open(str(source)) as src, av.open(str(destination), "w") as dst:
            _copier(src, dst, output_path=destination).drain()

        with av.open(str(source)) as src:
            source_samples = sum(frame.samples for frame in src.decode(audio=0))
        with av.open(str(destination)) as result:
            assert result.streams.audio[0].codec_context.name == "aac"
            assert sum(frame.samples for frame in result.decode(audio=0)) >= source_samples

    def test_copy_moves_packets_to_the_output_stream_and_skips_flush_packets(self, tmp_path):
        with av.open(str(tmp_path / "audio.mkv"), "w") as dst:
            out_stream = dst.add_stream("aac", rate=44_100)
            packet = av.Packet(b"\x21\x10\x04\x60")

            assert SourceStreamCopier._copy_packets(packet, out_stream, None) == [packet]
            assert packet.stream is out_stream
            assert SourceStreamCopier._copy_packets(av.Packet(), out_stream, None) == []

    def test_pump_muxes_source_packets_up_to_the_video_time(self, tmp_path):
        source = write_double_adts_aac_source(tmp_path)
        destination = tmp_path / "result.mkv"
        with av.open(str(source)) as src, av.open(str(destination), "w") as dst:
            _copier(src, dst, output_path=destination).pump(0.5)

        with av.open(str(destination)) as result:
            times = [float(packet.dts * packet.time_base) for packet in result.demux(audio=0) if packet.size]
        # One-second source; its AAC packets are 23 ms apart.
        assert 0.45 < max(times) <= 0.5

    @staticmethod
    def _mux_aac_packets(tmp_path, dts_values) -> list[av.Packet]:
        """Mux packets with ``dts_values`` (1/44100 s) into one MKV audio stream (1 ms time base)."""
        source = write_double_adts_aac_source(tmp_path)
        packets = []
        with av.open(str(source)) as src, av.open(str(tmp_path / "audio.mkv"), "w") as dst:
            copier = _copier(src, dst, smart_fragment=True)
            stream = dst.add_stream("aac", rate=44_100)
            stream.time_base = Fraction(1, 44_100)
            for dts in dts_values:
                packet = av.Packet(b"\x21\x10\x04\x60")
                packet.stream = stream
                packet.time_base = Fraction(1, 44_100)
                packet.pts = packet.dts = dts
                copier._mux(packet)
                packets.append(packet)
        return packets

    def test_copy_nudges_overlapping_timestamps_in_output_time_base(self, tmp_path):
        packets = self._mux_aac_packets(tmp_path, (0, 1024, 1024, 2048))

        assert [packet.dts for packet in packets] == [0, 23, 24, 46]
        assert [packet.pts for packet in packets] == [0, 23, 24, 46]

    def test_nudge_warns_once_per_stream(self, tmp_path, caplog):
        with caplog.at_level(logging.DEBUG, logger="jasna.media.source_streams"):
            self._mux_aac_packets(tmp_path, (4410, 0, 0, 0))

        nudges = [r for r in caplog.records if "nudging forward" in r.getMessage()]
        assert [r.levelno for r in nudges] == [logging.WARNING, logging.DEBUG, logging.DEBUG]

    def test_text_subtitle_transcode_preserves_cue_timing(self, tmp_path):
        source = _write_srt_subtitle_source(tmp_path, "00:00:00,500 --> 00:00:01,500")
        destination = tmp_path / "result.mp4"
        with av.open(str(source)) as src, av.open(str(destination), "w") as dst:
            _copier(src, dst, output_path=destination).drain()

        with av.open(str(destination)) as result:
            (cue,) = [packet for packet in result.demux(subtitles=0) if b"Cue" in bytes(packet)]
            assert result.streams.subtitles[0].codec_context.name == "mov_text"
        assert float(cue.pts * cue.time_base) == pytest.approx(0.5)
        assert float(cue.duration * cue.time_base) == pytest.approx(1.0)


class TestDropUnsupportedNvencOverrides:
    def _drop(self, codec, overrides, defaults=None):
        encoder_options._drop_unsupported_nvenc_overrides(
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


class TestResolveEncoderOptions:
    @pytest.mark.parametrize("codec", ["hevc", "av1"])
    def test_eight_bit_smart_fragment_matches_source_depth(self, codec):
        from jasna.accelerator import AcceleratorVendor
        from jasna.media.encoder_options import resolve_encoder_options

        spec, options = resolve_encoder_options(
            AcceleratorVendor.NVIDIA, codec, _fake_metadata(is_10bit=False), {}, smart_fragment=True
        )

        assert (spec.frame_format, spec.ten_bit) == ("nv12", False)
        assert options["forced-idr"] == "1"
        if codec == "hevc":
            assert options["profile"] == "main"

    @pytest.mark.parametrize(
        ("vendor", "codec", "settings", "expected"),
        [
            ("nvidia", "hevc", {"profile": "main"}, {"profile": "main"}),
            ("amd", "hevc", {"profile": "main"}, {"profile": "main", "bitdepth": None}),
            ("amd", "hevc", {"bitdepth": 8}, {"profile": "main", "bitdepth": "8"}),
            ("amd", "av1", {"bitdepth": 8}, {"profile": "main", "bitdepth": "8"}),
        ],
    )
    def test_eight_bit_request_encodes_eight_bit_frames(self, vendor, codec, settings, expected):
        from jasna.accelerator import AcceleratorVendor
        from jasna.media.encoder_options import resolve_encoder_options

        spec, options = resolve_encoder_options(
            AcceleratorVendor(vendor), codec, _fake_metadata(), settings, smart_fragment=False
        )

        assert spec.frame_format == "nv12"
        assert {key: options.get(key) for key in expected} == expected

    @pytest.mark.parametrize(
        ("vendor", "codec", "settings"),
        [
            ("nvidia", "hevc", {"profile": "main10"}),
            ("amd", "hevc", {"profile": "main10"}),
            ("amd", "av1", {"profile": "main"}),
        ],
    )
    def test_ten_bit_profiles_keep_ten_bit_frames(self, vendor, codec, settings):
        from jasna.accelerator import AcceleratorVendor
        from jasna.media.encoder_options import resolve_encoder_options

        spec, _ = resolve_encoder_options(
            AcceleratorVendor(vendor), codec, _fake_metadata(), settings, smart_fragment=False
        )

        assert spec.frame_format == "p010le"

    def test_amd_hevc_cq_maps_to_cqp_quantizers_without_bitrate_cap(self):
        from jasna.accelerator import AcceleratorVendor
        from jasna.media.encoder_options import resolve_encoder_options

        spec, options = resolve_encoder_options(
            AcceleratorVendor.AMD, "hevc", _fake_metadata(), {"cq": 21}, smart_fragment=False
        )

        assert spec.encoder_name == "hevc_amf"
        assert (options["qp_i"], options["qp_p"]) == ("21", "21")
        assert "cq" not in options
        assert "maxrate" not in options


def _nvenc_options(codec="hevc", settings=None, *, width, height):
    from jasna.accelerator import AcceleratorVendor
    from jasna.media.encoder_options import resolve_encoder_options

    return resolve_encoder_options(
        AcceleratorVendor.NVIDIA, codec, _fake_metadata(video_width=width, video_height=height), settings or {},
        smart_fragment=False,
    )


class TestNvencLookaheadVram:
    # Torch peaks of 8192x4096 encodes on an RTX 5090 differed by exactly 22 frames
    # between rc-lookahead 32 and 10 or 0 (bf 4), as FFmpeg 8.1's surface count says.
    @pytest.mark.parametrize(
        ("rc_lookahead", "b_frames", "held"), [(32, 4, 42), (10, 4, 20), (0, 4, 20), (0, 0, 4), (90, 4, 64)]
    )
    def test_held_frames_follow_ffmpeg_surfaces(self, rc_lookahead, b_frames, held):
        assert encoder_options.nvenc_held_frames(rc_lookahead, b_frames) == held

    @pytest.mark.parametrize("codec", ["hevc", "h264", "av1"])
    def test_8k_default_lookahead_holds_no_more_frames_than_its_b_frames(self, codec):
        _spec, options = _nvenc_options(codec, width=8192, height=4096)
        lookahead, b_frames = int(options["rc-lookahead"]), int(options["bf"])

        assert lookahead > 0
        assert encoder_options.nvenc_held_frames(lookahead, b_frames) == (
            encoder_options.nvenc_held_frames(0, b_frames)
        )

    def test_8k_without_b_frames_turns_lookahead_off(self):
        _spec, options = _nvenc_options(settings={"bf": 0}, width=8192, height=4096)
        assert options["rc-lookahead"] == "0"

    @pytest.mark.parametrize(("width", "height"), [(3840, 2160), (4096, 2160)])
    def test_4k_keeps_the_default_lookahead(self, width, height):
        _spec, options = _nvenc_options(width=width, height=height)
        assert options["rc-lookahead"] == DEFAULT_ENCODER_OPTIONS["rc-lookahead"]

    @pytest.mark.parametrize("settings", [{"rc-lookahead": 32}, {"lookahead_level": 2}])
    def test_8k_keeps_a_lookahead_the_user_asked_for(self, settings):
        _spec, options = _nvenc_options(settings=settings, width=8192, height=4096)
        assert options["rc-lookahead"] == "32"

    @pytest.mark.parametrize(
        ("width", "height", "reserved"), [(1920, 1080, False), (3840, 2160, False), (4096, 2160, False), (8192, 4096, True)]
    )
    def test_768_decision_reserves_the_encoder_only_above_4k(self, width, height, reserved):
        spec, options = _nvenc_options(width=width, height=height)
        beside_ltx = encoder_options.nvenc_vram_beside_ltx(spec, options, width=width, height=height)
        assert beside_ltx == (
            encoder_options.nvenc_vram_bytes(spec, options, width=width, height=height) if reserved else 0
        )
        assert (beside_ltx > 0) is reserved

    # NVML peak of 200-frame encodes on an RTX 5090 (2026-11-16), less the 14 RGB frames
    # waiting in VideoEncoder's queues.
    @pytest.mark.parametrize(
        ("codec", "width", "height", "rc_lookahead", "measured_mib"),
        [
            ("av1", 8192, 4096, 32, 6118),
            ("av1", 8192, 4096, 10, 3908),
            ("hevc", 8192, 4096, 32, 5756),
            ("hevc", 3840, 2160, 32, 1452),
            ("hevc", 1920, 1080, 32, 426),
        ],
    )
    def test_vram_estimate_matches_measured_encodes(self, codec, width, height, rc_lookahead, measured_mib):
        spec, options = _nvenc_options(codec, {"rc-lookahead": rc_lookahead}, width=width, height=height)
        estimate = encoder_options.nvenc_vram_bytes(spec, options, width=width, height=height) / 2**20
        assert estimate == pytest.approx(measured_mib, rel=0.15)
