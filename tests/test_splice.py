from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from av.video.reformatter import ColorRange, Colorspace

from jasna.media import VideoMetadata
from jasna.media.splice import (
    KeyframeIndex,
    SpliceSpan,
    SmartRenderCompatibilityError,
    _is_safe_random_access_packet,
    build_splice_plan,
    create_copy_fragment,
    validate_smart_render,
)
from jasna.segments import SegmentRange


def _metadata(codec: str = "h264", **overrides) -> VideoMetadata:
    values = dict(
        video_file="input.mp4",
        video_height=1080,
        video_width=1920,
        video_fps=30.0,
        average_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        codec_name=codec,
        duration=6.0,
        time_base=Fraction(1, 30),
        start_pts=0,
        color_range=ColorRange.MPEG,
        color_space=Colorspace.ITU709,
        num_frames=180,
        is_10bit=False,
        pixel_format="yuv420p",
        field_order="progressive",
    )
    values.update(overrides)
    return VideoMetadata(**values)


def _index(points=(0, 60, 120), end=180) -> KeyframeIndex:
    return KeyframeIndex(tuple(points), Fraction(1, 30), 0, end)


def test_plan_expands_to_keyframes_but_keeps_exact_effect_range() -> None:
    plan = build_splice_plan([SegmentRange(2.5, 3.0)], _index(), duration=6)
    assert [(span.kind, span.start_pts, span.end_pts) for span in plan.spans] == [
        ("copy", 0, 60),
        ("render", 60, 120),
        ("copy", 120, 180),
    ]
    assert plan.render_spans[0].effect_ranges == ((75, 90),)


def test_plan_merges_two_selections_in_one_gop_bridge() -> None:
    plan = build_splice_plan(
        [SegmentRange(2.2, 2.4), SegmentRange(3.0, 3.2)],
        _index(),
        duration=6,
    )
    assert len(plan.render_spans) == 1
    assert plan.render_spans[0].effect_ranges == ((66, 72), (90, 96))


def test_plan_offsets_effect_pts_by_stream_start_time() -> None:
    index = KeyframeIndex((900, 960, 1020), Fraction(1, 30), 900, 1080)
    plan = build_splice_plan([SegmentRange(2.5, 3)], index, duration=6)
    assert plan.render_spans[0].effect_ranges == ((975, 990),)
    assert plan.render_spans[0].start_pts == 960


def test_plan_rejects_when_only_full_reencode_is_possible() -> None:
    with pytest.raises(SmartRenderCompatibilityError, match="entire video"):
        build_splice_plan([SegmentRange(2, 3)], _index(points=(0,)), duration=6)


def test_plan_rejects_range_shorter_than_timestamp_interval() -> None:
    with pytest.raises(SmartRenderCompatibilityError, match="timestamp interval"):
        build_splice_plan([SegmentRange(2.001, 2.01)], _index(), duration=6)


@pytest.mark.parametrize("codec", ["h264", "hevc", "av1"])
def test_validation_accepts_supported_source_matched_codecs(codec: str) -> None:
    assert validate_smart_render(_metadata(codec), output_path="out.mp4", codec=codec) == codec


def test_validation_rejects_codec_mismatch() -> None:
    with pytest.raises(SmartRenderCompatibilityError, match="match the input codec"):
        validate_smart_render(_metadata("h264"), output_path="out.mp4", codec="hevc")


def test_validation_rejects_vfr_and_retargeting() -> None:
    with pytest.raises(SmartRenderCompatibilityError, match="constant-frame-rate"):
        validate_smart_render(
            _metadata(average_fps=29.0),
            output_path="out.mp4",
            codec="h264",
        )
    with pytest.raises(SmartRenderCompatibilityError, match="frame-rate retargeting"):
        validate_smart_render(
            _metadata(),
            output_path="out.mp4",
            codec="h264",
            retarget_high_fps=True,
        )


def test_random_access_classifier_distinguishes_hevc_idr_from_cra() -> None:
    hevc_idr = bytes([19 << 1, 1])
    hevc_cra = bytes([21 << 1, 1])
    assert _is_safe_random_access_packet(
        len(hevc_idr).to_bytes(4, "big") + hevc_idr,
        "hevc",
        4,
        length_prefixed=True,
    )
    assert not _is_safe_random_access_packet(
        len(hevc_cra).to_bytes(4, "big") + hevc_cra,
        "hevc",
        4,
        length_prefixed=True,
    )


def test_random_access_classifier_accepts_h264_idr() -> None:
    idr = b"\x65\x88"
    assert _is_safe_random_access_packet(
        len(idr).to_bytes(4, "big") + idr,
        "h264",
        4,
        length_prefixed=True,
    )


def test_length_prefixed_idr_is_not_misread_as_annex_b() -> None:
    idr = b"\x65" + bytes(255)
    packet = len(idr).to_bytes(4, "big") + idr

    assert packet[:4] == b"\x00\x00\x01\x00"
    assert _is_safe_random_access_packet(
        packet,
        "h264",
        4,
        length_prefixed=True,
    )


def test_random_access_classifier_accepts_annex_b_h264_idr() -> None:
    assert _is_safe_random_access_packet(
        b"\x00\x00\x00\x01\x65\x88",
        "h264",
        4,
        length_prefixed=False,
    )


def test_copy_fragment_seeks_before_demux(tmp_path: Path) -> None:
    source = MagicMock()
    destination = MagicMock()
    input_stream = MagicMock()
    output_stream = MagicMock()
    packet = MagicMock(pts=65, dts=63)
    source.__enter__.return_value = source
    source.__exit__.return_value = False
    source.streams.video = [input_stream]
    source.demux.return_value = [packet]
    destination.__enter__.return_value = destination
    destination.__exit__.return_value = False
    destination.add_stream_from_template.return_value = output_stream
    span = SpliceSpan("copy", 60, 120)

    with patch("jasna.media.splice.av.open", side_effect=[source, destination]):
        create_copy_fragment(
            Path("source.mp4"),
            span,
            _index(),
            tmp_path / "fragment.nut",
            codec="h264",
        )

    assert source.method_calls[:2] == [
        call.seek(60, stream=input_stream, backward=True),
        call.demux(input_stream),
    ]
    assert packet.pts == 5
    assert packet.dts == 3
