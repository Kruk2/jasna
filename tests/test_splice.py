from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
from av.video.reformatter import ColorRange, Colorspace

from jasna.accelerator import AcceleratorVendor
from jasna.media.probe import VideoMetadata
from jasna.media.splice import (
    HevcParameterSet,
    KeyframeIndex,
    OutputValidationError,
    SplicePlan,
    SpliceSpan,
    SmartRenderCompatibilityError,
    _analyze_packet_reordering,
    _commit_smart_output,
    canonical_codec,
    _decoded_frame_hashes,
    _fsync_directory,
    _fsync_file,
    _hevc_copy_windows_match,
    _is_safe_random_access_packet,
    _native_hevc_decoder_ffmpeg,
    _packet_timelines_match,
    _probe_hevc_parameter_set_sequences,
    _write_concat_manifest,
    build_splice_plan,
    create_copy_fragment,
    create_normalized_copy_fragment,
    mux_fragments_final_output,
    normalize_fragment,
    probe_keyframes,
    resolve_smart_encoder_settings,
    split_render_spans,
    sync_and_validate_final_output,
    strict_decode_hevc_smart_output,
    validate_hevc_fragment_parameter_sets,
    validate_hevc_smart_output,
    validate_video_output,
    validate_smart_render,
)
from jasna.segments import SegmentRange, SegmentRestoration


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
        profile="High",
        field_order="progressive",
    )
    values.update(overrides)
    return VideoMetadata(**values)


def _index(points=(0, 60, 120), end=180) -> KeyframeIndex:
    return KeyframeIndex(tuple(points), Fraction(1, 30), 0, end)


def test_packet_timeline_match_tolerates_origin_and_muxer_rounding() -> None:
    source = tuple(Fraction(index, 60) for index in range(6))
    fragment = tuple(
        Fraction(3, 100) + value + Fraction(1, 90_000)
        for value in source
    )

    assert _packet_timelines_match(source, fragment)


def test_packet_timeline_match_rejects_collapsed_stale_fragment_pts() -> None:
    source = tuple(Fraction(index, 60) for index in range(6))
    fragment = (
        Fraction(0),
        Fraction(1, 90_000),
        Fraction(2, 90_000),
        Fraction(3, 90_000),
        Fraction(4, 90_000),
        Fraction(1, 12),
    )

    assert not _packet_timelines_match(source, fragment)


class _Packet:
    def __init__(
        self,
        *,
        pts: int | None,
        dts: int | None = None,
        duration: int | None = None,
        is_keyframe: bool = False,
    ) -> None:
        self.pts = pts
        self.dts = dts
        self.duration = duration
        self.is_keyframe = is_keyframe

    def __bytes__(self) -> bytes:
        return b"packet"


def test_keyframe_index_defaults_to_no_decode_delay() -> None:
    assert _index().decode_delay_pts == 0


def test_decoded_frame_hashes_seeks_from_nonzero_stream_start(tmp_path) -> None:
    video = tmp_path / "offset.mkv"
    video.write_bytes(b"fixture")
    stream = MagicMock(start_time=21, time_base=Fraction(1, 1000))
    container = MagicMock()
    container.streams.video = [stream]
    opened = MagicMock()
    opened.__enter__.return_value = container
    completed = MagicMock(
        returncode=0,
        stdout="#tb 0: 1/1000\n0, 0, 2023, 17, 123, framehash\n",
        stderr="",
    )
    with (
        patch("jasna.media.splice.av.open", return_value=opened),
        patch("jasna.media.splice.resolve_executable", return_value="ffmpeg"),
        patch("jasna.media.splice.subprocess.run", return_value=completed) as run,
    ):
        hashes = _decoded_frame_hashes(video, start=2.002, duration=1.0)
    command = run.call_args.args[0]
    assert command[command.index("-ss") + 1] == "2.023000000"
    assert command[command.index("-to") + 1] == "3.023000000"
    assert hashes == ((Fraction(2002, 1000), Fraction(17, 1000), 123, "framehash"),)


def test_hevc_copy_windows_allow_only_boundary_and_constant_subframe_shift() -> None:
    duration = Fraction(1001, 60_000)
    source = tuple(
        (Fraction(index, 60), duration, 100 + index, f"hash-{index}")
        for index in range(4)
    )
    shift = Fraction(4, 1000)
    output = tuple((frame[0] + shift, *frame[1:]) for frame in source[:-1])
    assert _hevc_copy_windows_match(source, output)
    changed = (*output[:-1], (*output[-1][:-1], "changed"))
    assert not _hevc_copy_windows_match(source, changed)
    drifting = tuple(
        (frame[0] + shift + Fraction(index, 100), *frame[1:])
        for index, frame in enumerate(source[:-1])
    )
    assert not _hevc_copy_windows_match(source, drifting)


def test_hevc_smart_output_requires_matching_timeline_without_strict_decode(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    source_timeline = (Fraction(0), Fraction(1001, 60_000))
    output_timeline = (Fraction(0), Fraction(1502, 90_000))
    with (
        patch(
            "jasna.media.splice._video_packet_timeline",
            side_effect=[source_timeline, output_timeline],
        ),
        patch("jasna.media.splice.subprocess.run") as strict_decode,
    ):
        validate_hevc_smart_output(output, source)

    strict_decode.assert_not_called()


def test_explicit_hevc_acceptance_runs_native_strict_decode(tmp_path: Path) -> None:
    output = tmp_path / "output.mp4"
    with (
        patch(
            "jasna.media.splice._video_packet_timeline",
            return_value=(Fraction(0), Fraction(1001, 60_000)),
        ),
        patch(
            "jasna.media.splice._native_hevc_decoder_ffmpeg",
            return_value="strict-ffmpeg",
        ),
        patch(
            "jasna.media.splice.subprocess.run",
            return_value=MagicMock(returncode=0, stdout="", stderr=""),
        ) as strict_decode,
    ):
        strict_decode_hevc_smart_output(output)

    strict_decode.assert_called_once()
    command = strict_decode.call_args.args[0]
    assert command[:2] == ["strict-ffmpeg", "-hide_banner"]
    assert command[command.index("-c:v") + 1] == "hevc"
    assert command[command.index("-i") + 1] == str(output)


def test_hevc_smart_output_rejects_frame_count_before_decode(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    with (
        patch(
            "jasna.media.splice._video_packet_timeline",
            side_effect=[(Fraction(0), Fraction(1, 30)), (Fraction(0),)],
        ),
        patch("jasna.media.splice.subprocess.run") as strict_decode,
        pytest.raises(SmartRenderCompatibilityError) as rejected,
    ):
        validate_hevc_smart_output(output, source)

    assert rejected.value.reason == "hevc_final_timeline_invalid"
    strict_decode.assert_not_called()


def test_native_hevc_validator_skips_amf_only_runtime() -> None:
    amf_help = MagicMock(
        returncode=0,
        stdout="Decoder hevc_amf [HEVC AMD AMF video decoder]:\n",
        stderr="",
    )
    native_help = MagicMock(
        returncode=0,
        stdout="Decoder hevc [HEVC (High Efficiency Video Coding)]:\n",
        stderr="",
    )
    with (
        patch.dict("os.environ", {}, clear=True),
        patch("jasna.media.splice.resolve_executable", return_value="runtime-ffmpeg"),
        patch("jasna.media.splice.sys.platform", "linux"),
        patch(
            "jasna.media.splice.subprocess.run",
            side_effect=[amf_help, native_help],
        ) as run,
    ):
        assert _native_hevc_decoder_ffmpeg() == "/usr/bin/ffmpeg"

    assert run.call_args_list[0].args[0][0] == "runtime-ffmpeg"
    assert run.call_args_list[1].args[0][0] == "/usr/bin/ffmpeg"


def test_rejected_hevc_timeline_gate_is_not_published_or_deleted(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.mp4"
    fragment = tmp_path / "fragment.ts"
    destination = tmp_path / "output.mp4"
    source.write_bytes(b"source")
    fragment.write_bytes(b"fragment")
    destination.write_bytes(b"old output")

    def assemble(args, *, purpose):
        assert "assemble smart-render output" in purpose
        Path(args[-1]).write_bytes(b"rejected assembly")

    with (
        patch("jasna.media.splice._final_mux_args", return_value=["-fake"]),
        patch("jasna.media.splice._run_ffmpeg", side_effect=assemble),
        patch("jasna.media.splice.validate_hevc_copy_seams"),
        patch(
            "jasna.media.splice.validate_hevc_smart_output",
            side_effect=SmartRenderCompatibilityError(
                "timeline failure",
                reason="hevc_final_timeline_invalid",
            ),
        ),
        patch("jasna.media.splice._commit_smart_output") as commit,
        pytest.raises(SmartRenderCompatibilityError, match="timeline failure"),
    ):
        mux_fragments_final_output(
            [(fragment, 1.0)],
            source,
            destination,
            manifest=tmp_path / "fragments.ffconcat",
            codec="hevc",
            copy_validation_ranges=((0.0, 0.5),),
        )

    assert destination.read_bytes() == b"old output"
    rejected = tmp_path / ".output.smart-render.mp4"
    assert rejected.read_bytes() == b"rejected assembly"
    commit.assert_not_called()


def test_hevc_parameter_set_redefinition_is_deferred_to_final_gate(tmp_path) -> None:
    def signature(prefix: str) -> tuple[HevcParameterSet, ...]:
        return (
            HevcParameterSet(32, 0, None, f"{prefix}-vps"),
            HevcParameterSet(33, 0, 0, f"{prefix}-sps"),
            HevcParameterSet(34, 0, 0, f"{prefix}-pps"),
        )

    config_a = signature("a")
    config_b = signature("b")
    fragments = [
        (tmp_path / "copy-a.ts", "copy"),
        (tmp_path / "render.ts", "render"),
        (tmp_path / "copy-b.ts", "copy"),
    ]
    with patch(
        "jasna.media.splice._probe_hevc_parameter_set_sequences",
        side_effect=[(config_a,), (config_a, config_b), (config_b,)],
    ):
        validate_hevc_fragment_parameter_sets(fragments)


def test_hevc_parameter_set_validation_rejects_sps_before_vps(tmp_path) -> None:
    ordered = (
        HevcParameterSet(32, 0, None, "vps"),
        HevcParameterSet(33, 0, 0, "sps"),
        HevcParameterSet(34, 0, 0, "pps"),
    )
    misordered = (ordered[1], ordered[2], ordered[0])
    fragments = [
        (tmp_path / "copy.ts", "copy"),
        (tmp_path / "render.ts", "render"),
    ]

    with (
        patch(
            "jasna.media.splice._probe_hevc_parameter_set_sequences",
            side_effect=[(ordered,), (misordered,)],
        ),
        pytest.raises(SmartRenderCompatibilityError) as rejected,
    ):
        validate_hevc_fragment_parameter_sets(fragments)

    assert rejected.value.reason == "hevc_parameter_sets_unavailable"
    assert "dependency-ordered" in str(rejected.value)


def test_hevc_parameter_probe_stops_after_first_random_access_packet() -> None:
    class Packet:
        def __init__(self, *, keyframe: bool) -> None:
            self.size = 1
            self.is_keyframe = keyframe
            self.byte_reads = 0

        def __bytes__(self) -> bytes:
            self.byte_reads += 1
            return b"packet"

    non_keyframe = Packet(keyframe=False)
    keyframe = Packet(keyframe=True)
    later_keyframe = Packet(keyframe=True)
    stream = MagicMock()
    stream.codec_context.extradata = b""
    container = MagicMock()
    container.__enter__.return_value = container
    container.streams.video = [stream]
    container.demux.return_value = [non_keyframe, keyframe, later_keyframe]
    ordered = (
        HevcParameterSet(32, 0, None, "vps"),
        HevcParameterSet(33, 0, 0, "sps"),
        HevcParameterSet(34, 0, 0, "pps"),
    )

    with (
        patch("jasna.media.splice.av.open", return_value=container),
        patch("jasna.media.splice._hevc_configuration_nals", return_value=()),
        patch("jasna.media.splice._split_nal_units", return_value=(b"nal",)),
        patch(
            "jasna.media.splice._parameter_sets_from_nals",
            side_effect=lambda units: ordered if units else (),
        ),
    ):
        sequences = _probe_hevc_parameter_set_sequences("fragment.ts")

    assert sequences == (ordered,)
    assert non_keyframe.byte_reads == 0
    assert keyframe.byte_reads == 1
    assert later_keyframe.byte_reads == 0


@pytest.mark.parametrize(
    ("name", "expected_options"),
    [
        (
            "fragment.ts",
            {
                "probesize": "32768",
                "analyzeduration": "0",
                "fpsprobesize": "0",
            },
        ),
        ("fragment.mp4", None),
    ],
)
def test_hevc_parameter_probe_uses_fast_options_only_for_mpegts(
    name: str,
    expected_options: dict[str, str] | None,
) -> None:
    stream = MagicMock()
    stream.codec_context.extradata = b"configuration"
    container = MagicMock()
    container.__enter__.return_value = container
    container.streams.video = [stream]
    container.demux.return_value = []

    with (
        patch("jasna.media.splice.av.open", return_value=container) as open_media,
        patch(
            "jasna.media.splice._hevc_configuration_nals",
            return_value=(b"nal",),
        ),
        patch(
            "jasna.media.splice._parameter_sets_from_nals",
            return_value=(HevcParameterSet(32, 0, None, "vps"),),
        ),
    ):
        _probe_hevc_parameter_set_sequences(name)

    open_media.assert_called_once_with(name, options=expected_options)


def test_concat_manifest_applies_fast_probe_options_only_to_mpegts(
    tmp_path: Path,
) -> None:
    fragment_ts = tmp_path / "fragment one.ts"
    fragment_mkv = tmp_path / "fragment-two.mkv"
    manifest = tmp_path / "fragments.ffconcat"

    _write_concat_manifest(
        [(fragment_ts, 1.25), (fragment_mkv, 2.5)],
        manifest,
    )

    assert manifest.read_text(encoding="utf-8").splitlines() == [
        "ffconcat version 1.0",
        f"file '{fragment_ts.resolve()}'",
        "option probesize 32768",
        "option analyzeduration 0",
        "option fpsprobesize 0",
        "duration 1.250000000",
        f"file '{fragment_mkv.resolve()}'",
        "duration 2.500000000",
    ]


def test_smart_output_is_validated_and_synced_around_atomic_replace(
    tmp_path: Path,
) -> None:
    temporary = tmp_path / ".output.smart-render.mp4"
    destination = tmp_path / "output.mp4"
    source = tmp_path / "source.mp4"
    temporary.write_bytes(b"new")
    destination.write_bytes(b"old")
    source.write_bytes(b"source")
    events = []

    def replace(old, new):
        events.append(("replace", Path(old), Path(new)))
        Path(new).write_bytes(Path(old).read_bytes())
        Path(old).unlink()

    with (
        patch(
            "jasna.media.splice.validate_video_output",
            side_effect=lambda path, **_kwargs: events.append(("validate", Path(path))),
        ),
        patch(
            "jasna.media.splice._fsync_file",
            side_effect=lambda path: events.append(("fsync", Path(path))),
        ),
        patch(
            "jasna.media.splice._fsync_directory",
            side_effect=lambda path: events.append(("fsync-dir", Path(path))),
        ),
        patch("jasna.media.splice.os.replace", side_effect=replace),
    ):
        _commit_smart_output(
            temporary,
            destination,
            source=source,
            codec="h264",
        )

    assert events == [
        ("validate", temporary),
        ("fsync", temporary),
        ("replace", temporary, destination),
        ("fsync", destination),
        ("fsync-dir", tmp_path),
        ("validate", destination),
    ]
    assert destination.read_bytes() == b"new"


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



def test_plan_rejects_two_models_in_one_render_span() -> None:
    ltx = SegmentRestoration("ltx", 1)
    with pytest.raises(SmartRenderCompatibilityError) as exc:
        build_splice_plan(
            [SegmentRange(2.2, 2.4, ltx), SegmentRange(3.0, 3.2, SegmentRestoration("basicvsrpp", None))],
            _index(),
            duration=6,
        )
    assert exc.value.reason == "mixed_models"


def test_plan_keeps_each_segment_behind_its_render_span_effect_range() -> None:
    first, third = SegmentRestoration("ltx", 1), SegmentRestoration("ltx", 3)
    plan = build_splice_plan(
        [SegmentRange(2.2, 2.4, first), SegmentRange(3.0, 3.2), SegmentRange(5.0, 5.5, third)],
        _index(points=(0, 60, 120, 150)),
        duration=6,
    )
    assert [[segment.restoration for segment in segments] for segments in plan.render_span_segments()] == [
        [first, None],
        [third],
    ]

def test_plan_offsets_effect_pts_by_stream_start_time() -> None:
    index = KeyframeIndex((900, 960, 1020), Fraction(1, 30), 900, 1080)
    plan = build_splice_plan([SegmentRange(2.5, 3)], index, duration=6)
    assert plan.render_spans[0].effect_ranges == ((975, 990),)
    assert plan.render_spans[0].start_pts == 960


def test_split_render_spans_preserves_exact_frame_timeline_and_effects() -> None:
    plan = SplicePlan(
        index=KeyframeIndex((0, 300), Fraction(1, 30), 0, 300),
        spans=(
            SpliceSpan("render", 0, 300, ((15, 75), (120, 285))),
        ),
        segments=(SegmentRange(0.5, 9.5),),
    )

    split = split_render_spans(
        plan,
        video_fps=Fraction(30, 1),
        max_duration_seconds=4,
    )

    assert [
        (span.start_pts, span.end_pts, span.effect_ranges)
        for span in split.spans
    ] == [
        (0, 120, ((15, 75),)),
        (120, 240, ((120, 240),)),
        (240, 300, ((240, 285),)),
    ]
    assert split.index is plan.index
    assert split.segments == plan.segments


def test_split_render_spans_leaves_copy_and_short_render_spans_unchanged() -> None:
    plan = SplicePlan(
        index=KeyframeIndex((0, 60), Fraction(1, 30), 0, 120),
        spans=(
            SpliceSpan("copy", 0, 60),
            SpliceSpan("render", 60, 120, ((75, 90),)),
        ),
        segments=(SegmentRange(2.5, 3.0),),
    )

    assert (
        split_render_spans(
            plan,
            video_fps=Fraction(30, 1),
            max_duration_seconds=120,
        )
        is plan
    )


def test_plan_rejects_when_only_full_reencode_is_possible() -> None:
    with pytest.raises(SmartRenderCompatibilityError, match="entire video"):
        build_splice_plan([SegmentRange(2, 3)], _index(points=(0,)), duration=6)


def test_plan_rejects_range_shorter_than_timestamp_interval() -> None:
    with pytest.raises(SmartRenderCompatibilityError, match="timestamp interval"):
        build_splice_plan([SegmentRange(2.001, 2.01)], _index(), duration=6)


def test_packet_reordering_detects_flat_and_hierarchical_b_frames() -> None:
    assert _analyze_packet_reordering((0, 4, 1, 2, 3, 8, 5, 6, 7)) == (3, False)
    assert _analyze_packet_reordering((0, 5, 3, 1, 2, 4, 10, 8, 6, 7, 9)) == (4, True)


def test_probe_keyframes_uses_packet_tail_and_keyframe_decode_delay() -> None:
    stream = MagicMock()
    stream.codec_context.extradata = b""
    stream.time_base = Fraction(1, 30)
    stream.start_time = 100
    stream.duration = 90
    container = MagicMock()
    container.__enter__.return_value = container
    container.streams.video = [stream]
    container.demux.return_value = [
        _Packet(pts=100, dts=94, duration=3, is_keyframe=True),
        _Packet(pts=190, dts=188, duration=0),
    ]

    with (
        patch("jasna.media.splice.av.open", return_value=container),
        patch("jasna.media.splice._is_safe_random_access_packet", return_value=True),
    ):
        index = probe_keyframes(Path("input.mp4"), _metadata())

    assert index.pts == (100,)
    assert index.end_pts == 191
    assert index.decode_delay_pts == 6


@pytest.mark.parametrize(
    ("codec", "bitstream_filter", "muxer", "container_args"),
    [
        ("h264", "h264_mp4toannexb,dump_extra=freq=keyframe", "mpegts", ["-muxdelay", "0"]),
        ("hevc", "hevc_mp4toannexb", "mpegts", ["-muxdelay", "0"]),
        ("av1", "av1_metadata=td=insert,dump_extra=freq=keyframe", "matroska", ["-avoid_negative_ts", "make_zero"]),
    ],
)
def test_normalize_fragment_preserves_default_timestamps(
    codec: str,
    bitstream_filter: str,
    muxer: str,
    container_args: list[str],
) -> None:
    source = Path("raw.nut")
    destination = Path("normalized.media")
    with (
        patch("jasna.media.splice._fragment_pts_are_reordered") as reordered,
        patch("jasna.media.splice._run_ffmpeg") as run_ffmpeg,
    ):
        normalize_fragment(source, destination, codec=codec)
    reordered.assert_not_called()
    assert run_ffmpeg.call_args.args[0] == [
        "-i", str(source),
        "-map", "0:v:0",
        "-an",
        "-c:v", "copy",
        "-bsf:v", bitstream_filter,
        *container_args,
        "-f", muxer,
        str(destination),
    ]


def test_normalize_fragment_adds_decode_delay_to_unreordered_render_output() -> None:
    source = Path("raw.nut")
    destination = Path("normalized.ts")
    stream = MagicMock()
    stream.time_base = Fraction(1, 90_000)
    container = MagicMock()
    container.__enter__.return_value = container
    container.streams.video = [stream]
    with (
        patch("jasna.media.splice._fragment_pts_are_reordered", return_value=False),
        patch("jasna.media.splice.av.open", return_value=container),
        patch("jasna.media.splice._run_ffmpeg") as run_ffmpeg,
    ):
        normalize_fragment(
            source,
            destination,
            codec="h264",
            decode_delay=Fraction(1, 30),
        )
    assert run_ffmpeg.call_args.args[0] == [
        "-i", str(source),
        "-map", "0:v:0",
        "-an",
        "-c:v", "copy",
        "-bsf:v", "h264_mp4toannexb,dump_extra=freq=keyframe,setts=pts=PTS:dts=PTS-3000",
        "-muxdelay", "0",
        "-output_ts_offset", "0.033333333",
        "-avoid_negative_ts", "disabled",
        "-f", "mpegts",
        str(destination),
    ]


def test_normalize_fragment_skips_delay_for_reordered_packets() -> None:
    source = Path("raw.nut")
    destination = Path("normalized.ts")
    container = MagicMock()
    container.__enter__.return_value = container
    container.streams.video = [MagicMock()]
    container.demux.return_value = [
        _Packet(pts=0), _Packet(pts=4), _Packet(pts=1), _Packet(pts=2), _Packet(pts=3),
    ]
    with (
        patch("jasna.media.splice.av.open", return_value=container),
        patch("jasna.media.splice._run_ffmpeg") as run_ffmpeg,
    ):
        normalize_fragment(
            source,
            destination,
            codec="h264",
            decode_delay=Fraction(1, 30),
        )
    assert "setts=" not in " ".join(run_ffmpeg.call_args.args[0])


def test_smart_h264_settings_match_source_structure() -> None:
    index = KeyframeIndex(
        (0, 60, 120),
        Fraction(1, 30),
        0,
        180,
        max_b_frames=3,
        uses_b_references=False,
    )

    settings = resolve_smart_encoder_settings(
        "h264",
        _metadata("h264", profile="Main"),
        index,
        {"cq": 22, "profile": "high", "g": 250, "bf": 4, "b_ref_mode": "middle"},
        vendor=AcceleratorVendor.NVIDIA,
    )

    assert settings == {
        "cq": 22,
        "profile": "main",
        "g": 60,
        "bf": 3,
        "b_ref_mode": "disabled",
    }


@pytest.mark.parametrize("uses_b_references", [False, True])
def test_smart_h264_amf_settings_match_source_structure(
    uses_b_references: bool,
) -> None:
    index = KeyframeIndex(
        (0, 60, 120),
        Fraction(1, 30),
        0,
        180,
        max_b_frames=3,
        uses_b_references=uses_b_references,
    )

    settings = resolve_smart_encoder_settings(
        "h264",
        _metadata("h264", profile="Main"),
        index,
        {
            "cq": 22,
            "profile": "high",
            "g": 250,
            "bf": 2,
            "preanalysis": 1,
            "pa_adaptive_mini_gop": 1,
        },
        vendor=AcceleratorVendor.AMD,
    )

    assert settings == {
        "profile": "main",
        "g": 60,
        "bf": 3,
        "bf_ref": int(uses_b_references),
        "rc": "vbr_peak",
        "preanalysis": 0,
        "vbaq": 0,
    }


@pytest.mark.parametrize("source_profile", ["Baseline", "Constrained Baseline"])
def test_smart_h264_amf_uses_amf_baseline_profile(source_profile: str) -> None:
    settings = resolve_smart_encoder_settings(
        "h264",
        _metadata("h264", profile=source_profile),
        _index(),
        {},
        vendor=AcceleratorVendor.AMD,
    )

    assert settings["profile"] == "constrained_baseline"


def test_windows_amf_h264_smart_keeps_existing_qvbr_contract(monkeypatch) -> None:
    import jasna.media.splice as module

    monkeypatch.setattr(module.sys, "platform", "win32")
    settings = module.resolve_smart_encoder_settings(
        "h264",
        _metadata("h264", profile="High"),
        _index(),
        {"cq": 22},
        vendor=AcceleratorVendor.AMD,
    )

    assert settings["cq"] == 22
    assert settings["pa_adaptive_mini_gop"] == 0
    assert "rc" not in settings
    assert "preanalysis" not in settings


def test_smart_h264_amf_rejects_more_than_three_b_frames() -> None:
    index = KeyframeIndex(
        (0, 60, 120),
        Fraction(1, 30),
        0,
        180,
        max_b_frames=4,
        uses_b_references=True,
    )

    with pytest.raises(
        SmartRenderCompatibilityError,
        match="AMF H.264 smart rendering supports at most 3 consecutive B-frames; source uses 4",
    ):
        resolve_smart_encoder_settings(
            "h264",
            _metadata("h264"),
            index,
            {},
            vendor=AcceleratorVendor.AMD,
        )


@pytest.mark.parametrize("codec", ["hevc", "av1"])
@pytest.mark.parametrize(
    "vendor",
    [AcceleratorVendor.NVIDIA, AcceleratorVendor.AMD],
)
def test_smart_settings_match_source_gop_for_other_codecs(
    codec: str,
    vendor: AcceleratorVendor,
) -> None:
    settings = resolve_smart_encoder_settings(
        codec,
        _metadata(codec),
        _index(),
        {"cq": 22, "g": 250, "bf": 4},
        vendor=vendor,
    )

    assert settings == {"cq": 22, "g": 60, "bf": 4}


def test_smart_h264_settings_reject_unknown_source_profile() -> None:
    with pytest.raises(SmartRenderCompatibilityError, match="H.264 profile"):
        resolve_smart_encoder_settings(
            "h264",
            _metadata("h264", profile="Extended"),
            _index(),
            {},
            vendor=AcceleratorVendor.NVIDIA,
        )


@pytest.mark.parametrize("codec", ["h264", "hevc", "av1"])
def test_validation_accepts_supported_source_matched_codecs(codec: str) -> None:
    assert validate_smart_render(_metadata(codec), output_path="out.mp4", codec=codec, retarget_high_fps=False) == codec


def test_validation_rejects_codec_mismatch() -> None:
    with pytest.raises(SmartRenderCompatibilityError, match="match the input codec"):
        validate_smart_render(_metadata("h264"), output_path="out.mp4", codec="hevc", retarget_high_fps=False)


def test_validation_rejects_unsupported_h264_profile() -> None:
    with pytest.raises(SmartRenderCompatibilityError, match="H.264 profile"):
        validate_smart_render(
            _metadata("h264", profile="Extended"),
            output_path="out.mp4",
            codec="h264",
            retarget_high_fps=False,
        )


def test_validation_rejects_vfr_and_retargeting() -> None:
    with pytest.raises(SmartRenderCompatibilityError, match="constant-frame-rate"):
        validate_smart_render(
            _metadata(average_fps=29.0),
            output_path="out.mp4",
            codec="h264",
            retarget_high_fps=False,
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


def test_segment_render_span_widens_one_segment_to_its_keyframes() -> None:
    from fractions import Fraction

    from jasna.media.splice import KeyframeIndex, SpliceSpan, segment_render_span
    from jasna.segments import SegmentRange

    index = KeyframeIndex(pts=(0, 2000, 4000), time_base=Fraction(1, 1000), start_pts=0, end_pts=6000)

    assert segment_render_span(SegmentRange(2.5, 3.0), index) == SpliceSpan("render", 2000, 4000, ((2500, 3000),))
    assert segment_render_span(SegmentRange(4.5, 6.0), index) == SpliceSpan("render", 4000, 6000, ((4500, 6000),))
def test_normalized_copy_fragment_writes_mpegts_with_source_dts(tmp_path: Path) -> None:
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
    output = tmp_path / "fragment.ts"

    with patch("jasna.media.splice.av.open", side_effect=[source, destination]) as opened:
        create_normalized_copy_fragment(
            Path("source.mp4"),
            span,
            _index(),
            output,
            codec="hevc",
        )

    assert opened.call_args_list[1] == call(str(output), "w", format="mpegts")
    assert packet.pts == 5
    assert packet.dts == 3
    assert packet.stream is output_stream


def test_validate_video_output_rejects_missing_output(tmp_path: Path) -> None:
    with pytest.raises(OutputValidationError, match="missing"):
        validate_video_output(tmp_path / "missing.mp4")


@pytest.mark.parametrize(
    ("reported_name", "bitstream_codec"),
    [
        ("hevc_amf", "hevc"),
        ("h264_amf", "h264"),
        ("av1_amf", "av1"),
    ],
)
def testcanonical_codec_accepts_exact_amf_decoder_names(
    reported_name: str,
    bitstream_codec: str,
) -> None:
    assert canonical_codec(reported_name) == bitstream_codec


def testcanonical_codec_does_not_hide_unknown_amf_codec_mismatch() -> None:
    assert canonical_codec("vp9_amf") == "vp9_amf"
    assert canonical_codec("hevc_amf_experimental") == "hevc_amf_experimental"


def _completed_output_container(codec_name: str) -> MagicMock:
    stream = MagicMock()
    stream.codec_context.name = codec_name
    stream.duration = 60_000
    stream.time_base = Fraction(1, 1000)
    stream.start_time = 0
    packet = MagicMock()
    packet.size = 1
    packet.pts = 59_000
    packet.dts = 59_000
    packet.duration = 1000
    packet.time_base = Fraction(1, 1000)
    container = MagicMock()
    container.__enter__.return_value = container
    container.__exit__.return_value = False
    container.streams.video = [stream]
    container.demux.return_value = [packet]
    return container


def test_validate_video_output_accepts_hevc_amf_decoder_name(tmp_path: Path) -> None:
    output = tmp_path / "output.mp4"
    output.write_bytes(b"media")
    container = _completed_output_container("hevc_amf")

    with patch("jasna.media.splice.av.open", return_value=container):
        validate_video_output(output, expected_codec="hevc", expected_duration=60.0)


def test_validate_video_output_still_rejects_real_codec_mismatch(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output.mp4"
    output.write_bytes(b"media")
    container = _completed_output_container("vp9_amf")

    with (
        patch("jasna.media.splice.av.open", return_value=container),
        pytest.raises(OutputValidationError, match="vp9_amf, expected hevc"),
    ):
        validate_video_output(output, expected_codec="hevc", expected_duration=60.0)


def test_sync_and_validate_final_output_validates_before_and_after_sync(
    monkeypatch, tmp_path: Path
) -> None:
    import jasna.media.splice as module

    output = tmp_path / "output.mp4"
    source = tmp_path / "source.mp4"
    events = []

    monkeypatch.setattr(
        module,
        "validate_video_output",
        lambda path, **kwargs: events.append(("validate", Path(path), kwargs)),
    )
    monkeypatch.setattr(
        module,
        "_fsync_file",
        lambda path: events.append(("fsync-file", Path(path))),
    )
    monkeypatch.setattr(
        module,
        "_fsync_directory",
        lambda path: events.append(("fsync-directory", Path(path))),
    )

    sync_and_validate_final_output(
        output,
        source=source,
        expected_codec="hevc",
    )

    assert [event[0] for event in events] == [
        "validate",
        "fsync-file",
        "fsync-directory",
        "validate",
    ]
    assert events[0] == events[-1]
    assert events[0][1] == output
    assert events[0][2] == {
        "source": source,
        "expected_codec": "hevc",
        "expected_duration": None,
    }


def test_directory_sync_is_safe_noop_on_windows(monkeypatch, tmp_path: Path) -> None:
    import jasna.media.splice as module

    open_directory = MagicMock()
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(module.os, "open", open_directory)

    _fsync_directory(tmp_path)

    open_directory.assert_not_called()


def test_windows_file_sync_reopens_output_with_write_access(
    monkeypatch, tmp_path: Path
) -> None:
    import jasna.media.splice as module

    output = tmp_path / "output.mp4"
    output.write_bytes(b"completed output")
    opened_modes = []
    original_open = Path.open

    def record_open(path, mode="r", *args, **kwargs):
        opened_modes.append(mode)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setattr(Path, "open", record_open)
    monkeypatch.setattr(module.os, "fsync", lambda _fd: None)

    _fsync_file(output)

    assert opened_modes == ["r+b"]
