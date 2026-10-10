from __future__ import annotations

import threading
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from jasna.accelerator import AcceleratorVendor
from jasna.media.splice import KeyframeIndex, SmartRenderCompatibilityError, SplicePlan, SpliceSpan
from jasna.pipeline import Pipeline
from jasna.segments import SegmentRange, SegmentRestoration


def test_smart_run_processes_only_render_spans_and_assembles_full_output(tmp_path) -> None:
    pipeline = object.__new__(Pipeline)
    # Resume/checkpoint state (smart-run keeps finished fragments next to the
    # output): these hand-built instances must carry the attributes `_run_smart`
    # reads, the same way `Pipeline.__init__` sets them.
    pipeline.resume_enabled = False
    pipeline.resume_dir_override = None
    pipeline.restoration_model_name = "basicvsrpp"
    pipeline.ltx_seed = 0
    pipeline.input_video = tmp_path / "input.mp4"
    pipeline.output_video = tmp_path / "output.mp4"
    pipeline.codec = "h264"
    pipeline.encoder_settings = {"cq": 22}
    pipeline.device = torch.device("cuda:0")
    pipeline.disable_progress = True
    pipeline.progress_callback = None
    pipeline.lut_path = None
    pipeline.sharpen_strength = 0.0
    pipeline.retarget_high_fps = False
    pipeline.segments = (SegmentRange(2.5, 3.0),)
    pipeline.working_dir = None
    pipeline._cancel_event = threading.Event()
    pipeline._run_pass = MagicMock()

    metadata = MagicMock(
        video_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        duration=6.0,
        profile="Main",
    )
    index = KeyframeIndex(
        (0, 60, 120),
        Fraction(1, 30),
        0,
        180,
        max_b_frames=3,
        uses_b_references=False,
    )
    plan = SplicePlan(
        index=index,
        spans=(
            SpliceSpan("copy", 0, 60),
            SpliceSpan("render", 60, 120, ((75, 90),)),
            SpliceSpan("copy", 120, 180),
        ),
        segments=pipeline.segments,
    )
    pipeline.splice_plan = plan

    with (
        patch(
            "jasna.pipeline.vendor_for_device",
            return_value=AcceleratorVendor.NVIDIA,
        ),
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch("jasna.pipeline.probe_keyframes") as probe_keyframes,
        patch("jasna.pipeline.build_splice_plan") as build_splice_plan,
        patch("jasna.pipeline.VideoEncoder") as encoder,
        patch("jasna.pipeline.create_copy_fragment") as copy_fragment,
        patch("jasna.pipeline.normalize_fragment"),
        patch("jasna.pipeline.concatenate_fragments") as concatenate,
        patch("jasna.pipeline.mux_final_output") as mux,
    ):
        pipeline._run_smart(metadata)

    probe_keyframes.assert_not_called()
    build_splice_plan.assert_not_called()
    assert copy_fragment.call_count == 2
    encoder.assert_called_once()
    assert encoder.call_args.kwargs["codec"] == "h264"
    assert encoder.call_args.kwargs["pts_origin"] == 60
    assert encoder.call_args.kwargs["smart_fragment"] is True
    assert encoder.call_args.kwargs["encoder_settings"] == {
        "cq": 22,
        "profile": "main",
        "g": 60,
        "bf": 3,
        "b_ref_mode": "disabled",
    }
    pipeline._run_pass.assert_called_once()
    pass_args = pipeline._run_pass.call_args.kwargs
    assert pass_args["seek_ts"] == 2.0
    assert pass_args["end_pts"] == 120
    assert pass_args["effect_ranges"] == ((75, 90),)
    concatenate.assert_called_once()
    mux.assert_called_once()


def test_smart_run_uses_working_dir_for_temp_files(tmp_path) -> None:
    pipeline = object.__new__(Pipeline)
    # Resume/checkpoint state (smart-run keeps finished fragments next to the
    # output): these hand-built instances must carry the attributes `_run_smart`
    # reads, the same way `Pipeline.__init__` sets them.
    pipeline.resume_enabled = False
    pipeline.resume_dir_override = None
    pipeline.restoration_model_name = "basicvsrpp"
    pipeline.ltx_seed = 0
    pipeline.input_video = tmp_path / "input.mp4"
    pipeline.output_video = tmp_path / "out" / "output.mp4"
    pipeline.codec = "h264"
    pipeline.encoder_settings = {"cq": 22}
    pipeline.device = torch.device("cuda:0")
    pipeline.disable_progress = True
    pipeline.progress_callback = None
    pipeline.lut_path = None
    pipeline.sharpen_strength = 0.0
    pipeline.retarget_high_fps = False
    pipeline.segments = (SegmentRange(2.5, 3.0),)
    pipeline.working_dir = tmp_path / "scratch"
    pipeline._cancel_event = threading.Event()
    pipeline._run_pass = MagicMock()

    metadata = MagicMock(
        video_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        duration=6.0,
        profile="Main",
    )
    index = KeyframeIndex((0, 60, 120), Fraction(1, 30), 0, 180)
    pipeline.splice_plan = SplicePlan(
        index=index,
        spans=(SpliceSpan("copy", 0, 60), SpliceSpan("render", 60, 120, ((75, 90),)), SpliceSpan("copy", 120, 180)),
        segments=pipeline.segments,
    )

    with (
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch("jasna.pipeline.VideoEncoder"),
        patch("jasna.pipeline.create_copy_fragment"),
        patch("jasna.pipeline.normalize_fragment"),
        patch("jasna.pipeline.concatenate_fragments"),
        patch("jasna.pipeline.mux_final_output") as mux,
    ):
        pipeline._run_smart(metadata)

    assembled = mux.call_args.args[0]
    assert assembled.parent.parent == pipeline.working_dir
    assert pipeline.working_dir.is_dir()
    assert pipeline.output_video.parent.is_dir()


def test_amf_h264_full_reencode_preserves_selected_ranges(tmp_path) -> None:
    pipeline = object.__new__(Pipeline)
    # Resume/checkpoint state (smart-run keeps finished fragments next to the
    # output): these hand-built instances must carry the attributes `_run_smart`
    # reads, the same way `Pipeline.__init__` sets them.
    pipeline.resume_enabled = False
    pipeline.resume_dir_override = None
    pipeline.restoration_model_name = "basicvsrpp"
    pipeline.ltx_seed = 0
    pipeline.input_video = tmp_path / "input.mp4"
    pipeline.output_video = tmp_path / "output.mp4"
    pipeline.codec = "h264"
    pipeline.encoder_settings = {"cq": 22}
    pipeline.device = torch.device("cuda:0")
    pipeline.disable_progress = True
    pipeline.progress_callback = None
    pipeline.lut_path = None
    pipeline.sharpen_strength = 0.0
    pipeline.retarget_high_fps = False
    pipeline.fmp4 = False
    pipeline.segments = (SegmentRange(2.5, 3.0),)
    pipeline._run_pass = MagicMock()

    metadata = MagicMock(
        video_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        average_fps=30.0,
        num_frames=180,
        duration=6.0,
    )
    index = KeyframeIndex(
        (0, 60, 120), Fraction(1, 30), 0, 180, max_b_frames=4
    )
    pipeline.splice_plan = SplicePlan(
        index=index,
        spans=(
            SpliceSpan("copy", 0, 60),
            SpliceSpan("render", 60, 120, ((75, 90),)),
            SpliceSpan("copy", 120, 180),
        ),
        segments=pipeline.segments,
    )

    with (
        patch("jasna.pipeline.vendor_for_device", return_value=AcceleratorVendor.AMD),
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch("jasna.pipeline.VideoEncoder"),
    ):
        pipeline._run_smart(metadata)

    assert pipeline._run_pass.call_args.kwargs["effect_ranges"] == ((75, 90),)


class _SmartRenderReached(Exception):
    pass


@pytest.mark.parametrize(
    ("vendor", "max_b_frames"),
    [(AcceleratorVendor.AMD, 3), (AcceleratorVendor.NVIDIA, 4)],
)
def test_smart_run_keeps_smart_render_unless_amd_exceeds_b_frame_cap(
    vendor, max_b_frames
) -> None:
    pipeline = object.__new__(Pipeline)
    # Resume/checkpoint state (smart-run keeps finished fragments next to the
    # output): these hand-built instances must carry the attributes `_run_smart`
    # reads, the same way `Pipeline.__init__` sets them.
    pipeline.resume_enabled = False
    pipeline.resume_dir_override = None
    pipeline.restoration_model_name = "basicvsrpp"
    pipeline.ltx_seed = 0
    pipeline.input_video = Path("input.mp4")
    pipeline.output_video = Path("output.mp4")
    pipeline.codec = "h264"
    pipeline.encoder_settings = {}
    pipeline.device = torch.device("cuda:0")
    pipeline.retarget_high_fps = False
    pipeline.segments = (SegmentRange(2.5, 3.0),)
    pipeline.splice_plan = SplicePlan(
        index=KeyframeIndex((0, 60), Fraction(1, 30), 0, 120, max_b_frames=max_b_frames),
        spans=(SpliceSpan("render", 0, 60, ((15, 30),)), SpliceSpan("copy", 60, 120)),
        segments=pipeline.segments,
    )
    pipeline._run_full = MagicMock()

    with (
        patch("jasna.pipeline.vendor_for_device", return_value=vendor),
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch(
            "jasna.pipeline.resolve_smart_encoder_settings",
            side_effect=_SmartRenderReached,
        ),
        pytest.raises(_SmartRenderReached),
    ):
        pipeline._run_smart(MagicMock(duration=4.0, video_fps=30.0))

    pipeline._run_full.assert_not_called()


def test_smart_run_rejects_precomputed_plan_for_different_segments() -> None:
    pipeline = object.__new__(Pipeline)
    # Resume/checkpoint state (smart-run keeps finished fragments next to the
    # output): these hand-built instances must carry the attributes `_run_smart`
    # reads, the same way `Pipeline.__init__` sets them.
    pipeline.resume_enabled = False
    pipeline.resume_dir_override = None
    pipeline.restoration_model_name = "basicvsrpp"
    pipeline.ltx_seed = 0
    pipeline.input_video = Path("input.mp4")
    pipeline.output_video = Path("output.mp4")
    pipeline.codec = "h264"
    pipeline.retarget_high_fps = False
    pipeline.segments = (SegmentRange(1, 2),)
    pipeline.splice_plan = SplicePlan(
        index=KeyframeIndex((0, 60), Fraction(1, 30), 0, 120),
        spans=(SpliceSpan("render", 0, 60, ((15, 30),)), SpliceSpan("copy", 60, 120)),
        segments=(SegmentRange(0.5, 1),),
    )

    with (
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        pytest.raises(ValueError, match="does not match"),
    ):
        pipeline._run_smart(MagicMock(duration=4.0, video_fps=30.0))


def _mixed_pipeline(tmp_path, default_model: str, segments: tuple[SegmentRange, ...], spans) -> Pipeline:
    pipeline = object.__new__(Pipeline)
    # Resume/checkpoint state (smart-run keeps finished fragments next to the
    # output): these hand-built instances must carry the attributes `_run_smart`
    # reads, the same way `Pipeline.__init__` sets them.
    pipeline.resume_enabled = False
    pipeline.resume_dir_override = None
    pipeline.restoration_model_name = default_model
    pipeline.ltx_seed = 5
    pipeline.input_video = tmp_path / "input.mp4"
    pipeline.output_video = tmp_path / "output.mp4"
    pipeline.codec = "h264"
    pipeline.encoder_settings = {}
    pipeline.device = torch.device("cuda:0")
    pipeline.disable_progress = True
    pipeline.progress_callback = None
    pipeline.lut_path = None
    pipeline.sharpen_strength = 0.0
    pipeline.retarget_high_fps = False
    pipeline.working_dir = None
    pipeline.segments = segments
    pipeline.splice_plan = SplicePlan(
        index=KeyframeIndex((0, 60, 120, 180), Fraction(1, 30), 0, 240), spans=spans, segments=segments
    )
    pipeline._cancel_event = threading.Event()
    pipeline._run_pass = MagicMock()
    pipeline._run_ltx_spans = MagicMock()
    return pipeline


def _run_smart_mocked(pipeline):
    with (
        patch("jasna.pipeline.vendor_for_device", return_value=AcceleratorVendor.NVIDIA),
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch("jasna.pipeline.resolve_smart_encoder_settings", return_value={}),
        patch("jasna.pipeline.VideoEncoder"),
        patch("jasna.pipeline.create_copy_fragment"),
        patch("jasna.pipeline.normalize_fragment"),
        patch("jasna.pipeline.concatenate_fragments") as concatenate,
        patch("jasna.pipeline.mux_final_output"),
    ):
        pipeline._run_smart(MagicMock(duration=8.0, video_fps=30.0, video_fps_exact=Fraction(30, 1)))
    return concatenate


def test_mixed_job_batches_ltx_spans_and_keeps_fragments_in_span_order(tmp_path) -> None:
    ltx = SegmentRestoration("ltx", 42)
    segments = (SegmentRange(2.5, 3.0), SegmentRange(6.5, 7.0, ltx))
    spans = (
        SpliceSpan("copy", 0, 60),
        SpliceSpan("render", 60, 120, ((75, 90),)),
        SpliceSpan("copy", 120, 180),
        SpliceSpan("render", 180, 240, ((195, 210),)),
    )
    pipeline = _mixed_pipeline(tmp_path, "basicvsrpp", segments, spans)

    concatenate = _run_smart_mocked(pipeline)

    pipeline._run_pass.assert_called_once()
    assert pipeline._run_pass.call_args.kwargs["effect_ranges"] == ((75, 90),)
    (_, _, ltx_spans, _, _), _ = pipeline._run_ltx_spans.call_args
    assert [(span, segs) for span, segs, _ in ltx_spans] == [(spans[3], (SegmentRange(6.5, 7.0, ltx),))]
    fragments = concatenate.call_args.args[0]
    assert [path.name for path, _ in fragments] == ["0000.ts", "0001.ts", "0002.ts", "0003.ts"]



def test_mixed_job_reports_one_bar_weighted_by_estimated_work(tmp_path) -> None:
    ltx = SegmentRestoration("ltx", 42)
    segments = (SegmentRange(2.5, 3.0), SegmentRange(6.5, 7.0, ltx))
    spans = (
        SpliceSpan("copy", 0, 60),
        SpliceSpan("render", 60, 120, ((75, 90),)),
        SpliceSpan("copy", 120, 180),
        SpliceSpan("render", 180, 240, ((195, 210),)),
    )
    pipeline = _mixed_pipeline(tmp_path, "basicvsrpp", segments, spans)
    pipeline.progress_callback = MagicMock()

    _run_smart_mocked(pipeline)

    (*_, ltx_callback), _ = pipeline._run_ltx_spans.call_args
    ltx_callback(100.0, 0.0, 0.0, 0, 0, "compose")
    ltx_share = 30.0 * 60 / (30.0 * 60 + 60)
    assert pipeline.progress_callback.call_args.args[0] == pytest.approx(100.0 * ltx_share)
    assert pipeline._run_pass.call_args.kwargs["progress"].callback is not pipeline.progress_callback

def test_segments_on_the_job_model_follow_an_ltx_job(tmp_path) -> None:
    segments = (SegmentRange(2.5, 3.0),)
    spans = (SpliceSpan("copy", 0, 60), SpliceSpan("render", 60, 120, ((75, 90),)), SpliceSpan("copy", 120, 240))
    pipeline = _mixed_pipeline(tmp_path, "ltx", segments, spans)

    _run_smart_mocked(pipeline)

    pipeline._run_pass.assert_not_called()
    (_, _, ltx_spans, _, _), _ = pipeline._run_ltx_spans.call_args
    assert ltx_spans[0][1] == (SegmentRange(2.5, 3.0, SegmentRestoration("ltx", 5)),)


def test_a_render_span_resolving_to_two_models_is_rejected(tmp_path) -> None:
    segments = (SegmentRange(2.2, 2.4), SegmentRange(3.0, 3.2, SegmentRestoration("basicvsrpp", None)))
    spans = (SpliceSpan("copy", 0, 60), SpliceSpan("render", 60, 120, ((66, 72), (90, 96))), SpliceSpan("copy", 120, 240))
    pipeline = _mixed_pipeline(tmp_path, "ltx", segments, spans)

    with pytest.raises(SmartRenderCompatibilityError):
        _run_smart_mocked(pipeline)


@pytest.mark.parametrize(
    ("model", "segments", "expected"),
    [
        ("basicvsrpp", None, "_run_full"),
        ("ltx", None, "_run_ltx"),
        ("ltx", (SegmentRange(1, 2),), "_run_smart"),
    ],
)
def test_run_routes_by_the_job_model_and_segments(model, segments, expected) -> None:
    pipeline = object.__new__(Pipeline)
    # Resume/checkpoint state (smart-run keeps finished fragments next to the
    # output): these hand-built instances must carry the attributes `_run_smart`
    # reads, the same way `Pipeline.__init__` sets them.
    pipeline.resume_enabled = False
    pipeline.resume_dir_override = None
    pipeline.restoration_model_name = model
    pipeline.segments = segments
    pipeline.input_video = Path("input.mp4")
    pipeline.fmp4 = False
    pipeline.ltx_files = MagicMock()
    pipeline._cancel_event = threading.Event()
    pipeline.validate_metadata = MagicMock()
    pipeline.configure_vr = MagicMock()
    for name in ("_run_full", "_run_ltx", "_run_smart"):
        setattr(pipeline, name, MagicMock())

    with patch("jasna.pipeline.get_video_meta_data"):
        pipeline.run()

    assert [name for name in ("_run_full", "_run_ltx", "_run_smart") if getattr(pipeline, name).called] == [expected]
