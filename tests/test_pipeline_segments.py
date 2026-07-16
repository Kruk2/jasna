from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from jasna.media.splice import KeyframeIndex, SplicePlan, SpliceSpan
from jasna.pipeline import Pipeline
from jasna.segments import SegmentRange


def test_smart_run_processes_only_render_spans_and_assembles_full_output(tmp_path) -> None:
    pipeline = object.__new__(Pipeline)
    pipeline.input_video = tmp_path / "input.mp4"
    pipeline.output_video = tmp_path / "output.mp4"
    pipeline.codec = "h264"
    pipeline.encoder_settings = {"cq": 22}
    pipeline.device = torch.device("cuda:0")
    pipeline.disable_progress = True
    pipeline.progress_callback = None
    pipeline.lut_path = None
    pipeline.retarget_high_fps = False
    pipeline.segments = (SegmentRange(2.5, 3.0),)
    pipeline._run_pass = MagicMock()

    metadata = MagicMock(
        video_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        duration=6.0,
    )
    index = KeyframeIndex((0, 60, 120), Fraction(1, 30), 0, 180)
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
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch("jasna.pipeline.probe_keyframes") as probe_keyframes,
        patch("jasna.pipeline.build_splice_plan") as build_splice_plan,
        patch("jasna.pipeline.NvidiaVideoEncoder") as encoder,
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
    assert encoder.call_args.kwargs["mux_audio"] is False
    assert encoder.call_args.kwargs["pts_origin"] == 60
    assert encoder.call_args.kwargs["smart_fragment"] is True
    pipeline._run_pass.assert_called_once()
    pass_args = pipeline._run_pass.call_args.kwargs
    assert pass_args["seek_ts"] == 2.0
    assert pass_args["end_pts"] == 120
    assert pass_args["effect_ranges"] == ((75, 90),)
    concatenate.assert_called_once()
    mux.assert_called_once()


def test_smart_run_rejects_precomputed_plan_for_different_segments() -> None:
    pipeline = object.__new__(Pipeline)
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
