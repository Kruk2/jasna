from __future__ import annotations

import logging
import threading
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
from factories import make_pipeline

import pytest
import torch

from jasna.accelerator import AcceleratorVendor
from jasna.media.splice import KeyframeIndex, SmartRenderCompatibilityError, SplicePlan, SpliceSpan
from jasna.pipeline import Pipeline
from jasna.segments import SegmentRange, SegmentRestoration
from jasna.native_worker import (
    ISOLATED_VIDEO_JOB_ENV,
    NativeWorkerRecycleRequested,
)


def _mock_workspace(tmp_path: Path, name: str = "workspace") -> MagicMock:
    workspace = MagicMock()
    workspace.path = tmp_path / name
    workspace.path.mkdir(parents=True, exist_ok=True)
    workspace.raw_path.side_effect = lambda index: workspace.path / f"{index:04d}-raw.nut"
    workspace.fragment_path.side_effect = (
        lambda index, suffix: workspace.path / f"{index:04d}{suffix}"
    )
    workspace.reusable_fragment.return_value = None
    return workspace


def test_full_video_encode_keeps_only_scanned_ranges_effect_active(tmp_path) -> None:
    pipeline = make_pipeline()
    pipeline.input_video = tmp_path / "input.mp4"
    pipeline.output_video = tmp_path / "output.mp4"
    pipeline.codec = "hevc"
    pipeline.encoder_settings = {"cq": 22}
    pipeline.device = torch.device("cuda:0")
    pipeline.disable_progress = True
    pipeline.progress_callback = None
    pipeline.lut_path = None
    pipeline.sharpen_strength = 0.0
    pipeline.retarget_high_fps = False
    pipeline.auto_source_rate = True
    pipeline.fmp4 = False
    pipeline.effect_ranges = ((900, 1200), (1800, 2400))
    pipeline._run_pass = MagicMock()
    metadata = MagicMock(
        video_fps=30.0,
        average_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        num_frames=3000,
    )

    with (
        patch("jasna.pipeline.Progressbar"),
        patch("jasna.pipeline.VideoEncoder") as encoder,
    ):
        pipeline._run_full(metadata)

    assert encoder.call_args.kwargs["metadata"] is metadata
    assert encoder.call_args.kwargs["auto_source_rate"] is True
    assert encoder.call_args.kwargs["prefer_amf_host_native"] is False
    assert pipeline._run_pass.call_args.kwargs["effect_ranges"] == (
        (900, 1200),
        (1800, 2400),
    )
    assert pipeline._run_pass.call_args.kwargs["output_frame_count"] == 3000


def test_smart_run_processes_only_render_spans_and_assembles_full_output(tmp_path) -> None:
    pipeline = make_pipeline()
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
        decode_delay_pts=2,
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
    workspace = _mock_workspace(tmp_path)

    with (
        patch(
            "jasna.pipeline.vendor_for_device",
            return_value=AcceleratorVendor.NVIDIA,
        ),
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch("jasna.pipeline.workspace_signature", return_value={}),
        patch("jasna.pipeline.SmartRenderWorkspace.open", return_value=workspace),
        patch("jasna.pipeline.probe_keyframes") as probe_keyframes,
        patch("jasna.pipeline.build_splice_plan") as build_splice_plan,
        patch("jasna.pipeline.VideoEncoder") as encoder,
        patch("jasna.pipeline.create_copy_fragment") as copy_fragment,
        patch("jasna.pipeline.create_normalized_copy_fragment") as direct_copy,
        patch("jasna.pipeline.normalize_fragment") as normalize_fragment,
        patch("jasna.pipeline.mux_fragments_final_output") as mux,
    ):
        pipeline._run_smart(metadata)

    probe_keyframes.assert_not_called()
    build_splice_plan.assert_not_called()
    copy_fragment.assert_not_called()
    assert direct_copy.call_count == 2
    encoder.assert_called_once()
    assert encoder.call_args.kwargs["codec"] == "h264"
    assert encoder.call_args.kwargs["pts_origin"] == 60
    assert encoder.call_args.kwargs["smart_fragment"] is True
    assert encoder.call_args.kwargs["prefer_amf_host_native"] is False
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
    assert [call.kwargs for call in normalize_fragment.call_args_list] == [
        {"codec": "h264", "decode_delay": Fraction(1, 15)},
    ]
    mux.assert_called_once()
    assert mux.call_args.args[0][0][0].parent == mux.call_args.kwargs["manifest"].parent
    workspace.cleanup.assert_called_once_with()


def test_isolated_smart_run_recycles_after_pressured_span_before_next_decoder_pair(
    monkeypatch,
    tmp_path,
) -> None:
    pipeline = make_pipeline()
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
    pipeline.segments = (SegmentRange(0.5, 3.5),)
    pipeline.working_dir = tmp_path
    pipeline._cancel_event = threading.Event()
    pipeline._run_pass = MagicMock(
        return_value=SimpleNamespace(
            system_pressure_episodes=1,
            system_reclaim_count=0,
        )
    )
    metadata = MagicMock(
        video_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        duration=4.0,
        profile="Main",
    )
    pipeline.splice_plan = SplicePlan(
        index=KeyframeIndex((0, 60), Fraction(1, 30), 0, 120),
        spans=(
            SpliceSpan("render", 0, 60, ((15, 30),)),
            SpliceSpan("render", 60, 120, ((75, 90),)),
        ),
        segments=pipeline.segments,
    )
    workspace = _mock_workspace(tmp_path)
    monkeypatch.setenv(ISOLATED_VIDEO_JOB_ENV, "1")

    with (
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch("jasna.pipeline.workspace_signature", return_value={}),
        patch("jasna.pipeline.SmartRenderWorkspace.open", return_value=workspace),
        patch("jasna.pipeline.VideoEncoder"),
        patch("jasna.pipeline.normalize_fragment"),
        patch("jasna.pipeline.mux_fragments_final_output") as mux,
        pytest.raises(NativeWorkerRecycleRequested, match="span 0"),
    ):
        pipeline._run_smart(metadata)

    pipeline._run_pass.assert_called_once()
    workspace.mark_complete.assert_called_once()
    workspace.cleanup.assert_not_called()
    mux.assert_not_called()


def test_isolated_amd_h264_smart_run_recycles_after_each_render_span(
    monkeypatch,
    tmp_path,
) -> None:
    pipeline = make_pipeline()
    pipeline.input_video = tmp_path / "input.mp4"
    pipeline.output_video = tmp_path / "output.mp4"
    pipeline.codec = "h264"
    pipeline.encoder_settings = {"cq": 22}
    pipeline.device = torch.device("cuda:0")
    pipeline.batch_size = 4
    pipeline.disable_progress = True
    pipeline.progress_callback = None
    pipeline.lut_path = None
    pipeline.sharpen_strength = 0.0
    pipeline.retarget_high_fps = False
    pipeline.segments = (SegmentRange(0.5, 3.5),)
    pipeline.working_dir = tmp_path
    pipeline._cancel_event = threading.Event()
    pipeline._run_pass = MagicMock(
        return_value=SimpleNamespace(
            system_pressure_episodes=0,
            system_reclaim_count=0,
        )
    )
    metadata = SimpleNamespace(
        video_fps=30.0,
        average_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        duration=4.0,
        profile="High",
        is_10bit=False,
        video_width=4096,
        video_height=2048,
    )
    pipeline.splice_plan = SplicePlan(
        index=KeyframeIndex((0, 60), Fraction(1, 30), 0, 120),
        spans=(
            SpliceSpan("render", 0, 60, ((15, 30),)),
            SpliceSpan("render", 60, 120, ((75, 90),)),
        ),
        segments=pipeline.segments,
    )
    workspace = _mock_workspace(tmp_path)
    monkeypatch.setenv(ISOLATED_VIDEO_JOB_ENV, "1")

    with (
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch(
            "jasna.pipeline.vendor_for_device",
            return_value=AcceleratorVendor.AMD,
        ),
        patch("jasna.pipeline.resolve_smart_encoder_settings", return_value={}),
        patch("jasna.pipeline.workspace_signature", return_value={}),
        patch("jasna.pipeline.SmartRenderWorkspace.open", return_value=workspace),
        patch("jasna.pipeline.VideoEncoder"),
        patch("jasna.pipeline.normalize_fragment"),
        patch("jasna.pipeline.mux_fragments_final_output") as mux,
        pytest.raises(NativeWorkerRecycleRequested) as caught,
    ):
        pipeline._run_smart(metadata)

    assert caught.value.reason == "amf_session_limit"
    assert "H.264 Smart Render span 0" in str(caught.value)
    pipeline._run_pass.assert_called_once()
    workspace.mark_complete.assert_called_once()
    workspace.cleanup.assert_not_called()
    mux.assert_not_called()


def test_isolated_8k_hevc_run_splits_and_recycles_before_process_global_amf_limit(
    monkeypatch,
    tmp_path,
) -> None:
    pipeline = make_pipeline()
    pipeline.input_video = tmp_path / "input.mp4"
    pipeline.output_video = tmp_path / "output.mp4"
    pipeline.codec = "hevc"
    pipeline.encoder_settings = {"cq": 22}
    pipeline.device = torch.device("cuda:0")
    pipeline.batch_size = 4
    pipeline.disable_progress = True
    pipeline.progress_callback = None
    pipeline.lut_path = None
    pipeline.sharpen_strength = 0.0
    pipeline.retarget_high_fps = False
    pipeline.segments = (SegmentRange(0.5, 9.5),)
    pipeline.working_dir = tmp_path
    pipeline._cancel_event = threading.Event()
    pipeline._run_pass = MagicMock(
        return_value=SimpleNamespace(
            system_pressure_episodes=0,
            system_reclaim_count=0,
        )
    )
    metadata = SimpleNamespace(
        video_fps=30.0,
        average_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        duration=10.0,
        profile="Main 10",
        codec_name="hevc",
        is_10bit=True,
        video_width=8192,
        video_height=4096,
    )
    pipeline.splice_plan = SplicePlan(
        index=KeyframeIndex((0, 300), Fraction(1, 30), 0, 300),
        spans=(SpliceSpan("render", 0, 300, ((15, 285),)),),
        segments=pipeline.segments,
    )
    workspace = _mock_workspace(tmp_path)
    captured_signature_plan = []
    monkeypatch.setenv(ISOLATED_VIDEO_JOB_ENV, "1")

    with (
        patch("jasna.pipeline.validate_smart_render", return_value="hevc"),
        patch("jasna.pipeline.vendor_for_device", return_value=AcceleratorVendor.AMD),
        patch("jasna.pipeline.amf_render_session_seconds", return_value=4.0),
        patch("jasna.pipeline.resolve_smart_encoder_settings", return_value={}),
        patch(
            "jasna.pipeline.workspace_signature",
            side_effect=lambda **kwargs: captured_signature_plan.append(kwargs["plan"])
            or {},
        ),
        patch("jasna.pipeline.SmartRenderWorkspace.open", return_value=workspace),
        patch(
            "jasna.pipeline.resolve_hevc_smart_render_vui",
            return_value=(metadata, Fraction(30, 1)),
        ),
        patch("jasna.pipeline.VideoEncoder") as encoder,
        patch("jasna.pipeline.normalize_fragment"),
        patch("jasna.pipeline.mux_fragments_final_output") as mux,
        pytest.raises(NativeWorkerRecycleRequested) as caught,
    ):
        pipeline._run_smart(metadata)

    assert caught.value.reason == "amf_session_limit"
    assert [
        (span.start_pts, span.end_pts, span.effect_ranges)
        for span in captured_signature_plan[0].render_spans
    ] == [
        (0, 120, ((15, 120),)),
        (120, 240, ((120, 240),)),
        (240, 300, ((240, 285),)),
    ]
    assert encoder.call_args.kwargs["pts_origin"] == 0
    assert pipeline._run_pass.call_args.kwargs["end_pts"] == 120
    workspace.mark_complete.assert_called_once()
    workspace.cleanup.assert_not_called()
    mux.assert_not_called()


def test_isolated_8k_main8_dual_gop_uses_bounded_amf_sessions(
    monkeypatch,
) -> None:
    from jasna.pipeline import _bounded_amf_render_session_seconds

    metadata = SimpleNamespace(
        is_10bit=False,
        video_width=8192,
        video_height=4096,
    )
    monkeypatch.setenv(ISOLATED_VIDEO_JOB_ENV, "1")

    with patch("jasna.pipeline.amf_render_session_seconds", return_value=120.0):
        assert _bounded_amf_render_session_seconds(
            metadata=metadata,
            codec="hevc",
            vendor=AcceleratorVendor.AMD,
            batch_size=4,
            dual_gop_enabled=True,
        ) == 120.0
        assert _bounded_amf_render_session_seconds(
            metadata=metadata,
            codec="hevc",
            vendor=AcceleratorVendor.AMD,
            batch_size=4,
            dual_gop_enabled=False,
        ) is None
        assert _bounded_amf_render_session_seconds(
            metadata=metadata,
            codec="hevc",
            vendor=AcceleratorVendor.AMD,
            batch_size=4,
            dual_gop_enabled=True,
            retarget_high_fps=True,
        ) is None
        assert _bounded_amf_render_session_seconds(
            metadata=metadata,
            codec="h264",
            vendor=AcceleratorVendor.AMD,
            batch_size=4,
            dual_gop_enabled=False,
        ) is None


def test_bounded_full_render_reopens_encoder_per_fragment(monkeypatch, tmp_path):
    pipeline = make_pipeline()
    pipeline.input_video = tmp_path / "input.mp4"
    pipeline.output_video = tmp_path / "output.mp4"
    pipeline.codec = "hevc"
    pipeline.encoder_settings = {"rc": "vbr_peak"}
    pipeline.device = torch.device("cuda:0")
    pipeline.working_dir = tmp_path
    pipeline.lut_path = None
    pipeline.sharpen_strength = 0.0
    pipeline.auto_source_rate = True
    pipeline.amd_dual_gop_encode = True
    pipeline.effect_ranges = None
    pipeline._cancel_event = threading.Event()
    pipeline._run_pass = MagicMock()
    progress = MagicMock()
    # The source is 30 fps but the active output contract is 24 fps.  Bounded
    # full fragments must size their worker pass from output_fps, not the
    # source metadata float.
    frame_rate = SimpleNamespace(output_fps=Fraction(24, 1))
    index = KeyframeIndex(
        (0,),
        Fraction(1, 30),
        0,
        180,
    )
    plan = SplicePlan(
        index=index,
        spans=(SpliceSpan("render", 0, 180),),
        segments=(),
    )
    workspace = _mock_workspace(tmp_path)

    with (
        patch("jasna.pipeline.probe_keyframes", return_value=index),
        patch("jasna.pipeline.split_render_spans", return_value=SplicePlan(
            index=index,
            spans=(
                SpliceSpan("render", 0, 90),
                SpliceSpan("render", 90, 180),
            ),
            segments=(),
        )),
        patch("jasna.pipeline.VideoEncoder"),
        patch("jasna.pipeline.workspace_signature", return_value={}),
        patch("jasna.pipeline.SmartRenderWorkspace.open", return_value=workspace),
        patch("jasna.pipeline.normalize_fragment") as normalize,
        patch("jasna.pipeline.validate_hevc_fragment_parameter_sets"),
        patch("jasna.pipeline.mux_fragments_final_output") as mux,
    ):
        pipeline._run_bounded_full(
            SimpleNamespace(video_fps=30.0),
            frame_rate=frame_rate,
            progress=progress,
            output_frame_count=180,
            max_duration_seconds=3.0,
        )

    assert pipeline._run_pass.call_count == 2
    assert [call.kwargs["output_frame_count"] for call in pipeline._run_pass.call_args_list] == [72, 72]
    # The bounded-full plan uses an empty effect_ranges tuple for persistence,
    # but the decoder sentinel for "process the whole video" is None.  Passing
    # () here would suppress detection/restoration for every frame.
    assert all(
        call.kwargs["effect_ranges"] is None
        for call in pipeline._run_pass.call_args_list
    )
    assert normalize.call_count == 2
    assert all(
        call.kwargs["decode_delay"] == Fraction(0, 1)
        for call in normalize.call_args_list
    )
    mux.assert_called_once()


def test_bounded_full_workspace_identity_ignores_attempt_staging_path(
    monkeypatch,
    tmp_path,
):
    """A recycled child must reopen the canonical workspace, not a UUID path."""

    pipeline = make_pipeline()
    pipeline.input_video = tmp_path / "input.mp4"
    pipeline.output_video = tmp_path / ".output.jasna-full-attempt.mp4"
    pipeline.workspace_output = tmp_path / "output.mp4"
    pipeline.codec = "hevc"
    pipeline.encoder_settings = {"rc": "vbr_peak"}
    pipeline.device = torch.device("cuda:0")
    pipeline.working_dir = tmp_path
    pipeline.lut_path = None
    pipeline.sharpen_strength = 0.0
    pipeline.auto_source_rate = True
    pipeline.amd_dual_gop_encode = True
    pipeline.effect_ranges = None
    pipeline._cancel_event = threading.Event()
    pipeline._run_pass = MagicMock()
    workspace = _mock_workspace(tmp_path)
    index = KeyframeIndex((0,), Fraction(1, 30), 0, 90)
    plan = SplicePlan(
        index=index,
        spans=(SpliceSpan("render", 0, 90),),
        segments=(),
    )
    monkeypatch.delenv(ISOLATED_VIDEO_JOB_ENV, raising=False)

    with (
        patch("jasna.pipeline.probe_keyframes", return_value=index),
        patch("jasna.pipeline.split_render_spans", return_value=plan),
        patch("jasna.pipeline.workspace_signature", return_value={}) as signature,
        patch("jasna.pipeline.SmartRenderWorkspace.open", return_value=workspace) as open_workspace,
        patch("jasna.pipeline.VideoEncoder"),
        patch("jasna.pipeline.normalize_fragment"),
        patch("jasna.pipeline.validate_hevc_fragment_parameter_sets"),
        patch("jasna.pipeline.mux_fragments_final_output"),
    ):
        pipeline._run_bounded_full(
            SimpleNamespace(video_fps=30.0),
            frame_rate=SimpleNamespace(output_fps=Fraction(30, 1)),
            progress=MagicMock(),
            output_frame_count=90,
            max_duration_seconds=3.0,
        )

    assert signature.call_args.kwargs["output"] == pipeline.workspace_output
    assert open_workspace.call_args.kwargs["output"] == pipeline.workspace_output
    assert signature.call_args.kwargs["processing"][
        "bounded_full_effect_ranges_semantics"
    ] == "all-frames-none-v1"


def test_isolated_bounded_full_recycles_and_resumes_from_workspace(
    monkeypatch,
    tmp_path,
    caplog,
):
    pipeline = make_pipeline()
    pipeline.input_video = tmp_path / "input.mp4"
    pipeline.output_video = tmp_path / "output.mp4"
    pipeline.codec = "hevc"
    pipeline.encoder_settings = {"rc": "vbr_peak"}
    pipeline.device = torch.device("cuda:0")
    pipeline.working_dir = tmp_path
    pipeline.lut_path = None
    pipeline.sharpen_strength = 0.0
    pipeline.auto_source_rate = True
    pipeline.amd_dual_gop_encode = True
    pipeline.effect_ranges = None
    pipeline._cancel_event = threading.Event()
    pipeline._run_pass = MagicMock()
    progress = MagicMock()
    frame_rate = SimpleNamespace(output_fps=Fraction(30, 1))
    index = KeyframeIndex((0,), Fraction(1, 30), 0, 180)
    bounded_plan = SplicePlan(
        index=index,
        spans=(
            SpliceSpan("render", 0, 90),
            SpliceSpan("render", 90, 180),
        ),
        segments=(),
    )
    workspace = _mock_workspace(tmp_path)
    completed_zero = workspace.path / "0000.ts"
    workspace.reusable_fragment.side_effect = [None]
    monkeypatch.setenv(ISOLATED_VIDEO_JOB_ENV, "1")

    with (
        patch("jasna.pipeline.probe_keyframes", return_value=index),
        patch("jasna.pipeline.split_render_spans", return_value=bounded_plan),
        patch("jasna.pipeline.workspace_signature", return_value={}),
        patch("jasna.pipeline.SmartRenderWorkspace.open", return_value=workspace),
        patch("jasna.pipeline.VideoEncoder"),
        patch("jasna.pipeline.normalize_fragment"),
        patch("jasna.pipeline.mux_fragments_final_output") as mux,
        pytest.raises(NativeWorkerRecycleRequested, match="fragment 0"),
    ):
        pipeline._run_bounded_full(
            SimpleNamespace(video_fps=30.0),
            frame_rate=frame_rate,
            progress=progress,
            output_frame_count=180,
            max_duration_seconds=3.0,
        )

    assert pipeline._run_pass.call_count == 1
    workspace.mark_complete.assert_called_once_with(0, workspace.path / "0000.ts")
    mux.assert_not_called()
    workspace.cleanup.assert_not_called()
    assert not any(record.levelno >= logging.WARNING for record in caplog.records)

    # A fresh isolated child sees the durable completed fragment and only
    # renders the remaining suffix before assembling the final output.
    pipeline._run_pass.reset_mock()
    workspace.mark_complete.reset_mock()
    workspace.reusable_fragment.side_effect = [completed_zero, None]
    with (
        patch("jasna.pipeline.probe_keyframes", return_value=index),
        patch("jasna.pipeline.split_render_spans", return_value=bounded_plan),
        patch("jasna.pipeline.workspace_signature", return_value={}),
        patch("jasna.pipeline.SmartRenderWorkspace.open", return_value=workspace),
        patch("jasna.pipeline.VideoEncoder"),
        patch("jasna.pipeline.normalize_fragment"),
        patch("jasna.pipeline.validate_hevc_fragment_parameter_sets"),
        patch("jasna.pipeline.mux_fragments_final_output") as resumed_mux,
    ):
        pipeline._run_bounded_full(
            SimpleNamespace(video_fps=30.0),
            frame_rate=frame_rate,
            progress=progress,
            output_frame_count=180,
            max_duration_seconds=3.0,
        )

    pipeline._run_pass.assert_called_once()
    assert pipeline._run_pass.call_args.kwargs["seek_ts"] == 3.0
    resumed_mux.assert_called_once()
    workspace.cleanup.assert_called_once_with()

    # A genuine failure must still retain its workspace and surface an ERROR;
    # only the typed, successful bounded-session control flow is quiet.
    caplog.clear()
    workspace.cleanup.reset_mock()
    workspace.reusable_fragment.side_effect = [None]
    pipeline._run_pass.side_effect = RuntimeError("HIP illegal memory access")
    with (
        patch("jasna.pipeline.probe_keyframes", return_value=index),
        patch("jasna.pipeline.split_render_spans", return_value=bounded_plan),
        patch("jasna.pipeline.workspace_signature", return_value={}),
        patch("jasna.pipeline.SmartRenderWorkspace.open", return_value=workspace),
        patch("jasna.pipeline.VideoEncoder"),
        pytest.raises(RuntimeError, match="HIP illegal memory access"),
    ):
        pipeline._run_bounded_full(
            SimpleNamespace(video_fps=30.0), frame_rate=frame_rate,
            progress=progress, output_frame_count=180, max_duration_seconds=3.0,
        )
    workspace.cleanup.assert_not_called()
    assert any(record.levelno == logging.ERROR and "Preserving failed/resumable" in record.message
               for record in caplog.records)


def test_bounded_full_plan_rejects_pts_gap_or_overlap() -> None:
    index = KeyframeIndex((0,), Fraction(1, 30), 0, 180)
    plan = SplicePlan(
        index=index,
        spans=(
            SpliceSpan("render", 0, 90),
            SpliceSpan("render", 91, 180),
        ),
        segments=(),
    )

    with pytest.raises(RuntimeError, match="PTS gap or overlap"):
        Pipeline._validate_bounded_full_plan(plan)


def test_bounded_full_plan_rejects_uncovered_tail() -> None:
    index = KeyframeIndex((0,), Fraction(1, 30), 0, 180)
    plan = SplicePlan(
        index=index,
        spans=(SpliceSpan("render", 0, 90),),
        segments=(),
    )

    with pytest.raises(RuntimeError, match="does not cover"):
        Pipeline._validate_bounded_full_plan(plan)


def test_hevc_smart_run_checks_parameter_sets_and_bounded_copy_gops(tmp_path) -> None:
    pipeline = make_pipeline()
    pipeline.input_video = tmp_path / "input.mkv"
    pipeline.output_video = tmp_path / "output.mkv"
    pipeline.codec = "hevc"
    pipeline.encoder_settings = {"cq": 22}
    pipeline.device = torch.device("cuda:0")
    pipeline.disable_progress = True
    pipeline.progress_callback = None
    pipeline.lut_path = None
    pipeline.sharpen_strength = 0.0
    pipeline.retarget_high_fps = False
    pipeline.amd_dual_gop_encode = True
    pipeline.segments = (SegmentRange(2.5, 3.0),)
    pipeline.working_dir = tmp_path
    pipeline._cancel_event = threading.Event()
    pipeline._run_pass = MagicMock()
    metadata = MagicMock(
        video_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        duration=6.0,
        profile="Main 10",
    )
    plan = SplicePlan(
        index=KeyframeIndex(
            (0, 30, 60, 90, 120, 150),
            Fraction(1, 30),
            0,
            180,
        ),
        spans=(
            SpliceSpan("copy", 0, 60),
            SpliceSpan("render", 60, 120, ((75, 90),)),
            SpliceSpan("copy", 120, 180),
        ),
        segments=pipeline.segments,
    )
    pipeline.splice_plan = plan
    render_metadata = MagicMock(name="render_metadata")
    workspace = _mock_workspace(tmp_path)

    with (
        patch("jasna.pipeline.validate_smart_render", return_value="hevc"),
        patch("jasna.pipeline.resolve_smart_encoder_settings", return_value={}),
        patch("jasna.pipeline.workspace_signature", return_value={}),
        patch("jasna.pipeline.SmartRenderWorkspace.open", return_value=workspace),
        patch(
            "jasna.pipeline.resolve_hevc_smart_render_vui",
            return_value=(render_metadata, Fraction(60_000, 1_001)),
        ) as resolve_vui,
        patch("jasna.pipeline.VideoEncoder") as encoder,
        patch("jasna.pipeline.create_normalized_copy_fragment"),
        patch("jasna.pipeline.normalize_fragment"),
        patch("jasna.pipeline.validate_hevc_fragment_parameter_sets") as validate_sets,
        patch("jasna.pipeline.mux_fragments_final_output") as mux,
    ):
        pipeline._run_smart(metadata)

    resolve_vui.assert_called_once_with(metadata)
    assert encoder.call_args.kwargs["metadata"] is render_metadata
    assert encoder.call_args.kwargs["output_fps"] == Fraction(60_000, 1_001)
    assert encoder.call_args.kwargs["prefer_amf_host_native"] is True
    fragment_paths = [fragment for fragment, _duration in mux.call_args.args[0]]
    validate_sets.assert_called_once_with(
        [
            (fragment_paths[0], "copy"),
            (fragment_paths[1], "render"),
            (fragment_paths[2], "copy"),
        ]
    )
    assert mux.call_args.kwargs["copy_validation_ranges"] == (
        (1.0, 1.0),
        (4.0, 1.0),
    )


def test_hevc_copy_validation_ranges_cap_long_gops_to_one_second() -> None:
    plan = SplicePlan(
        index=KeyframeIndex(
            (0, 900, 1800),
            Fraction(1, 30),
            0,
            2700,
        ),
        spans=(
            SpliceSpan("copy", 0, 900),
            SpliceSpan("render", 900, 1800, ((1000, 1100),)),
            SpliceSpan("copy", 1800, 2700),
        ),
        segments=(SegmentRange(33.0, 36.0),),
    )

    assert Pipeline._hevc_copy_validation_ranges(plan) == (
        (29.0, 1.0),
        (60.0, 1.0),
    )


def test_hevc_smart_run_rebuilds_only_stale_reusable_copy_span(tmp_path) -> None:
    pipeline = make_pipeline()
    pipeline.input_video = tmp_path / "input.mkv"
    pipeline.output_video = tmp_path / "output.mkv"
    pipeline.codec = "hevc"
    pipeline.encoder_settings = {"cq": 22}
    pipeline.device = torch.device("cuda:0")
    pipeline.disable_progress = True
    pipeline.progress_callback = None
    pipeline.lut_path = None
    pipeline.sharpen_strength = 0.0
    pipeline.retarget_high_fps = False
    pipeline.segments = (SegmentRange(2.5, 3.0),)
    pipeline.working_dir = tmp_path
    pipeline._cancel_event = threading.Event()
    pipeline._run_pass = MagicMock()
    metadata = MagicMock(
        video_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        duration=4.0,
        profile="Main 10",
    )
    copy_span = SpliceSpan("copy", 0, 60)
    render_span = SpliceSpan("render", 60, 120, ((75, 90),))
    pipeline.splice_plan = SplicePlan(
        index=KeyframeIndex((0, 60), Fraction(1, 30), 0, 120),
        spans=(copy_span, render_span),
        segments=pipeline.segments,
    )
    workspace = _mock_workspace(tmp_path)
    stale_copy = workspace.fragment_path(0, ".ts")
    reusable_render = workspace.fragment_path(1, ".ts")
    workspace.reusable_fragment.side_effect = (stale_copy, reusable_render)

    with (
        patch("jasna.pipeline.validate_smart_render", return_value="hevc"),
        patch("jasna.pipeline.resolve_smart_encoder_settings", return_value={}),
        patch("jasna.pipeline.workspace_signature", return_value={}),
        patch("jasna.pipeline.SmartRenderWorkspace.open", return_value=workspace),
        patch(
            "jasna.pipeline.hevc_copy_fragment_timeline_matches_source",
            return_value=False,
        ) as timeline_matches,
        patch("jasna.pipeline.create_normalized_copy_fragment") as create_copy,
        patch("jasna.pipeline.normalize_fragment") as normalize,
        patch("jasna.pipeline.validate_hevc_fragment_parameter_sets"),
        patch("jasna.pipeline.mux_fragments_final_output") as mux,
    ):
        pipeline._run_smart(metadata)

    timeline_matches.assert_called_once_with(
        stale_copy,
        pipeline.input_video,
        copy_span,
        pipeline.splice_plan.index,
    )
    create_copy.assert_called_once()
    normalize.assert_not_called()
    workspace.mark_running.assert_called_once_with(0)
    workspace.mark_complete.assert_called_once_with(0, stale_copy)
    assert pipeline._run_pass.call_count == 0
    assert mux.call_args.args[0][1][0] == reusable_render


def test_smart_run_uses_working_dir_for_temp_files(tmp_path) -> None:
    pipeline = make_pipeline()
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
    workspace = _mock_workspace(tmp_path)

    with (
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch("jasna.pipeline.workspace_signature", return_value={}),
        patch("jasna.pipeline.SmartRenderWorkspace.open", return_value=workspace) as workspace_open,
        patch("jasna.pipeline.VideoEncoder"),
        patch("jasna.pipeline.create_normalized_copy_fragment"),
        patch("jasna.pipeline.normalize_fragment"),
        patch("jasna.pipeline.mux_fragments_final_output") as mux,
    ):
        pipeline._run_smart(metadata)

    workspace_open.assert_called_once_with(
        pipeline.working_dir,
        output=pipeline.output_video,
        signature={},
    )
    fragments = mux.call_args.args[0]
    assert all(fragment.parent == workspace.path for fragment, _ in fragments)
    assert mux.call_args.kwargs["manifest"].parent == workspace.path
    assert pipeline.working_dir.is_dir()
    assert pipeline.output_video.parent.is_dir()


def test_amf_h264_full_reencode_preserves_selected_ranges(tmp_path) -> None:
    pipeline = make_pipeline()
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
    pipeline = make_pipeline()
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
    pipeline = make_pipeline()
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
    pipeline = make_pipeline()
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
    workspace = _mock_workspace(pipeline.output_video.parent)
    with (
        patch("jasna.pipeline.vendor_for_device", return_value=AcceleratorVendor.NVIDIA),
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch("jasna.pipeline.resolve_smart_encoder_settings", return_value={}),
        patch("jasna.pipeline.workspace_signature", return_value={}),
        patch("jasna.pipeline.SmartRenderWorkspace.open", return_value=workspace),
        patch("jasna.pipeline.VideoEncoder"),
        patch("jasna.pipeline.create_copy_fragment"),
        patch("jasna.pipeline.create_normalized_copy_fragment"),
        patch("jasna.pipeline.normalize_fragment"),
        patch("jasna.pipeline.mux_fragments_final_output") as concatenate,
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
    pipeline = make_pipeline()
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
