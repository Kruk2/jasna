from __future__ import annotations

import threading
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from factories import make_pipeline
from jasna.accelerator import AcceleratorVendor
from jasna.media.splice import KeyframeIndex, SmartRenderCompatibilityError, SplicePlan, SpliceSpan
from jasna.pipeline import Pipeline
from jasna.segments import SegmentRange, SegmentRestoration


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
    pipeline.ltx_window_hop = 104
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



def test_mixed_job_restores_basicvsrpp_spans_before_the_ltx_spans(tmp_path) -> None:
    ltx = SegmentRestoration("ltx", 42)
    segments = (SegmentRange(2.5, 3.0, ltx), SegmentRange(6.5, 7.0))
    spans = (
        SpliceSpan("copy", 0, 60),
        SpliceSpan("render", 60, 120, ((75, 90),)),
        SpliceSpan("copy", 120, 180),
        SpliceSpan("render", 180, 240, ((195, 210),)),
    )
    pipeline = _mixed_pipeline(tmp_path, "basicvsrpp", segments, spans)
    order = MagicMock()
    order.attach_mock(pipeline._run_pass, "basicvsrpp")
    order.attach_mock(pipeline._run_ltx_spans, "ltx")

    concatenate = _run_smart_mocked(pipeline)

    assert [call[0] for call in order.mock_calls] == ["basicvsrpp", "ltx"]
    fragments = concatenate.call_args.args[0]
    assert [path.name for path, _ in fragments] == ["0000.ts", "0001.ts", "0002.ts", "0003.ts"]


def test_mixed_job_reports_time_left_from_the_ltx_estimate_until_ltx_reports_its_own(tmp_path) -> None:
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
    pipeline.ltx_files = MagicMock(transformer_name="t")

    with (
        patch("jasna.ltx.speed.segment_estimate_seconds", return_value=900.0) as estimate,
        patch("torch.cuda.get_device_name", return_value="Test GPU"),
    ):
        _run_smart_mocked(pipeline)

    assert estimate.call_args.kwargs["frames"] == 60
    assert estimate.call_args.kwargs["window_hop"] == 104
    standard_callback = pipeline._run_pass.call_args.kwargs["progress"].callback
    standard_callback(50.0, 30.0, 10.0, 30, 60, None)
    assert pipeline.progress_callback.call_args.args[2] == pytest.approx(910.0)
    (*_, ltx_callback), _ = pipeline._run_ltx_spans.call_args
    ltx_callback(0.0, 0.0, 0.0, 0, 0, "scan")
    assert pipeline.progress_callback.call_args.args[2] == pytest.approx(900.0)
    ltx_callback(50.0, 0.0, 120.0, 0, 0, "compose")
    assert pipeline.progress_callback.call_args.args[2] == pytest.approx(120.0)


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


def test_a_sharpened_job_encodes_the_parts_between_ranges_again_so_sharpening_covers_them(tmp_path) -> None:
    segments = (SegmentRange(2.5, 3.0),)
    spans = (SpliceSpan("copy", 0, 60), SpliceSpan("render", 60, 120, ((75, 90),)), SpliceSpan("copy", 120, 240))
    pipeline = _mixed_pipeline(tmp_path, "basicvsrpp", segments, spans)
    pipeline.sharpen_strength = 0.4
    pipeline._reencode_span = MagicMock()

    _run_smart_mocked(pipeline)

    assert [reencode.args[2] for reencode in pipeline._reencode_span.call_args_list] == [spans[0], spans[2]]


def test_a_lut_job_encodes_the_parts_between_ranges_again_so_the_lut_covers_them(tmp_path) -> None:
    ltx = SegmentRestoration("ltx", 42)
    segments = (SegmentRange(2.5, 3.0), SegmentRange(6.5, 7.0, ltx))
    spans = (
        SpliceSpan("copy", 0, 60),
        SpliceSpan("render", 60, 120, ((75, 90),)),
        SpliceSpan("copy", 120, 180),
        SpliceSpan("render", 180, 240, ((195, 210),)),
    )
    pipeline = _mixed_pipeline(tmp_path, "basicvsrpp", segments, spans)
    pipeline.lut_path = "look.cube"
    pipeline._reencode_span = MagicMock()

    with (
        patch("jasna.pipeline.vendor_for_device", return_value=AcceleratorVendor.NVIDIA),
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch("jasna.pipeline.resolve_smart_encoder_settings", return_value={}),
        patch("jasna.pipeline.VideoEncoder") as encoder,
        patch("jasna.pipeline.Progressbar") as progressbar,
        patch("jasna.pipeline.create_copy_fragment") as copy_fragment,
        patch("jasna.pipeline.normalize_fragment"),
        patch("jasna.pipeline.concatenate_fragments") as concatenate,
        patch("jasna.pipeline.mux_final_output"),
    ):
        pipeline._run_smart(MagicMock(duration=8.0, video_fps=30.0, video_fps_exact=Fraction(30, 1)))

    copy_fragment.assert_not_called()
    assert [reencode.args[2] for reencode in pipeline._reencode_span.call_args_list] == [spans[0], spans[2]]
    assert {made.kwargs["lut_path"] for made in encoder.call_args_list} == {"look.cube"}
    assert progressbar.call_args.kwargs["total_frames"] == 180
    assert len(concatenate.call_args.args[0]) == 4


def test_a_span_that_cannot_be_copied_is_encoded_again_without_restoring(tmp_path) -> None:
    segments = (SegmentRange(2.5, 3.0),)
    spans = (SpliceSpan("reencode", 0, 60), SpliceSpan("render", 60, 120, ((75, 90),)), SpliceSpan("copy", 120, 240))
    pipeline = _mixed_pipeline(tmp_path, "basicvsrpp", segments, spans)
    pipeline._reencode_span = MagicMock()

    with patch("jasna.pipeline.Progressbar") as progressbar:
        _run_smart_mocked(pipeline)

    assert [reencode.args[2] for reencode in pipeline._reencode_span.call_args_list] == [spans[0]]
    assert [restored.kwargs["end_pts"] for restored in pipeline._run_pass.call_args_list] == [120]
    assert progressbar.call_args.kwargs["total_frames"] == 120


def test_reencoded_span_writes_every_decoded_frame_and_closes_its_writer(tmp_path) -> None:
    pipeline = _mixed_pipeline(tmp_path, "basicvsrpp", (), ())
    pipeline.batch_size = 2
    index = KeyframeIndex((0, 60), Fraction(1, 30), 0, 120)
    batches = [(torch.zeros(2, 3, 4, 4), [60, 61]), (torch.zeros(1, 3, 4, 4), [62])]
    writer, progress = MagicMock(), MagicMock()

    with patch("jasna.ltx.restore.video_frames", return_value=lambda: iter(batches)) as frames:
        pipeline._reencode_span(MagicMock(), index, SpliceSpan("copy", 60, 120), writer, progress)

    assert (frames.call_args.kwargs["seek_ts"], frames.call_args.kwargs["end_pts"]) == (2.0, 120)
    assert [written.args[1] for written in writer.write.call_args_list] == [60, 61, 62]
    assert [update.args[0] for update in progress.update.call_args_list] == [2, 1]
    writer.finish.assert_called_once_with()
    writer.close.assert_called_once_with()


@pytest.mark.parametrize(
    ("model", "segments", "expected"),
    [
        ("basicvsrpp", None, "_run_full"),
        ("ltx", None, "_run_ltx"),
        ("ltx", (SegmentRange(1, 2),), "_run_smart"),
    ],
)
def test_run_routes_by_the_job_model_and_segments(tmp_path, model, segments, expected) -> None:
    pipeline = make_pipeline()
    pipeline.restoration_model_name = model
    pipeline.segments = segments
    pipeline.input_video = tmp_path / "input.mp4"
    pipeline.input_video.touch()
    pipeline.output_video = tmp_path / "output.mp4"
    pipeline.device = torch.device("cpu")
    pipeline.codec = "hevc"
    pipeline.fmp4 = False
    pipeline.ltx_files = MagicMock()
    pipeline._cancel_event = threading.Event()
    pipeline.validate_metadata = MagicMock()
    pipeline.configure_vr = MagicMock()
    for name in ("_run_full", "_run_ltx", "_run_smart"):
        setattr(pipeline, name, MagicMock())

    with (
        patch("jasna.pipeline.get_video_meta_data", return_value=MagicMock(num_frames=120, time_base=Fraction(1,30), duration=4.0)),
        patch("jasna.pipeline.require_ltx_input"),
    ):
        pipeline.run()

    assert [name for name in ("_run_full", "_run_ltx", "_run_smart") if getattr(pipeline, name).called] == [expected]


def test_ltx_smart_resume_keeps_completed_fragments_and_retries_partial_span(tmp_path):
    from jasna.job_resume import ResumeJob

    segments = (SegmentRange(2.5, 3.0, SegmentRestoration("ltx", 42)), SegmentRange(6.5, 7.0))
    spans = (
        SpliceSpan("copy", 0, 60), SpliceSpan("render", 60, 120, ((75, 90),)),
        SpliceSpan("copy", 120, 180), SpliceSpan("render", 180, 240, ((195, 210),)),
    )
    pipeline = _mixed_pipeline(tmp_path, "basicvsrpp", segments, spans)
    pipeline.input_video.touch()
    metadata = MagicMock(duration=8.0, num_frames=240, time_base=Fraction(1, 30), video_fps=30.0, video_fps_exact=Fraction(30, 1))
    pipeline.resume = ResumeJob(pipeline.config, pipeline.input_video, pipeline.output_video, metadata, segments)
    pipeline.resume.prepare()
    partial = pipeline.resume.directory / "segments" / "0001-partial.nut"
    completed = partial.with_name("0001-raw.nut")
    restored = []

    class Encoder:
        def __init__(self, path, **kwargs):
            self.path = Path(path)

        def __enter__(self):
            self.path.write_bytes(b"encoded")

        def encode(self, frame, pts):
            pass

        def __exit__(self, *exc):
            pass

    def ltx_run(_metadata, _index, pending, *_args):
        restored.extend(span for span, _, _ in pending)
        writer = pending[0][2]()
        writer.write(torch.zeros(1), 75)
        if len(restored) == 1:
            writer.close()
            raise RuntimeError("compose interrupted")
        writer.finish()

    pipeline._run_ltx_spans = ltx_run
    pipeline._run_pass = MagicMock(side_effect=lambda **kwargs: kwargs["encoder_ctx"].path.write_bytes(b"basicvsrpp"))
    with (
        patch("jasna.pipeline.vendor_for_device", return_value=AcceleratorVendor.NVIDIA),
        patch("jasna.pipeline.validate_smart_render", return_value="h264"),
        patch("jasna.pipeline.resolve_smart_encoder_settings", return_value={}),
        patch("jasna.pipeline.VideoEncoder", Encoder),
        patch("jasna.pipeline.create_copy_fragment", side_effect=lambda _source, _span, _index, path, **_k: path.write_bytes(b"copy")) as copies,
        patch("jasna.pipeline.normalize_fragment"), patch("jasna.pipeline.concatenate_fragments"), patch("jasna.pipeline.mux_final_output") as mux,
    ):
        with pytest.raises(RuntimeError, match="compose interrupted"):
            pipeline._run_smart(metadata)
        assert partial.exists() and not completed.exists()
        assert pipeline._run_pass.call_count == 1
        assert copies.call_count == 2
        pipeline._run_smart(metadata)
        assert completed.read_bytes() == b"encoded"
        assert pipeline._run_pass.call_count == 1
        assert copies.call_count == 2
        mux.assert_called_once()
    assert restored == [spans[1], spans[1]]


@pytest.mark.parametrize("cancelled", [False, True])
def test_pipeline_keeps_ltx_checkpoint_until_success(tmp_path, cancelled):
    source = tmp_path / "in.mp4"
    source.touch()
    pipeline = make_pipeline(input_video=source, output_video=tmp_path / "out.mp4", restoration_model_name="ltx")
    if cancelled:
        pipeline.cancel()
    with (
        patch("jasna.pipeline.get_video_meta_data", return_value=MagicMock(num_frames=120, duration=4.0, time_base=Fraction(1,30))),
        patch.object(pipeline, "validate_metadata"), patch.object(pipeline, "configure_vr"), patch.object(pipeline, "_run_ltx"),
        patch("jasna.pipeline.require_ltx_input"),
    ):
        pipeline.run()
    assert pipeline.resume.root.exists() is cancelled
    assert pipeline.completed is not cancelled
