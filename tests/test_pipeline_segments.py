from __future__ import annotations

from contextlib import contextmanager
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from factories import make_pipeline, video_metadata
from jasna.accelerator import AcceleratorVendor
from jasna.job_resume import ResumeJob
from jasna.media.splice import KeyframeIndex, SmartRenderCompatibilityError, SplicePlan, SpliceSpan
from jasna.pipeline import Pipeline
from jasna.segments import SegmentRange, SegmentRestoration

LTX = SegmentRestoration("ltx", 42)
METADATA = video_metadata(fps=30, num_frames=240, profile="Main")
FOUR_SPANS = (
    SpliceSpan("copy", 0, 60),
    SpliceSpan("render", 60, 120, ((75, 90),)),
    SpliceSpan("copy", 120, 180),
    SpliceSpan("render", 180, 240, ((195, 210),)),
)
ONE_RENDER = (SpliceSpan("copy", 0, 60), SpliceSpan("render", 60, 120, ((75, 90),)), SpliceSpan("copy", 120, 240))


def _smart_pipeline(tmp_path, segments, spans, *, max_b_frames=0, **overrides) -> Pipeline:
    settings = dict(
        input_video=tmp_path / "input.mp4",
        output_video=tmp_path / "output.mp4",
        codec="h264",
        encoder_settings={},
        disable_progress=True,
        ltx_seed=5,
        ltx_window_hop=104,
    )
    settings.update(overrides)
    return make_pipeline(
        segments=segments,
        splice_plan=SplicePlan(
            index=KeyframeIndex((0, 60, 120, 180), Fraction(1, 30), 0, 240, max_b_frames=max_b_frames),
            spans=spans,
            segments=segments,
        ),
        **settings,
    )


@contextmanager
def _smart_run(vendor=AcceleratorVendor.NVIDIA):
    """Patch out decoding, encoding and the restorers; yields their mocks by name."""
    with (
        patch("jasna.pipeline.vendor_for_device", return_value=vendor),
        patch("jasna.pipeline.VideoEncoder") as encoder,
        patch("jasna.pipeline.Progressbar") as progressbar,
        patch("jasna.pipeline.create_copy_fragment") as copy_fragment,
        patch("jasna.pipeline.normalize_fragment"),
        patch("jasna.pipeline.concatenate_fragments") as concatenate,
        patch("jasna.pipeline.mux_final_output") as mux,
        patch.object(Pipeline, "_run_full") as run_full,
        patch.object(Pipeline, "_run_pass") as run_pass,
        patch.object(Pipeline, "_run_ltx_spans") as run_ltx_spans,
        patch.object(Pipeline, "_reencode_span") as reencode_span,
    ):
        yield dict(
            encoder=encoder, progressbar=progressbar, copy_fragment=copy_fragment, concatenate=concatenate,
            mux=mux, run_full=run_full, run_pass=run_pass, run_ltx_spans=run_ltx_spans, reencode_span=reencode_span,
        )


def _fragment_names(mocks) -> list[str]:
    return [path.name for path, _ in mocks["concatenate"].call_args.args[0]]


def test_smart_run_restores_only_render_spans_and_assembles_every_fragment(tmp_path) -> None:
    pipeline = _smart_pipeline(
        tmp_path, (SegmentRange(2.5, 3.0),), ONE_RENDER, max_b_frames=3, encoder_settings={"cq": 22}
    )

    with _smart_run() as mocks:
        pipeline._run_smart(METADATA)

    assert mocks["copy_fragment"].call_count == 2
    encoder_kwargs = mocks["encoder"].call_args.kwargs
    assert (encoder_kwargs["pts_origin"], encoder_kwargs["smart_fragment"]) == (60, True)
    assert encoder_kwargs["encoder_settings"] == {
        "cq": 22, "profile": "main", "g": 60, "bf": 3, "b_ref_mode": "disabled",
    }
    pass_args = mocks["run_pass"].call_args.kwargs
    assert (pass_args["seek_ts"], pass_args["end_pts"], pass_args["effect_ranges"]) == (2.0, 120, ((75, 90),))
    assert _fragment_names(mocks) == ["0000.ts", "0001.ts", "0002.ts"]
    mocks["mux"].assert_called_once()


def test_smart_run_uses_working_dir_for_temp_files(tmp_path) -> None:
    pipeline = _smart_pipeline(
        tmp_path, (SegmentRange(2.5, 3.0),), ONE_RENDER,
        output_video=tmp_path / "out" / "output.mp4", working_dir=tmp_path / "scratch",
    )

    with _smart_run() as mocks:
        pipeline._run_smart(METADATA)

    assert mocks["mux"].call_args.args[0].parent.parent == tmp_path / "scratch"
    assert (tmp_path / "out").is_dir()


@pytest.mark.parametrize(
    ("vendor", "max_b_frames", "full_reencode"),
    [
        (AcceleratorVendor.AMD, 4, True),
        (AcceleratorVendor.AMD, 3, False),
        (AcceleratorVendor.NVIDIA, 4, False),
    ],
)
def test_smart_run_reencodes_everything_only_when_amd_h264_exceeds_its_b_frame_cap(
    tmp_path, vendor, max_b_frames, full_reencode
) -> None:
    pipeline = _smart_pipeline(tmp_path, (SegmentRange(2.5, 3.0),), ONE_RENDER, max_b_frames=max_b_frames)

    with _smart_run(vendor) as mocks:
        pipeline._run_smart(METADATA)

    assert mocks["run_full"].called is full_reencode
    assert mocks["run_pass"].called is not full_reencode
    if full_reencode:
        assert mocks["run_full"].call_args.kwargs["effect_ranges"] == ((75, 90),)


def test_mixed_job_batches_ltx_spans_and_keeps_fragments_in_span_order(tmp_path) -> None:
    pipeline = _smart_pipeline(tmp_path, (SegmentRange(2.5, 3.0), SegmentRange(6.5, 7.0, LTX)), FOUR_SPANS)

    with _smart_run() as mocks:
        pipeline._run_smart(METADATA)

    assert [call.kwargs["effect_ranges"] for call in mocks["run_pass"].call_args_list] == [((75, 90),)]
    (_, _, ltx_spans, _, _), _ = mocks["run_ltx_spans"].call_args
    assert [(span, segs) for span, segs, _ in ltx_spans] == [(FOUR_SPANS[3], (SegmentRange(6.5, 7.0, LTX),))]
    assert _fragment_names(mocks) == ["0000.ts", "0001.ts", "0002.ts", "0003.ts"]


def test_mixed_job_reports_time_left_from_the_ltx_estimate_until_ltx_reports_its_own(tmp_path) -> None:
    pipeline = _smart_pipeline(
        tmp_path, (SegmentRange(2.5, 3.0), SegmentRange(6.5, 7.0, LTX)), FOUR_SPANS, progress_callback=MagicMock()
    )
    pipeline.ltx_files = MagicMock(transformer_name="t")

    with (
        patch("jasna.ltx.speed.segment_estimate_seconds", return_value=900.0) as estimate,
        patch("torch.cuda.get_device_name", return_value="Test GPU"),
        _smart_run() as mocks,
    ):
        pipeline._run_smart(METADATA)

    assert estimate.call_args.kwargs["frames"] == 60
    standard_callback = mocks["progressbar"].call_args.kwargs["callback"]
    standard_callback(50.0, 30.0, 10.0, 30, 60, None)
    assert pipeline.progress_callback.call_args.args[2] == pytest.approx(910.0)
    (*_, ltx_callback), _ = mocks["run_ltx_spans"].call_args
    ltx_callback(0.0, 0.0, 0.0, 0, 0, "scan")
    assert pipeline.progress_callback.call_args.args[2] == pytest.approx(900.0)
    ltx_callback(50.0, 0.0, 120.0, 0, 0, "compose")
    assert pipeline.progress_callback.call_args.args[2] == pytest.approx(120.0)


def test_segments_on_the_job_model_follow_an_ltx_job(tmp_path) -> None:
    pipeline = _smart_pipeline(tmp_path, (SegmentRange(2.5, 3.0),), ONE_RENDER, restoration_model_name="ltx")

    with _smart_run() as mocks:
        pipeline._run_smart(METADATA)

    mocks["run_pass"].assert_not_called()
    (_, _, ltx_spans, _, _), _ = mocks["run_ltx_spans"].call_args
    assert ltx_spans[0][1] == (SegmentRange(2.5, 3.0, SegmentRestoration("ltx", 5)),)


def test_a_render_span_resolving_to_two_models_is_rejected(tmp_path) -> None:
    segments = (SegmentRange(2.2, 2.4), SegmentRange(3.0, 3.2, SegmentRestoration("basicvsrpp", None)))
    spans = (SpliceSpan("copy", 0, 60), SpliceSpan("render", 60, 120, ((66, 72), (90, 96))), SpliceSpan("copy", 120, 240))
    pipeline = _smart_pipeline(tmp_path, segments, spans, restoration_model_name="ltx")

    with _smart_run(), pytest.raises(SmartRenderCompatibilityError):
        pipeline._run_smart(METADATA)


@pytest.mark.parametrize("look", [{"sharpen_strength": 0.4}, {"lut_path": "look.cube"}], ids=["sharpen", "lut"])
def test_a_job_with_a_look_encodes_the_parts_between_ranges_again_so_the_look_covers_them(tmp_path, look) -> None:
    pipeline = _smart_pipeline(tmp_path, (SegmentRange(2.5, 3.0), SegmentRange(6.5, 7.0, LTX)), FOUR_SPANS, **look)
    pipeline.splice_plan = pipeline.splice_plan.reencoding_copies()

    with _smart_run() as mocks:
        pipeline._run_smart(METADATA)

    mocks["copy_fragment"].assert_not_called()
    assert [reencode.args[2].start_pts for reencode in mocks["reencode_span"].call_args_list] == [0, 120]
    assert mocks["progressbar"].call_args.kwargs["total_frames"] == 180
    assert len(_fragment_names(mocks)) == 4


def test_a_span_that_cannot_be_copied_is_encoded_again_without_restoring(tmp_path) -> None:
    spans = (SpliceSpan("reencode", 0, 60), SpliceSpan("render", 60, 120, ((75, 90),)), SpliceSpan("copy", 120, 240))
    pipeline = _smart_pipeline(tmp_path, (SegmentRange(2.5, 3.0),), spans)

    with _smart_run() as mocks:
        pipeline._run_smart(METADATA)

    assert [reencode.args[2] for reencode in mocks["reencode_span"].call_args_list] == [spans[0]]
    assert [restored.kwargs["end_pts"] for restored in mocks["run_pass"].call_args_list] == [120]
    assert mocks["progressbar"].call_args.kwargs["total_frames"] == 120


def test_reencoded_span_writes_every_decoded_frame_and_closes_its_writer(tmp_path) -> None:
    pipeline = _smart_pipeline(tmp_path, (), (), batch_size=2)
    batches = [(torch.zeros(2, 3, 4, 4), [60, 61]), (torch.zeros(1, 3, 4, 4), [62])]
    writer, progress = MagicMock(), MagicMock()

    with patch("jasna.ltx.restore.video_frames", return_value=lambda: iter(batches)) as frames:
        pipeline._reencode_span(METADATA, pipeline.splice_plan.index, SpliceSpan("copy", 60, 120), writer, progress)

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
        ("ltx", (SegmentRange(0.01, 0.02),), "_run_smart"),
    ],
)
def test_run_routes_by_the_job_model_and_segments(tmp_path, model, segments, expected) -> None:
    pipeline = make_pipeline(
        restoration_model_name=model, segments=segments,
        input_video=tmp_path / "input.mp4", output_video=tmp_path / "output.mp4",
    )
    pipeline.input_video.touch()

    with (
        patch("jasna.pipeline.get_video_meta_data", return_value=video_metadata()),
        patch.object(Pipeline, "_run_full") as run_full,
        patch.object(Pipeline, "_run_ltx") as run_ltx,
        patch.object(Pipeline, "_run_smart") as run_smart,
    ):
        pipeline.run()

    ran = {"_run_full": run_full, "_run_ltx": run_ltx, "_run_smart": run_smart}
    assert [name for name, method in ran.items() if method.called] == [expected]


def test_ltx_smart_resume_keeps_completed_fragments_and_retries_partial_span(tmp_path):
    segments = (SegmentRange(2.5, 3.0, LTX), SegmentRange(6.5, 7.0))
    pipeline = _smart_pipeline(tmp_path, segments, FOUR_SPANS)
    pipeline.input_video.touch()
    pipeline.resume = ResumeJob(pipeline.config, pipeline.input_video, pipeline.output_video, METADATA, pipeline.splice_plan)
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

    with (
        _smart_run() as mocks,
        patch("jasna.pipeline.VideoEncoder", Encoder),
    ):
        mocks["run_ltx_spans"].side_effect = ltx_run
        mocks["run_pass"].side_effect = lambda **kwargs: kwargs["encoder_ctx"].path.write_bytes(b"basicvsrpp")
        mocks["copy_fragment"].side_effect = lambda _source, _span, _index, path, **_k: path.write_bytes(b"copy")
        with pytest.raises(RuntimeError, match="compose interrupted"):
            pipeline._run_smart(METADATA)
        assert partial.exists() and not completed.exists()
        assert (mocks["run_pass"].call_count, mocks["copy_fragment"].call_count) == (1, 2)
        pipeline._run_smart(METADATA)
        assert completed.read_bytes() == b"encoded"
        assert (mocks["run_pass"].call_count, mocks["copy_fragment"].call_count) == (1, 2)
        mocks["mux"].assert_called_once()
    assert restored == [FOUR_SPANS[1], FOUR_SPANS[1]]


@pytest.mark.parametrize("cancelled", [False, True])
def test_pipeline_keeps_ltx_checkpoint_until_success(tmp_path, cancelled):
    source = tmp_path / "in.mp4"
    source.touch()
    pipeline = make_pipeline(input_video=source, output_video=tmp_path / "out.mp4", restoration_model_name="ltx")
    if cancelled:
        pipeline.cancel()
    with (
        patch("jasna.pipeline.get_video_meta_data", return_value=video_metadata()),
        patch.object(Pipeline, "_run_ltx"),
    ):
        pipeline.run()
    assert pipeline.resume.root.exists() is cancelled
    assert pipeline.completed is not cancelled
