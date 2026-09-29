from pathlib import Path
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from factories import make_pipeline
from jasna.vr180 import SbsDetectionAdapter
from jasna.vr_projection import FisheyeProjector, GnomonicProjector


def _make_pipeline(**overrides):
    defaults = dict(
        batch_size=4,
        max_clip_size=60,
        temporal_overlap=8,
        max_detection_gap=0,
        min_detection_duration=0,
        vr_mode="off",
    )
    defaults.update(overrides)
    return make_pipeline(**defaults)


class TestPipelineInit:
    def test_stores_basic_attributes(self):
        p = _make_pipeline(batch_size=2, max_clip_size=30, temporal_overlap=4)
        assert p.batch_size == 2
        assert p.max_clip_size == 30
        assert p.temporal_overlap == 4
        assert p.codec == "hevc"
        assert p.enable_crossfade is True

    def test_crossfade_disabled(self):
        p = _make_pipeline(enable_crossfade=False)
        assert p.enable_crossfade is False

    def test_codec_forwarded_unchanged(self):
        for codec in ("hevc", "h264", "av1"):
            assert _make_pipeline(codec=codec).codec == codec

    def test_progress_callback(self):
        cb = MagicMock()
        p = _make_pipeline(progress_callback=cb)
        assert p.progress_callback is cb

    def test_ltx_progress_reaches_the_progress_callback_with_its_stage(self):
        cb = MagicMock()
        p = _make_pipeline(progress_callback=cb, disable_progress=True)
        with p._ltx_progress(10).bar("scan", 10) as bar:
            bar.update(10)
        assert cb.call_args[0] == (pytest.approx(2.0), 0.0, 0.0, 0, 0, "scan")

    def test_retarget_high_fps_defaults_off_and_can_be_enabled(self):
        assert _make_pipeline().retarget_high_fps is False
        assert _make_pipeline(retarget_high_fps=True).retarget_high_fps is True

    def test_fmp4_defaults_off_and_can_be_enabled(self):
        assert _make_pipeline().fmp4 is False
        assert _make_pipeline(fmp4=True).fmp4 is True

    def test_scene_detection_defaults_on_and_can_be_disabled(self):
        assert _make_pipeline().scene_detection is True
        assert _make_pipeline(scene_detection=False).scene_detection is False

    def test_configure_vr_wraps_detector_for_direct_sbs(self):
        pipeline = _make_pipeline(
            input_video=Path("VRKM-0001.mp4"),
            vr_mode="auto",
        )
        metadata = SimpleNamespace(
            video_width=200,
            video_height=100,
            sample_aspect_ratio=Fraction(1, 1),
            stereo_layout="",
            spherical_projection="",
        )

        pipeline.configure_vr(metadata)

        assert pipeline.vr_resolution.resolved == "sbs"
        assert isinstance(pipeline.job_detection_model, SbsDetectionAdapter)
        assert pipeline.vr_projector is None

    def test_configure_vr_builds_fisheye_projector(self):
        pipeline = _make_pipeline(
            input_video=Path("FSVSS-0001.mp4"),
            vr_mode="auto",
        )
        metadata = SimpleNamespace(
            video_width=200,
            video_height=100,
            sample_aspect_ratio=Fraction(1, 1),
            stereo_layout="",
            spherical_projection="",
        )

        pipeline.configure_vr(metadata)

        assert pipeline.vr_resolution.resolved == "sbs"
        assert pipeline.vr_resolution.projection == "fisheye"
        assert isinstance(pipeline.job_detection_model, SbsDetectionAdapter)
        assert isinstance(pipeline.vr_projector, FisheyeProjector)
        assert pipeline.vr_projector.eye_width == 100

    def test_configure_vr_builds_gnomonic_projector_for_routed_studio(self):
        pipeline = _make_pipeline(
            input_video=Path("VRPRD-0108.mp4"),
            vr_mode="auto",
        )
        metadata = SimpleNamespace(
            video_width=200,
            video_height=100,
            sample_aspect_ratio=Fraction(1, 1),
            stereo_layout="",
            spherical_projection="",
        )

        pipeline.configure_vr(metadata)

        assert pipeline.vr_resolution.projection == "gnomonic"
        assert isinstance(pipeline.vr_projector, GnomonicProjector)

    def test_configure_vr_honors_per_job_projection_override(self):
        pipeline = _make_pipeline(
            input_video=Path("VRKM-0001.mp4"),
            vr_mode="auto",
            vr_projection="fisheye",
        )
        metadata = SimpleNamespace(
            video_width=200,
            video_height=100,
            sample_aspect_ratio=Fraction(1, 1),
            stereo_layout="",
            spherical_projection="",
        )

        pipeline.configure_vr(metadata)

        assert pipeline.vr_resolution.projection == "fisheye"
        assert isinstance(pipeline.vr_projector, FisheyeProjector)


def test_ltx_span_gives_each_effect_range_its_segment_seed() -> None:
    from jasna.ltx.restore import LtxSegment
    from jasna.media.splice import KeyframeIndex, SpliceSpan
    from jasna.segments import SegmentRange, SegmentRestoration

    p = _make_pipeline()
    index = KeyframeIndex(pts=(0, 2000), time_base=Fraction(1, 1000), start_pts=0, end_pts=6000)
    span = SpliceSpan("render", 2000, 6000, ((2500, 3000), (4000, 5000)))
    segments = (
        SegmentRange(2.5, 3.0, SegmentRestoration("ltx", 7)),
        SegmentRange(4.0, 5.0, SegmentRestoration("ltx", 9)),
    )
    opener = MagicMock()

    ltx = p.ltx_span(SimpleNamespace(), index, span, segments, opener)

    assert ltx.segments == (LtxSegment(2500, 3000, 7), LtxSegment(4000, 5000, 9))
    assert ltx.open_writer is opener
