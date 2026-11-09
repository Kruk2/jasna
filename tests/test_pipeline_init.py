from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from factories import make_pipeline, video_metadata
from jasna.ltx.restore import LtxSegment
from jasna.media.splice import KeyframeIndex, SpliceSpan
from jasna.progressbar import StageProgress
from jasna.segments import SegmentRange, SegmentRestoration
from jasna.vr180 import StereoDetectionAdapter
from jasna.vr_projection import FisheyeProjector, GnomonicProjector


def test_ltx_progress_reaches_the_progress_callback_with_its_stage():
    cb = MagicMock()
    p = make_pipeline(progress_callback=cb, disable_progress=True)
    with p._ltx_progress(10, cb).bar("scan", 10) as bar:
        bar.update(10)
    assert cb.call_args[0] == (pytest.approx(2.0), 0.0, 0.0, 0, 0, StageProgress("scan", 1, 4, 1.0, 0.0))


@pytest.mark.parametrize(
    ("file_name", "projection_override", "projection", "projector"),
    [
        ("VRKM-0001.mp4", "auto", "raw", type(None)),
        ("FSVSS-0001.mp4", "auto", "fisheye", FisheyeProjector),
        ("VRPRD-0001.mp4", "auto", "gnomonic", GnomonicProjector),
        ("VRKM-0001.mp4", "fisheye", "fisheye", FisheyeProjector),
    ],
    ids=["direct-sbs", "fisheye-prefix", "gnomonic-prefix", "per-job-override"],
)
def test_configure_vr_detects_per_eye_and_builds_the_prefix_projector(
    file_name, projection_override, projection, projector
):
    pipeline = make_pipeline(input_video=Path(file_name), vr_mode="auto", vr_projection=projection_override)

    pipeline.configure_vr(video_metadata(video_width=200, video_height=100))

    assert pipeline.vr_resolution.resolved == "sbs"
    assert pipeline.vr_resolution.projection == projection
    assert isinstance(pipeline.job_detection_model, StereoDetectionAdapter)
    assert isinstance(pipeline.vr_projector, projector)
    if pipeline.vr_projector is not None:
        assert pipeline.vr_projector.eye_width == 100


def test_ltx_span_gives_each_effect_range_its_segment_seed() -> None:
    index = KeyframeIndex(pts=(0, 2000), time_base=Fraction(1, 1000), start_pts=0, end_pts=6000)
    span = SpliceSpan("render", 2000, 6000, ((2500, 3000), (4000, 5000)))
    segments = (
        SegmentRange(2.5, 3.0, SegmentRestoration("ltx", 7)),
        SegmentRange(4.0, 5.0, SegmentRestoration("ltx", 9)),
    )

    ltx = make_pipeline().ltx_span(video_metadata(), index, span, segments, MagicMock())

    assert ltx.segments == (LtxSegment(2500, 3000, 7), LtxSegment(4000, 5000, 9))
