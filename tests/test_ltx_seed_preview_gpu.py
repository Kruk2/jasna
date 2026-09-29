"""A seed preview restores a range exactly as a segment job does.

Needs an NVIDIA GPU, the LTX model in model_weights/ltx-restore and a video with a
mosaic: set JASNA_LTX_TEST_VIDEO to its path and JASNA_LTX_TEST_RANGES to two ranges in
seconds, e.g. "9-11,21-24" (the first is previewed; the job restores both).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest
import torch

from jasna.engine_paths import default_restoration_model_path
from jasna.ltx.model_files import bundle_present

VIDEO = os.environ.get("JASNA_LTX_TEST_VIDEO", "")

pytestmark = pytest.mark.skipif(
    not VIDEO or not torch.cuda.is_available() or not bundle_present(default_restoration_model_path("ltx")),
    reason="needs JASNA_LTX_TEST_VIDEO, an NVIDIA GPU and the LTX model",
)


class _Capture:
    def __init__(self, effect_range: tuple[int, int]) -> None:
        self.start, self.end = effect_range
        self.frames: dict[int, torch.Tensor] = {}

    def write(self, frame: torch.Tensor, pts: int, *, apply_lut: bool = True) -> None:
        if self.start <= int(pts) < self.end:
            self.frames[int(pts)] = frame.cpu()

    def close(self) -> None:
        pass


def _ranges() -> list[tuple[float, float]]:
    spec = os.environ.get("JASNA_LTX_TEST_RANGES", "9-11,21-24")
    return [tuple(float(value) for value in part.split("-")) for part in spec.split(",")]


def test_seed_preview_restores_a_range_exactly_like_a_segment_job(tmp_path) -> None:
    from jasna.gui.ltx_seed_preview import SeedRenderer
    from jasna.gui.models import AppSettings
    from jasna.gui.video_session import build_video_session, video_session_config
    from jasna.media.probe import get_video_meta_data
    from jasna.media.splice import build_splice_plan, probe_keyframes
    from jasna.segments import SegmentRange, SegmentRestoration
    from jasna.session_factory import build_pipeline

    video = Path(VIDEO)
    (first_start, first_end), (second_start, second_end) = _ranges()
    previewed = SegmentRange(first_start, first_end, SegmentRestoration("ltx", 1234))
    other = SegmentRange(second_start, second_end, SegmentRestoration("ltx", 99))
    settings = AppSettings()
    metadata = get_video_meta_data(str(video))
    index = probe_keyframes(video, metadata)

    renderer = SeedRenderer(video, metadata, index, tmp_path)
    try:
        preview = renderer.render(
            previewed,
            1234,
            settings,
            writer_for=lambda _directory, effect: _Capture(effect),
            report=lambda *_args: None,
            cancel=threading.Event(),
        )
    finally:
        renderer.close()

    session = build_video_session(settings, log=lambda _msg: None)
    try:
        config = video_session_config(settings, codec="hevc", encoder_settings={})
        pipeline = build_pipeline(config, session, video, tmp_path / "out.mkv", segments=(previewed, other))
        pipeline.configure_vr(metadata)
        plan = build_splice_plan((previewed, other), index, duration=metadata.duration)
        captures = []

        def opener(span):
            capture = _Capture(span.effect_ranges[0])
            captures.append(capture)
            return lambda: capture

        pipeline._run_ltx_spans(
            metadata,
            index,
            [(span, segments, opener(span)) for span, segments in zip(plan.render_spans, plan.render_span_segments())],
            tmp_path,
            None,
        )
    finally:
        session.close()

    job = captures[0]
    assert preview.frames and preview.frames.keys() == job.frames.keys()
    differing = [pts for pts in job.frames if not torch.equal(job.frames[pts], preview.frames[pts])]
    assert not differing, f"{len(differing)} of {len(job.frames)} frames differ"
