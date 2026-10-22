import contextlib
import json
import tempfile
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest
import torch

from jasna.ltx import inference, restore
from jasna.ltx.camera import CropFrame
from jasna.ltx.plan import LESS_OVERLAP_WINDOW_HOP, Region, Window
from jasna.tensor_utils import half_dtype

FILES = restore.LtxModelFiles(Path("t"), Path("v"), Path("d"))
REAL_BAR = restore.Progress.bar
MOSAIC = Region(box=(10.0, 10.0, 40.0, 40.0), polygon=np.array([[10, 10], [40, 10], [40, 40], [10, 40]], dtype=np.int32))


def _video(frames: int, mosaic: range, pts_step: int = 10) -> list[tuple[torch.Tensor, int]]:
    out = []
    for i in range(frames):
        frame = torch.zeros(3, 64, 64, dtype=torch.uint8)
        frame[1, 0, 0] = i
        frame[0, 0, 0] = 1 if i in mosaic else 0
        out.append((frame, i * pts_step))
    return out


def _source(video, first: int = 0, batch: int = 4):
    def frames():
        for start in range(first, len(video), batch):
            chunk = video[start : start + batch]
            yield torch.stack([frame for frame, _ in chunk]), [pts for _, pts in chunk]

    return frames


class _Detector:
    def __init__(self):
        self.batches: list[list[int]] = []

    def __call__(self, batch, target_hw):
        self.batches.append([int(frame[1, 0, 0]) for frame in batch])
        return batch


class _Bar:
    def __init__(self, total=0):
        self.total, self.n = total, 0

    def update(self, n):
        self.n += n

    def reuse(self, done, frames, windows):
        self.n += done

    def window(self, canvas):
        return contextlib.nullcontext()

    def waiting(self):
        return contextlib.nullcontext()

    def windows_done(self, canvas, count):
        pass

    def track(self, canvas, steps):
        return self.update

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Transformer:
    seeds: list[list[int]] = []

    def __init__(self, path, device, free_vram):
        self.conditions = [None] * 8
        self.blocks = SimpleNamespace(resident=40, count=48)

    def denoise_chain(self, references, seeds, window_hop, advance):
        _Transformer.seeds.append(list(seeds))
        for _ in self.conditions:
            advance(len(references))
        return [torch.zeros(1) for _ in references]

    def close(self):
        pass


class _Writer:
    def __init__(self):
        self.frames: list[tuple[int, int]] = []
        self.closed = False

    def write(self, frame, pts):
        self.frames.append((pts, int(frame[2, 0, 0])))

    def finish(self):
        self.finished = True

    def close(self):
        self.closed = True


@pytest.fixture
def fakes(monkeypatch, tmp_path):
    """Fake models; every budget's speed profile has a rate, so no run starts with a speed test."""
    for budget in ("segment", "free VRAM"):
        restore.speed.save_rates(
            tmp_path / "speed.json", restore.speed.profile_key("Test GPU", "t", budget), {"denoise/window512": 4.0}
        )
    budgets = []
    bars = []

    def decode(decoder, latent, free_bytes, generator):
        budgets.append(free_bytes)
        return torch.zeros(1, 3, restore.WINDOW_FRAMES, 2, 2)

    def composite(source, candidates, *, feather):
        out = source.clone()
        out[2, 0, 0] = len(candidates)
        return out

    def bar(self, name, total, unit="frame", marks=None):
        bars.append(_Bar(total))
        return bars[-1]

    monkeypatch.setattr(
        restore, "regions_per_frame", lambda frames, h, w: [[MOSAIC] if int(f[0, 0, 0]) else [] for f in frames]
    )
    monkeypatch.setattr(restore, "_SpeedTestMosaics", _Detector)
    monkeypatch.setattr(restore, "crop_to_canvas", lambda frame, crop: torch.zeros(1))
    monkeypatch.setattr(restore, "_vae_pixels", lambda canvases, device: torch.zeros(1))
    monkeypatch.setattr(inference, "load_video_encoder", lambda path, device: lambda pixels: torch.zeros(1))
    monkeypatch.setattr(inference, "load_video_decoder", lambda vae, tuned, device: object())
    monkeypatch.setattr(inference, "TransformerSource", lambda path: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(inference, "LtxTransformer", _Transformer)
    monkeypatch.setattr(restore, "decode_latent", decode)
    monkeypatch.setattr(restore, "composite_frame", composite)
    monkeypatch.setattr(restore.Progress, "bar", bar)
    monkeypatch.setattr(restore, "device_free_bytes", lambda device: 123 << 20)
    monkeypatch.setattr(inference, "device_free_bytes", lambda device: 123 << 20)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "Test GPU")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 5 << 20)
    _Transformer.seeds = []
    return budgets, bars


def _render(tmp_path):
    return restore.LtxRender(
        detector=_Detector(),
        files=FILES,
        frame_h=64,
        frame_w=64,
        batch_size=4,
        large_canvas=False,
        window_hop=restore.WINDOW_HOP,
        budget=restore.segment_decode_budget(12 << 30),
        budget_name="segment",
        device=torch.device("cpu"),
        speed_profile=tmp_path / "speed.json",
    )


def _new_dir(tmp_path):
    return Path(tempfile.mkdtemp(dir=tmp_path))


def _run_spans(tmp_path, spans):
    restore.restore_spans(
        spans,
        _render(tmp_path),
        work_dir=tmp_path, resume_dir=_new_dir(tmp_path),
        progress=restore.Progress(0, disable=True, report=None),
        cancel=threading.Event(),
    )


def test_segment_batches_count_batches_from_each_segment_start():
    video = _video(12, range(0))
    segment = restore.LtxSegment(20, 90, seed=0)
    out = [
        (owner, [int(f[1, 0, 0]) for f in frames], pts)
        for owner, frames, pts in restore.segment_batches(_source(video)(), [segment], 4)
    ]
    assert out == [
        (None, [0, 1], [0, 10]),
        (0, [2, 3, 4, 5], [20, 30, 40, 50]),
        (0, [6, 7, 8], [60, 70, 80]),
        (None, [9, 10, 11], [90, 100, 110]),
    ]


def test_a_whole_video_segment_passes_decoded_batches_through():
    batches = list(_source(_video(10, range(0)))())
    whole = restore.LtxSegment(*restore.WHOLE_VIDEO_PTS, seed=0)
    out = list(restore.segment_batches(iter(batches), [whole], 4))
    assert [owner for owner, _, _ in out] == [0, 0, 0]
    assert all(torch.equal(frames, batch) and pts == batch_pts for (_, frames, pts), (batch, batch_pts) in zip(out, batches))


def test_a_segment_plans_the_same_wherever_decoding_starts(fakes):
    video = _video(30, range(3, 28))
    segment = restore.LtxSegment(50, 150, seed=0)
    results = []
    for first in (0, 3):
        detector = _Detector()
        plans = restore.scan_span(
            _source(video, first),
            [segment],
            detector,
            batch_size=4,
            frame_h=64,
            frame_w=64,
            large_canvas=False,
            window_hop=restore.WINDOW_HOP,
            bar=_Bar(),
            cancel=threading.Event(),
        )
        results.append((plans, detector.batches))
    assert results[0] == results[1]
    (plans,), batches = results[0]
    assert batches == [[5, 6, 7, 8], [9, 10, 11, 12], [13, 14]]
    assert [w.index for plan in plans for w in plan.windows] == [0]
    assert plans[0].start == 0 and len(plans[0].polygons) == 10


def test_restore_spans_seeds_segments_and_leaves_other_frames_untouched(fakes, tmp_path):
    budgets, bars = fakes
    video = _video(30, range(3, 28))
    writer = _Writer()
    segments = (restore.LtxSegment(50, 150, seed=100), restore.LtxSegment(200, 260, seed=7))
    _run_spans(tmp_path, [restore.LtxSpan(_source(video), segments, lambda: writer)])

    assert _Transformer.seeds == [[100], [7]]
    assert budgets == [6 << 30] * 2
    assert bars[2].total == bars[2].n == 16
    inside = {pts for pts in range(50, 150, 10)} | {pts for pts in range(200, 260, 10)}
    assert [pts for pts, _ in writer.frames] == list(range(0, 300, 10))
    for pts, marker in writer.frames:
        assert marker == (1 if pts in inside else 0)
    assert writer.closed


def test_restore_spans_opens_each_span_writer_when_it_is_composed(fakes, tmp_path):
    video = _video(30, range(3, 28))
    opened = []

    def opener(name):
        def open_writer():
            opened.append(name)
            return _Writer()

        return open_writer

    spans = [
        restore.LtxSpan(_source(video[:15]), (restore.LtxSegment(50, 150, seed=1),), opener("a")),
        restore.LtxSpan(_source(video, 15), (restore.LtxSegment(200, 260, seed=1),), opener("b")),
    ]
    _run_spans(tmp_path, spans)
    assert opened == ["a", "b"]
    assert _Transformer.seeds == [[1], [1]]


def test_restore_video_decodes_with_the_free_vram(fakes, tmp_path):
    budgets, bars = fakes
    video = _video(12, range(2, 10))
    writer = _Writer()
    restore.restore_video(
        _source(video),
        writer,
        detector=_Detector(),
        files=FILES,
        frame_h=64,
        frame_w=64,
        batch_size=4,
        large_canvas=False,
        window_hop=restore.WINDOW_HOP,
        seed=5,
        device=torch.device("cpu"),
        work_dir=tmp_path, resume_dir=_new_dir(tmp_path),
        speed_profile=tmp_path / "speed.json",
        progress=restore.Progress(12, disable=True, report=None),
        cancel=threading.Event(),
    )
    assert budgets == [123 << 20]
    assert _Transformer.seeds == [[5]]
    assert bars[2].total == bars[2].n == 8
    assert len(writer.frames) == 12


def test_decode_window_converts_a_float32_decode(monkeypatch):
    decoded = torch.tensor([-1.0, 0.0, 1.0]).view(1, 1, 3, 1, 1).expand(1, 3, 3, 1, 1).contiguous()
    monkeypatch.setattr(restore, "decode_latent", lambda decoder, latent, free_bytes, generator: decoded.clone())
    frames = restore._decode_window(None, torch.zeros(1, 128, 1, 1, 1), 0, 512, lambda canvas: 1, torch.device("cpu"))
    assert frames.dtype == torch.uint8 and frames.shape == (3, 3, 1, 1)
    assert frames[:, 0, 0, 0].tolist() == [0, 128, 255]


def _two_window_composer(monkeypatch, decode, composite):
    crop = CropFrame((0, 0, 4, 4), (4, 4), (0, 0, 0, 0))
    windows = (Window(0, 0, 0, 13, (crop,) * 13), Window(1, 0, 3, 13, (crop,) * 13))
    plan = restore.TrackPlan(0, 0, ((),) * 16, (1.0,) * 16, windows, 3)
    monkeypatch.setattr(restore, "_decode_window", decode)
    monkeypatch.setattr(restore, "composite_frame", composite)
    decodes = ThreadPoolExecutor(max_workers=1)
    composer = restore._SegmentComposer(
        [plan], None, SimpleNamespace(get=lambda index: index), seed=0, feather=1, budget=lambda canvas: 0,
        device=torch.device("cpu"), bar=_Bar(), decodes=decodes, stream=None,
    )
    return composer, decodes


def test_composer_decodes_the_next_window_while_earlier_frames_are_composited(monkeypatch):
    second_started = threading.Event()
    started_before_its_start = []

    def decode(decoder, latent, seed, canvas, budget, device):
        if latent == 1:
            second_started.set()
        return torch.full((13, 3, 4, 4), latent, dtype=torch.uint8)

    def composite(source, candidates, *, feather):
        if int(source[0, 0, 0]) == 2:
            started_before_its_start.append(second_started.wait(timeout=5))
        return torch.stack([c.canvas for c in candidates]).max(0).values

    composer, decodes = _two_window_composer(monkeypatch, decode, composite)
    try:
        out = [int(composer.frame(torch.full((3, 4, 4), i, dtype=torch.uint8))[0, 0, 0]) for i in range(16)]
    finally:
        decodes.shutdown()
    assert started_before_its_start == [True]
    assert out == [0, 0, 0] + [1] * 13


def test_composer_decodes_nothing_before_its_first_frame(monkeypatch):
    decoded = []

    def decode(decoder, latent, seed, canvas, budget, device):
        decoded.append(latent)
        return torch.full((13, 3, 4, 4), latent, dtype=torch.uint8)

    composer, decodes = _two_window_composer(monkeypatch, decode, lambda source, candidates, *, feather: source)
    decodes.shutdown()
    assert decoded == []


def test_composer_raises_a_failed_window_decode(monkeypatch):
    def decode(decoder, latent, seed, canvas, budget, device):
        raise RuntimeError("decode failed")

    composer, decodes = _two_window_composer(monkeypatch, decode, lambda source, candidates, *, feather: source)
    try:
        with pytest.raises(RuntimeError, match="decode failed"):
            composer.frame(torch.zeros(3, 4, 4, dtype=torch.uint8))
    finally:
        decodes.shutdown()


def test_segment_decode_budget_follows_the_gpu_size_not_free_vram(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1, 1))
    gib = 1 << 30
    for total_gib, budget_512, budget_768 in (
        (5.8, 5.5, 8), (7.8, 6, 8), (11.7, 6, 8), (15.5, 8, 10), (21.9, 8, 10), (23.6, 8, 16), (31.8, 8, 16)
    ):
        budget = restore.segment_decode_budget(int(total_gib * gib))
        assert (budget(512), budget(768)) == (budget_512 * gib, budget_768 * gib)


def test_each_window_decode_returns_the_cache_before_reading_its_budget(monkeypatch):
    events = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))
    monkeypatch.setattr(
        restore, "decode_latent", lambda decoder, latent, free_bytes, generator: torch.zeros(1, 3, 2, 4, 4)
    )
    budget = lambda canvas: events.append("budget") or 5 << 30
    for _ in range(2):
        restore._decode_window(object(), torch.zeros(1), 7, 512, budget, torch.device("cpu"))
    assert events == ["empty_cache", "budget"] * 2


def test_large_canvas_needs_twelve_gib_free():
    gib = 1 << 30
    assert restore.large_canvas_fits(True, 12 * gib)
    assert not restore.large_canvas_fits(True, 11 * gib)
    assert not restore.large_canvas_fits(False, 30 * gib)


def test_segment_large_canvas_follows_the_gpu_size():
    gib = 1 << 30
    assert restore.segment_large_canvas(True, 16 * gib)
    assert not restore.segment_large_canvas(True, 12 * gib)
    assert not restore.segment_large_canvas(False, 32 * gib)


class _Clock:
    """``time.monotonic`` and CUDA events on one settable clock."""

    def __init__(self, monkeypatch):
        self.now = 0.0
        clock = self
        monkeypatch.setattr(restore.time, "monotonic", lambda: clock.now)

        class Event:
            def __init__(self, enable_timing):
                self.at = None

            def record(self):
                self.at = clock.now

            def synchronize(self):
                pass

            def elapsed_time(self, end):
                return (end.at - self.at) * 1000.0

        monkeypatch.setattr(torch.cuda, "Event", Event)


def test_progress_reports_the_stage_and_the_share_of_the_whole_run(monkeypatch):
    clock = _Clock(monkeypatch)
    reports = []
    progress = restore.Progress(10, disable=True, report=lambda *args: reports.append(args))
    with progress.bar("scan", 10) as bar:
        clock.now = 100.0
        bar.update(10)
    with progress.bar("denoise", 4, unit="step") as bar:
        clock.now = 300.0
        bar.update(2)
    assert [(stage.name, round(fraction, 3)) for fraction, _eta, stage in reports] == [
        ("scan", 0.0),
        ("scan", 0.02),
        ("denoise", 0.02),
        ("denoise", 0.42),
    ]
    assert [eta for _fraction, eta, _stage in reports[:3]] == [0.0, 0.0, 0.0]
    assert reports[-1][1] == pytest.approx(300.0 * 0.58 / 0.42)


def test_progress_reports_each_pass_with_its_own_share_and_time_left(monkeypatch):
    clock = _Clock(monkeypatch)
    reports = []
    progress = restore.Progress(100, disable=True, report=lambda *args: reports.append(args))
    with progress.bar("scan", 100) as bar:
        clock.now = 1.0
        bar.update(10)
        clock.now = 5.0
        bar.update(20)
    first, second = reports[-2][2], reports[-1][2]
    assert (first.number, first.count, first.fraction, first.seconds_left) == (1, 4, 0.1, 0.0)
    assert second.fraction == pytest.approx(0.3)
    assert second.seconds_left == pytest.approx(4.0 / 20 * 70)


def _recorded_bar_options(monkeypatch) -> list[dict]:
    options = []
    real_tqdm = restore.tqdm

    def recording_tqdm(**kwargs):
        options.append(kwargs)
        return real_tqdm(**kwargs)

    monkeypatch.setattr(restore, "tqdm", recording_tqdm)
    return options


def test_speed_test_bars_are_labelled_as_the_speed_test(monkeypatch):
    options = _recorded_bar_options(monkeypatch)
    run = restore.Progress(1000, disable=True, report=None)
    with run.speed_test().bar("scan", 10):
        pass
    with run.bar("scan", 10):
        pass
    assert [option["desc"] for option in options] == ["LTX speed test 1/4 scan", "LTX 1/4 scan"]


@pytest.mark.parametrize(("video_frames", "speed_test_frames"), [(61, 61), (1000, restore.SPEED_TEST_FRAMES)])
def test_speed_test_counts_only_the_frames_the_video_has(video_frames, speed_test_frames):
    assert restore.Progress(video_frames, disable=True, report=None).speed_test().frames == speed_test_frames


def test_progress_begin_shows_the_next_pass_while_its_model_loads(monkeypatch):
    _Clock(monkeypatch)
    reports = []
    progress = restore.Progress(10, disable=True, report=lambda *args: reports.append(args))
    with progress.bar("scan", 10) as bar:
        bar.update(10)
    progress.begin("encode")
    fraction, _eta, stage = reports[-1]
    assert (stage.name, stage.number, stage.fraction) == ("encode", 2, 0.0)
    assert fraction == pytest.approx(0.02)


_RATES = {
    "encode/fixed": 5.0,
    "encode/frame": 0.01,
    "encode/window512": 0.5,
    "denoise/fixed": 20.0,
    "denoise/window512": 4.0,
    "compose/fixed": 3.0,
    "compose/crop": 0.1,
    "compose/window512": 1.0,
}


def test_progress_estimates_the_time_left_from_the_speed_profile_once_planned(monkeypatch):
    clock = _Clock(monkeypatch)
    reports = []
    progress = restore.Progress(100, disable=True, report=lambda *args: reports.append(args))
    progress.expect(_RATES)
    with progress.bar("scan", 100) as bar:
        clock.now = 10.0
        bar.update(100)
    assert reports[-1][:2] == (pytest.approx(0.02), 0.0)
    progress.plan(restore.speed.Work(frames=100, megapixels=2.0, windows={512: 10}, crop_megapixels=40.0))
    with progress.bar("encode", 100) as bar:
        clock.now = 20.0
        bar.update(50)

    encode_left = 50 * 2.0 * 0.01 + 10 * 0.5
    later = 20.0 + 10 * 4.0 + 3.0 + max(40.0 * 0.1, 10 * 1.0)
    fraction, left, stage = reports[-1]
    assert stage.name == "encode"
    assert stage.seconds_left == pytest.approx(encode_left)
    assert left == pytest.approx(encode_left + later)
    assert fraction == pytest.approx(20.0 / (20.0 + left))


def test_run_fraction_does_not_count_the_speed_test_before_the_run(monkeypatch):
    clock = _Clock(monkeypatch)
    reports = []
    progress = restore.Progress(100, disable=True, report=lambda *args: reports.append(args))
    clock.now = 300.0
    progress.expect(_RATES)
    with progress.bar("scan", 100) as bar:
        clock.now = 310.0
        bar.update(100)
    progress.plan(restore.speed.Work(frames=100, megapixels=2.0, windows={512: 10}, crop_megapixels=40.0))
    with progress.bar("encode", 100):
        pass

    fraction, left, _stage = reports[-1]
    assert fraction == pytest.approx(10.0 / (10.0 + left))


def test_pass_time_left_is_the_profile_work_left_scaled_by_the_measured_pace(monkeypatch):
    clock = _Clock(monkeypatch)
    reports = []
    progress = restore.Progress(100, disable=True, report=lambda *args: reports.append(args))
    progress.expect(_RATES)
    progress.plan(restore.speed.Work(frames=100, megapixels=2.0, windows={512: 10}, crop_megapixels=40.0))
    with progress.bar("compose", 100) as bar:
        bar.update(10)
        clock.now = 2.0
        bar.windows_done(512, 1)
        clock.now = 3.0
        bar.update(40)

    compose_work = max(40.0 * 0.1, 10 * 1.0)
    done_work = max(40.0 * 0.1 * 50 / 100, 1.0)
    warm_done_work = done_work - max(40.0 * 0.1 * 10 / 100, 0.0)
    fraction, left, stage = reports[-1]
    assert stage.seconds_left == pytest.approx((compose_work - done_work) * 3.0 / warm_done_work)
    assert left == pytest.approx(stage.seconds_left)
    assert fraction == pytest.approx(3.0 / (3.0 + left))


def _window(index: int, start: int, frames: int, canvas: int = 512) -> Window:
    return Window(index, 0, start, frames, (CropFrame((0, 0, 4, 4), (canvas, canvas), (0, 0, 0, 0)),) * frames)


def test_decode_shares_spread_each_window_over_the_frames_it_decodes_during():
    shares = restore._decode_shares([_window(0, 2, 4), _window(1, 2, 4), _window(2, 6, 4, canvas=768)])

    assert shares == {
        0: [(512, 0.5)],
        1: [(512, 0.5)],
        2: [(512, 1.0), (768, 0.25)],
        3: [(768, 0.25)],
        4: [(768, 0.25)],
        5: [(768, 0.25)],
    }


def test_compose_time_left_falls_steadily_while_windows_decode_ahead(monkeypatch):
    clock = _Clock(monkeypatch)
    reports = []
    progress = restore.Progress(100, disable=True, report=lambda *args: reports.append(args))
    progress.expect(_RATES)
    progress.plan(restore.speed.Work(frames=100, megapixels=2.0, windows={512: 10}, crop_megapixels=40.0))
    shares = restore._decode_shares([_window(index, index * 10, 10) for index in range(10)])
    first_window_blocking_decode_seconds = 1.0
    clock.now = 100.0
    with progress.bar("compose", 100) as bar:
        for batch_start in range(0, 100, 5):
            clock.now += 0.5
            bar.update(5)
            if batch_start == 0:
                clock.now += first_window_blocking_decode_seconds
            for frame_idx in range(batch_start, batch_start + 5):
                for canvas, share in shares.get(frame_idx, ()):
                    bar.windows_done(canvas, share)

    lefts = [left for _fraction, left, _stage in reports[2:]]
    assert all(later <= earlier + 1e-9 for earlier, later in zip(lefts, lefts[1:]))


def test_run_fraction_never_goes_back_when_the_pace_drops(monkeypatch):
    clock = _Clock(monkeypatch)
    reports = []
    progress = restore.Progress(100, disable=True, report=lambda *args: reports.append(args))
    progress.expect(_RATES)
    progress.plan(restore.speed.Work(frames=100, megapixels=2.0, windows={512: 10}, crop_megapixels=40.0))
    clock.now = 100.0
    with progress.bar("compose", 100) as bar:
        bar.update(10)
        clock.now = 101.0
        bar.windows_done(512, 1)
        bar.update(10)
        clock.now = 110.0
        bar.update(10)

    fractions = [fraction for fraction, _left, _stage in reports]
    assert reports[-1][1] > reports[-2][1]
    assert fractions == sorted(fractions)


def test_progress_without_rates_for_a_planned_canvas_keeps_the_fixed_shares(monkeypatch):
    _Clock(monkeypatch)
    reports = []
    progress = restore.Progress(100, disable=True, report=lambda *args: reports.append(args))
    progress.expect(_RATES)
    progress.plan(restore.speed.Work(frames=100, megapixels=2.0, windows={512: 3, 768: 1}, crop_megapixels=40.0))
    with progress.bar("encode", 100) as bar:
        bar.update(50)
    assert reports[-1][:2] == (pytest.approx(0.04), 0.0)
    assert reports[-1][2].name == "encode"


def test_stage_bars_time_warm_units_for_the_profile(monkeypatch):
    clock = _Clock(monkeypatch)
    progress = restore.Progress(10, disable=True, report=None)
    clock.now = 2.0
    with progress.bar("compose", 10) as bar:
        clock.now = 3.0
        bar.update(4)
        with bar.window(512):
            clock.now = 4.5
        with bar.waiting():
            clock.now = 5.0
        clock.now = 6.0
        bar.update(6)
    with progress.bar("denoise", 6, unit="step") as bar:
        advance = bar.track(512, 3)
        clock.now = 16.0
        advance(2)
        clock.now = 18.0
        advance(2)
        clock.now = 20.0
        advance(2)
        other = bar.track(512, 3)
        other(2)

    compose, denoise = progress.timings["compose"], progress.timings["denoise"]
    assert (compose.seconds, compose.warm_seconds, compose.warm_frames) == (6.0, 3.0, 6)
    assert compose.windows == [(512, 1.5)]
    assert compose.waited_seconds == 0.5
    assert compose.units == 10
    assert denoise.seconds == 14.0
    assert denoise.units == 8
    assert denoise.windows == [(512, pytest.approx(4.0 / 4 * 3))]
    assert denoise.steps == [(512, 1.0), (512, 1.0)]


def test_worker_passes_are_timed_on_the_wall_clock(monkeypatch):
    clock = _Clock(monkeypatch)

    class BatchedEvent:
        """Windows batches this process's event records when only the worker used the GPU."""

        def __init__(self, enable_timing):
            pass

        def record(self):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return 0.0

    monkeypatch.setattr(torch.cuda, "Event", BatchedEvent)

    class Worker:
        def request(self, op, params, tensors, advance=None):
            for _ in range(3):
                clock.now += 0.5
                advance(len(tensors))
            return {}, tensors

    transformer = inference.WorkerTransformer(Worker(), {"steps": 3, "resident": 48, "count": 48})
    encoder = inference.WorkerEncoder(Worker())
    progress = restore.Progress(10, disable=True, report=None)
    with progress.bar("encode", 10, marks=restore.timing_marks(encoder)) as bar:
        bar.update(10)
        for _ in range(2):
            with bar.window(512):
                clock.now += 2.0
    with progress.bar("denoise", 6, unit="step", marks=restore.timing_marks(transformer)) as bar:
        transformer.denoise_chain([torch.zeros(1), torch.zeros(1)], [1, 2], restore.WINDOW_HOP, bar.track(512, 3))

    assert progress.timings["encode"].windows == [(512, 2.0), (512, 2.0)]
    assert progress.timings["denoise"].steps == [(512, 0.25), (512, 0.25)]


def test_restore_spans_logs_the_run_summary(fakes, tmp_path, caplog):
    video = _video(12, range(2, 10))
    with caplog.at_level("INFO", logger=restore.logger.name):
        _run_spans(tmp_path, [restore.LtxSpan(_source(video), (restore.LtxSegment(*restore.WHOLE_VIDEO_PTS, 1),), _Writer)])
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("[timing] LTX")]
    assert lines == [
        "[timing] LTX: 0.0s, 0 frames, windows 1 at 512, 0 window-steps, peak VRAM 5.0 MiB",
        f"[timing] LTX profile: {restore.speed.profile_key('Test GPU', 't', 'segment')}",
    ]


def test_restore_spans_saves_the_speed_profile(fakes, tmp_path):
    (tmp_path / "speed.json").unlink()
    video = _video(12, range(2, 10))
    _run_spans(tmp_path, [restore.LtxSpan(_source(video), (restore.LtxSegment(*restore.WHOLE_VIDEO_PTS, 1),), _Writer)])
    key = restore.speed.profile_key("Test GPU", "t", "segment")
    assert restore.speed.load_rates(tmp_path / "speed.json", key) == {}
    assert key in json.loads((tmp_path / "speed.json").read_text())


def test_a_whole_video_keeps_its_speed_profile_apart_from_segment_renders(fakes, tmp_path):
    (tmp_path / "speed.json").unlink()
    restore.restore_video(
        _source(_video(12, range(2, 10))),
        _Writer(),
        detector=_Detector(),
        files=FILES,
        frame_h=64,
        frame_w=64,
        batch_size=4,
        large_canvas=False,
        window_hop=restore.WINDOW_HOP,
        seed=1,
        device=torch.device("cpu"),
        work_dir=tmp_path, resume_dir=_new_dir(tmp_path),
        speed_profile=tmp_path / "speed.json",
        progress=restore.Progress(0, disable=True, report=None),
        cancel=threading.Event(),
    )
    assert list(json.loads((tmp_path / "speed.json").read_text())) == [
        restore.speed.profile_key("Test GPU", "t", "free VRAM")
    ]


def test_progress_without_a_report_only_drives_the_console_bar():
    progress = restore.Progress(3, disable=True, report=None)
    with progress.bar("compose", 3) as bar:
        bar.update(3)


def test_the_planned_work_counts_each_window_crop_on_every_frame_it_covers(fakes, tmp_path, monkeypatch):
    planned = []
    monkeypatch.setattr(restore.Progress, "plan", lambda self, work: planned.append(work))
    span = restore.LtxSpan(_source(_video(30, range(3, 28))), (restore.LtxSegment(*restore.WHOLE_VIDEO_PTS, 1),), _Writer)
    progress = restore.Progress(30, disable=True, report=None)
    restore.prepare_spans([span], _render(tmp_path), tmp_path, progress=progress, cancel=threading.Event())
    (work,) = planned
    assert (work.frames, work.windows) == (30, {512: 1})
    assert work.crop_megapixels == pytest.approx(25 * 64 * 64 / 1e6)


def test_a_prepared_span_denoises_with_many_seeds(fakes, tmp_path):
    video = _video(30, range(3, 28))
    span = restore.LtxSpan(_source(video), (restore.LtxSegment(50, 150, seed=1),), _Writer)
    progress = restore.Progress(30, disable=True, report=None)
    cancel = threading.Event()
    (prepared,) = restore.prepare_spans([span], _render(tmp_path), tmp_path, progress=progress, cancel=cancel)
    transformer = _Transformer(None, None, None)
    _Transformer.seeds = []

    for seed in (11, 22):
        reseeded = restore.PreparedSpan(
            restore.LtxSpan(span.frames, (restore.LtxSegment(50, 150, seed=seed),), span.open_writer),
            prepared.plans,
            prepared.references,
        )
        finals = restore.final_stores([reseeded], tmp_path / str(seed))
        (tmp_path / str(seed)).mkdir()
        restore.denoise_spans([reseeded], finals, transformer, keep_references=True, progress=progress, cancel=cancel)

    assert _Transformer.seeds == [[11], [22]]
    assert prepared.windows == 1
    assert prepared.largest_canvas == 512


def test_restore_spans_shows_each_pass_before_its_model_loads(fakes, monkeypatch, tmp_path):
    reports = []
    shown_at_load = []

    def loading(model, made):
        def load(*args):
            shown_at_load.append((model, reports[-1][2].name))
            return made(*args)

        return load

    monkeypatch.setattr(inference, "load_video_encoder", loading("encoder", inference.load_video_encoder))
    monkeypatch.setattr(inference, "LtxTransformer", loading("transformer", inference.LtxTransformer))
    monkeypatch.setattr(inference, "load_video_decoder", loading("decoder", inference.load_video_decoder))
    video = _video(12, range(2, 10))
    restore.restore_spans(
        [restore.LtxSpan(_source(video), (restore.LtxSegment(*restore.WHOLE_VIDEO_PTS, 1),), _Writer)],
        _render(tmp_path),
        work_dir=tmp_path, resume_dir=_new_dir(tmp_path),
        progress=restore.Progress(12, disable=True, report=lambda *args: reports.append(args)),
        cancel=threading.Event(),
    )
    assert shown_at_load == [("encoder", "encode"), ("transformer", "denoise"), ("decoder", "compose")]


def test_restore_spans_frees_the_decoder_and_its_cached_memory_after_compose(fakes, monkeypatch, tmp_path):
    class Decoder:
        pass

    decoders = []
    released = []

    def load_decoder(vae, tuned, device):
        decoders.append(weakref.ref(decoder := Decoder()))
        return decoder

    monkeypatch.setattr(inference, "load_video_decoder", load_decoder)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: released.append(decoders[0]() is None if decoders else None))
    _run_spans(tmp_path, [restore.LtxSpan(_source(_video(30, range(3, 28))), (restore.LtxSegment(50, 150, seed=1),), _Writer)])
    assert decoders and released[-1] is True


_CUDA = pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"))


@pytest.mark.parametrize("device", ["cpu", _CUDA])
def test_segment_encoder_feeds_each_window_its_crops_padded_with_the_last(device):
    crop = CropFrame(crop_box=(0, 0, 8, 8), resize_hw=(8, 8), pad=(0, 0, 0, 0))
    windows = (Window(0, 0, 1, 3, (crop,) * 3), Window(1, 0, 2, 2, (crop,) * 2), Window(2, 0, 4, 1, (crop,)))
    encoded = {}
    store = SimpleNamespace(put=lambda index, latent: encoded.__setitem__(index, latent.cpu()))
    buffers = restore._WindowBuffers()
    encoder = restore._SegmentEncoder(
        [SimpleNamespace(windows=windows)], lambda pixels: pixels, store, buffers, torch.device(device), _Bar()
    )
    for value in range(5):
        encoder.frame(torch.full((3, 16, 16), value * 10, dtype=torch.uint8, device=device))
    encoder.finish()

    def expected(values):
        frames = values + [values[-1]] * (restore.WINDOW_FRAMES - len(values))
        pixels = torch.tensor(frames, dtype=torch.float32).view(1, 1, -1, 1, 1).expand(1, 3, -1, 8, 8)
        return pixels.div(127.5).sub(1.0).to(half_dtype(torch.device(device)))

    assert torch.equal(encoded[0], expected([10, 20, 30]))
    assert torch.equal(encoded[1], expected([20, 30]))
    assert torch.equal(encoded[2], expected([40]))


def test_window_buffers_reuse_an_encoded_windows_buffer():
    buffers = restore._WindowBuffers()
    first = buffers.take(8)
    assert first.shape == (3, restore.WINDOW_FRAMES, 8, 8) and not first.is_pinned()
    buffers.give_back(8, first)
    assert buffers.take(8) is first
    assert buffers.take(8) is not first


def test_encrypted_span_uses_worker_for_every_model_pass(fakes, monkeypatch, tmp_path):
    requests = []

    def forbidden(*_args):
        pytest.fail("licensed weights loaded in the parent")

    def request(self, op, params, tensors, advance=None):
        requests.append((op, params, tensors))
        if op == "load_transformer":
            return {"steps": 8, "resident": 40, "count": 48}, []
        if op == "encode":
            assert tensors[0].dtype == torch.uint8
            assert tensors[0].shape == (1, 3, restore.WINDOW_FRAMES, 512, 512)
            return {}, [torch.zeros(1)]
        if op == "denoise_chain":
            advance(8 * len(tensors))
            return {}, [torch.zeros(1) for _ in tensors]
        if op == "decode":
            return {}, [torch.zeros(restore.WINDOW_FRAMES, 3, 2, 2, dtype=torch.uint8)]
        return {}, []

    monkeypatch.setattr(inference, "load_video_encoder", forbidden)
    monkeypatch.setattr(inference, "load_video_decoder", forbidden)
    monkeypatch.setattr(inference, "LtxTransformer", forbidden)
    monkeypatch.setattr(inference.ModelRuntime, "request", request)
    monkeypatch.setattr(restore, "crop_to_canvas", lambda _frame, crop: torch.zeros(3, crop.canvas, crop.canvas, dtype=torch.uint8))
    files = restore.LtxModelFiles(Path("t.enc"), Path("v.enc"), Path("d.enc"))
    render = replace(_render(tmp_path), files=files, window_hop=LESS_OVERLAP_WINDOW_HOP)
    writer = _Writer()
    restore.restore_spans(
        [restore.LtxSpan(_source(_video(12, range(2, 10))), (restore.LtxSegment(*restore.WHOLE_VIDEO_PTS, 5),), lambda: writer)],
        render, work_dir=tmp_path, resume_dir=_new_dir(tmp_path), progress=restore.Progress(12, disable=True, report=None), cancel=threading.Event(),
    )
    assert writer.closed
    assert [op for op, _, _ in requests] == [
        "prepare_transformer", "load_encoder", "encode", "unload_encoder", "load_transformer", "denoise_chain",
        "unload_transformer", "load_decoder", "decode", "unload_decoder",
    ]
    assert requests[5][1] == {"seeds": [5], "window_hop": LESS_OVERLAP_WINDOW_HOP}


def test_encrypted_worker_starts_on_its_own_thread_while_the_scan_runs(fakes, monkeypatch, tmp_path):
    scanning, started = threading.Event(), threading.Event()
    requests = []
    scan = restore.scan_span

    def slow_scan(*args, **kwargs):
        scanning.set()
        assert started.wait(5), "the worker did not start during the scan"
        return scan(*args, **kwargs)

    def request(self, op, params, tensors, advance=None):
        requests.append((op, threading.current_thread() is threading.main_thread()))
        if op == "prepare_transformer":
            assert scanning.wait(5)
            started.set()
        if op == "load_transformer":
            return {"steps": 8, "resident": 40, "count": 48}, []
        if op == "encode":
            return {}, [torch.zeros(1)]
        if op == "denoise_chain":
            advance(8 * len(tensors))
            return {}, [torch.zeros(1) for _ in tensors]
        if op == "decode":
            return {}, [torch.zeros(restore.WINDOW_FRAMES, 3, 2, 2, dtype=torch.uint8)]
        return {}, []

    monkeypatch.setattr(restore, "scan_span", slow_scan)
    monkeypatch.setattr(inference.ModelRuntime, "request", request)
    monkeypatch.setattr(restore, "crop_to_canvas", lambda _frame, crop: torch.zeros(3, crop.canvas, crop.canvas, dtype=torch.uint8))
    files = restore.LtxModelFiles(Path("t.enc"), Path("v.enc"), Path("d.enc"))
    restore.restore_spans(
        [restore.LtxSpan(_source(_video(12, range(2, 10))), (restore.LtxSegment(*restore.WHOLE_VIDEO_PTS, 5),), _Writer)],
        replace(_render(tmp_path), files=files), work_dir=tmp_path, resume_dir=_new_dir(tmp_path),
        progress=restore.Progress(12, disable=True, report=None), cancel=threading.Event(),
    )
    assert requests[0] == ("prepare_transformer", False)
    assert requests[1:3] == [("load_encoder", True), ("encode", True)]


def test_a_failed_scan_waits_for_the_worker_start_before_it_raises(fakes, monkeypatch, tmp_path):
    finished = []

    def request(self, op, params, tensors, advance=None):
        time.sleep(0.2)
        finished.append(op)
        return {}, []

    def failing_scan(*_args, **_kwargs):
        raise RuntimeError("synthetic scan failure")

    monkeypatch.setattr(restore, "scan_span", failing_scan)
    monkeypatch.setattr(inference.ModelRuntime, "request", request)
    files = restore.LtxModelFiles(Path("t.enc"), Path("v.enc"), Path("d.enc"))
    runtime = inference.ModelRuntime(files, torch.device("cpu"), threading.Event())
    with pytest.raises(RuntimeError, match="synthetic scan failure"):
        restore.prepare_spans([restore.LtxSpan(_source(_video(12, range(2, 10))), (restore.LtxSegment(*restore.WHOLE_VIDEO_PTS, 5),), _Writer)],
                              replace(_render(tmp_path), files=files), tmp_path, progress=restore.Progress(12, disable=True, report=None),
                              cancel=threading.Event(), runtime=runtime)
    assert finished == ["prepare_transformer"]


def test_a_run_without_a_speed_profile_runs_a_speed_test_first(fakes, tmp_path):
    (tmp_path / "speed.json").unlink()
    reports = []
    writer = _Writer()
    video = _video(12, range(2, 10))
    restore.restore_spans(
        [restore.LtxSpan(_source(video), (restore.LtxSegment(*restore.WHOLE_VIDEO_PTS, 7),), lambda: writer)],
        _render(tmp_path),
        work_dir=tmp_path, resume_dir=_new_dir(tmp_path),
        progress=restore.Progress(12, disable=True, report=lambda *args: reports.append(args)),
        cancel=threading.Event(),
    )
    assert _Transformer.seeds == [[0], [7]]
    assert len(writer.frames) == 12
    assert reports[0][:2] == (0.0, 0.0)
    assert (reports[0][2].name, reports[0][2].number, reports[0][2].count) == ("speed_test", 1, 1)
    assert [report[2].name for report in reports if report[2].name != "speed_test"] == ["encode", "denoise", "compose"]
    assert reports.index(next(r for r in reports if r[2].name == "encode")) > max(
        i for i, r in enumerate(reports) if r[2].name == "speed_test"
    )


def test_a_fresh_encode_pass_counts_frames_and_warms_up_after_its_first_batch(fakes, monkeypatch, tmp_path):
    monkeypatch.setattr(
        restore.Progress, "bar",
        lambda self, stage, total, unit="frame", marks=None: REAL_BAR(self, stage, total, unit, restore._WallClockMark),
    )
    progress = restore.Progress(12, disable=True, report=None)
    restore.restore_spans(
        [restore.LtxSpan(_source(_video(12, range(2, 10))), (restore.LtxSegment(*restore.WHOLE_VIDEO_PTS, 7),), _Writer)],
        _render(tmp_path), work_dir=tmp_path, resume_dir=_new_dir(tmp_path), progress=progress, cancel=threading.Event(),
    )
    timing = progress.timings["encode"]
    assert (timing.units, timing.warm_frames) == (12, 8)


def test_skipping_reused_passes_keeps_the_profile_paced_fraction(monkeypatch):
    estimates = {"encode": (0.0, 100.0), "denoise": (0.0, 1000.0), "compose": (0.0, 100.0)}
    monkeypatch.setattr(restore.speed, "pass_estimates", lambda _rates, _work: dict(estimates))
    reports = []
    progress = restore.Progress(100, disable=True, report=lambda fraction, _left, _step: reports.append(fraction))
    progress.plan(restore.speed.Work(100, 1.0, {512: 1}, 1.0))
    progress.skip("scan", "encode")
    progress.begin("denoise")
    assert reports[-1] < 0.01


def test_a_run_with_a_speed_profile_skips_the_speed_test(fakes, tmp_path):
    _run_spans(
        tmp_path,
        [restore.LtxSpan(_source(_video(12, range(2, 10))), (restore.LtxSegment(*restore.WHOLE_VIDEO_PTS, 7),), _Writer)],
    )
    assert _Transformer.seeds == [[7]]


def _resume_spans(tmp_path, spans):
    restore.restore_spans(
        spans, _render(tmp_path), work_dir=tmp_path, resume_dir=tmp_path / "saved",
        progress=restore.Progress(50, disable=True, report=None), cancel=threading.Event(),
    )


def _two_tracks():
    video = _video(50, range(0))
    for index in [*range(2, 10), *range(30, 38)]:
        video[index][0][0, 0, 0] = 1
    return [restore.LtxSpan(_source(video), (restore.LtxSegment(*restore.WHOLE_VIDEO_PTS, 7),), _Writer)]


def test_resume_reuses_plans_and_encoded_windows_after_encode_failure(fakes, monkeypatch, tmp_path):
    spans = _two_tracks()
    put = restore.LatentStore.put
    encoded = []

    def interrupted(store, index, latent):
        if store._kind.endswith("reference"):
            encoded.append(index)
            if len(encoded) == 2:
                raise RuntimeError("encode interrupted")
        put(store, index, latent)

    monkeypatch.setattr(restore.LatentStore, "put", interrupted)
    with pytest.raises(RuntimeError, match="encode interrupted"):
        _resume_spans(tmp_path, spans)
    encoded.clear()
    monkeypatch.setattr(restore, "scan_span", lambda *_a, **_k: pytest.fail("scan repeated"))
    monkeypatch.setattr(restore.LatentStore, "put", lambda store, index, latent: (
        encoded.append(index) if store._kind.endswith("reference") else None, put(store, index, latent)
    )[-1])
    _resume_spans(tmp_path, spans)
    assert encoded == [1]
    assert _Transformer.seeds == [[7], [8]]


def test_resume_skips_finished_tracks_and_does_not_load_encoder(fakes, monkeypatch, tmp_path):
    spans = _two_tracks()
    denoise = restore.denoise_track

    def interrupted(plan, *args, **kwargs):
        if plan.track_id == 1:
            raise RuntimeError("denoise interrupted")
        return denoise(plan, *args, **kwargs)

    monkeypatch.setattr(restore, "denoise_track", interrupted)
    with pytest.raises(RuntimeError, match="denoise interrupted"):
        _resume_spans(tmp_path, spans)
    assert _Transformer.seeds == [[7]]
    monkeypatch.setattr(restore, "denoise_track", denoise)
    monkeypatch.setattr(inference, "load_video_encoder", lambda *_a: pytest.fail("encoder loaded again"))
    monkeypatch.setattr(restore, "scan_span", lambda *_a, **_k: pytest.fail("scan repeated"))
    _Transformer.seeds.clear()
    _resume_spans(tmp_path, spans)
    assert _Transformer.seeds == [[8]]


def test_resume_after_compose_failure_loads_only_decoder(fakes, monkeypatch, tmp_path):
    spans = _two_tracks()
    compose = restore.compose_spans
    monkeypatch.setattr(restore, "compose_spans", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("compose failed")))
    with pytest.raises(RuntimeError, match="compose failed"):
        _resume_spans(tmp_path, spans)
    monkeypatch.setattr(restore, "compose_spans", compose)
    monkeypatch.setattr(inference, "load_video_encoder", lambda *_a: pytest.fail("encoder loaded again"))
    monkeypatch.setattr(inference, "LtxTransformer", lambda *_a: pytest.fail("transformer loaded again"))
    _resume_spans(tmp_path, spans)


def test_incomplete_track_keeps_all_references_until_finals_are_committed(fakes, monkeypatch, tmp_path):
    spans = [restore.LtxSpan(_source(_video(180, range(2, 178))), (restore.LtxSegment(*restore.WHOLE_VIDEO_PTS, 7),), _Writer)]
    put = restore.LatentStore.put
    written = []

    def interrupted(store, index, latent):
        if store._kind.endswith("final"):
            written.append(index)
            if len(written) == 2:
                raise RuntimeError("final write interrupted")
        put(store, index, latent)

    monkeypatch.setattr(restore.LatentStore, "put", interrupted)
    with pytest.raises(RuntimeError, match="final write interrupted"):
        _resume_spans(tmp_path, spans)
    assert len(list((tmp_path / "saved").glob("*.reference.pt"))) == 2
    assert not list((tmp_path / "saved").glob("*.done"))
    monkeypatch.setattr(restore.LatentStore, "put", put)
    monkeypatch.setattr(inference, "load_video_encoder", lambda *_a: pytest.fail("references were lost"))
    _Transformer.seeds.clear()
    _resume_spans(tmp_path, spans)
    assert _Transformer.seeds == [[7, 8]]


def test_resume_replays_whole_video_canvas_decision(fakes, monkeypatch, tmp_path):
    kwargs = dict(
        detector=_Detector(), files=FILES, frame_h=64, frame_w=64, batch_size=4,
        window_hop=88, seed=7, device=torch.device("cpu"), work_dir=tmp_path,
        speed_profile=tmp_path / "speed.json", resume_dir=tmp_path / "saved",
    )
    def run():
        restore.restore_video(
            _source(_video(12, range(2, 10))), _Writer(), large_canvas=True,
            progress=restore.Progress(12, disable=True, report=None), cancel=threading.Event(), **kwargs,
        )

    run()
    monkeypatch.setattr(restore, "large_canvas_fits", lambda *_a: pytest.fail("canvas decision repeated"))
    run()


def test_resume_of_saved_768_px_windows_fails_plainly_when_vram_is_short(fakes, tmp_path):
    from jasna.ltx.plan import TrackPlan, plans_to_arrays

    resume_dir = tmp_path / "saved"
    resume_dir.mkdir()
    crop = CropFrame((0, 0, 700, 700), (768, 768), (0, 0, 0, 0))
    polygon = np.array([[0, 0], [700, 0], [700, 700]], dtype=np.int32)
    plan = TrackPlan(0, 0, ((polygon,),), (1.0,), (Window(0, 0, 0, 1, (crop,)),), 88)
    span = restore.LtxSpan(_source(_video(12, range(0))), (restore.LtxSegment(*restore.WHOLE_VIDEO_PTS, 7),), _Writer)
    np.savez(restore._plan_path(span, resume_dir), frames=np.array(12), **plans_to_arrays([[plan]]))
    with pytest.raises(RuntimeError, match="768 px"):
        restore.restore_video(
            span.frames, _Writer(), detector=_Detector(), files=FILES, frame_h=720, frame_w=1280, batch_size=4,
            large_canvas=True, window_hop=88, seed=7, device=torch.device("cpu"), work_dir=tmp_path,
            speed_profile=tmp_path / "speed.json", progress=restore.Progress(12, disable=True, report=None),
            resume_dir=resume_dir, cancel=threading.Event(),
        )


def test_resuming_only_the_remaining_span_uses_its_original_latents(fakes, monkeypatch, tmp_path):
    video = _video(50, range(2, 48))
    first = restore.LtxSpan(_source(video[:20]), (restore.LtxSegment(20, 180, 7),), _Writer)

    class FailedWriter(_Writer):
        def write(self, frame, pts):
            raise RuntimeError("second span interrupted")

    second = restore.LtxSpan(_source(video, 20), (restore.LtxSegment(220, 480, 9),), FailedWriter)
    with pytest.raises(RuntimeError, match="second span interrupted"):
        _resume_spans(tmp_path, [first, second])
    monkeypatch.setattr(restore, "scan_span", lambda *_a, **_k: pytest.fail("span rescanned"))
    monkeypatch.setattr(inference, "load_video_encoder", lambda *_a: pytest.fail("span reencoded"))
    monkeypatch.setattr(inference, "LtxTransformer", lambda *_a: pytest.fail("span denoised again"))
    _resume_spans(tmp_path, [replace(second, open_writer=_Writer)])


@pytest.mark.parametrize(("frame_h", "frame_w"), [(720, 1280), (1080, 1920), (2160, 3840)])
def test_the_speed_test_mosaics_plan_two_windows_on_each_canvas(frame_h, frame_w):
    detections = restore._SpeedTestMosaics()(torch.zeros(1, 3, 8, 8), target_hw=(frame_h, frame_w))
    regions = restore.regions_per_frame(detections, frame_h, frame_w) * restore.SPEED_TEST_FRAMES
    plans = restore.plan_video(
        regions, set(), frame_w=frame_w, frame_h=frame_h, large_canvas=True, window_hop=restore.WINDOW_HOP
    )
    assert sorted(w.canvas for plan in plans for w in plan.windows) == [512, 512, 768, 768]
