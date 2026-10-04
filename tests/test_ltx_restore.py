import contextlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from jasna.ltx import restore
from jasna.ltx.plan import Region

FILES = restore.LtxModelFiles(Path("t"), Path("v"), Path("d"))
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

    def window(self, canvas):
        return contextlib.nullcontext()

    def track(self, canvas, steps):
        return self.update

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Transformer:
    seeds: list[list[int]] = []

    def __init__(self, path, device):
        self.conditions = [None] * 8
        self.blocks = SimpleNamespace(resident=40, count=48)

    def denoise_chain(self, references, seeds, advance):
        _Transformer.seeds.append(list(seeds))
        for _ in self.conditions:
            advance(len(references))
        return [torch.zeros(1) for _ in references]

    def close(self):
        pass


class _Writer:
    def __init__(self):
        self.frames: list[tuple[int, int, bool]] = []
        self.closed = False

    def write(self, frame, pts, *, apply_lut=True):
        self.frames.append((pts, int(frame[2, 0, 0]), apply_lut))

    def close(self):
        self.closed = True


@pytest.fixture
def fakes(monkeypatch):
    budgets = []
    bars = []

    def decode(decoder, latent, free_bytes, generator):
        budgets.append(free_bytes)
        return torch.zeros(1, 3, restore.WINDOW_FRAMES, 2, 2)

    def composite(source, candidates, *, feather):
        out = source.clone()
        out[2, 0, 0] = len(candidates)
        return out

    def bar(self, name, total, unit="frame"):
        bars.append(_Bar(total))
        return bars[-1]

    monkeypatch.setattr(
        restore, "regions_per_frame", lambda frames, h, w: [[MOSAIC] if int(f[0, 0, 0]) else [] for f in frames]
    )
    monkeypatch.setattr(restore, "crop_to_canvas", lambda frame, crop: torch.zeros(1))
    monkeypatch.setattr(restore, "_vae_pixels", lambda canvases, device: torch.zeros(1))
    monkeypatch.setattr(restore, "load_video_encoder", lambda path, device: lambda pixels: torch.zeros(1))
    monkeypatch.setattr(restore, "load_video_decoder", lambda vae, tuned, device: object())
    monkeypatch.setattr(restore, "LtxTransformer", _Transformer)
    monkeypatch.setattr(restore, "decode_latent", decode)
    monkeypatch.setattr(restore, "composite_frame", composite)
    monkeypatch.setattr(restore.Progress, "bar", bar)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (123 << 20, 1 << 40))
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
        budget=restore.segment_decode_budget,
        budget_name="segment",
        device=torch.device("cpu"),
        speed_profile=tmp_path / "speed.json",
    )


def _run_spans(tmp_path, spans):
    restore.restore_spans(
        spans,
        _render(tmp_path),
        work_dir=tmp_path,
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
    assert budgets == [restore.SEGMENT_DECODE_BUDGET_BYTES[512]] * 2
    assert bars[2].total == bars[2].n == 16
    inside = {pts for pts in range(50, 150, 10)} | {pts for pts in range(200, 260, 10)}
    assert [pts for pts, _, _ in writer.frames] == list(range(0, 300, 10))
    for pts, marker, apply_lut in writer.frames:
        assert (marker, apply_lut) == ((1, True) if pts in inside else (0, False))
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
        seed=5,
        device=torch.device("cpu"),
        work_dir=tmp_path,
        speed_profile=tmp_path / "speed.json",
        progress=restore.Progress(12, disable=True, report=None),
        cancel=threading.Event(),
    )
    assert budgets == [123 << 20]
    assert _Transformer.seeds == [[5]]
    assert bars[2].total == bars[2].n == 8
    assert [apply_lut for _, _, apply_lut in writer.frames] == [True] * 12


def test_decode_window_converts_a_float32_decode(monkeypatch):
    decoded = torch.tensor([-1.0, 0.0, 1.0]).view(1, 1, 3, 1, 1).expand(1, 3, 3, 1, 1).contiguous()
    monkeypatch.setattr(restore, "decode_latent", lambda decoder, latent, free_bytes, generator: decoded.clone())
    frames = restore._decode_window(None, torch.zeros(1, 128, 1, 1, 1), 0, 512, lambda canvas: 1, torch.device("cpu"))
    assert frames.dtype == torch.uint8 and frames.shape == (3, 3, 1, 1)
    assert frames[:, 0, 0, 0].tolist() == [0, 128, 255]


def test_segment_decode_budget_ignores_free_vram(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1, 1))
    assert restore.segment_decode_budget(512) == 6 << 30
    assert restore.segment_decode_budget(768) == 8 << 30


def test_large_canvas_needs_ten_gib_free():
    gib = 1 << 30
    assert restore.large_canvas_fits(True, 10 * gib)
    assert not restore.large_canvas_fits(True, 9 * gib)
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
    "compose/frame": 0.02,
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
    progress.plan(restore.speed.Work(frames=100, megapixels=2.0, windows={512: 10}))
    with progress.bar("encode", 100) as bar:
        clock.now = 20.0
        bar.update(50)

    encode_left = (100 * 2.0 * 0.01 + 10 * 0.5) / 2
    later = 20.0 + 10 * 4.0 + 3.0 + 100 * 2.0 * 0.02 + 10 * 1.0
    fraction, left, stage = reports[-1]
    assert stage.name == "encode"
    assert stage.seconds_left == pytest.approx(encode_left)
    assert left == pytest.approx(encode_left + later)
    assert fraction == pytest.approx(20.0 / (20.0 + left))


def test_progress_without_rates_for_a_planned_canvas_keeps_the_fixed_shares(monkeypatch):
    _Clock(monkeypatch)
    reports = []
    progress = restore.Progress(100, disable=True, report=lambda *args: reports.append(args))
    progress.expect(_RATES)
    progress.plan(restore.speed.Work(frames=100, megapixels=2.0, windows={512: 3, 768: 1}))
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
    assert compose.units == 10
    assert denoise.seconds == 14.0
    assert denoise.units == 8
    assert denoise.windows == [(512, pytest.approx(4.0 / 4 * 3))]
    assert denoise.steps == [(512, 1.0), (512, 1.0)]


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
    video = _video(12, range(2, 10))
    _run_spans(tmp_path, [restore.LtxSpan(_source(video), (restore.LtxSegment(*restore.WHOLE_VIDEO_PTS, 1),), _Writer)])
    key = restore.speed.profile_key("Test GPU", "t", "segment")
    assert restore.speed.load_rates(tmp_path / "speed.json", key) == {}
    assert key in json.loads((tmp_path / "speed.json").read_text())


def test_a_whole_video_keeps_its_speed_profile_apart_from_segment_renders(fakes, tmp_path):
    restore.restore_video(
        _source(_video(12, range(2, 10))),
        _Writer(),
        detector=_Detector(),
        files=FILES,
        frame_h=64,
        frame_w=64,
        batch_size=4,
        large_canvas=False,
        seed=1,
        device=torch.device("cpu"),
        work_dir=tmp_path,
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


def test_a_prepared_span_denoises_with_many_seeds(fakes, tmp_path):
    video = _video(30, range(3, 28))
    span = restore.LtxSpan(_source(video), (restore.LtxSegment(50, 150, seed=1),), _Writer)
    progress = restore.Progress(30, disable=True, report=None)
    cancel = threading.Event()
    (prepared,) = restore.prepare_spans([span], _render(tmp_path), tmp_path, progress=progress, cancel=cancel)
    transformer = _Transformer(None, None)
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


def test_restore_spans_shows_each_pass_before_its_model_loads(fakes, monkeypatch, tmp_path):
    reports = []
    shown_at_load = []

    def loading(model, made):
        def load(*args):
            shown_at_load.append((model, reports[-1][2].name))
            return made(*args)

        return load

    monkeypatch.setattr(restore, "load_video_encoder", loading("encoder", restore.load_video_encoder))
    monkeypatch.setattr(restore, "LtxTransformer", loading("transformer", restore.LtxTransformer))
    monkeypatch.setattr(restore, "load_video_decoder", loading("decoder", restore.load_video_decoder))
    video = _video(12, range(2, 10))
    restore.restore_spans(
        [restore.LtxSpan(_source(video), (restore.LtxSegment(*restore.WHOLE_VIDEO_PTS, 1),), _Writer)],
        _render(tmp_path),
        work_dir=tmp_path,
        progress=restore.Progress(12, disable=True, report=lambda *args: reports.append(args)),
        cancel=threading.Event(),
    )
    assert shown_at_load == [("encoder", "encode"), ("transformer", "denoise"), ("decoder", "compose")]
