import threading
from pathlib import Path

import pytest
import torch

from jasna.ltx import restore
from jasna.ltx.plan import Region

FILES = restore.LtxModelFiles(Path("t"), Path("v"), Path("d"))
MOSAIC = Region(box=(10.0, 10.0, 40.0, 40.0), polygon=[[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]])


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

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Transformer:
    seeds: list[list[int]] = []

    def __init__(self, path, device):
        self.conditions = [None] * 8

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
    _Transformer.seeds = []
    return budgets, bars


def _run_spans(tmp_path, spans):
    restore.restore_spans(
        spans,
        detector=_Detector(),
        files=FILES,
        frame_h=64,
        frame_w=64,
        batch_size=4,
        large_canvas=False,
        budget=restore.segment_decode_budget,
        device=torch.device("cpu"),
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


def test_progress_reports_the_stage_and_the_share_of_the_whole_run(monkeypatch):
    clock = iter([0.0, 100.0, 100.0, 100.0, 300.0, 300.0])
    monkeypatch.setattr(restore.time, "monotonic", lambda: next(clock))
    reports = []
    progress = restore.Progress(10, disable=True, report=lambda *args: reports.append(args))
    with progress.bar("scan", 10) as bar:
        bar.update(10)
    with progress.bar("denoise", 4, unit="step") as bar:
        bar.update(2)
    assert [(stage, round(fraction, 3)) for stage, fraction, _eta in reports] == [
        ("scan", 0.0),
        ("scan", 0.02),
        ("denoise", 0.02),
        ("denoise", 0.42),
    ]
    assert [eta for *_rest, eta in reports[:3]] == [0.0, 0.0, 0.0]
    assert reports[-1][2] == pytest.approx(300.0 * 0.58 / 0.42)


def test_progress_without_a_report_only_drives_the_console_bar():
    progress = restore.Progress(3, disable=True, report=None)
    with progress.bar("compose", 3) as bar:
        bar.update(3)
