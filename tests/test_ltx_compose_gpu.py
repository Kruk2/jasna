"""The LTX compose pass on the GPU: the colour-match histogram median against torch.quantile, the
frame loop's streams and the compose kernels against the Torch path."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

from jasna.ltx.compose import masked_median
from jasna.media.cuda_kernel import has_precompiled_kernels

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_masked_median_equals_quantile_without_a_host_sync():
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)
    for spread in (0, 1, 7, 255):
        for density in (0.0, 1e-5, 0.3, 1.0):
            difference = torch.randint(-spread, spread + 1, (3, 300, 500), generator=generator, device=device).float()
            mask = torch.rand(300, 500, generator=generator, device=device) < density
            torch.cuda.set_sync_debug_mode("error")
            try:
                median = masked_median(difference, mask)
            finally:
                torch.cuda.set_sync_debug_mode("default")
            if mask.any():
                expected = torch.quantile(difference[:, mask], 0.5, dim=1)
            else:
                expected = torch.zeros(3, device=device)
            assert torch.equal(median, expected)


def test_compose_span_decodes_on_a_thread_below_the_priority_of_the_frame_loop(monkeypatch):
    import contextlib
    import threading
    from types import SimpleNamespace

    from jasna.ltx import restore
    from jasna.ltx.camera import CropFrame
    from jasna.ltx.plan import TrackPlan, Window

    device = torch.device("cuda")
    crop = CropFrame((0, 0, 4, 4), (4, 4), (0, 0, 0, 0))
    plan = TrackPlan(0, 0, ((),) * 6, (1.0,) * 6, (Window(0, 0, 0, 6, (crop,) * 6, 0),), 3)
    decodes, writes = [], []

    def decode(decoder, latent, seed, canvas, budget, device):
        decodes.append((threading.current_thread() is threading.main_thread(), torch.cuda.current_stream().priority))
        return torch.zeros(6, 3, 4, 4, dtype=torch.uint8, device=device)

    class Writer:
        def write(self, frame, pts):
            writes.append(torch.cuda.current_stream().priority)

        def finish(self):
            pass

        def close(self):
            pass

    def frames():
        yield torch.zeros(6, 3, 4, 4, dtype=torch.uint8, device=device), list(range(6))

    monkeypatch.setattr(restore, "_decode_window", decode)
    monkeypatch.setattr(restore, "composite_frame", lambda source, candidates, *, feather, projector: source)
    bar = SimpleNamespace(
        update=lambda n: None,
        window=lambda canvas: contextlib.nullcontext(),
        waiting=contextlib.nullcontext,
        windows_done=lambda canvas, count: None,
    )
    span = restore.LtxSpan(frames, (restore.LtxSegment(*restore.WHOLE_VIDEO_PTS, 0),), Writer)
    restore.compose_span(
        span, [[plan]], None, [SimpleNamespace(get=lambda index: index)], frame_h=64, batch_size=6,
        budget=lambda canvas: 0, device=device, bar=bar, cancel=threading.Event(), super_res=None, projector=None,
    )
    assert decodes == [(False, 0)]
    assert len(writes) == 6 and all(priority < 0 for priority in writes)


def test_decoded_windows_stay_on_the_gpu_from_a_roomy_decode_budget_on(monkeypatch):
    from jasna.ltx import restore

    device = torch.device("cuda")
    monkeypatch.setattr(
        restore, "decode_latent", lambda decoder, latent, free_bytes, generator: torch.zeros(1, 3, 6, 4, 4, device=device)
    )

    def decoded(free_bytes):
        return restore._decode_window(None, torch.zeros(1, 128, 1, 1, 1), 0, 512, lambda canvas: free_bytes, device)

    assert decoded(restore.GPU_WINDOWS_MIN_FREE_BYTES).is_cuda
    assert decoded(restore.GPU_WINDOWS_MIN_FREE_BYTES - 1).is_cpu


@pytest.mark.parametrize("roomy", [False, True], ids=["host-memory windows", "gpu windows"])
def test_frames_take_decoded_windows_without_waiting_for_the_gpu(monkeypatch, roomy):
    import contextlib
    import threading
    from types import SimpleNamespace

    from jasna.ltx import restore
    from jasna.ltx.camera import CropFrame
    from jasna.ltx.plan import TrackPlan, Window

    device = torch.device("cuda")
    monkeypatch.setattr(
        restore, "decode_latent", lambda decoder, latent, free_bytes, generator: torch.zeros(1, 3, 6, 4, 4, device=device)
    )
    budget = restore.GPU_WINDOWS_MIN_FREE_BYTES if roomy else 1
    crop = CropFrame((0, 0, 4, 4), (4, 4), (0, 0, 0, 0))
    plan = TrackPlan(0, 0, ((),) * 6, (1.0,) * 6, (Window(0, 0, 0, 6, (crop,) * 6, 0),), 3)
    canvases = []

    def composite(source, candidates, *, feather, projector):
        canvases.extend(candidate.canvas.device.type for candidate in candidates)
        return source

    monkeypatch.setattr(restore, "composite_frame", composite)
    bar = SimpleNamespace(
        update=lambda n: None, window=lambda canvas: contextlib.nullcontext(), waiting=contextlib.nullcontext,
        windows_done=lambda canvas, count: None,
    )
    decodes = ThreadPoolExecutor(max_workers=1)
    try:
        composer = restore._SegmentComposer(
            [plan], None, SimpleNamespace(get=lambda index: torch.zeros(1)), seed=0, feather=1,
            budget=lambda canvas: budget, device=device, bar=bar, decodes=decodes, stream=torch.cuda.Stream(device),
            super_res=None, projector=None,
        )
        composer.frame(torch.zeros(3, 4, 4, dtype=torch.uint8, device=device))
        torch.cuda.set_sync_debug_mode("error")
        try:
            for _ in range(5):
                composer.frame(torch.zeros(3, 4, 4, dtype=torch.uint8, device=device))
        finally:
            torch.cuda.set_sync_debug_mode("default")
    finally:
        decodes.shutdown()
    assert canvases == ["cuda"] * 6


def test_composite_frame_queues_its_work_without_waiting_for_the_gpu():
    from jasna.ltx.camera import letterbox
    from jasna.ltx.compose import Candidate, composite_frame

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(0)
    source = torch.randint(0, 256, (3, 360, 640), dtype=torch.uint8, generator=generator, device=device)
    canvas = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8, generator=generator, device=device)
    left = [[150.0, 120.0], [230.0, 120.0], [230.0, 200.0], [150.0, 200.0]]
    right = [[420.0, 140.0], [500.0, 140.0], [500.0, 220.0]]
    candidates = [
        Candidate(canvas, letterbox((100, 60, 300, 260), 256), 1.0, 0, [left], 1.0),
        Candidate(canvas, letterbox((380, 80, 560, 300), 256), 0.6, 1, [right], 0.5),
    ]
    expected = composite_frame(source, candidates, feather=8, projector=None)
    torch.cuda.set_sync_debug_mode("error")
    try:
        out = composite_frame(source, candidates, feather=8, projector=None)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    assert torch.equal(out, expected)
    assert not torch.equal(out, source)


@pytest.mark.parametrize("feather", [7, 12, 40])
def test_crop_edge_ramps_are_bit_identical_on_the_gpu_and_the_cpu(feather):
    from jasna.ltx.compose import crop_edge_ramp

    shape, box, window = (300, 500), (37, 23, 431, 277), (10, 5, 480, 290)
    on_gpu = crop_edge_ramp(box, shape, feather, window, torch.device("cuda"))
    assert on_gpu.is_cuda
    assert torch.equal(on_gpu.cpu(), crop_edge_ramp(box, shape, feather, window, torch.device("cpu")))


@pytest.mark.parametrize("out_size, in_size", [(1, 512), (37, 64), (300, 333), (511, 512), (700, 768)])
def test_area_weights_are_bit_identical_on_the_gpu_and_the_cpu(out_size, in_size):
    from jasna.ltx.compose import _area_weights

    on_gpu = _area_weights(out_size, in_size, torch.device("cuda"))
    assert on_gpu.is_cuda
    assert torch.equal(on_gpu.cpu(), _area_weights(out_size, in_size, torch.device("cpu")))


def _random_polygon(rng: np.random.Generator, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    angles = np.sort(rng.uniform(0, 2 * np.pi, rng.integers(3, 12)))
    radii = rng.uniform(0.3, 1.0, len(angles))
    centre, half = np.array([(x1 + x2) / 2, (y1 + y2) / 2]), np.array([(x2 - x1) / 2, (y2 - y1) / 2])
    return (centre + half * radii[:, None] * np.stack([np.cos(angles), np.sin(angles)], 1)).astype(np.float32)


def _random_candidates(rng: np.random.Generator, height: int, width: int, device: torch.device):
    """Up to three tracks, each with one or two windows: crops cut by the frame edge, canvases that
    shrink and enlarge onto their crops, crossfade weights and tracks without polygons."""
    from jasna.ltx.camera import letterbox
    from jasna.ltx.compose import Candidate

    candidates = []
    for track in range(rng.integers(1, 4)):
        side = int(rng.integers(40, min(height, width) // 2))
        x1 = max(int(rng.integers(-side // 3, width - side + side // 3)), 0)
        y1 = max(int(rng.integers(-side // 3, height - side + side // 3)), 0)
        box = (x1, y1, min(x1 + side + int(rng.integers(-20, 20)), width), min(y1 + side, height))
        polygons = [] if rng.random() < 0.1 else [_random_polygon(rng, box) for _ in range(rng.integers(1, 3))]
        for _ in range(rng.integers(1, 3)):
            canvas_side = int(rng.choice([64, 512]))
            canvas = torch.from_numpy(rng.integers(0, 256, (3, canvas_side, canvas_side), dtype=np.uint8)).to(device)
            weight = float(np.float32(rng.choice([1.0, rng.uniform(0.01, 1.0)])))
            candidates.append(Candidate(canvas, letterbox(box, canvas_side), weight, track, polygons, 1.0))
    return candidates


@pytest.fixture
def compositing_caches_released():
    """Compositing keeps its Gaussian bands and resampling weights on the GPU. Left there after
    full-size frames they pin reserved memory, which the tests that measure it would see."""
    yield
    from jasna.ltx import compose
    from jasna.tracking import blending

    for cached in (compose._gaussian_band, compose._gaussian, compose._area_weights, blending.ellipse_row_reach):
        cached.cache_clear()
    torch.cuda.empty_cache()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not has_precompiled_kernels(torch.device("cuda")),
    reason="needs a GPU the precompiled kernels support",
)
@pytest.mark.parametrize("height, width", [(360, 640), (1080, 1920), (2160, 3840)])
def test_composite_on_the_compose_kernels_is_bit_identical_to_torch(height, width, compositing_caches_released):
    from jasna.ltx.compose import composite_on_kernels, composite_on_torch
    from jasna.ltx.plan import feather_pixels

    device = torch.device("cuda")
    rng = np.random.default_rng(height)
    feather = feather_pixels(height)
    frames = [
        (torch.from_numpy(rng.integers(0, 256, (3, height, width), dtype=np.uint8)).to(device),
         _random_candidates(rng, height, width, device))
        for _ in range(40)
    ]
    on_kernels = [composite_on_kernels(source, live, feather=feather) for source, live in frames]
    on_torch = [composite_on_torch(source, live, feather=feather, projector=None) for source, live in frames]
    assert all(torch.equal(k, t) for k, t in zip(on_kernels, on_torch, strict=True))
    assert sum(not torch.equal(k, source) for k, (source, _) in zip(on_kernels, frames)) > 30
