"""The histogram median of the LTX colour match on the GPU against torch.quantile."""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from jasna.ltx.compose import masked_median

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
    plan = TrackPlan(0, 0, ((),) * 6, (1.0,) * 6, (Window(0, 0, 0, 6, (crop,) * 6),), 3)
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
    monkeypatch.setattr(restore, "composite_frame", lambda source, candidates, *, feather: source)
    bar = SimpleNamespace(
        update=lambda n: None,
        window=lambda canvas: contextlib.nullcontext(),
        waiting=contextlib.nullcontext,
        windows_done=lambda canvas, count: None,
    )
    span = restore.LtxSpan(frames, (restore.LtxSegment(*restore.WHOLE_VIDEO_PTS, 0),), Writer)
    restore.compose_span(
        span, [[plan]], None, [SimpleNamespace(get=lambda index: index)], frame_h=64, batch_size=6,
        budget=lambda canvas: 0, device=device, bar=bar, cancel=threading.Event(),
    )
    assert decodes == [(False, 0)]
    assert len(writes) == 6 and all(priority < 0 for priority in writes)


def test_decoded_windows_wait_in_host_memory_and_composite_on_the_gpu(monkeypatch):
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
    assert restore._decode_window(None, torch.zeros(1, 128, 1, 1, 1), 0, 512, lambda canvas: 1, device).is_cpu
    crop = CropFrame((0, 0, 4, 4), (4, 4), (0, 0, 0, 0))
    plan = TrackPlan(0, 0, ((),) * 6, (1.0,) * 6, (Window(0, 0, 0, 6, (crop,) * 6),), 3)
    canvases = []

    def composite(source, candidates, *, feather):
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
            budget=lambda canvas: 1, device=device, bar=bar, decodes=decodes, stream=torch.cuda.Stream(device),
        )
        for _ in range(6):
            composer.frame(torch.zeros(3, 4, 4, dtype=torch.uint8, device=device))
    finally:
        decodes.shutdown()
    assert canvases == ["cuda"] * 6
