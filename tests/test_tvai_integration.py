"""Integration tests for TvaiSecondaryRestorer using real Topaz Video AI ffmpeg.

These tests require:
- CUDA GPU
- Topaz Video AI installed at the default path
- TVAI_MODEL_DATA_DIR and TVAI_MODEL_DIR env vars set

Run with: pytest tests/test_tvai_integration.py -v
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

TVAI_FFMPEG_PATH = r"C:\Program Files\Topaz Labs LLC\Topaz Video AI\ffmpeg.exe"
TVAI_ARGS_SCALE1 = "model=iris-2:scale=1:preblur=0:noise=0:details=0:halo=0:blur=0:compression=0:estimate=8:blend=0.2:device=-2:vram=1:instances=1"
TVAI_ARGS_SCALE4 = "model=iris-2:scale=4:preblur=0:noise=0:details=0:halo=0:blur=0:compression=0:estimate=8:blend=0.2:device=-2:vram=1:instances=1"
MAX_CLIP_SIZE = 60

_skip_reason = None
if not torch.cuda.is_available():
    _skip_reason = "CUDA not available"
elif not Path(TVAI_FFMPEG_PATH).is_file():
    _skip_reason = f"TVAI ffmpeg not found at {TVAI_FFMPEG_PATH}"
elif not os.environ.get("TVAI_MODEL_DATA_DIR"):
    _skip_reason = "TVAI_MODEL_DATA_DIR env var not set"
elif not os.environ.get("TVAI_MODEL_DIR"):
    _skip_reason = "TVAI_MODEL_DIR env var not set"

pytestmark = pytest.mark.skipif(_skip_reason is not None, reason=_skip_reason or "")


def _make_restorer(*, num_workers: int = 1, tvai_args: str = TVAI_ARGS_SCALE4):
    from jasna.restorer.tvai_secondary_restorer import TvaiSecondaryRestorer
    return TvaiSecondaryRestorer(
        device=torch.device("cuda:0"),
        ffmpeg_path=TVAI_FFMPEG_PATH,
        tvai_args=tvai_args,
        max_clip_size=MAX_CLIP_SIZE,
        num_workers=num_workers,
    )


def _make_raw_restorer(*, tvai_args: str = TVAI_ARGS_SCALE4, max_clip_size: int = MAX_CLIP_SIZE):
    from jasna.restorer.tvai_secondary_restorer import _TvaiFfmpegRestorer
    return _TvaiFfmpegRestorer(
        device=torch.device("cuda:0"),
        ffmpeg_path=TVAI_FFMPEG_PATH,
        tvai_args=tvai_args,
        max_clip_size=max_clip_size,
    )


def _make_colored_frames(n: int, *, value: float = 0.5) -> torch.Tensor:
    """Create (n, 3, 256, 256) float tensor in [0, 1] on CUDA."""
    return torch.full((n, 3, 256, 256), value, dtype=torch.float32, device="cuda:0")


def _drain_all(restorer, *, timeout_s: float = 120.0) -> list:
    """Drain all completed items, including flushing."""
    import time
    items = restorer.drain_completed()
    restorer.flush(timeout_s=timeout_s)
    deadline = time.perf_counter() + timeout_s
    while True:
        batch = restorer.drain_completed()
        items.extend(batch)
        if time.perf_counter() >= deadline:
            break
        if not batch:
            break
    return items


_TVAI_EXTRA_ARGS = "preblur=0:noise=0:details=0:halo=0:blur=0:compression=0:estimate=8:blend=0.2:device=-2:vram=1:instances=1"


@pytest.mark.parametrize(
    "model,scale,clip_size",
    [
        ("iris-2", 1, 60),
        ("iris-2", 2, 60),
        ("iris-2", 4, 60),
        pytest.param("iris-3", 1, 60, marks=pytest.mark.xfail(reason="iris-3 scale=1 barely modifies input, no first-frame artifact")),
        ("iris-3", 2, 60),
        ("iris-3", 4, 60),
        ("prob-4", 1, 60),
        ("prob-4", 2, 60),
        ("prob-4", 4, 60),
        ("iris-2", 1, 180),
        ("iris-2", 2, 180),
        ("iris-2", 4, 180),
        pytest.param("iris-3", 1, 180, marks=pytest.mark.xfail(reason="iris-3 scale=1 barely modifies input, no first-frame artifact")),
        ("iris-3", 2, 180),
        ("iris-3", 4, 180),
        ("prob-4", 1, 180),
        ("prob-4", 2, 180),
        ("prob-4", 4, 180),
    ],
)
def test_tvai_raw_first_frame_is_garbage(model: str, scale: int, clip_size: int) -> None:
    """Prove that _TvaiFfmpegRestorer (no priming) produces a garbage first frame.

    Feeds clip_size copies of the same textured frame directly to the low-level
    restorer, flushes to collect all outputs, and verifies the very first
    output differs significantly from settled mid-sequence output.

    With identical input frames the temporal filter should converge to a stable
    output, but the very first frame lacks prior temporal context and will
    differ visibly (the artifact the user reported).
    """
    tvai_args = f"model={model}:scale={scale}:{_TVAI_EXTRA_ARGS}"
    rest = _make_raw_restorer(tvai_args=tvai_args, max_clip_size=clip_size)

    torch.manual_seed(42)
    one_frame = torch.rand(1, 3, 256, 256, dtype=torch.float32, device="cuda:0") * 0.6 + 0.2
    frames = one_frame.expand(clip_size, -1, -1, -1).contiguous()

    drained = rest.restore(frames, keep_start=0, keep_end=clip_size)
    drained.extend(rest.flush())

    assert len(drained) == clip_size, f"expected {clip_size} outputs, got {len(drained)}"

    first_frame = drained[0].float()
    ref_idx = clip_size // 2
    ref_frame = drained[ref_idx].float()

    first_mean = first_frame.mean().item()
    ref_mean = ref_frame.mean().item()

    first_vs_ref_mae = (first_frame - ref_frame).abs().mean().item()

    settled_a = drained[ref_idx].float()
    settled_b = drained[ref_idx + 1].float()
    baseline_mae = (settled_a - settled_b).abs().mean().item()

    print(f"[{model} scale={scale} clip={clip_size}] first_mean={first_mean:.1f}  ref_mean={ref_mean:.1f}  "
          f"first_vs_ref_mae={first_vs_ref_mae:.2f}  settled_baseline_mae={baseline_mae:.2f}")

    assert first_vs_ref_mae > baseline_mae + 0.5, (
        f"[{model} scale={scale} clip={clip_size}] Expected first raw TVAI frame to differ from settled output. "
        f"first_vs_ref MAE={first_vs_ref_mae:.2f}, settled baseline MAE={baseline_mae:.2f}. "
        f"first_mean={first_mean:.1f}, ref_mean={ref_mean:.1f}."
    )

    rest.close()


def test_tvai_first_frame_is_not_black_single_worker() -> None:
    """The very first output frame from TVAI must not be black/garbage.
    The priming frame should absorb the garbage first output."""
    restorer = _make_restorer(num_workers=1)

    frames = _make_colored_frames(MAX_CLIP_SIZE, value=0.5)
    meta = list(range(MAX_CLIP_SIZE))
    restorer.submit(frames, keep_start=0, keep_end=MAX_CLIP_SIZE, meta=meta, track_id=1)

    items = _drain_all(restorer)
    assert len(items) == MAX_CLIP_SIZE, f"expected {MAX_CLIP_SIZE} items, got {len(items)}"

    for item in items:
        frame_u8 = item.to_frame_u8(torch.device("cuda:0"))
        mean_val = frame_u8.float().mean().item()
        assert mean_val > 10, (
            f"frame meta={item.meta}: mean pixel value {mean_val:.1f} is near-black. "
            f"TVAI first-frame garbage was not absorbed by priming."
        )


def test_tvai_first_frame_is_not_black_two_workers() -> None:
    """Same check with 2 workers - both workers' first frames must be valid."""
    restorer = _make_restorer(num_workers=2)

    frames = _make_colored_frames(MAX_CLIP_SIZE, value=0.5)

    meta1 = [("t1", i) for i in range(MAX_CLIP_SIZE)]
    restorer.submit(frames, keep_start=0, keep_end=MAX_CLIP_SIZE, meta=meta1, track_id=1)

    meta2 = [("t2", i) for i in range(MAX_CLIP_SIZE)]
    restorer.submit(frames, keep_start=0, keep_end=MAX_CLIP_SIZE, meta=meta2, track_id=2)

    items = _drain_all(restorer)
    assert len(items) == 2 * MAX_CLIP_SIZE, f"expected {2 * MAX_CLIP_SIZE} items, got {len(items)}"

    for item in items:
        frame_u8 = item.to_frame_u8(torch.device("cuda:0"))
        mean_val = frame_u8.float().mean().item()
        assert mean_val > 10, (
            f"frame meta={item.meta}: mean pixel value {mean_val:.1f} is near-black. "
            f"TVAI first-frame garbage was not absorbed by priming."
        )


def test_tvai_first_frame_after_flush_track_is_not_black() -> None:
    """After flush_track (mosaic disappears then reappears), the first frame
    of the new track must not be garbage."""
    import time
    restorer = _make_restorer(num_workers=1)

    # First track: 20 frames
    frames1 = _make_colored_frames(20, value=0.6)
    meta1 = [("t1", i) for i in range(20)]
    restorer.submit(frames1, keep_start=0, keep_end=20, meta=meta1, track_id=1)

    # Flush track 1 (mosaic disappeared)
    restorer.flush_track(track_id=1)

    # Give worker time to process the flush
    time.sleep(2.0)
    items1 = restorer.drain_completed()

    # Second track on same worker: 20 frames
    frames2 = _make_colored_frames(20, value=0.4)
    meta2 = [("t2", i) for i in range(20)]
    restorer.submit(frames2, keep_start=0, keep_end=20, meta=meta2, track_id=2)

    items_rest = _drain_all(restorer)
    all_items = items1 + items_rest

    t2_items = [it for it in all_items if isinstance(it.meta, tuple) and it.meta[0] == "t2"]
    assert len(t2_items) == 20, f"expected 20 t2 items, got {len(t2_items)}"

    for item in t2_items:
        frame_u8 = item.to_frame_u8(torch.device("cuda:0"))
        mean_val = frame_u8.float().mean().item()
        assert mean_val > 10, (
            f"frame meta={item.meta}: mean pixel value {mean_val:.1f} is near-black. "
            f"TVAI garbage after flush_track was not absorbed by priming."
        )


def test_tvai_output_resembles_input() -> None:
    """Sanity check: TVAI output for a solid-color input should have similar
    mean brightness (not wildly different or black)."""
    restorer = _make_restorer(num_workers=1)

    input_value = 0.5
    expected_u8 = int(round(input_value * 255))
    frames = _make_colored_frames(30, value=input_value)
    meta = list(range(30))
    restorer.submit(frames, keep_start=0, keep_end=30, meta=meta, track_id=1)

    items = _drain_all(restorer)
    assert len(items) == 30

    for item in items:
        frame_u8 = item.to_frame_u8(torch.device("cuda:0"))
        mean_val = frame_u8.float().mean().item()
        assert abs(mean_val - expected_u8) < 60, (
            f"frame meta={item.meta}: mean pixel value {mean_val:.1f} deviates too much "
            f"from expected ~{expected_u8} (input was solid {input_value})."
        )
