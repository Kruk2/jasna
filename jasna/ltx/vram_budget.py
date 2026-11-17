"""The VRAM decisions of an LTX run: the 768 px canvas and the window decode budget."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import torch

from jasna.accelerator import device_free_bytes
from jasna.ltx.camera import PLAN_CANVAS
from jasna.ltx.plan import LARGE_CANVAS
from jasna.ltx.resume_files import plans_file, pts_key, read_plans
from jasna.ltx.spans import WHOLE_VIDEO_PTS

logger = logging.getLogger(__name__)

DecodeBudget = Callable[[int], int]


LARGE_CANVAS_MIN_FREE_BYTES = 9 << 30
"""Free VRAM a whole-video run needs for 768 px windows; a 12 GB card with a desktop on it
has ~9.7 GiB. With less, more transformer blocks stream from RAM and decode tiles shrink; from
~7.5 GiB the small tiles slow decoding sharply, and a protected run's LTX worker takes ~0.5 GiB
beyond the figure read before it starts."""


LARGE_CANVAS_MIN_TOTAL_BYTES = 15 << 30


SEGMENT_DECODE_BUDGETS = (
    (22 << 30, {PLAN_CANVAS: 8 << 30, LARGE_CANVAS: 16 << 30}),
    (15 << 30, {PLAN_CANVAS: 8 << 30, LARGE_CANVAS: 10 << 30}),
    (9 << 30, {PLAN_CANVAS: 6 << 30, LARGE_CANVAS: 8 << 30}),
    (7 << 30, {PLAN_CANVAS: 5 << 30, LARGE_CANVAS: 8 << 30}),
    (0, {PLAN_CANVAS: 4 << 30, LARGE_CANVAS: 8 << 30}),
)
"""(least total VRAM, decode budget per canvas): 8 GiB decodes a 512 window in one tile,
16 GiB a 768 window; 5 GiB keeps an 8 GB card's whole job under 7 GiB while BasicVSR++
and NVENC share it (#697), and a 6 GB card gets less again for the same reason (768 px
windows need 15 GiB, see ``segment_large_canvas``)."""


DECODE_BUDGET_STEP_BYTES = 512 << 20


DECODE_BUDGET_EXACT_BELOW = 4 << 30


def free_vram_budget(device: torch.device) -> DecodeBudget:
    """Free VRAM rounded down to ``DECODE_BUDGET_STEP_BYTES``: decode tiling follows the budget and
    each tile shape is a compiled graph, so an exact figure that wanders (8 GB cards) runs past
    the compile limit (#690). Under ``DECODE_BUDGET_EXACT_BELOW`` the exact figure stays, so the
    step never drops a decode that the smallest tiles (~3.3 GiB at 512) would fit."""

    def budget(_canvas: int) -> int:
        free = device_free_bytes(device)
        rounded = free // DECODE_BUDGET_STEP_BYTES * DECODE_BUDGET_STEP_BYTES
        return rounded if rounded >= DECODE_BUDGET_EXACT_BELOW else free

    return budget


def segment_decode_budget(total_bytes: int) -> DecodeBudget:
    """A decode budget fixed by the GPU's size, so a segment's tile layout (and pixels) never
    depend on the VRAM free at that moment."""
    return next(budgets for least, budgets in SEGMENT_DECODE_BUDGETS if total_bytes >= least).__getitem__


def large_canvas_fits(requested: bool, free_bytes: int) -> bool:
    """Below ``LARGE_CANVAS_MIN_FREE_BYTES`` free every window runs at 512 px instead."""
    if requested and free_bytes < LARGE_CANVAS_MIN_FREE_BYTES:
        logger.warning("LTX: %.1f GiB free VRAM, restoring large mosaics at 512 px", free_bytes / 2**30)
        return False
    return requested


def whole_video_large_canvas(requested: bool, free: int, resume_dir: Path, *, encoder_bytes: int) -> bool:
    """The 768 px decision of a whole-video run: saved work keeps the one it was made with,
    else ``free``, the VRAM free before the job's LTX worker started, decides, less the
    ``encoder_bytes`` the video encoder will hold beside LTX (#790)."""
    beside_encoder = free - encoder_bytes
    saved = read_plans(plans_file(pts_key(*WHOLE_VIDEO_PTS), resume_dir))
    if saved is None:
        return large_canvas_fits(requested, beside_encoder)
    if saved.large_canvas and beside_encoder < LARGE_CANVAS_MIN_FREE_BYTES:
        raise RuntimeError(
            f"The saved LTX work for this video uses 768 px windows, which need "
            f"{LARGE_CANVAS_MIN_FREE_BYTES / 2**30:.0f} GiB of free GPU memory beside the video encoder; "
            f"{beside_encoder / 2**30:.1f} GiB is free. Close other programs that use the GPU and try again."
        )
    return saved.large_canvas


def segment_large_canvas(requested: bool, total_bytes: int) -> bool:
    """The 768 px decision for segments, from the GPU's size so it is the same every run."""
    if requested and total_bytes < LARGE_CANVAS_MIN_TOTAL_BYTES:
        logger.warning("LTX: %.1f GiB GPU, restoring large mosaics at 512 px", total_bytes / 2**30)
        return False
    return requested
