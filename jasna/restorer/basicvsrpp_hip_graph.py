"""HIP-graph replay for the AMD BasicVSR++ forward.

On AMD the split forward still launches roughly one hundred and fifty thousand
small kernels per clip; the launches, not the math, are the cost. Capturing the
whole forward once per clip length and replaying it measured 534 -> 292 ms at
T=60 on an RX 7900 XT with bit-identical output.

Graphs need a static shape and a quiet GPU. Capture therefore happens once, in
the restorer's constructor, before any pipeline thread exists: capturing while
another thread submits work poisons its calls ("operation not permitted when
stream is capturing") and aborts the decode and encode threads on this ROCm
build. Clips whose length was captured replay; every other length runs eager.
A short clip is never padded up to a captured length - the model is
bidirectional, so copies of its last frame at the tail feed back into the
frames that are kept and smear the end of the clip.

JASNA_BV_HIP_GRAPH=0 disables the feature entirely.
"""
from __future__ import annotations

import logging
import os
import threading

import torch

from jasna.accelerator import (
    hip_graph_pool_generation,
    register_graph_pool,
    unregister_graph_pool,
)
from jasna.os_utils import env_flag

log = logging.getLogger(__name__)

HIP_GRAPH_ENV = "JASNA_BV_HIP_GRAPH"
# Extra clip lengths to capture, comma separated (e.g. "52,40"). A length that
# was not captured runs eager, ~1.8x slower per frame on the same card, so a run
# whose clips end early pays that on every such clip. Capture costs one eager
# forward plus the graph's reserved activations per length, so add only the
# lengths the clip histogram shows.
HIP_GRAPH_LENGTHS_ENV = "JASNA_BV_HIP_GRAPH_LENGTHS"
# Capture reserves one private pool per clip length. Those pools cannot be
# excluded from empty_cache(), and at 8K the pipeline reserve with graphs alive
# reaches the card's ceiling, so a defragmenting release would take the pools
# with it and the next replay would read re-mapped pages. Keeping the reserve
# below this fraction of the card means a release never has to run while graphs
# are held, so the pools stay valid; a machine that cannot afford that (8K on a
# 20 GB card) drops the graphs and runs eager, which is slower but correct.
HIP_GRAPH_MAX_RESERVE_ENV = "JASNA_BV_HIP_GRAPH_MAX_RESERVE"
HIP_GRAPH_MAX_RESERVE_DEFAULT = 0.55


def graph_lengths(max_clip_size: int, temporal_overlap: int) -> list[int]:
    """Clip lengths worth capturing for this restorer configuration."""
    lengths: list[int] = []
    t1 = int(max_clip_size)
    t2 = t1 - 2 * int(temporal_overlap)
    if t1 > 1:
        lengths.append(t1)
    if 1 < t2 < t1:
        lengths.append(t2)
    # Short anchors: a scene cut or a brief mosaic produces clips of a few
    # frames. Without an anchor the only captured length is the full clip size,
    # multiplying their cost by 20-80x.
    for anchor in (16, 32, 64, 128):
        if 1 < anchor < t1 and anchor not in lengths:
            lengths.append(anchor)
    for token in os.environ.get(HIP_GRAPH_LENGTHS_ENV, "").split(","):
        token = token.strip()
        if not token.isdigit():
            continue
        extra = int(token)
        if extra > 1 and extra not in lengths:
            lengths.append(extra)
    return lengths


def reserve_cap(device: torch.device) -> int:
    try:
        fraction = float(
            os.environ.get(HIP_GRAPH_MAX_RESERVE_ENV, "").strip()
            or HIP_GRAPH_MAX_RESERVE_DEFAULT
        )
    except ValueError:
        fraction = HIP_GRAPH_MAX_RESERVE_DEFAULT
    if not 0.1 <= fraction <= 1.0:
        fraction = HIP_GRAPH_MAX_RESERVE_DEFAULT
    total = torch.cuda.get_device_properties(device).total_memory
    return int(total * fraction)


class HipGraphReplay:
    """Captured graphs for one forward callable, one per clip length."""

    def __init__(self, device: torch.device):
        self.device = torch.device(device)
        self.enabled = (
            env_flag(HIP_GRAPH_ENV, default=True) and self.device.type == "cuda"
        )
        self._graphs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.cuda.CUDAGraph]] = {}
        self._generation = 0
        self._guard_held = False

    # -- capture ---------------------------------------------------------
    def capture(self, forward, lengths: list[int], *, dtype: torch.dtype) -> None:
        """Capture ``forward`` for each length. Constructor-time only."""
        if not self.enabled:
            return
        self._generation = hip_graph_pool_generation()
        for t in lengths:
            if not self._capture_one(forward, int(t), dtype=dtype):
                continue
            # At 8K the first capture alone can push the reserve past the cap,
            # so the graphs would be dropped on the very first replay anyway.
            # Stop instead of paying the capture cost for every remaining length.
            if self._drop_if_over_reserve():
                log.info(
                    "BasicVSR++ HIP graphs not affordable on this card at this "
                    "resolution; the remaining clip lengths run eager."
                )
                return
        if self._graphs:
            log.info(
                "BasicVSR++ HIP graphs ready for clip lengths %s (%s=0 disables)",
                sorted(self._graphs),
                HIP_GRAPH_ENV,
            )

    def _capture_one(self, forward, t: int, *, dtype: torch.dtype) -> bool:
        if t in self._graphs:
            return False
        try:
            free, _total = torch.cuda.mem_get_info(self.device)
        except Exception:  # noqa: BLE001
            free = 0
        if free < 2 * 1024**3:
            log.info(
                "BasicVSR++ graph capture skipped for T=%d: only %.1f GiB VRAM free",
                t,
                free / 1024**3,
            )
            return False
        reserved_before = torch.cuda.memory_reserved(self.device)
        try:
            with torch.inference_mode():
                static_input = torch.zeros(
                    (1, t, 3, 256, 256), device=self.device, dtype=dtype
                )
                # The eager call doubles as the capture warm-up: MIOpen kernel
                # selection must happen outside the capture.
                forward(static_input)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_output = forward(static_input)
        except Exception:  # noqa: BLE001
            log.warning(
                "BasicVSR++ HIP graph capture failed for T=%d; that clip length "
                "will run eager.",
                t,
                exc_info=True,
            )
            torch.cuda.synchronize(self.device)
            return False
        self._graphs[t] = (static_input, static_output, graph)
        self._hold_guard(True)
        log.info(
            "BasicVSR++ HIP graph captured for T=%d (+%.0f MiB reserved)",
            t,
            (torch.cuda.memory_reserved(self.device) - reserved_before) / (1024 * 1024),
        )
        return True

    # -- pool guard ------------------------------------------------------
    def _hold_guard(self, hold: bool) -> None:
        """Register the captured pools while they exist.

        The guard is what makes a defragmenting release skip empty_cache(): the
        pools cannot be spared and freeing one under a replay corrupts the clip.
        It must follow the graphs, not the restorer - dropping the graphs has to
        hand the release back, otherwise every later release is a no-op and the
        reserve walks up to the card's ceiling (measured at 8K: 19.98 GB used of
        20.46 GB, 194 offload calls that could not free anything).
        """
        token = id(self)
        if hold and not self._guard_held:
            register_graph_pool(token)
            self._guard_held = True
        elif not hold and self._guard_held:
            unregister_graph_pool(token)
            self._guard_held = False

    def _drop_if_over_reserve(self) -> bool:
        if not self._graphs:
            return False
        reserved = torch.cuda.memory_reserved(self.device)
        cap = reserve_cap(self.device)
        if reserved <= cap:
            return False
        log.warning(
            "BasicVSR++ HIP graphs dropped: %.1f GiB reserved is over the %.1f GiB "
            "cap for keeping graph pools alive (a defragmenting release would free "
            "them under a replay and corrupt its output). Clips run eager; raise "
            "%s to keep them if this card has the headroom.",
            reserved / 1024**3,
            cap / 1024**3,
            HIP_GRAPH_MAX_RESERVE_ENV,
        )
        self.close()
        return True

    # -- replay ----------------------------------------------------------
    def replay(self, batched: torch.Tensor) -> torch.Tensor | None:
        """Replay for this clip length, or None when it must run eager."""
        if not self._graphs:
            return None
        if self._generation != hip_graph_pool_generation():
            # A release slipped past the guard (a direct empty_cache elsewhere)
            # so the captured pools may have been re-mapped. Give the graphs up:
            # re-capturing needs a quiet GPU, which a pipeline thread cannot
            # provide, and a wrong replay is worse than a slow eager forward.
            log.warning(
                "VRAM cold-cache release ran outside the graph guard; dropping "
                "BasicVSR++ HIP graphs for lengths %s",
                sorted(self._graphs),
            )
            self.close()
            return None
        self._drop_if_over_reserve()
        length = int(batched.shape[1])
        entry = self._graphs.get(length)
        if entry is None:
            return None
        static_input, static_output, graph = entry
        static_input.copy_(batched)
        graph.replay()
        # static_output is a fixed buffer the next replay overwrites; callers
        # may hold the result long after, so hand out a copy.
        return static_output.clone()

    def close(self) -> None:
        self._graphs.clear()
        self._hold_guard(False)

    def __len__(self) -> int:
        return len(self._graphs)


def capture_thread_is_safe() -> bool:
    """Capture must run before the pipeline threads exist."""
    return threading.current_thread() is threading.main_thread()
