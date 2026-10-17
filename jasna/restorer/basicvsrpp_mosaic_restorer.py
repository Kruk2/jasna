import json
import logging
import threading
import time
import os
from pathlib import Path

import torch

logger = logging.getLogger(__name__)
from torch import Tensor

from jasna.accelerator import (
    hip_graph_pool_generation,
    is_nvidia_device,
    register_graph_pool,
    unregister_graph_pool,
)
from jasna.models.basicvsrpp.inference import load_model
from jasna.os_utils import env_flag, get_user_config_dir

# On AMD (no TensorRT sub-engines) the BasicVSR++ forward runs ~150k small CUDA
# ops per call, which makes it launch/dispatch bound: HIP-graph replay of the
# captured forward measured 534 -> 292 ms at T=60 (bit-identical output) on an
# RX 7900 XT (torch 2.14+rocm10.1, Windows).
#
# Graphs need static shapes AND a quiet GPU: capturing while the pipeline's
# other threads submit CUDA work poisons their calls ("operation not permitted
# when stream is capturing" aborts the decode/encode threads on this ROCm
# build). Capture therefore happens once here, at construction time - before
# any pipeline thread exists - for the two clip lengths that cover virtually
# every clip of a run: max_clip_size and max_clip_size - 2*temporal_overlap.
# Any other length (final partial clip, scene-cut clip) is padded with copies
# of its last frame to the next captured length and replayed there, so no clip
# ever pays a first-touch MIOpen compile in the middle of a job.
# JASNA_BV_HIP_GRAPH=0 disables the feature entirely.
_HIP_GRAPH_ENV = "JASNA_BV_HIP_GRAPH"
# Extra clip lengths to capture, comma separated (e.g. "52,40"). Any clip whose
# length was not captured runs eager, ~1.8x slower per frame on the same card, so
# a run whose clips end early (scene cuts, mosaics that disappear) pays that on
# every such clip. Capture cost is one eager forward plus the graph's reserved
# activations per length, so add only the lengths the clip histogram shows.
_HIP_GRAPH_LENGTHS_ENV = "JASNA_BV_HIP_GRAPH_LENGTHS"
# Clip lengths seen in past runs, persisted across processes. Tail clips (scene
# cuts, mosaics that end early) are shorter than max_clip_size and otherwise run
# eager; the first eager forward for a new length compiles/tunes MIOpen kernels
# and measured 10-30 s, which collapses the pipeline's rolling fps. The next
# session captures a graph for the most frequent learned lengths before the
# pipeline threads start, so that cost is paid during model loading instead.
_HISTORY_FILE = "bv_clip_lengths.json"
_HISTORY_MAX_EXTRA = 2
_HISTORY_MIN_FREE_BYTES = 6 * 1024 ** 3


def _length_history_path() -> Path:
    return get_user_config_dir("jasna") / _HISTORY_FILE


def _load_length_history() -> dict[int, int]:
    try:
        raw = json.loads(_length_history_path().read_text(encoding="utf-8"))
        return {int(k): int(v) for k, v in raw.items() if int(v) > 0}
    except Exception:
        return {}


def _record_length_history(length: int) -> None:
    path = _length_history_path()
    history = _load_length_history()
    history[int(length)] = history.get(int(length), 0) + 1
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({str(k): v for k, v in sorted(history.items())}),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    except Exception:
        logger.debug("Could not persist BasicVSR++ clip length history", exc_info=True)


class BasicvsrppMosaicRestorer:
    def __init__(
        self,
        checkpoint_path: str,
        device: torch.device,
        max_clip_size: int,
        use_tensorrt: bool,
        fp16: bool,
        config: str | dict | None = None,
        temporal_overlap: int = 0,
    ):
        self.device = torch.device(device)
        self.max_clip_size = int(max_clip_size)
        self.input_dtype = torch.float16 if fp16 else torch.float32

        self._split_forward = None
        self.model = None
        self._graphs: dict[int, tuple[Tensor, Tensor, "torch.cuda.CUDAGraph"]] = {}
        self._graph_pool_generation = 0
        self._recorded_lengths: set[int] = set()
        self._graphs_enabled = (
            env_flag(_HIP_GRAPH_ENV, default=True)
            and self.device.type == "cuda"
        )

        if use_tensorrt and is_nvidia_device(self.device):
            from jasna.restorer.basicvsrpp_sub_engines import create_split_forward

            pytorch_model = load_model(config, checkpoint_path, self.device, fp16)
            self._split_forward = create_split_forward(
                model=pytorch_model,
                model_weights_path=checkpoint_path,
                device=self.device,
                fp16=fp16,
            )
            if self._split_forward is not None:
                self._graphs_enabled = False
                logger.info("BasicVSR++ using TRT sub-engines (fp16=%s)", fp16)
            else:
                self.model = pytorch_model
                logger.info("BasicVSR++ sub-engines not found, using PyTorch model (fp16=%s)", fp16)
        else:
            self.model = load_model(config, checkpoint_path, self.device, fp16)
            logger.info("BasicVSR++ loaded from checkpoint: %s (fp16=%s)", checkpoint_path, fp16)

        if self._graphs_enabled:
            self._prewarm_graphs(int(temporal_overlap))

    def _prewarm_graphs(self, temporal_overlap: int) -> None:
        """Capture graphs for the two dominant clip lengths before the
        pipeline threads start (mid-pipeline capture is not safe on ROCm)."""
        self._temporal_overlap = int(temporal_overlap)
        register_graph_pool(id(self))
        # Construction happens before any pipeline thread exists, so this is
        # safe on a worker thread too: the GUI builds its session on the
        # processing thread, and deferring here left every clip eager for the
        # whole job (was ~2-3x slower on mosaic-heavy footage).
        self._capture_graphs(prewarm=True)

    def _capture_graphs(self, *, prewarm: bool = False) -> None:
        """(Re-)capture the graphs for the dominant clip lengths. Capture must
        happen off the pipeline threads; when invoked from a replay thread
        after a slipped release, defer - clips run eager and the graphs
        return on the next pass. ``prewarm`` marks the construction-time call,
        which precedes the pipeline threads and is therefore always safe."""
        self._graph_pool_generation = hip_graph_pool_generation()
        if not prewarm and threading.current_thread() is not threading.main_thread():
            logger.warning(
                "HIP graph re-capture skipped: not on the main thread; "
                "clips run eager until the next pass"
            )
            return
        temporal_overlap = getattr(self, "_temporal_overlap", 0)
        lengths: list[int] = []
        t1 = self.max_clip_size
        t2 = t1 - 2 * int(temporal_overlap)
        if t1 > 1:
            lengths.append(t1)
        if 1 < t2 < t1:
            lengths.append(t2)
        # Short anchors: a scene cut or a brief mosaic produces clips of a few
        # frames. Without an anchor the smallest captured length (e.g. 84) is
        # used as the pad target, multiplying their cost by 20-80x. A power-of-
        # two ladder keeps every pad at most ~2x the real clip length.
        for anchor in (16, 32, 64, 128):
            if 1 < anchor < t1 and anchor not in lengths:
                lengths.append(anchor)
        for token in os.environ.get(_HIP_GRAPH_LENGTHS_ENV, "").split(","):
            token = token.strip()
            if not token.isdigit():
                continue
            extra = int(token)
            if extra > 1 and extra not in lengths:
                lengths.append(extra)
        for t in lengths:
            self._capture_length(t)
        self._capture_learned_lengths()
        if self._graphs:
            logger.info(
                "BasicVSR++ HIP graphs ready for clip lengths %s (JASNA_BV_HIP_GRAPH=0 disables)",
                sorted(self._graphs),
            )

    def _capture_length(self, t: int) -> bool:
        if t in self._graphs or t <= 1:
            return False
        try:
            free, _total = torch.cuda.mem_get_info(self.device)
        except Exception:
            free = 0
        if free < 2 * 1024 ** 3:
            logger.info(
                "BasicVSR++ graph capture skipped for T=%d: only %.1f GiB VRAM free",
                t,
                free / 1024 ** 3,
            )
            return False
        reserved_before = torch.cuda.memory_reserved(self.device)
        try:
            with torch.inference_mode():
                static_input = torch.zeros(
                    (1, t, 3, 256, 256), device=self.device, dtype=self.input_dtype
                )
                # the eager call doubles as the capture warmup (MIOpen
                # kernel selection must happen outside the capture)
                self.model(inputs=static_input)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    static_output = self.model(inputs=static_input)
        except Exception:
            logger.warning(
                "BasicVSR++ HIP graph capture failed for T=%d; that clip"
                " length will run eager.", t, exc_info=True,
            )
            torch.cuda.synchronize(self.device)
            return False
        self._graphs[t] = (static_input, static_output, graph)
        logger.info(
            "BasicVSR++ HIP graph captured for T=%d (+%.0f MiB reserved)",
            t,
            (torch.cuda.memory_reserved(self.device) - reserved_before) / (1024 * 1024),
        )
        return True

    def _remember_length(self, length: int) -> None:
        """Persist a new tail length once per process (never per clip)."""
        if length in self._recorded_lengths:
            return
        self._recorded_lengths.add(length)
        _record_length_history(length)

    def _capture_learned_lengths(self) -> None:
        """Graph the tail lengths this machine actually meets.

        The dominant lengths are fixed, but a track that ends early produces a
        clip of any length, which otherwise runs eager; its first forward
        compiles MIOpen kernels (~10-30 s) and freezes the pipeline. The
        persisted histogram turns that one-off into a build-time graph.
        """
        try:
            free, _total = torch.cuda.mem_get_info(self.device)
        except Exception:
            return
        if free < _HISTORY_MIN_FREE_BYTES:
            logger.info(
                "BasicVSR++ learned clip lengths skipped: only %.1f GiB VRAM free",
                free / 1024 ** 3,
            )
            return
        history = _load_length_history()
        candidates = [
            length
            for length, count in sorted(history.items(), key=lambda kv: (-kv[1], kv[0]))
            if count >= 1 and 1 < length < self.max_clip_size
        ][:_HISTORY_MAX_EXTRA]
        for length in sorted(candidates):
            if self._capture_length(int(length)):
                logger.info(
                    "BasicVSR++ captured learned clip length T=%d (seen %d times)",
                    length,
                    history[length],
                )

    def close(self) -> None:
        if self._split_forward is not None:
            self._split_forward.close()
            self._split_forward = None
        unregister_graph_pool(id(self))
        self._graphs.clear()
        self._graphs_enabled = False
        self.model = None

    def _forward_graphed(self, batched: Tensor) -> Tensor:
        # A cold-cache release that slipped past the accelerator's graph-pool
        # guard (a direct torch.cuda.empty_cache elsewhere) can free the pool
        # a captured graph replays against; replay would then read re-mapped
        # pages. Drop the graphs and re-capture before trusting one again.
        if self._graphs and self._graph_pool_generation != hip_graph_pool_generation():
            logger.warning(
                "VRAM cold-cache release ran outside the graph guard; "
                "re-capturing BasicVSR++ HIP graphs for lengths %s",
                sorted(self._graphs),
            )
            self._graphs.clear()
            self._capture_graphs()
        length = int(batched.shape[1])
        entry = self._graphs.get(length)
        t0 = time.perf_counter()
        if entry is None and self._graphs:
            # A track that ends early yields a clip of any length; there is one
            # per track, so capturing every possible length is impossible. Pad
            # it to the smallest captured length with copies of its last frame
            # and replay that graph: the first ``length`` outputs differ from
            # the eager forward by ~60 dB PSNR (measured on real crops), while
            # eager would pay a one-off MIOpen compile of 10-30 s mid-video.
            target = next((n for n in sorted(self._graphs) if n >= length), None)
            if target is not None:
                self._remember_length(length)
                extra = target - length
                padded = torch.cat(
                    (batched, batched[:, -1:].expand(-1, extra, -1, -1, -1)), dim=1
                )
                static_input, static_output, graph = self._graphs[target]
                static_input.copy_(padded)
                graph.replay()
                result = static_output[:, :length].clone()
                logger.debug(
                    "BasicVSR++ clip T=%d padded to %d graph %.0f ms",
                    length,
                    target,
                    (time.perf_counter() - t0) * 1000.0,
                )
                return result
        if entry is None:
            self._remember_length(length)
            result = self.model(inputs=batched)
            logger.debug(
                "BasicVSR++ clip T=%d eager %.0f ms (length not captured)",
                length,
                (time.perf_counter() - t0) * 1000.0,
            )
            return result
        static_input, static_output, graph = entry
        static_input.copy_(batched)
        graph.replay()
        # static_output is a fixed buffer that the next replay overwrites;
        # callers may hold the result until long after, so hand out a copy.
        result = static_output.clone()
        logger.debug(
            "BasicVSR++ clip T=%d graph %.0f ms",
            int(batched.shape[1]),
            (time.perf_counter() - t0) * 1000.0,
        )
        return result

    def raw_process(self, video: list[Tensor]) -> torch.Tensor:
        """
        Args:
            video: list of (C, H, W) tensors in RGB format, [0, 255]
        Returns:
            (T, C, 256, 256) float tensor in [0, 1]
        """
        with torch.inference_mode():
            stacked = torch.stack(video).to(device=self.device, dtype=self.input_dtype, memory_format=torch.contiguous_format).div_(255.0)
            batched = stacked.unsqueeze(0)

            if self._split_forward is not None:
                result = self._split_forward(batched)
            else:
                result = self._forward_graphed(batched)
            return result.squeeze(0)
