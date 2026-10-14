from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch

from jasna.restorer.basicvsrpp_sub_engines import compile_basicvsrpp_engines
from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
from jasna.restorer.basicvsrpp_sub_engines import BasicVSRPlusPlusNetSplit

CLIP_LENGTH = 60
SIZE = 256
WARMUP = 3
RUNS = 100


def _timed(label: str, fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  {label:30s} {dt*1000:8.1f} ms")
    return result


def _profile_split_forward(split: BasicVSRPlusPlusNetSplit, device: torch.device, dtype: torch.dtype) -> None:
    T = CLIP_LENGTH
    lqs = torch.randn(1, T, 3, SIZE, SIZE, device=device, dtype=dtype)

    for _ in range(WARMUP):
        split(lqs)
    torch.cuda.synchronize()

    print(f"\n=== Profiling BasicVSRPlusPlusNetSplit (T={T}) ===")
    n, t, c, h, w = lqs.size()

    lqs_flat = lqs.view(-1, c, h, w)
    feats_, flows_fwd, flows_bwd = _timed("preprocess (TRT)", split._preprocess, lqs_flat)
    h_f, w_f = feats_.shape[2:]
    feats_ = feats_.view(n, t, -1, h_f, w_f)
    flows_fwd = flows_fwd.view(n, t - 1, 2, h // 4, w // 4)
    flows_bwd = flows_bwd.view(n, t - 1, 2, h // 4, w // 4)

    feats: dict[str, list[torch.Tensor]] = {"spatial": [feats_[:, i, :, :, :] for i in range(t)]}
    grid = BasicVSRPlusPlusNetSplit._make_identity_grid(h_f, w_f, device, dtype)

    flows_by_direction = {"backward": flows_bwd, "forward": flows_fwd}
    flow_data = _timed(
        "propagate/precompute_all",
        lambda: {d: split._precompute_flow_data(f, d, grid) for d, f in flows_by_direction.items()},
    )

    def propagate_all() -> dict[str, list[torch.Tensor]]:
        propagated = feats
        for iter_ in [1, 2]:
            for direction, flows in flows_by_direction.items():
                module_name = f"{direction}_{iter_}"
                propagated[module_name] = []
                propagated = split.propagate(propagated, flows, module_name, grid, *flow_data[direction])
        return propagated

    feats = _timed("propagate (TRT loop body)", propagate_all)

    _timed("upsample (TRT)", split.upsample, lqs, feats)

    durations: list[float] = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        split(lqs)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - t0)

    med = statistics.median(durations)
    print(f"\n  {'FULL FORWARD median':30s} {med*1000:8.1f} ms  ({RUNS} runs)")


def benchmark_basicvsrpp_restoration(
    *,
    device: torch.device,
    fp16: bool,
    restoration_model_path: Path | None,
    compile_basicvsrpp: bool,
    **_: object,
) -> None:
    if restoration_model_path is None or not restoration_model_path.resolve().exists():
        return
    path = restoration_model_path.resolve()

    use_tensorrt = compile_basicvsrpp and compile_basicvsrpp_engines(str(path), device, fp16)
    restorer = BasicvsrppMosaicRestorer(
        checkpoint_path=str(path),
        device=device,
        max_clip_size=CLIP_LENGTH,
        use_tensorrt=use_tensorrt,
        fp16=fp16,
    )

    dtype = torch.float16 if fp16 else torch.float32

    if restorer._split_forward is not None:
        with torch.inference_mode():
            _profile_split_forward(restorer._split_forward, device, dtype)
    else:
        print("\nNo split forward available (engines missing?), skipping detailed profiling.")

        durations: list[float] = []
        video = [
            torch.randint(0, 256, (SIZE, SIZE, 3), dtype=torch.uint8, device=device)
            for _ in range(CLIP_LENGTH)
        ]
        with torch.inference_mode():
            for _ in range(RUNS):
                start = time.perf_counter()
                restorer.raw_process(video)
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - start)

        med = statistics.median(durations)
        print(f"\n  {'raw_process median':30s} {med*1000:8.1f} ms  ({RUNS} runs)")
    print()
