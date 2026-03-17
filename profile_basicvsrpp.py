"""Profile BasicVSR++ split forward to find actual bottleneck."""
from __future__ import annotations

import gc
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision

from jasna.restorer.basicvrspp_tenorrt_compilation import basicvsrpp_startup_policy
from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
from jasna.restorer.basicvsrpp_sub_engines import BasicVSRPlusPlusNetSplit, get_sub_engine_paths


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


def _timed_no_print(fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return result, dt


def _profile_stages_once(
    split: BasicVSRPlusPlusNetSplit,
    lqs: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    stage_ms: dict[str, float] = {}
    n, t, c, h, w = lqs.size()

    lqs_flat = lqs.view(-1, c, h, w)
    (feats_, flows_fwd, flows_bwd), dt = _timed_no_print(split._preprocess_engine, lqs_flat)
    stage_ms["preprocess (TRT)"] = dt
    h_f, w_f = feats_.shape[2:]
    feats_ = feats_.view(n, t, -1, h_f, w_f)
    flows_fwd = flows_fwd.view(n, t - 1, 2, h // 4, w // 4)
    flows_bwd = flows_bwd.view(n, t - 1, 2, h // 4, w // 4)

    feats: dict[str, list[torch.Tensor]] = {"spatial": [feats_[:, i, :, :, :] for i in range(t)]}
    total_loop_body = 0.0
    total_precompute = 0.0

    grid = BasicVSRPlusPlusNetSplit._make_identity_grid(h_f, w_f, device, dtype)

    flow_data: dict[str, tuple] = {}
    for direction in ["backward", "forward"]:
        flows = flows_bwd if direction == "backward" else flows_fwd
        torch.cuda.synchronize()
        tp0 = time.perf_counter()
        flow_data[direction] = split._precompute_flow_data(flows, direction, grid)
        torch.cuda.synchronize()
        total_precompute += time.perf_counter() - tp0

    for iter_ in [1, 2]:
        for direction in ["backward", "forward"]:
            module_name = f"{direction}_{iter_}"
            feats[module_name] = []
            flows = flows_bwd if direction == "backward" else flows_fwd
            frame_idx, flow_idx, acc_flows, flows_grid, acc_grids = flow_data[direction]

            n2, t2, _, h2, w2 = flows.size()
            mid = split.mid_channels
            mapping_idx = list(range(0, len(feats["spatial"])))
            mapping_idx += mapping_idx[::-1]

            lbe = split._loop_body_engines[module_name]
            backbone_pt = split.backbone[module_name]
            other_keys = [k for k in feats if k not in ["spatial", module_name]]

            zero_feat = flows.new_zeros(n2, mid, h2, w2)
            zero_flow = flows.new_zeros(n2, 2, h2, w2)
            feat_prop = flows.new_zeros(n2, mid, h2, w2)

            for i, idx in enumerate(frame_idx):
                feat_current = feats["spatial"][mapping_idx[idx]]
                backbone_prefix = torch.cat(
                    [feat_current] + [feats[k][idx] for k in other_keys],
                    dim=1,
                )
                if i > 0:
                    flow_n1 = flows[:, flow_idx[i], :, :, :]
                    g_n1 = flows_grid[:, flow_idx[i]]

                    if i > 1:
                        feat_n2 = feats[module_name][-2]
                        flow_n2 = acc_flows[i]
                        g_n2 = acc_grids[i]
                    else:
                        feat_n2 = zero_feat
                        flow_n2 = zero_flow
                        g_n2 = grid

                    torch.cuda.synchronize()
                    tlb0 = time.perf_counter()
                    feat_prop = lbe(feat_prop, g_n1, feat_n2, g_n2, feat_current, flow_n1, flow_n2, backbone_prefix)
                    torch.cuda.synchronize()
                    total_loop_body += time.perf_counter() - tlb0
                else:
                    feat = torch.cat([backbone_prefix, feat_prop], dim=1)
                    feat_prop = feat_prop + backbone_pt(feat)
                feats[module_name].append(feat_prop)

            if "backward" in module_name:
                feats[module_name] = feats[module_name][::-1]

    stage_ms["propagate/precompute_all"] = total_precompute
    stage_ms["propagate/loop_body (TRT)"] = total_loop_body

    _, dt = _timed_no_print(split.upsample, lqs, feats)
    stage_ms["upsample (TRT)"] = dt
    return stage_ms


def _print_stage_medians(stage_medians: dict[str, float]) -> None:
    for label, dt in stage_medians.items():
        print(f"  {label:30s} {dt*1000:8.1f} ms")

    print(f"\n=== Top stages by median ({RUNS} runs, warmup={WARMUP}) ===")
    total_stage = sum(stage_medians.values())
    ranked = sorted(stage_medians.items(), key=lambda item: item[1], reverse=True)
    for rank, (label, dt) in enumerate(ranked, start=1):
        pct = 0.0 if total_stage == 0 else (dt / total_stage) * 100.0
        print(f"  {rank:2d}. {label:30s} {dt*1000:8.1f} ms  ({pct:5.1f}%)")


def _print_engine_vram(model_weights_path: str, fp16: bool, device: torch.device, max_clip_size: int) -> None:
    from jasna.trt.torch_tensorrt_export import load_torchtrt_export
    paths = get_sub_engine_paths(model_weights_path, fp16, max_clip_size)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

    print(f"\n=== Engine VRAM (driver-level delta) ===")
    total_delta = 0
    for name, p in paths.items():
        torch.cuda.synchronize()
        free_before, _ = torch.cuda.mem_get_info()
        engine = load_torchtrt_export(checkpoint_path=p, device=device)
        torch.cuda.synchronize()
        free_after, _ = torch.cuda.mem_get_info()
        delta_mb = (free_before - free_after) / (1024 * 1024)
        total_delta += delta_mb
        print(f"  {name:26s} {delta_mb:8.1f} MB")
        del engine
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
    print(f"  {'TOTAL':26s} {total_delta:8.1f} MB")


def profile_split_forward(split, device, dtype, model_weights_path: str, fp16: bool, max_clip_size: int):
    T = CLIP_LENGTH
    lqs = torch.randn(1, T, 3, SIZE, SIZE, device=device, dtype=dtype)

    for _ in range(WARMUP):
        _profile_stages_once(split, lqs, device, dtype)
    torch.cuda.synchronize()

    stage_runs: list[dict[str, float]] = []
    for _ in range(RUNS):
        stage_runs.append(_profile_stages_once(split, lqs, device, dtype))
    stage_medians = {
        label: statistics.median(run[label] for run in stage_runs)
        for label in stage_runs[0]
    }

    print(f"\n=== Profiling BasicVSRPlusPlusNetSplit (T={T}) ===")
    _print_stage_medians(stage_medians)

    for _ in range(WARMUP):
        split(lqs)
    torch.cuda.synchronize()

    durations = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        split(lqs)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - t0)

    med = statistics.median(durations)
    print(f"\n  {'FULL FORWARD median':30s} {med*1000:8.1f} ms  ({RUNS} runs)")

    _print_engine_vram(model_weights_path, fp16, device, max_clip_size)


def main():
    device = torch.device("cuda:0")
    fp16 = True
    dtype = torch.float16

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_weights_path")
    parser.add_argument("--opt-level", type=int, default=5)
    args = parser.parse_args()

    path = Path(args.model_weights_path).resolve()
    use_trt = basicvsrpp_startup_policy(
        restoration_model_path=str(path), device=device, fp16=fp16,
        compile_basicvsrpp=True, max_clip_size=CLIP_LENGTH,
        optimization_level=args.opt_level,
    )
    restorer = BasicvsrppMosaicRestorer(
        checkpoint_path=str(path), device=device, max_clip_size=CLIP_LENGTH,
        use_tensorrt=use_trt, fp16=fp16,
    )

    if restorer._split_forward is not None:
        with torch.inference_mode():
            profile_split_forward(restorer._split_forward, device, dtype, str(path), fp16, CLIP_LENGTH)
    else:
        print("No split forward available (engines missing?)")


if __name__ == "__main__":
    main()
