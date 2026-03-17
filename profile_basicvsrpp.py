"""Profile BasicVSR++ split forward to find actual bottleneck."""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision

from jasna.restorer.basicvrspp_tenorrt_compilation import basicvsrpp_startup_policy
from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
from jasna.restorer.basicvsrpp_sub_engines import BasicVSRPlusPlusNetSplit
from jasna.trt.trt_runner import TrtRunner


CLIP_LENGTH = 60
SIZE = 256
WARMUP = 3
RUNS = 10


def _timed(label: str, fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"  {label:30s} {dt*1000:8.1f} ms")
    return result


def profile_split_forward(split, device, dtype):
    T = CLIP_LENGTH
    lqs = torch.randn(1, T, 3, SIZE, SIZE, device=device, dtype=dtype)

    for _ in range(WARMUP):
        split(lqs)
    torch.cuda.synchronize()

    print(f"\n=== Profiling BasicVSRPlusPlusNetSplit (T={T}) ===")
    n, t, c, h, w = lqs.size()

    # feat_extract
    lqs_flat = lqs.view(-1, c, h, w)
    feats_ = _timed("feat_extract (TRT)", split._feat_extract_engine, lqs_flat)
    h_f, w_f = feats_.shape[2:]
    feats_ = feats_.view(n, t, -1, h_f, w_f)

    # downsample
    lqs_ds = _timed("downsample (bicubic)", lambda: F.interpolate(
        lqs.view(-1, c, h, w), scale_factor=0.25, mode="bicubic"
    ).view(n, t, c, h // 4, w // 4))

    # compute_flow (SPyNet)
    flows_fwd, flows_bwd = _timed("compute_flow (SPyNet)", split.compute_flow, lqs_ds)

    # propagate - detailed timing
    feats = {"spatial": [feats_[:, i, :, :, :] for i in range(t)]}

    total_deform_align = 0.0
    total_backbone = 0.0
    total_grid_sample = 0.0
    total_precompute = 0.0

    grid = BasicVSRPlusPlusNetSplit._make_identity_grid(h_f, w_f, device, dtype)

    for iter_ in [1, 2]:
        for direction in ["backward", "forward"]:
            module_name = f"{direction}_{iter_}"
            feats[module_name] = []
            flows = flows_bwd if direction == "backward" else flows_fwd

            n2, t2, _, h2, w2 = flows.size()
            mid = split.mid_channels
            frame_idx = list(range(0, t2 + 1))
            flow_idx = list(range(-1, t2))
            mapping_idx = list(range(0, len(feats["spatial"])))
            mapping_idx += mapping_idx[::-1]
            if "backward" in module_name:
                frame_idx = frame_idx[::-1]
                flow_idx = frame_idx

            torch.cuda.synchronize()
            tp0 = time.perf_counter()
            acc_flows = split._precompute_accumulated_flows(
                flows, flow_idx, len(frame_idx), grid,
            )
            scale_x = 2.0 / max(w2 - 1, 1)
            scale_y = 2.0 / max(h2 - 1, 1)
            flows_grid = flows.permute(0, 1, 3, 4, 2).contiguous()
            flows_grid[..., 0].mul_(scale_x)
            flows_grid[..., 1].mul_(scale_y)
            flows_grid.add_(grid.unsqueeze(1))
            acc_grids = {}
            if acc_flows:
                acc_keys = sorted(acc_flows.keys())
                acc_batch = torch.cat([acc_flows[k] for k in acc_keys], dim=0)
                acc_nhwc = acc_batch.permute(0, 2, 3, 1).contiguous()
                acc_nhwc[..., 0].mul_(scale_x)
                acc_nhwc[..., 1].mul_(scale_y)
                acc_nhwc.add_(grid)
                acc_grids = {k: acc_nhwc[j : j + 1] for j, k in enumerate(acc_keys)}
            torch.cuda.synchronize()
            total_precompute += time.perf_counter() - tp0

            backbone_engine = split._backbone_engines[module_name]
            dae = split._deform_align_engines[module_name]
            other_keys = [k for k in feats if k not in ["spatial", module_name]]

            zero_feat = flows.new_zeros(n2, mid, h2, w2)
            zero_flow = flows.new_zeros(n2, 2, h2, w2)
            zero_cond = zero_feat

            feat_prop = flows.new_zeros(n2, mid, h2, w2)

            for i, idx in enumerate(frame_idx):
                feat_current = feats["spatial"][mapping_idx[idx]]
                if i > 0:
                    flow_n1 = flows[:, flow_idx[i], :, :, :]

                    torch.cuda.synchronize()
                    tw0 = time.perf_counter()
                    cond_n1 = F.grid_sample(
                        feat_prop, flows_grid[:, flow_idx[i]],
                        mode="bilinear", padding_mode="zeros", align_corners=True,
                    )
                    if i > 1:
                        feat_n2 = feats[module_name][-2]
                        flow_n2 = acc_flows[i]
                        cond_n2 = F.grid_sample(
                            feat_n2, acc_grids[i],
                            mode="bilinear", padding_mode="zeros",
                            align_corners=True,
                        )
                    else:
                        feat_n2 = zero_feat
                        flow_n2 = zero_flow
                        cond_n2 = zero_cond
                    torch.cuda.synchronize()
                    total_grid_sample += time.perf_counter() - tw0

                    cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)

                    torch.cuda.synchronize()
                    tda0 = time.perf_counter()
                    feat_prop = dae(cond, flow_n1, flow_n2, feat_prop, feat_n2)
                    torch.cuda.synchronize()
                    total_deform_align += time.perf_counter() - tda0

                torch.cuda.synchronize()
                tb0 = time.perf_counter()
                feat = [feat_current] + [
                    feats[k][idx] for k in other_keys
                ] + [feat_prop]
                feat = torch.cat(feat, dim=1)
                feat_prop = feat_prop + backbone_engine(feat)
                torch.cuda.synchronize()
                total_backbone += time.perf_counter() - tb0

                feats[module_name].append(feat_prop)

            if "backward" in module_name:
                feats[module_name] = feats[module_name][::-1]

    print(f"  {'propagate/precompute_all':30s} {total_precompute*1000:8.1f} ms")
    print(f"  {'propagate/grid_sample':30s} {total_grid_sample*1000:8.1f} ms")
    print(f"  {'propagate/deform_align (TRT)':30s} {total_deform_align*1000:8.1f} ms")
    print(f"  {'propagate/backbone (TRT)':30s} {total_backbone*1000:8.1f} ms")

    # upsample
    _timed("upsample (TRT)", split.upsample, lqs, feats)

    # full forward timing
    durations = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        split(lqs)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - t0)

    import statistics
    med = statistics.median(durations)
    print(f"\n  {'FULL FORWARD median':30s} {med*1000:8.1f} ms  ({RUNS} runs)")


def profile_monolithic_engine(engine_path: str, device: torch.device, dtype: torch.dtype):
    T = CLIP_LENGTH
    runner = TrtRunner(
        engine_path=Path(engine_path),
        input_shape=(1, T, 3, SIZE, SIZE),
        device=device,
    )
    x = torch.randn(1, T, 3, SIZE, SIZE, device=device, dtype=dtype)

    for _ in range(WARMUP):
        runner.infer(x)
    torch.cuda.synchronize()

    print(f"\n=== Profiling Monolithic TRT Engine (T={T}) ===")
    durations = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        runner.infer(x)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - t0)

    import statistics
    med = statistics.median(durations)
    print(f"  {'MONOLITHIC FORWARD median':30s} {med*1000:8.1f} ms  ({RUNS} runs)")


def main():
    device = torch.device("cuda:0")
    fp16 = True
    dtype = torch.float16

    import sys
    if len(sys.argv) < 2:
        print("Usage: python profile_basicvsrpp.py <model_weights_path> [monolithic_engine_path]")
        sys.exit(1)

    path = Path(sys.argv[1]).resolve()
    use_trt = basicvsrpp_startup_policy(
        restoration_model_path=str(path), device=device, fp16=fp16, compile_basicvsrpp=True,
    )
    restorer = BasicvsrppMosaicRestorer(
        checkpoint_path=str(path), device=device, max_clip_size=CLIP_LENGTH,
        use_tensorrt=use_trt, fp16=fp16,
    )

    if restorer._split_forward is not None:
        with torch.inference_mode():
            profile_split_forward(restorer._split_forward, device, dtype)
    else:
        print("No split forward available (engines missing?)")

    if len(sys.argv) >= 3:
        with torch.inference_mode():
            profile_monolithic_engine(sys.argv[2], device, dtype)


if __name__ == "__main__":
    main()
