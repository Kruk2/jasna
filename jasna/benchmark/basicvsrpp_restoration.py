from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from jasna.restorer.basicvrspp_tenorrt_compilation import basicvsrpp_startup_policy
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
    feats_ = _timed("feat_extract (TRT)", split._feat_extract_engine, lqs_flat)
    h_f, w_f = feats_.shape[2:]
    feats_ = feats_.view(n, t, -1, h_f, w_f)

    lqs_ds = _timed("downsample (bicubic)", lambda: F.interpolate(
        lqs.view(-1, c, h, w), scale_factor=0.25, mode="bicubic"
    ).view(n, t, c, h // 4, w // 4))

    flows_fwd, flows_bwd = _timed("compute_flow (SPyNet)", split.compute_flow, lqs_ds)

    feats: dict[str, list[torch.Tensor]] = {"spatial": [feats_[:, i, :, :, :] for i in range(t)]}

    total_loop_body = 0.0
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
            acc_grids: dict[int, torch.Tensor] = {}
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

    print(f"  {'propagate/precompute_all':30s} {total_precompute*1000:8.1f} ms")
    print(f"  {'propagate/loop_body (TRT)':30s} {total_loop_body*1000:8.1f} ms")

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

    use_tensorrt = basicvsrpp_startup_policy(
        restoration_model_path=str(path),
        device=device,
        fp16=fp16,
        compile_basicvsrpp=compile_basicvsrpp,
        max_clip_size=CLIP_LENGTH,
    )
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
