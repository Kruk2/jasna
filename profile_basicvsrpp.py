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

    total_deform_offset = 0.0
    total_deform_conv = 0.0
    total_backbone = 0.0
    total_flow_warp = 0.0
    total_precompute = 0.0

    grid = BasicVSRPlusPlusNetSplit._make_identity_grid(h_f, w_f, device, dtype)

    for iter_ in [1, 2]:
        for direction in ["backward", "forward"]:
            module_name = f"{direction}_{iter_}"
            feats[module_name] = []
            flows = flows_bwd if direction == "backward" else flows_fwd

            n2, t2, _, h2, w2 = flows.size()
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
            torch.cuda.synchronize()
            total_precompute += time.perf_counter() - tp0

            backbone_engine = split._backbone_engines[module_name]
            feat_prop = flows.new_zeros(n2, split.mid_channels, h2, w2)

            for i, idx in enumerate(frame_idx):
                feat_current = feats["spatial"][mapping_idx[idx]]
                if i > 0:
                    flow_n1 = flows[:, flow_idx[i], :, :, :]

                    torch.cuda.synchronize()
                    tw0 = time.perf_counter()
                    cond_n1 = split._flow_warp_cached(
                        feat_prop, flow_n1.permute(0, 2, 3, 1), grid,
                    )
                    feat_n2 = torch.zeros_like(feat_prop)
                    flow_n2 = torch.zeros_like(flow_n1)
                    cond_n2 = torch.zeros_like(cond_n1)
                    if i > 1:
                        feat_n2 = feats[module_name][-2]
                        flow_n2 = acc_flows[i]
                        cond_n2 = split._flow_warp_cached(
                            feat_n2, flow_n2.permute(0, 2, 3, 1), grid,
                        )
                    torch.cuda.synchronize()
                    total_flow_warp += time.perf_counter() - tw0

                    cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)

                    torch.cuda.synchronize()
                    tdo0 = time.perf_counter()
                    offset, mask = split._deform_offset_engines[module_name](
                        cond, flow_n1, flow_n2,
                    )
                    torch.cuda.synchronize()
                    total_deform_offset += time.perf_counter() - tdo0

                    torch.cuda.synchronize()
                    tdc0 = time.perf_counter()
                    da = split.deform_align[module_name]
                    feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                    feat_prop = torchvision.ops.deform_conv2d(
                        feat_prop, offset, da.weight, da.bias,
                        da.stride, da.padding, da.dilation, mask,
                    )
                    torch.cuda.synchronize()
                    total_deform_conv += time.perf_counter() - tdc0

                torch.cuda.synchronize()
                tb0 = time.perf_counter()
                feat = [feat_current] + [
                    feats[k][idx] for k in feats if k not in ["spatial", module_name]
                ] + [feat_prop]
                feat = torch.cat(feat, dim=1)
                feat_prop = feat_prop + backbone_engine(feat)
                torch.cuda.synchronize()
                total_backbone += time.perf_counter() - tb0

                feats[module_name].append(feat_prop)

            if "backward" in module_name:
                feats[module_name] = feats[module_name][::-1]

    print(f"  {'propagate/precompute_flows':30s} {total_precompute*1000:8.1f} ms")
    print(f"  {'propagate/flow_warp':30s} {total_flow_warp*1000:8.1f} ms")
    print(f"  {'propagate/deform_offset (TRT)':30s} {total_deform_offset*1000:8.1f} ms")
    print(f"  {'propagate/deform_conv2d':30s} {total_deform_conv*1000:8.1f} ms")
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
