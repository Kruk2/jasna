"""loop_body structural optimization experiments for BasicVSR++ sub-engines.

The T-frame forward spends ~90% of its time in 4*(T-1) sequential batch-1
loop_body calls, each split TRT -> torch deform_conv2d fallback -> TRT.
Experiments here:

  decomp   replace torchvision.ops.deform_conv2d with 9x grid_sample + 1x1 conv
           (TRT-native ops, exact same math) -> single-partition TRT engine
  unroll   chain K decomposed body steps inside one engine -> 1/K launches

Subcommands: bench (compile experimental engines in-memory, compare timings),
eval (end-to-end quality of decomposed engines vs fp32 reference).

Run from the model-opt venv:
    ~/.virtualenvs/model-opt/bin/python scripts/loop_body_fusion.py bench
    ~/.virtualenvs/model-opt/bin/python scripts/loop_body_fusion.py eval --data <calib-clip-dir>
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch
import torch.nn as nn
import torch.nn.functional as F

from quantize_basicvsrpp import (
    DEFAULT_WEIGHTS,
    DIRECTIONS,
    FEATURE_SIZE,
    INPUT_SIZE,
    MID_CHANNELS,
    _load_engine_set,
    _load_clips,
    _load_generator,
    _make_pytorch_split,
    _static_loop_body_inputs,
    _psnr,
    _ssim,
    _clip_to_lqs,
)

FP16_DIR = REPO_ROOT / "model_weights" / "lada_mosaic_restoration_model_generic_v1.2_sub_engines"


def _sample_grids(offset, mask, h, w, base_y, base_x):
    n = offset.shape[0]
    K = 9
    g = offset.shape[1] // (2 * K)
    off = offset.view(n, g, K, 2, h, w)
    m = mask.view(n, g, K, 1, h, w)
    grids = []
    for k in range(K):
        i, j = divmod(k, 3)
        py = base_y + (i - 1) + off[:, :, k, 0].reshape(n * g, h, w)
        px = base_x + (j - 1) + off[:, :, k, 1].reshape(n * g, h, w)
        gy = 2.0 * py / max(h - 1, 1) - 1.0
        gx = 2.0 * px / max(w - 1, 1) - 1.0
        grids.append(torch.stack((gx, gy), dim=-1))
    return grids, m


def deform_conv2d_decomposed(x, offset, w1x1, bias, mask, base_y, base_x):
    n, cin, h, w = x.shape
    K = 9
    g = offset.shape[1] // (2 * K)
    cg = cin // g
    grids, m = _sample_grids(offset, mask, h, w, base_y, base_x)
    xg = x.reshape(n * g, cg, h, w)
    samples = []
    for k in range(K):
        s = F.grid_sample(xg, grids[k], mode="bilinear", padding_mode="zeros", align_corners=True)
        samples.append(s.view(n, g, cg, h, w) * m[:, :, k])
    S = torch.stack(samples, dim=3).reshape(n, cin * K, h, w)
    return F.conv2d(S, w1x1, bias)


class _DecomposedBodyCore(nn.Module):
    """Shared parameters + single propagate-body step with decomposed deform conv."""

    def __init__(self, deform_align: nn.Module, backbone: nn.Module):
        super().__init__()
        self.conv_offset = deform_align.conv_offset
        self._max_res = int(deform_align.max_residue_magnitude)
        self._flow_rep = int(deform_align.deform_groups * 3 * 3 // 2)
        w = deform_align.weight
        cout, cin = w.shape[0], w.shape[1]
        self.register_buffer("w1x1", w.detach().reshape(cout, cin * 9, 1, 1).clone())
        self.dc_bias = deform_align.bias
        self.backbone = backbone
        ys = torch.arange(FEATURE_SIZE, dtype=torch.float32)
        self.register_buffer("base_y", ys.view(1, FEATURE_SIZE, 1).expand(1, FEATURE_SIZE, FEATURE_SIZE).clone())
        self.register_buffer("base_x", ys.view(1, 1, FEATURE_SIZE).expand(1, FEATURE_SIZE, FEATURE_SIZE).clone())

    def step(self, feat_prop, grid_n1, feat_n2, grid_n2, feat_current, flow_1, flow_2, backbone_prefix):
        cond_n1 = F.grid_sample(feat_prop, grid_n1, mode="bilinear", padding_mode="zeros", align_corners=True)
        cond_n2 = F.grid_sample(feat_n2, grid_n2, mode="bilinear", padding_mode="zeros", align_corners=True)

        x = torch.cat([cond_n1, feat_current, cond_n2, flow_1, flow_2], dim=1)
        out = self.conv_offset(x)
        o1, o2, mask = torch.chunk(out, 3, dim=1)

        offset = self._max_res * torch.tanh(torch.cat((o1, o2), dim=1))
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        offset_1 = offset_1 + flow_1.flip(1).repeat(1, self._flow_rep, 1, 1)
        offset_2 = offset_2 + flow_2.flip(1).repeat(1, self._flow_rep, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)

        mask = torch.sigmoid(mask)

        inp = torch.cat([feat_prop, feat_n2], dim=1)
        feat_prop_new = deform_conv2d_decomposed(
            inp, offset, self.w1x1, self.dc_bias, mask,
            self.base_y.to(inp.dtype), self.base_x.to(inp.dtype),
        )
        feat = torch.cat([backbone_prefix, feat_prop_new], dim=1)
        return feat_prop_new + self.backbone(feat)


class DecomposedBodyWrapper(nn.Module):
    def __init__(self, core: _DecomposedBodyCore):
        super().__init__()
        self.core = core

    def forward(self, feat_prop, grid_n1, feat_n2, grid_n2, feat_current, flow_1, flow_2, backbone_prefix):
        return self.core.step(feat_prop, grid_n1, feat_n2, grid_n2, feat_current, flow_1, flow_2, backbone_prefix)


class UnrolledBodyWrapper(nn.Module):
    """K chained body steps. Stacked inputs have leading dim K; steps s..s+K-1
    with the second-order recurrence resolved internally:
    step 0 uses (prev1, prev2), step 1 uses (out0, prev1), step k uses (out[k-1], out[k-2])."""

    def __init__(self, core: _DecomposedBodyCore, k_steps: int):
        super().__init__()
        self.core = core
        self.k_steps = k_steps

    def forward(self, prev1, prev2, grids_n1, grids_n2, feats_current, flows_1, flows_2, backbone_prefixes):
        outs = []
        fp, fn2 = prev1, prev2
        for k in range(self.k_steps):
            out = self.core.step(
                fp, grids_n1[k:k + 1].squeeze(0), fn2, grids_n2[k:k + 1].squeeze(0),
                feats_current[k:k + 1], flows_1[k:k + 1], flows_2[k:k + 1],
                backbone_prefixes[k:k + 1],
            )
            outs.append(out)
            fn2 = fp if k == 0 else outs[k - 1]
            fp = out
        return torch.cat(outs, dim=0)


def _compile(module, inputs, device, opt_level=5):
    import torch_tensorrt

    from jasna.trt.torch_tensorrt_export import _mute_torch_tensorrt

    _mute_torch_tensorrt()
    with torch.cuda.device(device):
        return torch_tensorrt.compile(
            module, ir="dynamo", inputs=inputs, min_block_size=1,
            workspace_size=8 << 30, enabled_precisions={torch.float16},
            use_fp32_acc=False, use_explicit_typing=False, sparse_weights=False,
            optimization_level=opt_level, hardware_compatible=False,
            use_python_runtime=False, cache_built_engines=False,
            reuse_cached_engines=False, truncate_double=True,
        )


def _timeit(label, fn, iters=50):
    with torch.inference_mode():
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(iters):
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
    ts.sort()
    print(f"{label:44s} median {ts[len(ts) // 2]:7.3f} ms  min {ts[0]:7.3f} ms")
    return ts[len(ts) // 2]


def _unroll_inputs(k, device, dtype, prefix_channels):
    fs, mid = FEATURE_SIZE, MID_CHANNELS
    return [
        torch.randn(1, mid, fs, fs, dtype=dtype, device=device),
        torch.randn(1, mid, fs, fs, dtype=dtype, device=device),
        torch.randn(k, 1, fs, fs, 2, dtype=dtype, device=device),
        torch.randn(k, 1, fs, fs, 2, dtype=dtype, device=device),
        torch.randn(k, mid, fs, fs, dtype=dtype, device=device),
        torch.randn(k, 2, fs, fs, dtype=dtype, device=device),
        torch.randn(k, 2, fs, fs, dtype=dtype, device=device),
        torch.randn(k, prefix_channels, fs, fs, dtype=dtype, device=device),
    ]


def cmd_bench(args) -> None:
    import torch_tensorrt

    device = torch.device(args.device)
    dtype = torch.float16
    _m, gen = _load_generator(args.weights, device, fp16=True)
    direction = "backward_1"
    core = _DecomposedBodyCore(gen.deform_align[direction], gen.backbone[direction]).to(device, dtype).eval()

    lb_inputs = _static_loop_body_inputs(device, dtype, 0)

    from jasna.restorer.basicvsrpp_sub_engines import _PropagateBodyWrapper

    ref_wrapper = _PropagateBodyWrapper(gen.deform_align[direction], gen.backbone[direction]).to(device, dtype).eval()
    with torch.inference_mode():
        ref = ref_wrapper(*lb_inputs)
        dec = DecomposedBodyWrapper(core)(*lb_inputs)
        print(f"decomp vs torchvision (fp16): max|d|={(dec - ref).abs().max().item():.4f}")

    prod = _load_engine_set(FP16_DIR, None, device, 60)["loop_body_backward_1"]
    _timeit("prod fp16 engine (torch fallback) x1", lambda: prod(*lb_inputs))

    print("compiling decomposed single-step engine...")
    eng1 = _compile(DecomposedBodyWrapper(core), lb_inputs, device, args.optimization_level)
    with torch.inference_mode():
        d = (eng1(*lb_inputs) - ref).abs().max().item()
    print(f"decomp engine vs torchvision wrapper: max|d|={d:.4f}")
    _timeit("decomposed single-partition engine x1", lambda: eng1(*lb_inputs))

    engines_k = {}
    for k in args.unroll:
        print(f"compiling K={k} unrolled engine...")
        uk_inputs = _unroll_inputs(k, device, dtype, MID_CHANNELS)
        engines_k[k] = _compile(UnrolledBodyWrapper(core, k), uk_inputs, device, args.optimization_level)
        med = _timeit(f"unrolled K={k} engine x1 ({k} steps)", lambda: engines_k[k](*uk_inputs))
        print(f"  -> per-step {med / k:.3f} ms")

    torch_tensorrt.runtime.set_cudagraphs_mode(True)
    _timeit("prod engine x1 + cudagraphs", lambda: prod(*lb_inputs))
    _timeit("decomposed engine x1 + cudagraphs", lambda: eng1(*lb_inputs))
    for k in args.unroll:
        uk_inputs = _unroll_inputs(k, device, dtype, MID_CHANNELS)
        med = _timeit(f"unrolled K={k} + cudagraphs", lambda: engines_k[k](*uk_inputs))
        print(f"  -> per-step {med / k:.3f} ms")
    torch_tensorrt.runtime.set_cudagraphs_mode(False)


def cmd_eval(args) -> None:
    """End-to-end quality: split forward with decomposed loop_body engines."""
    device = torch.device(args.device)
    dtype = torch.float16
    _m, gen = _load_generator(args.weights, device, fp16=True)
    engines = _load_engine_set(FP16_DIR, None, device, 60)
    for i, d in enumerate(DIRECTIONS):
        core = _DecomposedBodyCore(gen.deform_align[d], gen.backbone[d]).to(device, dtype).eval()
        inputs = _static_loop_body_inputs(device, dtype, i)
        print(f"compiling decomposed loop_body_{d}...")
        engines[f"loop_body_{d}"] = _compile(DecomposedBodyWrapper(core), inputs, device)
    split = _make_pytorch_split(gen, engines)
    ref_model, _ = _load_generator(args.weights, device, fp16=False)

    clips = _load_clips(Path(args.data), args.eval_clips, skip=args.skip_clips)
    psnrs, ssims = [], []
    with torch.inference_mode():
        for i, clip in enumerate(clips):
            out_q = split(_clip_to_lqs(clip, device, dtype)).squeeze(0).float().clamp(0, 1)
            out_ref = ref_model(inputs=_clip_to_lqs(clip, device, torch.float32)).squeeze(0).float().clamp(0, 1)
            psnrs.append(_psnr(out_q, out_ref))
            ssims.append(_ssim(out_q, out_ref))
    print(f"decomposed loop_body engines vs fp32 reference over {len(clips)} clips:")
    print(f"  PSNR mean {sum(psnrs) / len(psnrs):6.2f}  min {min(psnrs):6.2f} dB")
    print(f"  SSIM mean {sum(ssims) / len(ssims):.4f}  min {min(ssims):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("bench", help="compile decomposed/unrolled engines, compare timings")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--unroll", type=int, nargs="*", default=[5, 10])
    p.add_argument("--optimization-level", type=int, default=5)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("eval", help="e2e quality with decomposed loop_body engines")
    p.add_argument("--data", required=True)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval-clips", type=int, default=16)
    p.add_argument("--skip-clips", type=int, default=32)
    p.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
