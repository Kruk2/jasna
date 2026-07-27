"""BasicVSR++ TensorRT sub-engine quantization experiments (INT8 / FP8).

Investigation driver for PTQ of the restoration sub-engines with NVIDIA Model
Optimizer + torch-tensorrt explicit typing. Writes engines to its own output
directory and never touches the production sub-engine cache.

Run from the model-opt venv (nvidia-modelopt, torch+cu130, TensorRT 10.16):

    PY=~/.virtualenvs/model-opt/bin/python
    $PY scripts/quantize_basicvsrpp.py selftest
    $PY scripts/quantize_basicvsrpp.py synth --video in.mp4 --out calib
    $PY scripts/quantize_basicvsrpp.py capture --out calib -- <jasna CLI args>
    $PY scripts/quantize_basicvsrpp.py compile --data calib --precision int8 \
        --out engines_int8 [--exclude "*conv_offset*,*spynet*"]
    $PY scripts/quantize_basicvsrpp.py eval --data calib --engines engines_int8 \
        --fallback model_weights/lada_mosaic_restoration_model_generic_v1.2_sub_engines
    $PY scripts/quantize_basicvsrpp.py bench --engines engines_int8 --fallback <fp16 dir>

`synth` and `selftest` are CPU-only; everything else needs the GPU.
"""

import argparse
import copy
import itertools
import math
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_WEIGHTS = str(REPO_ROOT / "model_weights" / "lada_mosaic_restoration_model_generic_v1.2.pth")
DIRECTIONS = ("backward_1", "forward_1", "backward_2", "forward_2")
LOOP_BODY_NAMES = tuple(f"loop_body_{d}" for d in DIRECTIONS)
ALL_TARGETS = LOOP_BODY_NAMES + ("preprocess", "upsample")
INPUT_SIZE = 256
FEATURE_SIZE = 64
MID_CHANNELS = 64


def _engine_filename(name: str, precision: str, max_clip_size: int) -> str:
    if name.startswith("loop_body_"):
        return f"{name}.trt_{precision}.linux.engine"
    if name == "preprocess":
        return f"preprocess_b{max_clip_size}.trt_{precision}.linux.engine"
    if name == "upsample":
        return f"upsample_dyn_b{max_clip_size}.trt_{precision}.linux.engine"
    raise ValueError(name)


def _find_engine(directory: Path, name: str, max_clip_size: int) -> Path | None:
    prefix = name if name.startswith("loop_body_") else (
        "preprocess_b" if name == "preprocess" else "upsample_dyn_b"
    )
    matches = sorted(directory.glob(f"{prefix}*.engine"))
    exact = [m for m in matches if f"_b{max_clip_size}." in m.name or name.startswith("loop_body_")]
    if exact:
        return exact[0]
    return matches[0] if matches else None


def _load_generator(weights: str, device, fp16: bool):
    from jasna.models.basicvsrpp.inference import load_model
    from jasna.restorer.basicvsrpp_sub_engines import _get_inference_generator

    model = load_model(None, weights, device, fp16)
    return model, _get_inference_generator(model)


def _build_wrappers(generator, device, dtype) -> dict:
    from jasna.restorer.basicvsrpp_sub_engines import (
        _PreprocessWrapper,
        _PropagateBodyWrapper,
        _UpsampleWrapper,
    )

    wrappers = {}
    for d in DIRECTIONS:
        wrappers[f"loop_body_{d}"] = _PropagateBodyWrapper(
            generator.deform_align[d], generator.backbone[d],
        ).to(device=device, dtype=dtype).eval()
    wrappers["preprocess"] = _PreprocessWrapper(
        generator.feat_extract, generator.spynet,
    ).to(device=device, dtype=dtype).eval()
    wrappers["upsample"] = _UpsampleWrapper(
        generator.reconstruction, generator.upsample1, generator.upsample2,
        generator.conv_hr, generator.conv_last,
    ).to(device=device, dtype=dtype).eval()
    return wrappers


def _make_pytorch_split(generator, wrappers):
    from jasna.restorer.basicvsrpp_sub_engines import BasicVSRPlusPlusNetSplit

    return BasicVSRPlusPlusNetSplit(
        generator,
        {d: wrappers[f"loop_body_{d}"] for d in DIRECTIONS},
        wrappers["preprocess"],
        wrappers["upsample"],
    )


def _load_clips(data_dir: Path, limit: int, skip: int = 0) -> list:
    import torch

    files = sorted(data_dir.glob("clip_*.pt"))[skip:skip + limit]
    if not files:
        raise SystemExit(f"no clip_*.pt files in {data_dir}")
    return [torch.load(f, weights_only=True) for f in files]


def _clip_to_lqs(clip, device, dtype):
    return clip.to(device=device, dtype=dtype).div(255.0).unsqueeze(0)


# ── quality metrics ──────────────────────────────────────────────────────────

def _psnr(a, b) -> float:
    import torch

    mse = torch.mean((a.float() - b.float()) ** 2).item()
    return 99.0 if mse == 0 else -10.0 * math.log10(mse)


def _ssim(a, b) -> float:
    import torch
    import torch.nn.functional as F

    a = a.float()
    b = b.float()
    coords = torch.arange(11, dtype=torch.float32, device=a.device) - 5
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    kernel_1d = (g / g.sum()).view(1, 1, 1, 11)
    c = a.shape[1]

    def blur(x):
        x = F.conv2d(x, kernel_1d.expand(c, 1, 1, 11), padding=(0, 5), groups=c)
        return F.conv2d(x, kernel_1d.view(1, 1, 11, 1).expand(c, 1, 11, 1), padding=(5, 0), groups=c)

    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / (
        (mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)
    )
    return ssim_map.mean().item()


# ── synth: CPU-only synthetic calibration clips ──────────────────────────────

def cmd_synth(args) -> None:
    import av
    import numpy as np
    import torch
    import torch.nn.functional as F

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames: list = []
    with av.open(args.video) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        total = stream.frames or 0
        step = max(1, total // max(args.clips * args.clip_len * 2, 1)) if total else 1
        for i, frame in enumerate(container.decode(stream)):
            if i % step == 0 or total == 0:
                frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= args.clips * args.clip_len:
                break
    if len(frames) < args.clip_len:
        raise SystemExit(f"decoded only {len(frames)} frames, need at least {args.clip_len}")

    h, w = frames[0].shape[:2]
    written = 0
    for ci in range(args.clips):
        start = rng.randrange(0, max(len(frames) - args.clip_len, 1))
        crop = rng.randrange(96, min(h, w, 512) + 1)
        y0 = rng.randrange(0, h - crop + 1)
        x0 = rng.randrange(0, w - crop + 1)
        block = rng.randrange(4, 33)
        cy, cx = rng.uniform(0.35, 0.65), rng.uniform(0.35, 0.65)
        ry, rx = rng.uniform(0.3, 0.55), rng.uniform(0.3, 0.55)

        yy, xx = np.mgrid[0:INPUT_SIZE, 0:INPUT_SIZE]
        mask = (((yy / INPUT_SIZE - cy) / ry) ** 2 + ((xx / INPUT_SIZE - cx) / rx) ** 2) <= 1.0
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        clip_frames = []
        for t in range(args.clip_len):
            f = frames[min(start + t, len(frames) - 1)][y0:y0 + crop, x0:x0 + crop]
            img = torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1).float().unsqueeze(0)
            img = F.interpolate(img, size=(INPUT_SIZE, INPUT_SIZE), mode="bilinear", align_corners=False)
            small = F.avg_pool2d(img, block, block, ceil_mode=True)
            mosaic = F.interpolate(small, size=(INPUT_SIZE, INPUT_SIZE), mode="nearest")
            img = torch.where(mask_t, mosaic[0], img[0])
            clip_frames.append(img.round().clamp(0, 255).to(torch.uint8))
        torch.save(torch.stack(clip_frames), out / f"clip_{ci:05d}.pt")
        written += 1
    print(f"wrote {written} synthetic clips (T={args.clip_len}) to {out}")


# ── capture: dump real clips from a jasna CLI run ────────────────────────────

def cmd_capture(args) -> None:
    import torch

    from jasna.restorer import basicvsrpp_mosaic_restorer as bmr

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counter = itertools.count()
    original = bmr.BasicvsrppMosaicRestorer.raw_process

    def patched(self, video):
        clip = torch.stack([f.detach().round().clamp(0, 255).to(torch.uint8).cpu() for f in video])
        torch.save(clip, out / f"clip_{next(counter):05d}.pt")
        return original(self, video)

    bmr.BasicvsrppMosaicRestorer.raw_process = patched
    sys.argv = ["jasna"] + args.jasna_args
    from jasna.main import main as jasna_main

    jasna_main()
    print(f"captured {next(counter)} clips to {out}")


# ── compile: calibrate + build quantized engines ─────────────────────────────

def _build_quant_cfg(precision: str, excludes: list[str]) -> dict:
    import modelopt.torch.quantization as mtq

    base = {"int8": mtq.INT8_DEFAULT_CFG, "fp8": mtq.FP8_DEFAULT_CFG}[precision]
    cfg = copy.deepcopy(base)
    quant_cfg = cfg["quant_cfg"]
    for pattern in excludes:
        entry = {"quantizer_name": f"{pattern}*_quantizer", "enable": False}
        if isinstance(quant_cfg, list):
            quant_cfg.append(entry)
        else:
            quant_cfg[entry["quantizer_name"]] = {"enable": False}
    return cfg


def _capture_wrapper_inputs(generator, wrappers, clips, device, dtype, targets, cap_per_target):
    import torch

    split = _make_pytorch_split(generator, wrappers)
    captured: dict[str, list] = {name: [] for name in targets}
    hooks = []

    def make_hook(name):
        def hook(_module, inputs):
            bucket = captured[name]
            if len(bucket) < cap_per_target:
                bucket.append(tuple(t.detach().clone() for t in inputs))

        return hook

    for name in targets:
        hooks.append(wrappers[name].register_forward_pre_hook(make_hook(name)))
    with torch.inference_mode():
        for clip in clips:
            split(_clip_to_lqs(clip, device, dtype))
    for h in hooks:
        h.remove()
    for name in targets:
        random.Random(0).shuffle(captured[name])
        print(f"  calib inputs for {name}: {len(captured[name])} calls")
    return captured


def _compile_quantized(module, trt_inputs, output_path: Path, device, dtype, opt_level: int) -> None:
    import torch
    import torch_tensorrt
    from modelopt.torch.quantization.utils import export_torch_mode
    from torch.export import Dim

    from jasna.trt.torch_tensorrt_export import _save_with_dynamic_shapes, get_workspace_size_bytes

    sample_args: list = []
    dyn_shapes: list = []
    has_dynamic = False
    for inp in trt_inputs:
        if isinstance(inp, torch_tensorrt.Input):
            has_dynamic = True
            shape = inp.shape
            sample_args.append(torch.randn(*shape["opt_shape"], dtype=dtype, device=device))
            dim_map = {
                d: Dim(f"d{d}", min=int(shape["min_shape"][d]), max=int(shape["max_shape"][d]))
                for d in range(len(shape["opt_shape"]))
                if shape["min_shape"][d] != shape["max_shape"][d]
            }
            dyn_shapes.append(dim_map or None)
        else:
            sample_args.append(inp)
            dyn_shapes.append(None)

    with torch.no_grad(), export_torch_mode():
        ep = torch.export.export(
            module, tuple(sample_args),
            dynamic_shapes=tuple(dyn_shapes) if has_dynamic else None,
            strict=False,
        )
        trt_gm = torch_tensorrt.dynamo.compile(
            ep,
            inputs=trt_inputs,
            device=device,
            min_block_size=1,
            workspace_size=get_workspace_size_bytes(),
            use_explicit_typing=True,
            use_fp32_acc=False,
            optimization_level=opt_level,
            truncate_double=True,
            cache_built_engines=False,
            reuse_cached_engines=False,
        )
    if has_dynamic:
        _save_with_dynamic_shapes(trt_gm, str(output_path), trt_inputs, device, dtype)
    else:
        torch_tensorrt.save(trt_gm, str(output_path), inputs=sample_args)


def _static_loop_body_inputs(device, dtype, direction_index: int) -> list:
    import torch

    mid = MID_CHANNELS
    prefix_channels = (1 + direction_index) * mid
    fs = FEATURE_SIZE
    return [
        torch.randn(1, mid, fs, fs, dtype=dtype, device=device),
        torch.randn(1, fs, fs, 2, dtype=dtype, device=device),
        torch.randn(1, mid, fs, fs, dtype=dtype, device=device),
        torch.randn(1, fs, fs, 2, dtype=dtype, device=device),
        torch.randn(1, mid, fs, fs, dtype=dtype, device=device),
        torch.randn(1, 2, fs, fs, dtype=dtype, device=device),
        torch.randn(1, 2, fs, fs, dtype=dtype, device=device),
        torch.randn(1, prefix_channels, fs, fs, dtype=dtype, device=device),
    ]


def _dynamic_input(channels_spec, max_clip_size: int, min_batch: int, dtype):
    import torch_tensorrt

    return torch_tensorrt.Input(
        min_shape=[min_batch, *channels_spec],
        opt_shape=[max_clip_size, *channels_spec],
        max_shape=[max_clip_size, *channels_spec],
        dtype=dtype,
    )


def cmd_compile(args) -> None:
    import modelopt.torch.quantization as mtq
    import torch

    device = torch.device(args.device)
    dtype = torch.float16
    targets = args.targets.split(",")
    excludes = [p for p in args.exclude.split(",") if p]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"precision={args.precision} targets={targets} excludes={excludes}")
    _model, generator = _load_generator(args.weights, device, fp16=True)
    wrappers = _build_wrappers(generator, device, dtype)
    clips = _load_clips(Path(args.data), args.calib_clips)
    print(f"calibrating on {len(clips)} clips")
    captured = _capture_wrapper_inputs(
        generator, wrappers, clips, device, dtype, targets, args.calib_calls,
    )

    for name in targets:
        wrapper = wrappers[name]
        calib_inputs = captured[name]
        cfg = _build_quant_cfg(args.precision, excludes)

        def forward_loop(mod):
            with torch.no_grad():
                for inp in calib_inputs:
                    mod(*[t.to(device) for t in inp])

        print(f"quantizing {name} ({args.precision}) on {len(calib_inputs)} calls")
        mtq.quantize(wrapper, cfg, forward_loop)
        if args.print_summary:
            mtq.print_quant_summary(wrapper)

        if name.startswith("loop_body_"):
            direction = name.removeprefix("loop_body_")
            trt_inputs = _static_loop_body_inputs(device, dtype, DIRECTIONS.index(direction))
        elif name == "preprocess":
            trt_inputs = [_dynamic_input((3, INPUT_SIZE, INPUT_SIZE), args.max_clip_size, 3, dtype)]
        else:
            trt_inputs = [_dynamic_input((5 * MID_CHANNELS, FEATURE_SIZE, FEATURE_SIZE), args.max_clip_size, 1, dtype)]

        path = out / _engine_filename(name, args.precision, args.max_clip_size)
        print(f"compiling {name} -> {path}")
        started = time.perf_counter()
        _compile_quantized(wrapper, trt_inputs, path, device, dtype, args.optimization_level)
        print(f"  done in {time.perf_counter() - started:.1f}s")
        torch.cuda.empty_cache()
    print(f"engines written to {out}")


# ── eval: quality vs fp32 PyTorch reference ──────────────────────────────────

def _load_engine_set(engines_dir: Path, fallback_dir: Path | None, device, max_clip_size: int):
    from jasna.trt.torch_tensorrt_export import load_torchtrt_export

    loaded = {}
    for name in ALL_TARGETS:
        path = _find_engine(engines_dir, name, max_clip_size)
        origin = "engines"
        if path is None and fallback_dir is not None:
            path = _find_engine(fallback_dir, name, max_clip_size)
            origin = "fallback"
        if path is None:
            raise SystemExit(f"engine for {name} not found in {engines_dir} or fallback")
        print(f"  {name}: [{origin}] {path.name}")
        loaded[name] = load_torchtrt_export(checkpoint_path=str(path), device=device)
    return loaded


def cmd_eval(args) -> None:
    import torch

    device = torch.device(args.device)
    engines = _load_engine_set(Path(args.engines), Path(args.fallback) if args.fallback else None, device, args.max_clip_size)

    _model16, generator16 = _load_generator(args.weights, device, fp16=True)
    split = _make_pytorch_split(generator16, engines)

    ref_model, _ = _load_generator(args.weights, device, fp16=False)

    clips = _load_clips(Path(args.data), args.eval_clips, skip=args.skip_clips)
    psnrs, ssims = [], []
    with torch.inference_mode():
        for i, clip in enumerate(clips):
            out_q = split(_clip_to_lqs(clip, device, torch.float16)).squeeze(0).float().clamp(0, 1)
            out_ref = ref_model(inputs=_clip_to_lqs(clip, device, torch.float32)).squeeze(0).float().clamp(0, 1)
            p = _psnr(out_q, out_ref)
            s = _ssim(out_q, out_ref)
            psnrs.append(p)
            ssims.append(s)
            print(f"clip {i:03d} T={clip.shape[0]:3d}  PSNR {p:6.2f} dB  SSIM {s:.4f}")
    print(f"\nvs fp32 PyTorch reference over {len(clips)} clips:")
    print(f"  PSNR mean {sum(psnrs) / len(psnrs):6.2f}  min {min(psnrs):6.2f} dB")
    print(f"  SSIM mean {sum(ssims) / len(ssims):.4f}  min {min(ssims):.4f}")


# ── bench: per-stage latency ─────────────────────────────────────────────────

def cmd_bench(args) -> None:
    import torch

    device = torch.device(args.device)
    dtype = torch.float16
    engines = _load_engine_set(Path(args.engines), Path(args.fallback) if args.fallback else None, device, args.max_clip_size)
    _model, generator = _load_generator(args.weights, device, fp16=True)
    split = _make_pytorch_split(generator, engines)

    t = args.clip_len
    lqs = torch.rand(1, t, 3, INPUT_SIZE, INPUT_SIZE, dtype=dtype, device=device)
    lb_inputs = _static_loop_body_inputs(device, dtype, 0)
    pre_input = lqs.view(-1, 3, INPUT_SIZE, INPUT_SIZE)
    up_input = torch.randn(t, 5 * MID_CHANNELS, FEATURE_SIZE, FEATURE_SIZE, dtype=dtype, device=device)

    def timeit(label, fn, iters):
        with torch.inference_mode():
            for _ in range(3):
                fn()
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            times = []
            for _ in range(iters):
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize(device)
                times.append(start.elapsed_time(end))
        times.sort()
        print(f"{label:28s} median {times[len(times) // 2]:8.3f} ms  min {times[0]:8.3f} ms")

    timeit(f"preprocess (T={t})", lambda: engines["preprocess"](pre_input), args.iters)
    timeit("loop_body (1 call)", lambda: engines["loop_body_backward_1"](*lb_inputs), args.iters * 10)
    timeit(f"upsample (T={t})", lambda: engines["upsample"](up_input), args.iters)
    timeit(f"full forward (T={t})", lambda: split(lqs), max(args.iters // 2, 3))
    print(f"peak VRAM {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB")


# ── selftest: CPU-only sanity of the quantize→export chain ───────────────────

def cmd_selftest(_args) -> None:
    import modelopt.torch.quantization as mtq
    import torch
    from modelopt.torch.quantization.utils import export_torch_mode

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_offset = torch.nn.Conv2d(8, 8, 3, padding=1)
            self.body = torch.nn.Conv2d(8, 8, 3, padding=1)

        def forward(self, x):
            return self.body(torch.relu(self.conv_offset(x)))

    for precision in ("int8", "fp8"):
        model = Tiny().eval()
        cfg = _build_quant_cfg(precision, ["*conv_offset*"])
        data = [torch.randn(2, 8, 16, 16) for _ in range(4)]

        def forward_loop(mod):
            with torch.no_grad():
                for x in data:
                    mod(x)

        mtq.quantize(model, cfg, forward_loop)
        offset_disabled = not model.conv_offset.input_quantizer.is_enabled
        body_enabled = model.body.input_quantizer.is_enabled
        checks = [f"exclude-respected={offset_disabled and body_enabled}"]
        ok = offset_disabled and body_enabled
        if precision == "fp8":
            with torch.no_grad(), export_torch_mode():
                ep = torch.export.export(model, (data[0],), strict=False)
            ops = {str(n.target) for n in ep.graph.nodes if n.op == "call_function"}
            has_qdq = any("quantize_op" in op for op in ops)
            checks.append(f"quantize_op-in-graph={has_qdq}")
            ok = ok and has_qdq
        else:
            checks.append("export-check=skipped (int8 quantize_op needs CUDA inputs)")
        print(f"{precision}: {' '.join(checks)} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(1)
    print("selftest passed (CPU chain: mtq.quantize -> export_torch_mode -> torch.export)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="CPU: synthetic mosaic calibration clips from a video")
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--clips", type=int, default=48)
    p.add_argument("--clip-len", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("capture", help="GPU: dump real clips while running the jasna CLI")
    p.add_argument("--out", required=True)
    p.add_argument("jasna_args", nargs=argparse.REMAINDER, help="args after -- go to jasna CLI")
    p.set_defaults(func=cmd_capture)

    p = sub.add_parser("compile", help="GPU: calibrate + compile quantized sub-engines")
    p.add_argument("--data", required=True, help="dir with clip_*.pt calibration clips")
    p.add_argument("--precision", choices=("int8", "fp8"), required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--targets", default=",".join(ALL_TARGETS))
    p.add_argument("--exclude", default="", help="comma list of module patterns, e.g. *conv_offset*,*spynet*,*conv_last*")
    p.add_argument("--max-clip-size", type=int, default=60)
    p.add_argument("--calib-clips", type=int, default=32)
    p.add_argument("--calib-calls", type=int, default=128, help="max captured calls per sub-engine")
    p.add_argument("--optimization-level", type=int, default=5)
    p.add_argument("--print-summary", action="store_true")
    p.set_defaults(func=cmd_compile)

    p = sub.add_parser("eval", help="GPU: PSNR/SSIM vs fp32 PyTorch reference")
    p.add_argument("--data", required=True)
    p.add_argument("--engines", required=True)
    p.add_argument("--fallback", default=None, help="fp16 sub-engine dir for engines not in --engines")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval-clips", type=int, default=16)
    p.add_argument("--skip-clips", type=int, default=32, help="skip calibration clips")
    p.add_argument("--max-clip-size", type=int, default=60)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("bench", help="GPU: per-stage latency")
    p.add_argument("--engines", required=True)
    p.add_argument("--fallback", default=None)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--clip-len", type=int, default=30)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--max-clip-size", type=int, default=60)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("selftest", help="CPU: sanity-check quantize->export chain")
    p.set_defaults(func=cmd_selftest)

    args = parser.parse_args()
    if getattr(args, "jasna_args", None) and args.jasna_args[0] == "--":
        args.jasna_args = args.jasna_args[1:]
    args.func(args)


if __name__ == "__main__":
    main()
