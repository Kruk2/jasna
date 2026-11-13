"""Tune ``conv_triton`` launches for the LTX VAE encoder's fp16 head on this GPU.

For each conv shape of the head (one 121-frame window at ``--canvas``), times comfy-kitchen's
``fp16_conv3d`` (with the padded input copy it needs), cuDNN bf16, and ``jasna::causal_conv3d`` over a
grid of launches, then prints an ``EncoderHead`` entry for ``video_encoder.ENCODER_HEADS``: each
(in, out) key whose best launch beats kitchen (or, for the 1024-channel convs kitchen declines,
cuDNN bf16) over all the window's convs with that key, weighted by their count (the (256, 256) key
serves 12 residual convs and one downsample). Random data.

``--verify`` then loads the encoder as the encode pass does with the kitchen head alone (6 down
blocks, no Triton conv), with that entry, and with the entry without its 1024-channel convs, times
them in turn on one window of ``--clip`` at 512 and 768 fed bf16 pixels like the pass, measures each
latent's distance from an fp32 encoder's, and prints the fastest entry that is faster than the kitchen
head at both sizes and whose distance rises by at most 1 % at each size and not on average. The
1024-channel convs, summed over 27648 products in fp16, cost accuracy on RTX 40 and 30 cards.

Needs an otherwise idle NVIDIA GPU, Triton and comfy-kitchen; the sweep takes ~5-20 minutes.

Usage:
    python scripts/benchmark_ltx_encoder_convs.py [--canvas 512|768] [--layers res6,down1] [--json out.json] \\
        [--verify [--repeats 7] [--vae model_weights/ltx-restore/vae.safetensors] [--clip assets/test_clip1_1080p.mp4]]
"""

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from benchmark_ltx_encode import clip_pixels, encode, variant_encoder
from jasna._suppress_noise import install as install_noise_filters
from jasna.engine_paths import configure_torch_compile_cache
from jasna.models.ltx_vae import conv_triton, load_video_encoder, video_encoder
from jasna.models.ltx_vae.video_encoder import ConvLaunch, EncoderHead

REPO_ROOT = Path(__file__).resolve().parent.parent
WINDOW_FRAMES = 121
GRID = [ConvLaunch(*values) for values in itertools.product((64, 128, 256), (64, 128, 256), (32, 64), (4, 8), (3, 4), (1, 8))]
"""(block voxels, block out, block in, warps, stages, group) tried per shape."""
KITCHEN_HEAD = EncoderHead(6, {})
"""The fp16 head without the Triton conv: kitchen declines ``down_blocks[6]``, so it stays bf16."""
VERIFY_CANVASES = (512, 768)
ACCURACY_TOLERANCE = 0.01
PROMOTED_TRIES = 5
"""The fastest plain launches of the 1024-channel conv that are also timed promoted."""


def head_convs(canvas: int) -> list[tuple[str, int, int, int, int, int]]:
    """(name, in channels, out channels, frames, side, convs per window) of the head's convs after ``conv_in``."""
    side = canvas // 4
    return [
        ("res0", 128, 128, WINDOW_FRAMES, side, 8),
        ("down1", 128, 64, WINDOW_FRAMES, side, 1),
        ("res2", 256, 256, WINDOW_FRAMES, side // 2, 12),
        ("down3", 256, 256, WINDOW_FRAMES + 1, side // 2, 1),
        ("res4", 512, 512, (WINDOW_FRAMES + 1) // 2, side // 2, 8),
        ("down5", 512, 128, (WINDOW_FRAMES + 1) // 2 + 1, side // 2, 1),
        ("res6", 1024, 1024, (WINDOW_FRAMES + 3) // 4, side // 4, 4),
    ]


def milliseconds(fn, repeats: int = 7) -> float:
    for _ in range(2):
        fn()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def causal(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((x[:, :, :1], x[:, :, :1], x), dim=2)


def sweep(canvas: int, layers: set[str] | None) -> tuple[list[dict], dict[tuple[int, int], ConvLaunch], ConvLaunch | None]:
    """Per head conv the timings and best launch, the launches that beat kitchen (or cuDNN), and the
    fastest promoted launch of the 1024-channel conv if it beats cuDNN."""
    torch.manual_seed(0)
    rows, shapes, promoted = [], [], None
    for name, cin, cout, frames, side, count in head_convs(canvas):
        if layers and name not in layers:
            continue
        row, launches, baseline_ms, layer_promoted = time_layer(name, cin, cout, frames, side, count)
        torch.cuda.empty_cache()  # the layer's tensors are gone with time_layer's frame: 12 GB cards fit 768
        rows.append(row)
        shapes.append(((cin, cout), count, baseline_ms, launches))
        promoted = layer_promoted or promoted
    return rows, key_launches(shapes), promoted


def time_layer(
    name: str, cin: int, cout: int, frames: int, side: int, count: int
) -> tuple[dict, dict[ConvLaunch, float], float, ConvLaunch | None]:
    """One head conv's row, ms per launch, ms of the conv it would replace (kitchen, cuDNN bf16 for the
    1024-channel convs), and for those the fastest promoted launch if it beats cuDNN."""
    channels_last = torch.channels_last_3d
    x = (torch.randn(1, cin, frames, side, side, device="cuda") * 0.7).half().contiguous(memory_format=channels_last)
    weight = (torch.randn(cout, cin, 3, 3, 3, device="cuda") / math.sqrt(27 * cin)).half()
    weight = weight.contiguous(memory_format=channels_last)
    bias = (torch.randn(cout, device="cuda") * 0.1).half()

    def kitchen():
        padded = F.pad(causal(x), (1, 1, 1, 1)).contiguous(memory_format=channels_last)
        return torch.ops.comfy_kitchen.fp16_conv3d(padded, weight, bias, None, [1, 1, 1])

    x_bf16, weight_bf16, bias_bf16 = x.bfloat16(), weight.bfloat16(), bias.bfloat16()

    def cudnn():
        return F.conv3d(causal(x_bf16), weight_bf16, bias_bf16, padding=(0, 1, 1))

    def triton(launch: ConvLaunch, repeats: int) -> float:
        return milliseconds(lambda: conv_triton.causal_conv3d(x, weight, bias, None, launch.arguments()), repeats)

    kitchen_ms = milliseconds(kitchen) if cin < 1024 else math.inf
    cudnn_ms = milliseconds(cudnn)
    launches = {}
    for launch in GRID:
        if launch.block_out > cout or cout % launch.block_out or cin % launch.block_in:
            continue
        try:
            launches[launch] = triton(launch, 3)
        except Exception as error:  # a launch that exceeds this GPU's shared memory or registers
            if "OutOfResources" not in type(error).__name__:
                raise
    timed = sorted((ms, launch) for launch, ms in launches.items())
    best = timed[0][1]
    triton_ms = triton(best, 15)
    tflops = 2 * 27 * cin * cout * frames * side * side / triton_ms / 1e9
    row = {"layer": name, "in": cin, "out": cout, "per_window": count, "kitchen_ms": round(kitchen_ms, 3),
           "cudnn_bf16_ms": round(cudnn_ms, 3), "triton_ms": round(triton_ms, 3), "triton_tflops": round(tflops, 1),
           "launch": best.arguments()}
    print(f"  {name:<6} {cin:>4}->{cout:<4} x{count:<2} kitchen {kitchen_ms:7.3f} ms  cuDNN bf16 {cudnn_ms:7.3f} ms  "
          f"triton {triton_ms:7.3f} ms ({tflops:5.1f} TFLOPS) {tuple(best)}", flush=True)
    promoted = None
    if cin == 1024:
        promoted_ms, promoted_launch = min((triton(launch._replace(promoted=True), 7), launch._replace(promoted=True))
                                           for _, launch in timed[:PROMOTED_TRIES])
        row.update(promoted_ms=round(promoted_ms, 3), promoted_launch=promoted_launch.arguments())
        print(f"  {name:<6} promoted {promoted_ms:7.3f} ms {tuple(promoted_launch)}", flush=True)
        if promoted_ms < cudnn_ms:
            promoted = promoted_launch
    return row, launches, cudnn_ms if cin == 1024 else kitchen_ms, promoted


def key_launches(shapes: list[tuple[tuple[int, int], int, float, dict[ConvLaunch, float]]]) -> dict[tuple[int, int], ConvLaunch]:
    """For each (in, out) key of ``shapes`` (key, convs per window, ms of the conv it would replace,
    ms per launch), the launch with the least time over every conv of that key, kept when that beats
    the replaced convs' time: layers sharing a key share its launch."""
    entry = {}
    for key in dict.fromkeys(key for key, _, _, _ in shapes):
        group = [(count, baseline, launches) for shape_key, count, baseline, launches in shapes if shape_key == key]
        common = set.intersection(*(set(launches) for _, _, launches in group))
        totals = {launch: sum(count * launches[launch] for count, _, launches in group) for launch in common}
        best = min(totals, key=totals.get, default=None)
        if best is not None and totals[best] < sum(count * baseline for count, baseline, _ in group):
            entry[key] = best
    return entry


def triton_head(entry: dict[tuple[int, int], ConvLaunch]) -> EncoderHead:
    """The head with ``entry``'s Triton convs: ``down_blocks[6]`` joins it when its 1024-channel convs run on Triton."""
    return EncoderHead(7 if (1024, 1024) in entry else 6, entry)


def shipped_encoder(vae: Path, head: EncoderHead):
    """The encoder as the encode pass loads it with ``head`` in place of this GPU's table entry,
    called on uint8 pixels converted to bf16 like the pass's."""
    architecture = torch.cuda.get_device_capability()
    saved = video_encoder.ENCODER_HEADS.get(architecture)
    video_encoder.ENCODER_HEADS[architecture] = head
    try:
        encoder = load_video_encoder(vae, torch.device("cuda"))
    finally:
        if saved is None:
            del video_encoder.ENCODER_HEADS[architecture]
        else:
            video_encoder.ENCODER_HEADS[architecture] = saved
    return lambda pixels: encoder(pixels.float().div_(127.5).sub_(1.0).to(torch.bfloat16))


def candidates(entry: dict[tuple[int, int], ConvLaunch], promoted: ConvLaunch | None) -> dict[str, EncoderHead]:
    """The kitchen head and the heads to choose from: the whole entry, and without the 1024-channel
    convs (their fp16 sums cost accuracy on some GPUs), and with them on ``promoted``."""
    heads = {"kitchen head": KITCHEN_HEAD, "entry": triton_head(entry)}
    if (1024, 1024) in entry:
        heads["entry without 1024"] = triton_head({key: launch for key, launch in entry.items() if key != (1024, 1024)})
    if promoted is not None:
        heads["entry, 1024 promoted"] = triton_head({**entry, (1024, 1024): promoted})
    return heads


def acceptable(kitchen: dict[int, float], head: dict[int, float]) -> bool:
    """``ACCURACY_TOLERANCE``: distance from fp32 (per canvas) up by at most 1 % at each canvas and not
    at all on average."""
    rises = [head[canvas] / kitchen[canvas] - 1 for canvas in kitchen]
    return max(rises) <= ACCURACY_TOLERANCE and statistics.mean(rises) <= 0


def verify(entry: dict[tuple[int, int], ConvLaunch], promoted: ConvLaunch | None, vae: Path, clip: Path,
           repeats: int) -> dict:
    """Times every candidate head at 512 and 768 on ``clip`` against an fp32 encoder and picks the
    fastest one within the accuracy tolerance (the kitchen head when none is)."""
    heads = candidates(entry, promoted)
    fns = {name: shipped_encoder(vae, head) for name, head in heads.items()}
    seconds = {name: {} for name in heads}
    distance = {name: {} for name in heads}
    for canvas in VERIFY_CANVASES:
        pixels = clip_pixels(clip, canvas, torch.device("cuda"))
        fp32, stack = variant_encoder("fp32", vae, torch.device("cuda"))
        with stack:
            reference, _ = encode(fp32, pixels)
        del fp32
        torch.cuda.empty_cache()
        times = {name: [] for name in fns}
        for name, fn in fns.items():
            latent, _ = encode(fn, pixels)
            distance[name][canvas] = float((latent.float() - reference.float()).norm() / reference.float().norm())
        for _ in range(repeats):
            for name, fn in fns.items():
                times[name].append(encode(fn, pixels)[1])
        for name in fns:
            seconds[name][canvas] = statistics.median(times[name])
            print(f"  {canvas} {name:<20} {seconds[name][canvas]:.4f} s/window  "
                  f"rel L2 vs fp32 {distance[name][canvas]:.3e}", flush=True)
    passing = [name for name in heads if name != "kitchen head" and acceptable(distance["kitchen head"], distance[name])]
    faster = [name for name in passing if all(seconds[name][c] < seconds["kitchen head"][c] for c in VERIFY_CANVASES)]
    chosen = min(faster, key=lambda name: sum(seconds[name].values()), default="kitchen head")
    print(f"chosen: {chosen}")
    print_entry(heads[chosen])
    return {"seconds": seconds, "rel_l2_vs_fp32": distance, "within_tolerance": passing, "chosen": chosen,
            "chosen_entry": {f"{cin},{cout}": launch.arguments() for (cin, cout), launch in heads[chosen].triton_convs.items()},
            "chosen_down_blocks": heads[chosen].down_blocks}


def print_entry(head: EncoderHead) -> None:
    print(f"{torch.cuda.get_device_capability()}: EncoderHead({head.down_blocks}, {{  # {torch.cuda.get_device_name()}")
    for (cin, cout), launch in head.triton_convs.items():
        print(f"    ({cin}, {cout}): ConvLaunch{tuple(launch)},")
    print("}),")


def main() -> None:
    import comfy_kitchen  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--canvas", type=int, choices=(512, 768), default=512)
    parser.add_argument("--layers")
    parser.add_argument("--json", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--vae", type=Path, default=REPO_ROOT / "model_weights/ltx-restore/vae.safetensors")
    parser.add_argument("--clip", type=Path, default=REPO_ROOT / "assets/test_clip1_1080p.mp4")
    args = parser.parse_args()
    install_noise_filters()
    configure_torch_compile_cache()
    print(f"{torch.cuda.get_device_name()} | SM {torch.cuda.get_device_capability()} | canvas {args.canvas}")
    rows, entry, promoted = sweep(args.canvas, set(args.layers.split(",")) if args.layers else None)
    print("fastest launches:")
    print_entry(triton_head(entry))
    result = {"gpu": torch.cuda.get_device_name(), "capability": list(torch.cuda.get_device_capability()),
              "canvas": args.canvas, "convs": rows,
              "entry": {f"{cin},{cout}": launch.arguments() for (cin, cout), launch in entry.items()}}
    if args.verify:
        result["verify"] = verify(entry, promoted, args.vae, args.clip, args.repeats)
    if args.json:
        args.json.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
