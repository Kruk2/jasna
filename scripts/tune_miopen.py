"""Tune MIOpen exhaustively for the LTX VAE encoder and the whole-video upscalers on this AMD GPU and
store the results in Jasna.

MIOpen keys its tuning by conv shape and memory layout, so the encoder runs as both kernel paths do in
production: the prebuilt path of the releases (eager, NCDHW tail) and the Triton path of Linux source
installs (compiled, channels-last). Every encoder size (full, half, max) runs at both canvases (512, 768),
with placeholder weights: tuning depends on shapes only. The upscalers (SPAN, RealPLKSR) run every
input size their passes see for common sources (480p to 1080p) at each output height. Each run happens in
its own process with an empty MIOpen user database and ``MIOPEN_FIND_ENFORCE=SEARCH``; the solvers it
settles on are merged into ``jasna/media/miopen/<arch>.HIP.fdb.txt`` and ``<arch>.db.txt``, so a user's
first job skips MIOpen's own search. Takes 30-60 minutes.

``--runs prebuilt upscale`` skips the Triton encoder path (Windows needs a Visual Studio build
environment for it). ``--install`` instead merges the committed results into the MIOpen databases of the
installed ROCm (``jasna.miopen_tuning.install``), as release builds do.

Usage (an otherwise idle AMD GPU):
    python scripts/tune_miopen.py [--runs prebuilt upscale]
    python scripts/tune_miopen.py --install
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from jasna import accelerator
from jasna.miopen_tuning import DATABASE_SUFFIXES, TUNED_DIR, install, merged, read_entries

CANVASES = (512, 768)
ENCODERS = ("full", "half", "max")
BACKENDS = ("prebuilt", "triton")
RUNS = (*BACKENDS, "upscale")
UPSCALE_SOURCES = ((1920, 1080), (1280, 720), (960, 540), (854, 480))
USER_DATABASE_SUFFIXES = {".HIP.fdb.txt": ".ufdb.txt", ".db.txt": ".udb.txt"}


def run_encoders(backend: str) -> None:
    """One encode window per encoder size and canvas through the production loader and input."""
    if backend == "prebuilt":
        accelerator.has_triton = lambda: False
    accelerator.configure_rocm_process_env()
    from jasna.engine_paths import configure_torch_compile_cache
    from jasna.ltx.model_files import LtxModelFiles
    from jasna.ltx.plan import WINDOW_FRAMES
    from jasna.ltx.restore import _vae_pixels
    from jasna.models.ltx_vae import load_video_encoder

    configure_torch_compile_cache()
    device = torch.device("cuda")
    print(f"{torch.cuda.get_device_name(device)}: {accelerator.kernel_backend(device)}", flush=True)
    for name in ENCODERS:
        files = LtxModelFiles.placeholder("distilled2", fast=False, encoder=name)
        encoder = load_video_encoder(files.vae, device, files.encoder)
        for canvas in CANVASES:
            pixels = torch.randint(0, 256, (3, WINDOW_FRAMES, canvas, canvas), dtype=torch.uint8)
            with torch.inference_mode():
                encoder(_vae_pixels(pixels, device))
            torch.cuda.synchronize()
            print(f"  {name} encoder at {canvas}: tuned", flush=True)
        del encoder
        torch.cuda.empty_cache()


def upscale_input_sizes(network) -> list[tuple[int, int]]:
    """Every pass input size of ``network`` for ``UPSCALE_SOURCES`` at each output height."""
    from jasna.session_config import UPSCALE_HEIGHTS
    from jasna.upscale import upscaled_size
    from jasna.upscale.network import network_input_sizes

    sizes = []
    for source in UPSCALE_SOURCES:
        for height in UPSCALE_HEIGHTS:
            output = upscaled_size(*source, height)
            if output is not None:
                sizes += [size for size in network_input_sizes(network, source, output) if size not in sizes]
    return sizes


def run_upscalers() -> None:
    """One fp16 frame per pass input size through each upscale network, as AMD runs them."""
    accelerator.configure_rocm_process_env()
    from jasna.upscale.network import UPSCALE_NETWORKS

    device = torch.device("cuda")
    for name, network in UPSCALE_NETWORKS.items():
        module = network.network().to(device=device, dtype=torch.float16)
        for width, height in upscale_input_sizes(network):
            with torch.inference_mode():
                module(torch.rand(1, 3, height, width, device=device, dtype=torch.float16))
            torch.cuda.synchronize()
            print(f"  {name} at {width}x{height}: tuned", flush=True)
        del module
        torch.cuda.empty_cache()


def tuned_entries(user_db: Path, user_suffix: str) -> dict[str, str]:
    entries = {}
    for path in sorted(user_db.glob(f"*{user_suffix}")):
        entries |= read_entries(path.read_text(encoding="utf-8"))
    return entries


def tune(runs: list[str]) -> None:
    architecture = accelerator.amd_gpu_architecture(torch.device("cuda"))
    collected = {suffix: {} for suffix in DATABASE_SUFFIXES}
    for run in runs:
        with tempfile.TemporaryDirectory(prefix="miopen-tune-") as user_db:
            environment = {
                **os.environ, "MIOPEN_FIND_ENFORCE": "SEARCH", "MIOPEN_FIND_MODE": "NORMAL",
                "MIOPEN_USER_DB_PATH": user_db, "MIOPEN_CUSTOM_CACHE_DIR": user_db,
            }
            subprocess.run([sys.executable, __file__, "--run", run], env=environment, check=True)
            for suffix, user_suffix in USER_DATABASE_SUFFIXES.items():
                collected[suffix] |= tuned_entries(Path(user_db), user_suffix)
    TUNED_DIR.mkdir(parents=True, exist_ok=True)
    for suffix, entries in collected.items():
        target = TUNED_DIR / f"{architecture}{suffix}"
        text = target.read_text(encoding="utf-8") if target.is_file() else ""
        target.write_text(merged(text, entries), encoding="utf-8", newline="\n")
        print(f"{target.relative_to(TUNED_DIR.parents[2])}: {len(entries)} tuned convs")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", choices=RUNS, help=argparse.SUPPRESS)
    parser.add_argument("--runs", nargs="+", choices=RUNS, default=list(RUNS), help="encoder paths and upscalers to tune")
    parser.add_argument("--install", action="store_true", help="merge the committed results into the installed ROCm")
    args = parser.parse_args()
    if args.run == "upscale":
        run_upscalers()
    elif args.run:
        run_encoders(args.run)
    elif args.install:
        for path in install(accelerator.rocm_libraries_dir() / "share" / "miopen" / "db"):
            print(path)
    else:
        tune(args.runs)


if __name__ == "__main__":
    main()
