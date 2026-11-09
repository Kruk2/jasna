"""Tune MIOpen exhaustively for the LTX VAE encoder on this AMD GPU and store the results in Jasna.

MIOpen keys its tuning by conv shape and memory layout, so the encoder runs as both kernel paths do in
production: the prebuilt path of the releases (eager, NCDHW tail) and the Triton path of Linux source
installs (compiled, channels-last). Every encoder size (full, half, max) runs at both canvases (512, 768),
with placeholder weights: tuning depends on shapes only. Each path runs in its own process with an empty
MIOpen user database and ``MIOPEN_FIND_ENFORCE=SEARCH``; the solvers it settles on are merged into
``jasna/media/miopen/<arch>.HIP.fdb.txt`` and ``<arch>.db.txt``. Takes 30-60 minutes.

``--backends prebuilt`` tunes only the release path (Windows needs a Visual Studio build environment for
the Triton one). ``--install`` instead merges the committed results into the MIOpen databases of the
installed ROCm (``jasna.miopen_tuning.install``), as release builds do.

Usage (an otherwise idle AMD GPU):
    python scripts/tune_miopen.py [--backends prebuilt]
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


def tuned_entries(user_db: Path, user_suffix: str) -> dict[str, str]:
    entries = {}
    for path in sorted(user_db.glob(f"*{user_suffix}")):
        entries |= read_entries(path.read_text(encoding="utf-8"))
    return entries


def tune(backends: list[str]) -> None:
    architecture = accelerator.amd_gpu_architecture(torch.device("cuda"))
    collected = {suffix: {} for suffix in DATABASE_SUFFIXES}
    for backend in backends:
        with tempfile.TemporaryDirectory(prefix="miopen-tune-") as user_db:
            environment = {
                **os.environ, "MIOPEN_FIND_ENFORCE": "SEARCH", "MIOPEN_FIND_MODE": "NORMAL",
                "MIOPEN_USER_DB_PATH": user_db, "MIOPEN_CUSTOM_CACHE_DIR": user_db,
            }
            subprocess.run([sys.executable, __file__, "--backend", backend], env=environment, check=True)
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
    parser.add_argument("--backend", choices=BACKENDS, help=argparse.SUPPRESS)
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=list(BACKENDS), help="kernel paths to tune")
    parser.add_argument("--install", action="store_true", help="merge the committed results into the installed ROCm")
    args = parser.parse_args()
    if args.backend:
        run_encoders(args.backend)
    elif args.install:
        for path in install(accelerator.rocm_libraries_dir() / "share" / "miopen" / "db"):
            print(path)
    else:
        tune(args.backends)


if __name__ == "__main__":
    main()
