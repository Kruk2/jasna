"""Choose supported AMD adapters before importing PyTorch."""
from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import subprocess
import sys

from jasna._frozen import is_frozen

log = logging.getLogger(__name__)

TRITON_HIP_ARCHITECTURES = ("gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1200", "gfx1201")
"""The architectures of every prebuilt Triton kernel, and so the AMD GPUs LTX runs on."""

INTEGRATED_HIP_ARCHITECTURES = frozenset({"gfx1033", "gfx1035", "gfx1036", "gfx1103", "gfx1150", "gfx1151", "gfx1152", "gfx1153"})
"""AMD integrated GPUs; a supported discrete GPU in the same PC is used instead."""


def kernel_pack_architectures(torch_dir: Path) -> frozenset[str] | None:
    """The gfx targets whose PyTorch device pack is installed, or None for a PyTorch without device packs.

    AMD's ROCm wheels report every compiled target in ``torch.cuda.get_arch_list()`` but ship each
    target's kernels in its own ``torch/.kpack`` pack; without it the first kernel launch fails with
    hipErrorInvalidKernelFile."""
    packs = frozenset(path.stem.removeprefix("torch_") for path in (torch_dir / ".kpack").glob("torch_gfx*.kpack"))
    return packs or None


def configure_rocm_devices() -> None:
    """Keep unsupported adapters out of the bundled HIP runtime; explicit selections win."""
    if sys.platform not in ("linux", "win32") or not is_frozen():
        return
    if any(name in os.environ for name in (
        "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
    )):
        return
    if any(arg == "--device" or arg.startswith("--device=") for arg in sys.argv[1:]):
        return
    # The LTX worker inherits the app's adapter selection.
    if sys.argv[1:2] == ["--ltx-worker"]:
        return

    from jasna.os_utils import get_application_executable

    root = Path(get_application_executable()).resolve().parent
    windows = sys.platform == "win32"
    relative = "lib/llvm/bin/offload-arch.exe" if windows else "bin/rocminfo"
    tool = root / "_rocm_sdk_core" / relative
    if not tool.is_file():
        return
    try:
        result = subprocess.run(
            [str(tool), "--only=amdgpu"] if windows else [str(tool)],
            capture_output=True, text=True, check=not windows, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if windows else 0,
        )
    except (OSError, subprocess.SubprocessError):
        log.warning("Could not select supported AMD GPUs before startup", exc_info=True)
        return

    if windows:
        architectures = result.stdout.splitlines()
        if not architectures or any(not re.fullmatch(r"gfx[0-9a-f]+(?::[^\s]+)*", arch) for arch in architectures):
            log.warning("Could not read AMD GPU architectures: %r", result.stdout)
            return
        devices = [(arch, str(i)) for i, arch in enumerate(architectures)]
        # offload-arch stops with an error at the first adapter it can't read; the ones it listed work.
        device_count = len(devices) + (result.returncode != 0)
    else:
        name = r"^[ \t]*Name:[ \t]+(gfx[0-9a-f]+(?::[^\s]+)*)[ \t]*$"
        architectures = re.findall(name, result.stdout, re.MULTILINE)
        devices = re.findall(
            name + r"\n[ \t]*Uuid:[ \t]+(GPU-[0-9a-fA-F]{16})[ \t]*$",
            result.stdout, re.MULTILINE,
        )
        if not devices or len(devices) != len(architectures) or len({uuid for _, uuid in devices}) != len(devices):
            log.warning("Could not read unique AMD GPU UUIDs from rocminfo")
            return
        device_count = len(devices)
    shipped = kernel_pack_architectures(root / "torch") or frozenset()
    supported = [(arch.split(":")[0], identifier) for arch, identifier in devices if arch.split(":")[0] in shipped]
    # comfy-kitchen's matrix-core kernels, which LTX needs, load only when every visible adapter has them.
    def rank(arch: str) -> tuple[bool, bool]:
        return arch in INTEGRATED_HIP_ARCHITECTURES, arch not in TRITON_HIP_ARCHITECTURES

    best = min((rank(arch) for arch, _ in supported), default=None)
    chosen = [identifier for arch, identifier in supported if rank(arch) == best]
    if chosen and len(chosen) < device_count:
        # Filter before torch import: HIP may load kernels for every visible adapter.
        os.environ["HIP_VISIBLE_DEVICES" if windows else "ROCR_VISIBLE_DEVICES"] = ",".join(chosen)
