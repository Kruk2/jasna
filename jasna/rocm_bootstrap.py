"""Choose supported Windows AMD adapters before importing PyTorch."""
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
"""The AMD release targets and architectures of every prebuilt Triton kernel."""


def configure_windows_rocm_devices() -> None:
    """Keep unsupported adapters out of the bundled HIP runtime; explicit selections win."""
    if sys.platform != "win32" or not is_frozen():
        return
    if any(name in os.environ for name in (
        "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL",
    )):
        return
    if any(arg == "--device" or arg.startswith("--device=") for arg in sys.argv[1:]):
        return

    root = Path(sys.executable).resolve().parent
    tool = root / "_rocm_sdk_core" / "lib" / "llvm" / "bin" / "offload-arch.exe"
    if not tool.is_file():
        return
    try:
        result = subprocess.run(
            [str(tool), "--only=amdgpu"],
            capture_output=True, text=True, check=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        log.warning("Could not select supported AMD GPUs before startup", exc_info=True)
        return

    architectures = result.stdout.splitlines()
    if not architectures or any(not re.fullmatch(r"gfx[0-9a-f]+(?::[^\s]+)*", arch) for arch in architectures):
        log.warning("Could not read AMD GPU architectures: %r", result.stdout)
        return
    supported = [str(i) for i, arch in enumerate(architectures)
                 if arch.split(":")[0] in TRITON_HIP_ARCHITECTURES]
    if supported and len(supported) < len(architectures):
        # Filter before torch import: HIP may load kernels for every visible adapter.
        os.environ["HIP_VISIBLE_DEVICES"] = ",".join(supported)
