"""Explicit source-runtime launch preparation for Windows GUI video workers."""
from dataclasses import dataclass
from pathlib import Path
import os
import re
import sys

from jasna._frozen import is_frozen
from jasna.runtime_contract import build_runtime_environment


@dataclass(frozen=True)
class WindowsUnifiedWorkerRuntime:
    repo_root: Path
    runtime_root: Path
    python_executable: Path

    def __post_init__(self):
        for path in (self.repo_root, self.runtime_root, self.python_executable):
            if not isinstance(path, Path) or not path.is_absolute():
                raise ValueError("worker runtime paths must be explicit absolute Path objects")

    def prepare(self, request_path: Path, base_environment: dict[str, str]):
        """Keep DLL activation and loaded-ABI validation in the existing launcher.

        The GUI parent validates the runtime layout without loading native media;
        the child revalidates the actual loaded DLLs while keeping their handles.
        Frozen distributions need a separately accepted packaging entry point.
        """
        if sys.platform != "win32" or is_frozen():
            raise RuntimeError("this unified worker launcher requires Windows source mode")
        if not isinstance(request_path, Path) or not request_path.is_absolute() or not request_path.is_file():
            raise ValueError("an existing absolute worker request is required")
        launcher = self.repo_root / "scripts/run_jasna_unified.py"
        if not launcher.is_file() or not self.python_executable.is_file():
            raise ValueError("selected worker Python or unified launcher is missing")
        environment = build_runtime_environment(self.runtime_root, self.repo_root,
            python_executable=self.python_executable, platform="win32", base_environment=base_environment)
        environment.pop("JASNA_MAIN_PID", None)
        environment["PYTHONIOENCODING"] = "utf-8"
        environment["JASNA_WINDOWS_WORKER_GPU_IDENTITY"] = "1"
        command = [str(self.python_executable), "-B", str(launcher), "--_product-child",
            "--runtime-root", str(self.runtime_root), "--repo-root", str(self.repo_root),
            "--", "--isolated-video-job", str(request_path)]
        return command, environment


def report_windows_worker_gpu_identity(output_stream):
    """Report the source GUI worker's actual cuda:0 identity before processing.

    Only the guarded launcher opts into this. HIP initialization happens in
    the disposable worker, never in its parent's recovery sampling path.
    """
    if os.environ.get("JASNA_WINDOWS_WORKER_GPU_IDENTITY") != "1":
        return
    token = os.environ.get("JASNA_WINDOWS_WORKER_ATTEMPT_TOKEN", "")
    if sys.platform != "win32" or re.fullmatch(r"[0-9a-f]{32}", token) is None:
        raise RuntimeError("Windows worker identity requires its guarded attempt token")
    import torch
    if not getattr(torch.version, "hip", None):
        raise RuntimeError("Windows worker identity requires the selected AMD HIP runtime")
    from jasna.media.hip_kernel import hip_runtime
    from jasna.windows_global_vram import WindowsGlobalVramReader
    from jasna.gui.video_job_process import _emit_event
    # video_session_config selects cuda:0. Do not substitute DXGI's largest
    # card without proving it matches that actual HIP device.
    reader = WindowsGlobalVramReader(0, hip_runtime())
    try:
        identity = reader.identity
    finally:
        reader.close()
    _emit_event(output_stream, dict(type="windows_gpu_identity", attempt_token=token,
        adapter_marker=identity.adapter_marker, node_index=identity.node_index))
