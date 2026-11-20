"""Compile TensorRT engines in a subprocess to guarantee full VRAM release; UNet 4x compiles in this process."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import typing
from dataclasses import dataclass, field, replace
from pathlib import Path

from jasna._frozen import is_frozen
from jasna.engine_paths import all_basicvsrpp_sub_engines_exist
from jasna.license_api import require_protection
from jasna.os_utils import get_application_executable, subprocess_no_window_kwargs

logger = logging.getLogger(__name__)



@dataclass
class EngineCompilationRequest:
    device: str
    fp16: bool

    basicvsrpp: bool = False
    basicvsrpp_model_path: str = ""

    detection: bool = False
    detection_model_name: str = ""
    detection_model_path: str = ""
    detection_batch_size: int = 4

    unet4x: bool = False

    upscale_model: str = ""
    upscale_input_sizes: list[tuple[int, int]] = field(default_factory=list)
    """``(width, height)`` of each pass of the ``upscale_model`` network that needs an engine."""

    def to_json(self) -> str:
        return json.dumps(self.__dict__)

    @staticmethod
    def from_json(s: str) -> EngineCompilationRequest:
        fields = json.loads(s)
        fields["upscale_input_sizes"] = [tuple(size) for size in fields.get("upscale_input_sizes", [])]
        return EngineCompilationRequest(**fields)


@dataclass
class EngineCompilationResult:
    use_basicvsrpp_tensorrt: bool = False


def _detection_engine_exists(
    detection_model_name: str,
    detection_model_path: str,
    batch_size: int,
    fp16: bool,
    device: str,
) -> bool:
    import torch

    from jasna.accelerator import is_amd_device
    from jasna.mosaic.detection_registry import detection_engine_path

    if is_amd_device(torch.device(device)):
        # AMD runs both RF-DETR (rfdetr torch model) and YOLO through PyTorch;
        # there is no compiled engine artifact to check for.
        return True
    return detection_engine_path(detection_model_name, Path(detection_model_path), batch_size, fp16=fp16).exists()


def _unet4x_engine_exists(fp16: bool) -> bool:
    from jasna.engine_paths import (
        expected_unet4x_engine_path,
        get_unet4x_encrypted_engine_path,
        unet4x_plaintext_available,
    )

    if unet4x_plaintext_available():
        return expected_unet4x_engine_path(fp16=fp16).exists()

    engine_path = get_unet4x_encrypted_engine_path(fp16=fp16)
    if not engine_path.exists():
        return False

    require_protection()
    from jasna.restorer.unet4x_secondary_restorer import decrypt_unet4x_engine
    return decrypt_unet4x_engine(engine_path) is not None


class _MissingEngines(typing.NamedTuple):
    basicvsrpp: bool
    detection: bool
    unet4x: bool
    upscale_input_sizes: tuple[tuple[int, int], ...] = ()


def _missing_engines(req: EngineCompilationRequest, *, nvidia: bool) -> _MissingEngines:
    return _MissingEngines(
        basicvsrpp=nvidia and req.basicvsrpp and req.fp16
        and not all_basicvsrpp_sub_engines_exist(req.basicvsrpp_model_path, req.fp16),
        detection=req.detection and not _detection_engine_exists(
            req.detection_model_name,
            req.detection_model_path,
            req.detection_batch_size,
            req.fp16,
            req.device,
        ),
        unet4x=nvidia and req.unet4x and not _unet4x_engine_exists(req.fp16),
        upscale_input_sizes=_missing_upscale_engines(req) if nvidia and req.upscale_model else (),
    )


def _missing_upscale_engines(req: EngineCompilationRequest) -> tuple[tuple[int, int], ...]:
    from jasna.upscale.network import UPSCALE_NETWORKS, missing_network_engines

    network = UPSCALE_NETWORKS[req.upscale_model]
    return missing_network_engines(network, tuple(req.upscale_input_sizes), fp16=req.fp16)


def _adopt_legacy_engines(req: EngineCompilationRequest, device) -> None:
    """Keep the engines Jasna 0.12.3 and older named without their GPU architecture. Only on a single-GPU PC:
    there loading an engine proves it was built for this GPU, while with more GPUs torch_tensorrt
    loads it on another GPU it matches."""
    import torch

    from jasna.engine_paths import (
        adopt_legacy_engine,
        expected_unet4x_engine_path,
        get_basicvsrpp_sub_engine_paths,
        unet4x_plaintext_available,
    )

    if torch.cuda.device_count() != 1:
        return
    if req.basicvsrpp and req.fp16:
        for path in get_basicvsrpp_sub_engine_paths(req.basicvsrpp_model_path, req.fp16).values():
            adopt_legacy_engine(path, lambda legacy: _torchtrt_export_runs_here(legacy, device))
    if req.detection:
        from jasna.mosaic.detection_registry import detection_engine_path

        engine = detection_engine_path(
            req.detection_model_name, Path(req.detection_model_path), req.detection_batch_size, fp16=req.fp16
        )
        adopt_legacy_engine(engine, _engine_file_runs_here)
    if req.unet4x:
        runs_here = _engine_file_runs_here if unet4x_plaintext_available() else _encrypted_unet4x_engine_runs_here
        adopt_legacy_engine(expected_unet4x_engine_path(fp16=req.fp16), runs_here)
    if req.upscale_model:
        from jasna.upscale.network import UPSCALE_NETWORKS, network_engine_path

        for size in req.upscale_input_sizes:
            engine = network_engine_path(UPSCALE_NETWORKS[req.upscale_model], size, fp16=req.fp16)
            adopt_legacy_engine(engine, _engine_file_runs_here)


def _engine_file_runs_here(path: Path) -> bool:
    from jasna.trt import engine_runs_here

    return engine_runs_here(path.read_bytes())


def _encrypted_unet4x_engine_runs_here(path: Path) -> bool:
    require_protection()
    from jasna.restorer.unet4x_secondary_restorer import decrypt_unet4x_engine
    from jasna.trt import engine_runs_here

    engine_bytes = decrypt_unet4x_engine(path)
    return engine_bytes is not None and engine_runs_here(engine_bytes)


def _torchtrt_export_runs_here(path: Path, device) -> bool:
    from jasna.engine_paths import basicvsrpp_sub_engine_exists
    from jasna.trt.torch_tensorrt_export import load_torchtrt_export

    if not basicvsrpp_sub_engine_exists(str(path)):
        return False
    try:
        load_torchtrt_export(checkpoint_path=str(path), device=device)
    except RuntimeError:
        logger.debug("Legacy engine %s does not load on this GPU", path, exc_info=True)
        return False
    return True


def ensure_engines_compiled(
    req: EngineCompilationRequest,
    log_callback: typing.Callable[[str], None] | None = None,
) -> EngineCompilationResult:
    import torch

    from jasna.accelerator import is_nvidia_device

    result = EngineCompilationResult()
    device = torch.device(req.device)
    nvidia = is_nvidia_device(device)

    if nvidia:
        _adopt_legacy_engines(req, device)
    missing = _missing_engines(req, nvidia=nvidia)
    if missing.unet4x:
        from jasna.engine_paths import unet4x_plaintext_available
        if not unet4x_plaintext_available():
            require_protection()
            from jasna.protection import license_store

            if not license_store.is_licensed():
                raise RuntimeError("unet-4x is a supporter feature. Enter your license to enable it.")

    if req.basicvsrpp and nvidia:
        if not req.fp16:
            result.use_basicvsrpp_tensorrt = False
        elif not missing.basicvsrpp:
            result.use_basicvsrpp_tensorrt = True

    if not any(missing):
        return result

    _show("Compiling TensorRT engines (this may take several minutes)...", log_callback)
    if any(missing._replace(unet4x=False)):
        _run_compiler_subprocess(replace(req, unet4x=False), log_callback)
        if req.basicvsrpp and nvidia:
            result.use_basicvsrpp_tensorrt = all_basicvsrpp_sub_engines_exist(
                req.basicvsrpp_model_path, req.fp16
            )

    if missing.unet4x:
        # In this process: a compiler subprocess would need its own license check, which online activation counts
        # as another device.
        from jasna.restorer.unet4x_secondary_restorer import compile_unet4x_engine
        _show("Compiling Unet4x engine...", log_callback)
        compile_unet4x_engine(device, fp16=req.fp16)
        _show("Unet4x engine compiled.", log_callback)

    return result


def _show(message: str, log_callback: typing.Callable[[str], None] | None) -> None:
    """The frozen GUI drops its console (FreeConsole), leaving stdout invalid — an unconditional print() there
    raises WinError 6. Print only on the CLI (no callback)."""
    if log_callback:
        log_callback(message)
    else:
        print(message)


def _compiler_command(req: EngineCompilationRequest, frozen: bool) -> list[str]:
    """A frozen build runs its own binary, a source run this module."""
    if frozen:
        return [get_application_executable(), "--compile-engines", req.to_json()]
    return [sys.executable, "-m", "jasna.engine_compiler", req.to_json()]


def _run_compiler_subprocess(
    req: EngineCompilationRequest,
    log_callback: typing.Callable[[str], None] | None,
) -> None:
    logger.info("Spawning GPU model compilation subprocess...")
    frozen = is_frozen()
    cmd = _compiler_command(req, frozen)

    kwargs: dict = {
        "stdin": subprocess.DEVNULL,  # don't inherit the GUI's detached (invalid) stdin
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
    }
    if frozen:
        from jasna.protection.runtime import subprocess_environment
        kwargs["env"] = subprocess_environment()
    proc = subprocess.Popen(cmd, **kwargs, **subprocess_no_window_kwargs())
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n\r")
        if line:
            _show(line, log_callback)
            logger.debug("[compiler] %s", line)
    returncode = proc.wait()

    if returncode != 0:
        raise RuntimeError(f"Engine compilation subprocess failed (exit code {returncode})")


def compile_missing_engines(req: EngineCompilationRequest) -> None:
    import logging as _logging
    import warnings
    warnings.filterwarnings("ignore")
    _logging.disable(_logging.WARNING)

    # Before torch_tensorrt (-> torch._inductor) imports: the frozen binary's source-introspection
    # shims must be in place.
    from jasna.bootstrap import configure_process
    configure_process()
    import torch
    from jasna.accelerator import is_nvidia_device, set_device

    device = torch.device(req.device)
    set_device(device)  # engine names carry the current device's GPU architecture
    missing = _missing_engines(req, nvidia=is_nvidia_device(device))

    if missing.basicvsrpp:
        from jasna.restorer.basicvsrpp_sub_engines import compile_basicvsrpp_engines
        print("Compiling BasicVSR++ sub-engines...")
        compile_basicvsrpp_engines(req.basicvsrpp_model_path, device, req.fp16)
        print("BasicVSR++ sub-engines compiled.")

    if missing.detection:
        from jasna.mosaic.detection_registry import precompile_detection_engine
        print(f"Compiling detection engine ({req.detection_model_name})...")
        precompile_detection_engine(
            detection_model_name=req.detection_model_name,
            detection_model_path=Path(req.detection_model_path),
            batch_size=req.detection_batch_size,
            device=device,
            fp16=req.fp16,
        )
        print("Detection engine compiled.")

    for width, height in missing.upscale_input_sizes:
        from jasna.upscale.network import UPSCALE_NETWORKS, compile_network_engine
        print(f"Compiling {req.upscale_model} upscale engine for {width}x{height}...")
        compile_network_engine(UPSCALE_NETWORKS[req.upscale_model], (width, height), device, fp16=req.fp16)
        print("Upscale engine compiled.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m jasna.engine_compiler <json_request>", file=sys.stderr)
        sys.exit(1)
    req = EngineCompilationRequest.from_json(sys.argv[1])
    compile_missing_engines(req)
