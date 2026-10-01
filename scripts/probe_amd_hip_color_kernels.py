#!/usr/bin/env python3
"""Isolated correctness/performance probe for AMD fused colour kernels.

This script deliberately does not alter product routing.  It compiles the
existing, FFmpeg-oracle-tested CUDA colour kernels as a gfx1100 HIP code object,
loads them through the HIP module API, and compares them with the current ROCm
eager Torch implementations before reporting timings.

Run only on an otherwise idle AMD GPU::

    python scripts/probe_amd_hip_color_kernels.py --height 4096 --width 8192
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import torch
from av.video.reformatter import Colorspace as AvColorspace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jasna.accelerator import is_amd_device
from jasna.media.hip_kernel import (
    AMD_HIP_COLOR_KERNELS_ENV,
    hip_runtime,
    hip_runtime_identity,
)
from jasna.media.rgb_to_yuv import RgbToYuvConverter
from jasna.media.yuv_to_rgb import YuvToRgbConverter


MEDIA = ROOT / "jasna" / "media"
VARIANTS = (
    "bt601_limited",
    "bt601_full",
    "bt709_limited",
    "bt709_full",
    "bt2020_limited",
    "bt2020_full",
)
COLOR_SPACES = {
    "bt601": AvColorspace.ITU601,
    "bt709": AvColorspace.ITU709,
    "bt2020": AvColorspace.BT2020,
}


@contextmanager
def _force_eager_reference():
    """Keep the oracle on Torch after the product route became automatic."""

    previous = os.environ.get(AMD_HIP_COLOR_KERNELS_ENV)
    os.environ[AMD_HIP_COLOR_KERNELS_ENV] = "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(AMD_HIP_COLOR_KERNELS_ENV, None)
        else:
            os.environ[AMD_HIP_COLOR_KERNELS_ENV] = previous


def _check_even(height: int, width: int) -> None:
    if height <= 0 or width <= 0 or height % 2 or width % 2:
        raise ValueError(f"expected positive even dimensions, got {height}x{width}")


def _compile(
    source: Path, destination: Path, architecture: str, hipcc: str
) -> list[str]:
    compiler = Path(hipcc).resolve()
    sdk_root = compiler.parent.parent
    windows_arguments: list[str] = []
    environment = None
    if sys.platform == "win32":
        device_libraries = sdk_root / "lib" / "llvm" / "amdgcn" / "bitcode"
        if not (device_libraries / "ocml.bc").is_file():
            raise RuntimeError(
                f"ROCm device libraries are missing beside {compiler}"
            )
        windows_arguments = [
            "--no-gpu-bundle-output",
            "-fuse-cuid=none",
            f"--rocm-device-lib-path={device_libraries}",
        ]
        environment = dict(os.environ)
        environment["HIP_PATH"] = str(sdk_root)
        environment["ROCM_PATH"] = str(sdk_root)
    command = [
        str(compiler),
        "--genco",
        *windows_arguments,
        f"--offload-arch={architecture}",
        "-O3",
        "-std=c++17",
        "-include",
        "hip/hip_runtime.h",
        str(source),
        "-o",
        str(destination),
    ]
    subprocess.run(
        command,
        check=True,
        env=environment,
    )
    return command


class HipModule:
    def __init__(self, path: Path) -> None:
        self._lib = hip_runtime()
        self._configure()
        self._module = ctypes.c_void_p()
        self._check(
            self._lib.hipModuleLoad(ctypes.byref(self._module), str(path).encode()),
            f"hipModuleLoad({path.name})",
        )
        self._functions: dict[str, ctypes.c_void_p] = {}

    def _configure(self) -> None:
        lib = self._lib
        lib.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
        lib.hipModuleLoad.restype = ctypes.c_int
        lib.hipModuleUnload.argtypes = [ctypes.c_void_p]
        lib.hipModuleUnload.restype = ctypes.c_int
        lib.hipModuleGetFunction.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_char_p,
        ]
        lib.hipModuleGetFunction.restype = ctypes.c_int
        lib.hipModuleLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        lib.hipModuleLaunchKernel.restype = ctypes.c_int
        lib.hipGetErrorName.argtypes = [ctypes.c_int]
        lib.hipGetErrorName.restype = ctypes.c_char_p
        lib.hipGetErrorString.argtypes = [ctypes.c_int]
        lib.hipGetErrorString.restype = ctypes.c_char_p

    def _check(self, result: int, operation: str) -> None:
        if result == 0:
            return
        name = self._lib.hipGetErrorName(result)
        message = self._lib.hipGetErrorString(result)
        raise RuntimeError(
            f"{operation} failed: "
            f"{name.decode(errors='replace') if name else result}: "
            f"{message.decode(errors='replace') if message else 'unknown error'}"
        )

    def function(self, name: str) -> ctypes.c_void_p:
        cached = self._functions.get(name)
        if cached is not None:
            return cached
        function = ctypes.c_void_p()
        self._check(
            self._lib.hipModuleGetFunction(
                ctypes.byref(function), self._module, name.encode("ascii")
            ),
            f"hipModuleGetFunction({name})",
        )
        self._functions[name] = function
        return function

    def launch(
        self,
        name: str,
        grid: tuple[int, int, int],
        block: tuple[int, int, int],
        values: list[ctypes._SimpleCData],
    ) -> None:
        params = (ctypes.c_void_p * len(values))(
            *(ctypes.cast(ctypes.byref(value), ctypes.c_void_p) for value in values)
        )
        stream = int(torch.cuda.current_stream().cuda_stream)
        self._check(
            self._lib.hipModuleLaunchKernel(
                self.function(name),
                *grid,
                *block,
                0,
                ctypes.c_void_p(stream),
                params,
                None,
            ),
            f"hipModuleLaunchKernel({name})",
        )

    def close(self) -> None:
        if self._module.value:
            self._check(self._lib.hipModuleUnload(self._module), "hipModuleUnload")
            self._module = ctypes.c_void_p()
            self._functions.clear()


class HipYuvToRgb:
    def __init__(self, module: HipModule, name: str) -> None:
        self.module = module
        self.name = name

    def __call__(self, packed: torch.Tensor, out: torch.Tensor) -> None:
        batch, packed_height, width = packed.shape
        height = out.shape[-2]
        if packed_height != height + height // 2:
            raise ValueError("packed YUV and RGB heights do not match")
        if out.shape != (batch, 3, height, width):
            raise ValueError("unexpected RGB output shape")
        y_ptrs = [packed[i, :height].data_ptr() for i in range(batch)]
        uv_ptrs = [packed[i, height:].data_ptr() for i in range(batch)]
        if not 1 <= batch <= 8:
            raise ValueError("YUV probe kernel supports batches 1 through 8")
        values: list[ctypes._SimpleCData] = []
        values.extend(ctypes.c_uint64(y_ptrs[i] if i < batch else 0) for i in range(8))
        values.extend(ctypes.c_uint64(uv_ptrs[i] if i < batch else 0) for i in range(8))
        values.extend(
            [
                ctypes.c_int(packed.stride(1)),
                ctypes.c_int(packed.stride(1)),
                ctypes.c_uint64(out.data_ptr()),
                ctypes.c_int64(out.stride(0)),
                ctypes.c_int64(out.stride(1)),
                ctypes.c_int64(out.stride(2)),
                ctypes.c_int(batch),
                ctypes.c_int(height),
                ctypes.c_int(width),
            ]
        )
        threads = 256
        pixels = batch * height * width
        self.module.launch(
            self.name,
            ((pixels + threads - 1) // threads, 1, 1),
            (threads, 1, 1),
            values,
        )


class HipRgbToYuv:
    def __init__(self, module: HipModule, name: str) -> None:
        self.module = module
        self.name = name

    def __call__(self, rgb: torch.Tensor, packed: torch.Tensor) -> None:
        _, height, width = rgb.shape
        if packed.shape != (height + height // 2, width):
            raise ValueError("unexpected packed YUV output shape")
        values: list[ctypes._SimpleCData] = [
            ctypes.c_uint64(rgb.data_ptr()),
            ctypes.c_int64(rgb.stride(0)),
            ctypes.c_int64(rgb.stride(1)),
            ctypes.c_uint64(packed[:height].data_ptr()),
            ctypes.c_int64(packed.stride(0)),
            ctypes.c_uint64(packed[height:].data_ptr()),
            ctypes.c_int64(packed.stride(0)),
            ctypes.c_int(height),
            ctypes.c_int(width),
        ]
        quads_x = (width + 1) // 2
        quads_y = (height + 1) // 2
        self.module.launch(
            self.name,
            ((quads_x + 15) // 16, (quads_y + 15) // 16, 1),
            (16, 16, 1),
            values,
        )


def _event_ms(fn: Callable[[], None], *, warmup: int, rounds: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rounds):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / rounds


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> int:
    # The production P010 converter intentionally exposes its uint16 backing
    # storage as int16.  Reinterpret it before widening so a one-code delta
    # across 0x7fff is not mistaken for a 65535-code error.
    if left.dtype == torch.int16:
        left = left.view(torch.uint16)
    if right.dtype == torch.int16:
        right = right.view(torch.uint16)
    return int((left.to(torch.int32) - right.to(torch.int32)).abs().max().item())


def _small_correctness(
    yuv_module: HipModule,
    rgb_module: HipModule,
    device: torch.device,
) -> list[dict[str, object]]:
    height, width = 64, 96
    rows: list[dict[str, object]] = []
    generator = torch.Generator(device=device).manual_seed(20260904)
    for bits in (8, 10):
        storage_dtype = torch.uint8 if bits == 8 else torch.uint16
        code_count = 256 if bits == 8 else 1024
        packed = torch.randint(
            0,
            code_count,
            (2, height + height // 2, width),
            dtype=torch.int32,
            device=device,
            generator=generator,
        )
        if bits == 10:
            # P010 stores each 10-bit code in the high bits of a 16-bit word.
            packed = packed.bitwise_left_shift(6)
        packed = packed.to(storage_dtype)
        for variant in VARIANTS:
            matrix, value_range = variant.split("_", 1)
            with _force_eager_reference():
                converter = YuvToRgbConverter(
                    height,
                    width,
                    COLOR_SPACES[matrix],
                    value_range == "full",
                    bits == 10,
                    device,
                )
            expected = torch.empty((2, 3, height, width), dtype=torch.uint8, device=device)
            for index in range(2):
                converter.convert_into(
                    packed[index, :height],
                    packed[index, height:].view(height // 2, width // 2, 2),
                    expected[index],
                )
            actual = torch.empty_like(expected)
            HipYuvToRgb(yuv_module, f"yuv{bits}_{variant}")(packed, actual)
            difference = _max_abs(actual, expected)
            if difference > 1:
                raise RuntimeError(
                    f"YUV->RGB parity failed for {bits}-bit {variant}: {difference}"
                )
            rows.append(
                {"direction": "yuv_to_rgb", "variant": f"{bits}-bit {variant}", "max_abs": difference}
            )

    rgb = torch.randint(
        0,
        256,
        (3, height, width),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    for prefix, storage_dtype, tolerance in (
        ("nv12", torch.uint8, 1),
        ("p010", torch.int16, 64),
    ):
        for variant in VARIANTS:
            name = f"{prefix}_{variant}"
            with _force_eager_reference():
                converter = RgbToYuvConverter(name, device=device)
            expected = converter.convert(rgb)
            actual = torch.empty(
                (height + height // 2, width), dtype=storage_dtype, device=device
            )
            HipRgbToYuv(rgb_module, name)(rgb, actual)
            difference = _max_abs(actual, expected)
            if difference > tolerance:
                raise RuntimeError(
                    f"RGB->YUV parity failed for {name}: {difference} > {tolerance}"
                )
            rows.append(
                {"direction": "rgb_to_yuv", "variant": name, "max_abs": difference}
            )
    torch.cuda.synchronize()
    return rows


def _performance(
    yuv_module: HipModule,
    rgb_module: HipModule,
    device: torch.device,
    *,
    height: int,
    width: int,
    rounds: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for bits, storage_dtype in ((8, torch.uint8), (10, torch.uint16)):
        packed = torch.randint(
            0,
            1024 if bits == 10 else 256,
            (1, height + height // 2, width),
            dtype=torch.int32,
            device=device,
        )
        if bits == 10:
            packed = packed.bitwise_left_shift(6)
        packed = packed.to(storage_dtype)
        out = torch.empty((1, 3, height, width), dtype=torch.uint8, device=device)
        allocated_before_reference = int(torch.cuda.memory_allocated(device))
        with _force_eager_reference():
            reference = YuvToRgbConverter(
                height, width, AvColorspace.ITU709, False, bits == 10, device
            )
        eager_converter_bytes = max(
            0, int(torch.cuda.memory_allocated(device)) - allocated_before_reference
        )

        def eager_decode() -> None:
            reference.convert_into(
                packed[0, :height],
                packed[0, height:].view(height // 2, width // 2, 2),
                out[0],
            )

        hip_decode = HipYuvToRgb(yuv_module, f"yuv{bits}_bt709_limited")
        eager_ms = _event_ms(eager_decode, warmup=2, rounds=rounds)
        fused_ms = _event_ms(lambda: hip_decode(packed, out), warmup=2, rounds=rounds)
        rows.append(
            {
                "direction": "yuv_to_rgb",
                "format": "NV12" if bits == 8 else "P010",
                "eager_ms": eager_ms,
                "fused_ms": fused_ms,
                "speedup": eager_ms / fused_ms,
                "eager_converter_torch_bytes": eager_converter_bytes,
            }
        )
        del reference, packed, out
        torch.cuda.empty_cache()

    rgb = torch.randint(
        0, 256, (3, height, width), dtype=torch.uint8, device=device
    )
    for name, storage_dtype in (
        ("nv12_bt709_limited", torch.uint8),
        ("p010_bt709_limited", torch.int16),
    ):
        packed = torch.empty(
            (height + height // 2, width), dtype=storage_dtype, device=device
        )
        with _force_eager_reference():
            reference = RgbToYuvConverter(name, device=device)
        fused = HipRgbToYuv(rgb_module, name)
        # Materialize the eager converter's lazy reusable scratch before
        # measuring its resident Torch allocation and steady-state latency.
        allocated_before_reference = int(torch.cuda.memory_allocated(device))
        reference.convert_into(rgb, packed[:height], packed[height:])
        torch.cuda.synchronize(device)
        eager_converter_bytes = max(
            0, int(torch.cuda.memory_allocated(device)) - allocated_before_reference
        )
        eager_ms = _event_ms(
            lambda: reference.convert_into(rgb, packed[:height], packed[height:]),
            warmup=2,
            rounds=rounds,
        )
        fused_ms = _event_ms(lambda: fused(rgb, packed), warmup=2, rounds=rounds)
        rows.append(
            {
                "direction": "rgb_to_yuv",
                "format": "NV12" if name.startswith("nv12") else "P010",
                "eager_ms": eager_ms,
                "fused_ms": fused_ms,
                "speedup": eager_ms / fused_ms,
                "eager_converter_torch_bytes": eager_converter_bytes,
            }
        )
        del reference, packed
        torch.cuda.empty_cache()
    del rgb
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", type=int, default=4096)
    parser.add_argument("--width", type=int, default=8192)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--architecture", default="gfx1100")
    parser.add_argument("--hipcc", default=shutil.which("hipcc"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _check_even(args.height, args.width)
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if not args.hipcc:
        raise RuntimeError("hipcc was not found; pass --hipcc explicitly")

    device = torch.device("cuda:0")
    if not torch.cuda.is_available() or not is_amd_device(device):
        raise RuntimeError("this probe requires a ROCm-backed AMD GPU")
    torch.cuda.set_device(device)
    actual_arch = str(getattr(torch.cuda.get_device_properties(device), "gcnArchName", ""))
    if not actual_arch.startswith(args.architecture):
        raise RuntimeError(
            f"compiled architecture {args.architecture!r} does not match {actual_arch!r}"
        )

    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="jasna-hip-colour-") as temporary:
        directory = Path(temporary)
        suffix = ".windows.co" if sys.platform == "win32" else ".co"
        yuv_code = directory / f"yuv_to_rgb{suffix}"
        rgb_code = directory / f"rgb_to_yuv{suffix}"
        yuv_command = _compile(
            MEDIA / "yuv_to_rgb.cu", yuv_code, args.architecture, args.hipcc
        )
        rgb_command = _compile(
            MEDIA / "rgb_to_yuv.cu", rgb_code, args.architecture, args.hipcc
        )
        # Establish Torch's primary HIP context before loading external modules.
        torch.empty(1, device=device)
        yuv_module = HipModule(yuv_code)
        rgb_module = HipModule(rgb_code)
        try:
            correctness = _small_correctness(yuv_module, rgb_module, device)
            performance = _performance(
                yuv_module,
                rgb_module,
                device,
                height=args.height,
                width=args.width,
                rounds=args.rounds,
            )
        finally:
            rgb_module.close()
            yuv_module.close()
        artifact_sha256 = {
            "yuv_to_rgb": hashlib.sha256(yuv_code.read_bytes()).hexdigest(),
            "rgb_to_yuv": hashlib.sha256(rgb_code.read_bytes()).hexdigest(),
        }

    report = {
        "schema": "jasna.amd-hip-colour-kernel-probe.v1",
        "device": torch.cuda.get_device_name(device),
        "architecture": actual_arch,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "hip_runtime": hip_runtime_identity(),
        "compiler": (lambda completed: completed.stdout or completed.stderr)(subprocess.run(
            [args.hipcc, "--version"],
            capture_output=True,
            text=True,
            check=True,
            env={
                **os.environ,
                "HIP_PATH": str(Path(args.hipcc).resolve().parent.parent),
                "ROCM_PATH": str(Path(args.hipcc).resolve().parent.parent),
            },
        )).splitlines()[0],
        "compile_commands": [yuv_command, rgb_command],
        "source_sha256": {
            name: hashlib.sha256((MEDIA / name).read_bytes()).hexdigest()
            for name in ("yuv_to_rgb.cu", "rgb_to_yuv.cu")
        },
        "artifact_sha256": artifact_sha256,
        "height": args.height,
        "width": args.width,
        "rounds": args.rounds,
        "correctness": correctness,
        "performance": performance,
        "wall_seconds": time.perf_counter() - started,
    }
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
