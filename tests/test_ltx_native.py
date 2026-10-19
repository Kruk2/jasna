"""Native LTX block wiring that needs no GPU: launch policy, kernel variants, capability
checks, the constants handed to the native side and the sampler's choice of block function."""

from __future__ import annotations

import os
import sys
import tomllib
from importlib.machinery import ModuleSpec
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

from jasna.ltx import kernels, native, sampler
from jasna.ltx import transformer as T
from jasna.ltx.model_files import PlaceholderHandle, PlaceholderTensors
from jasna.ltx.native import aot


def _block_specs(quantization: str) -> dict:
    specs = PlaceholderHandle(PlaceholderTensors(quantization, "distilled")).tensor_specs()
    return {name.split(".", 2)[2]: spec for name, spec in specs.items() if name.startswith("blocks.0.")}


def test_stream_k_splits_only_a_short_grids_idle_last_wave():
    # 512 canvas Q/K/V on an RTX 5090 (170 SMs): 1024 tiles leave 4 for a last wave
    assert kernels.stream_k_splits(8192, 4096, 4096, 128, 256, 128, 170) == 32
    assert kernels.stream_k_splits(8192, 4096, 4096, 128, 256, 128, 1024) == 0  # no partial wave
    assert kernels.stream_k_splits(4096, 16384, 4096, 128, 256, 128, 170) == 0  # many waves
    assert kernels.stream_k_splits(8192, 32, 4096, 32, 32, 256, 170) == 0  # gate logits
    assert kernels.stream_k_splits(4096, 4096, 16384, 128, 128, 128, 170) == 42  # FF down chunk


@pytest.mark.parametrize("quantization", ["int8", "nvfp4"])
def test_block_dims_come_from_the_weights(quantization):
    assert aot.BlockDims.from_specs(_block_specs(quantization)) == aot.BlockDims(dim=4096, ff_dim=16384, heads=32)
    assert native.block_quantization(_block_specs(quantization)) == quantization


def test_kernel_variants_follow_the_tile_tables():
    dims = aot.BlockDims(dim=4096, ff_dim=16384, heads=32)
    for capability, tiles in kernels.INT8_MM_TILES.items():
        found = aot.variants(capability, dims, fp4=False)
        keys = [variant.key for variant in found]
        assert len(keys) == len(set(keys))
        gemm_tiles = {(dict(v.constexprs)["BM"], dict(v.constexprs)["BN"], dict(v.constexprs)["BK"])
                      for v in found if v.family == "mm"}
        assert gemm_tiles == {tile[:3] for tile in tiles} | {kernels.SMALL_MM_TILE[:3]}
        assert any(v.family == "mm_stream_k" for v in found) == kernels.has_tma(capability)
        assert not any(v.family == "mm_stream_k" and dict(v.constexprs)["BN"] == kernels.SMALL_MM_TILE[1] for v in found)
    for capability in kernels.CUBLAS_INT8_MM:
        found = aot.variants(capability, dims, fp4=False)
        assert {v.family for v in found} == {"rotate_quantize", "epilogue"}
        epilogue_tiles = {(dict(v.constexprs)["BM"], dict(v.constexprs)["BN"]) for v in found if v.family == "epilogue"}
        assert epilogue_tiles == {kernels.CUBLAS_EPILOGUE_TILE}
    assert {v.family for v in aot.variants((12, 0), dims, fp4=True)} == {"quantize_fp4"}


def test_every_projection_has_a_variant_with_and_without_bias():
    found = aot.variants((12, 0), aot.BlockDims(dim=4096, ff_dim=16384, heads=32), fp4=False)
    down = [dict(v.constexprs) for v in found if v.family == "mm" and dict(v.constexprs)["BN"] == 128]
    assert {(flags["HAS_BIAS"], flags["RESIDUAL"]) for flags in down} == {(False, True), (True, True)}


def _cuda_capability(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(torch.version, "cuda", "13.0")


@pytest.mark.parametrize("capability, quantization, reason", [
    ((9, 0), "int8", "no tuned kernels for SM 9.0"),
    ((8, 9), "nvfp4", "NVFP4 blocks need SM 12.0 / 12.1"),
    ((7, 5), "nvfp4", "NVFP4 blocks need SM 12.0 / 12.1"),
])
def test_native_block_refuses_gpus_without_its_kernels(monkeypatch, capability, quantization, reason):
    _cuda_capability(monkeypatch, capability)
    assert reason in native.unavailable_reason(torch.device("cuda"), _block_specs(quantization))


def test_native_block_runs_its_own_int8_attention_whatever_comfy_kitchen_is_installed(monkeypatch):
    _cuda_capability(monkeypatch, (12, 0))
    monkeypatch.setattr(native, "_prebuilt_extension", lambda: native.PREBUILT / "jasna_ltx_native.so")
    monkeypatch.setitem(sys.modules, "comfy_kitchen", None)
    assert native.unavailable_reason(torch.device("cuda"), _block_specs("int8")) is None


def test_extension_sources_look_up_no_launchers_at_runtime():
    sources = [path for path in native._HERE.rglob("*") if path.suffix in (".cpp", ".cu", ".cuh", ".h")]
    assert {path.name for path in sources} >= {"block.cpp", "glue.cu", *native._ATTENTION_SOURCES}
    for path in sources:
        text = path.read_text()
        for lookup in ("dlopen", "dlsym", "GetProcAddress", "GetModuleHandle", 'extern "C" void', "kitchen_library"):
            assert lookup not in text, f"{path.name} uses {lookup}"
    assert '"launch_' not in (native._HERE / "block.cpp").read_text()


def test_package_data_ships_every_native_source():
    pyproject = tomllib.loads((native._HERE.parents[2] / "pyproject.toml").read_text())
    shipped = {path for pattern in pyproject["tool"]["setuptools"]["package-data"]["jasna.ltx.native"]
               for path in native._HERE.glob(pattern)}
    sources = {path for path in native._HERE.rglob("*")
               if path.is_file() and path.suffix != ".py" and not {"__pycache__", "prebuilt"} & set(path.parts)}
    assert sources and sources == shipped


def test_attention_tile_follows_comfy_kitchen():
    sage_attention = pytest.importorskip("comfy_kitchen.sage_attention")
    for head_dim in (64, 128, 256):
        for rows in (16, 1024, 1025, 8192):
            assert native.attention_cta_k(head_dim, rows) == sage_attention._select_cta_k(head_dim, rows, has_mask=False)


def test_vendored_attention_compiles_once_with_comfy_kitchens_flags(monkeypatch, tmp_path):
    from torch.utils import cpp_extension

    commands = []

    def compile_object(command, check):
        commands.append(command)
        Path(command[-1]).write_bytes(b"object")

    monkeypatch.setattr(native.subprocess, "run", compile_object)
    monkeypatch.setattr(cpp_extension, "_get_cuda_arch_flags", lambda: ["-gencode=arch=compute_120,code=sm_120"])
    monkeypatch.setattr(native, "_IS_WINDOWS", False)
    monkeypatch.setenv("CC", "gcc-15")
    objects = native._attention_objects(tmp_path, PurePosixPath("/cuda"))
    assert [path.name for path in objects] == ["quant_qk_int8.o", "quant_v_int8.o", "sage_attn_launcher.o"]
    assert all(path.read_bytes() == b"object" for path in objects)
    assert [command[-3] for command in commands] == [str(native._ATTENTION / source) for source in native._ATTENTION_SOURCES]
    assert commands[0][:3] == ["/cuda/bin/nvcc", "-ccbin", "gcc-15"]
    assert {"--use_fast_math", "-O3", "-DNDEBUG", "-Xcompiler=-fPIC", "-gencode=arch=compute_120,code=sm_120"} <= set(commands[0])
    assert native._attention_objects(tmp_path, PurePosixPath("/cuda")) == objects and len(commands) == 3

    monkeypatch.setattr(cpp_extension, "_get_cuda_arch_flags", lambda: ["-gencode=arch=compute_89,code=sm_89"])
    assert native._attention_objects(tmp_path, PurePosixPath("/cuda"))[0].parent != objects[0].parent


def test_vendored_attention_compiles_for_the_msvc_runtime_on_windows(monkeypatch):
    from torch.utils import cpp_extension

    monkeypatch.setattr(cpp_extension, "_get_cuda_arch_flags", lambda: [])
    monkeypatch.setattr(native, "_IS_WINDOWS", True)
    monkeypatch.delenv("CC", raising=False)
    command = native._attention_nvcc_command(PureWindowsPath("C:/cuda"))
    assert command[0] == "C:\\cuda\\bin\\nvcc.exe"
    assert {"-Xcompiler=/MD", "-Xcompiler=/Zc:preprocessor", "-std=c++17", "--use_fast_math"} <= set(command)


def test_native_block_without_a_toolchain_keeps_the_compiled_path(monkeypatch):
    from torch.utils import cpp_extension

    _cuda_capability(monkeypatch, (12, 0))
    monkeypatch.setattr(native, "_prebuilt_extension", lambda: None)
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", None)
    assert "nvcc" in native.unavailable_reason(torch.device("cuda"), _block_specs("int8"))


@pytest.mark.parametrize("windows, nvcc", [(True, "nvcc.exe"), (False, "nvcc")])
def test_source_build_finds_the_platform_nvcc(monkeypatch, tmp_path, windows, nvcc):
    from torch.utils import cpp_extension

    _cuda_capability(monkeypatch, (12, 0))
    monkeypatch.setattr(native, "_prebuilt_extension", lambda: None)
    monkeypatch.setattr(native, "_IS_WINDOWS", windows)
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", str(tmp_path))
    monkeypatch.setattr(native.shutil, "which", lambda name: name)
    monkeypatch.setattr(native, "_ninja_available", lambda: True)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / ("nvcc" if windows else "nvcc.exe")).touch()
    assert "nvcc" in native.unavailable_reason(torch.device("cuda"), _block_specs("int8"))
    (tmp_path / "bin" / nvcc).touch()
    assert native.unavailable_reason(torch.device("cuda"), _block_specs("int8")) is None


def test_source_build_looks_for_the_host_compiler_torch_builds_with(monkeypatch, tmp_path):
    from torch.utils import cpp_extension

    _cuda_capability(monkeypatch, (12, 0))
    monkeypatch.setattr(native, "_prebuilt_extension", lambda: None)
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", str(tmp_path))
    native._nvcc(tmp_path).parent.mkdir()
    native._nvcc(tmp_path).touch()
    monkeypatch.delenv("CXX", raising=False)
    looked_up = []
    monkeypatch.setattr(native.shutil, "which", looked_up.append)
    host = "cl" if sys.platform == "win32" else "c++"
    reason = native.unavailable_reason(torch.device("cuda"), _block_specs("int8"))
    assert reason == f"no C++ compiler ({host}) found to build it" and looked_up == [host]


def test_release_without_a_prebuilt_block_keeps_the_compiled_path_without_cpp_extension(monkeypatch):
    _cuda_capability(monkeypatch, (12, 0))
    monkeypatch.setattr(native, "is_frozen", lambda: True)
    monkeypatch.setattr(native, "_prebuilt_extension", lambda: None)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", None)
    assert "release" in native.unavailable_reason(torch.device("cuda"), _block_specs("int8"))


def test_release_with_a_prebuilt_block_runs_it(monkeypatch):
    _cuda_capability(monkeypatch, (12, 0))
    monkeypatch.setattr(native, "is_frozen", lambda: True)
    monkeypatch.setattr(native, "_prebuilt_extension", lambda: native.PREBUILT / "jasna_ltx_native.so")
    assert native.unavailable_reason(torch.device("cuda"), _block_specs("int8")) is None


def test_extension_build_finds_ninja_from_its_package_when_off_path(monkeypatch, tmp_path):
    from torch.utils import cpp_extension

    ninja = ModuleType("ninja")
    ninja.__spec__ = ModuleSpec("ninja", None)
    ninja.BIN_DIR = str(tmp_path / "ninja-bin")
    monkeypatch.setitem(sys.modules, "ninja", ninja)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", "/cuda")
    monkeypatch.setattr(native, "_attention_objects", lambda build_directory, cuda_home: [])
    monkeypatch.setattr(cpp_extension, "load", lambda **kwargs: os.environ["PATH"].split(os.pathsep))
    assert ninja.BIN_DIR in native._build_extension(tmp_path / "build")


def test_extension_links_the_vendored_attention_objects(monkeypatch, tmp_path):
    from torch.utils import cpp_extension

    objects = [tmp_path / "quant_qk_int8.o", tmp_path / "sage_attn_launcher.o"]
    monkeypatch.setattr(native, "_ninja_available", lambda: True)
    monkeypatch.setattr(native, "_IS_WINDOWS", False)
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", "/cuda")
    monkeypatch.setattr(native, "_attention_objects", lambda build_directory, cuda_home: objects)
    monkeypatch.setattr(cpp_extension, "load", lambda **kwargs: kwargs)
    built = native._build_extension(tmp_path / "build")
    assert built["extra_ldflags"][-2:] == [str(path) for path in objects]
    assert built["sources"] == [str(native._HERE / source) for source in native._SOURCES]


def test_build_flags_link_the_platform_cuda_driver_library(monkeypatch):
    monkeypatch.setattr(native, "_IS_WINDOWS", True)
    windows = native._build_flags(PureWindowsPath("C:/cuda"))
    assert windows["extra_cflags"] == ["/O2"]
    assert windows["extra_ldflags"] == ["C:\\cuda\\lib\\x64\\cuda.lib"]

    monkeypatch.setattr(native, "_IS_WINDOWS", False)
    linux = native._build_flags(PurePosixPath("/cuda"))
    assert linux["extra_cflags"] == ["-O3", "-fvisibility=hidden"]
    assert linux["extra_ldflags"] == ["-L/cuda/lib64/stubs", "-lcuda"]


def test_release_arch_list_covers_every_sm_with_tuned_int8_kernels():
    archs = {tuple(int(part) for part in arch.split(".")) for arch in native.cuda_arch_list().split(";")}
    assert archs == {*kernels.INT8_MM_TILES, *kernels.CUBLAS_INT8_MM}


def test_native_block_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("JASNA_LTX_NATIVE", "0")
    assert native.load(torch.device("cuda"), _block_specs("int8")) is None
    monkeypatch.delenv("JASNA_LTX_NATIVE")
    assert "NVIDIA GPU" in native.unavailable_reason(torch.device("cpu"), _block_specs("int8"))


def test_python_hands_every_constant_to_the_native_block(monkeypatch, tmp_path):
    _cuda_capability(monkeypatch, (12, 0))
    monkeypatch.setattr(kernels, "programs", lambda device: 170)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    built = []
    module = SimpleNamespace(Config=SimpleNamespace, Block=lambda config: built.append(config) or config)
    forward = native.NativeBlockForward(module, torch.device("cuda", 0), aot.BlockDims(4096, 16384, 32), manifest)
    config = built[0]
    assert (config.heads, config.eps, config.chunk_rows) == (T.HEADS, T.EPS, T.CHUNK_ROWS)
    assert (tuple(config.narrow_tile), tuple(config.wide_tile)) == kernels.INT8_MM_TILES[(12, 0)]
    assert tuple(config.small_tile) == kernels.SMALL_MM_TILE and config.small_max_n == kernels.SMALL_MM_MAX_N
    assert (config.wide_k, config.group_m, config.stream_k_max_waves) == (
        kernels.WIDE_K, kernels.GROUP_M, kernels.STREAM_K_MAX_WAVES)
    assert config.programs == 170 and config.has_tma and not config.cublas_int8
    assert config.fp4_tensor_scale_divisor == T.FP4_MAX * T.FP8_E4M3_MAX and config.fp4_block == T.FP4_BLOCK
    assert config.fp4_recipe == [F.ScalingType.BlockWise1x16.value, F.ScalingType.TensorWise.value]
    assert config.kernels == [] and config.kernel_directory == str(tmp_path)
    assert forward.head_dim == 128


def _expected_block_parameters(capability, heads):
    tiles = kernels.INT8_MM_TILES.get(capability, (kernels.SMALL_MM_TILE,) * 2)
    return {
        "heads": heads, "eps": T.EPS, "chunk_rows": T.CHUNK_ROWS,
        "cublas_int8": capability in kernels.CUBLAS_INT8_MM,
        "narrow_tile": list(tiles[0]), "wide_tile": list(tiles[1]), "small_tile": list(kernels.SMALL_MM_TILE),
        "small_max_n": kernels.SMALL_MM_MAX_N, "wide_k": kernels.WIDE_K, "group_m": kernels.GROUP_M,
        "stream_k_max_waves": kernels.STREAM_K_MAX_WAVES,
        "has_tma": kernels.has_tma(capability), "epilogue_tile": list(kernels.CUBLAS_EPILOGUE_TILE),
        "fp4_tensor_scale_divisor": T.FP4_MAX * T.FP8_E4M3_MAX, "fp4_block": T.FP4_BLOCK,
        "fp4_quantize_rows": kernels.FP4_QUANTIZE_ROWS, "fp4_quantize_cols": kernels.FP4_QUANTIZE_COLS,
        "fp4_scale_tile_rows": kernels.FP4_SCALE_TILE_ROWS, "fp4_scale_tile_cols": kernels.FP4_SCALE_TILE_COLS,
        "fp4_recipe": [F.ScalingType.BlockWise1x16.value, F.ScalingType.TensorWise.value],
        "fp4_swizzle": [F.SwizzleType.SWIZZLE_32_4_4.value, F.SwizzleType.NO_SWIZZLE.value],
    }


@pytest.mark.parametrize("capability", [*native.release_architectures(), (9, 0), (0, 0), (99, 9)])
def test_block_parameters_are_pure_capability_policy(monkeypatch, capability):
    expected = _expected_block_parameters(capability, 32)

    class ForbiddenCuda:
        def __getattr__(self, name):
            pytest.fail(f"Pure block parameters accessed torch.cuda.{name}")

    monkeypatch.setattr(torch, "cuda", ForbiddenCuda())
    monkeypatch.setattr(kernels, "programs", lambda *args: pytest.fail("GPU program count queried"))
    monkeypatch.setattr(native, "_kernel_manifest", lambda *args, **kwargs: pytest.fail("Kernel manifest queried"))
    monkeypatch.setattr(native, "build_kernels", lambda *args, **kwargs: pytest.fail("Kernels built"))
    values = native.block_parameters(capability, 32)
    assert values == expected
    assert not {"device", "programs", "kernels", "kernel_directory"} & values.keys()
    values["narrow_tile"][0] = -1
    assert native.block_parameters(capability, 32) == expected
    assert native.block_parameters(capability, 16)["heads"] == 16


@pytest.mark.parametrize("capability", [*native.release_architectures(), (9, 0)])
@pytest.mark.parametrize("explicit_device", [False, True])
def test_config_values_preserves_complete_runtime_dictionary(monkeypatch, tmp_path, capability, explicit_device):
    _cuda_capability(monkeypatch, capability)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(kernels, "programs", lambda device: 123)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('[{"key":"owned-kernel","file":"owned.cubin"}]')
    device = torch.device("cuda", 2) if explicit_device else torch.device("cuda")
    expected = {"device": 2 if explicit_device else 7, **_expected_block_parameters(capability, 32),
                "programs": 123, "kernels": [{"key": "owned-kernel", "file": "owned.cubin"}],
                "kernel_directory": str(tmp_path)}
    assert native.config_values(device, aot.BlockDims(4096, 16384, 32), manifest) == expected


def test_config_values_consumes_shared_public_block_parameters(monkeypatch, tmp_path):
    _cuda_capability(monkeypatch, (12, 0))
    monkeypatch.setattr(kernels, "programs", lambda device: 170)
    calls = []

    def parameters(capability, heads):
        calls.append((capability, heads))
        return {"heads": heads, "owned_policy_marker": "shared"}

    monkeypatch.setattr(native, "block_parameters", parameters)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    result = native.config_values(torch.device("cuda", 0), aot.BlockDims(4096, 16384, 32), manifest)
    assert calls == [((12, 0), 32)]
    assert result == {"device": 0, "heads": 32, "owned_policy_marker": "shared", "programs": 170,
                      "kernels": [], "kernel_directory": str(tmp_path)}


def test_rtx_3090_native_block_launches_plain_triton_gemms(monkeypatch, tmp_path):
    _cuda_capability(monkeypatch, (8, 6))
    monkeypatch.setattr(kernels, "programs", lambda device: 82)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    built = []
    module = SimpleNamespace(Config=SimpleNamespace, Block=lambda config: built.append(config) or config)
    native.NativeBlockForward(module, torch.device("cuda", 0), aot.BlockDims(4096, 16384, 32), manifest)
    config = built[0]
    assert not config.has_tma and not config.cublas_int8
    assert (tuple(config.narrow_tile), tuple(config.wide_tile)) == kernels.INT8_MM_TILES[(8, 6)]
    assert {v.family for v in aot.variants((8, 6), aot.BlockDims(4096, 16384, 32), fp4=False)} == {"rotate_quantize", "mm"}
    assert "NVFP4 blocks need SM 12.0 / 12.1" in native.unavailable_reason(torch.device("cuda"), _block_specs("nvfp4"))


def test_linear_slots_follow_the_transformer_names():
    weights = {"a.qdata": torch.zeros(2, 2, dtype=torch.int8), "a.scales": torch.ones(2),
               "b.weight": torch.zeros(2, 1, dtype=torch.uint8), "b.weight_scale": torch.ones(2),
               "b.weight_scale_2": torch.ones(()), "b.bias": torch.ones(2)}
    qdata, scales, scale_2, bias = native._linear_tensors(weights, "a")
    assert qdata is weights["a.qdata"] and scales is weights["a.scales"] and scale_2 is None and bias is None
    assert native._linear_tensors(weights, "b") == tuple(weights[f"b.{part}"] for part in
                                                          ("weight", "weight_scale", "weight_scale_2", "bias"))


def test_sampler_runs_native_blocks_with_eager_prompt_projections(monkeypatch):
    native_forward = object()
    monkeypatch.setattr(native, "load", lambda device, specs: native_forward)
    monkeypatch.setattr(T, "enable_int8_attention", lambda: pytest.fail("compiled-path attention enabled"))
    monkeypatch.setattr(sampler, "compiled_block_forward", lambda device: pytest.fail("block forward compiled"))
    assert sampler._block_functions(torch.device("cuda"), {}) == (native_forward, T.prompt_key_value, "native")


def test_sampler_compiles_blocks_when_native_cannot_run(monkeypatch):
    enabled = []
    monkeypatch.setattr(native, "load", lambda device, specs: None)
    monkeypatch.setattr(T, "enable_int8_attention", lambda: enabled.append(True))
    monkeypatch.setattr(sampler, "compiled_block_forward", lambda device: "compiled block")
    monkeypatch.setattr(sampler, "compile_static", lambda function, mode, device: ("compiled", function))
    assert sampler._block_functions(torch.device("cuda"), {}) == (
        "compiled block", ("compiled", T.prompt_key_value), "torch.compile")
    assert enabled


def test_native_package_imports_without_the_protection_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "jasna.protection", None)
    assert native.load(torch.device("cpu"), {}) is None


def test_each_source_checkout_builds_its_extension_in_its_own_directory(monkeypatch, tmp_path):
    """Two checkouts sharing one compile cache would otherwise rebuild the extension on every
    switch (~12 s), since the build commands name each checkout's source paths."""
    built = []
    monkeypatch.setattr(native, "_prebuilt_extension", lambda: None)
    monkeypatch.setattr(native, "_cache", lambda: tmp_path)
    monkeypatch.setattr(native, "_build_extension", built.append)
    for checkout in ("/work/jasna-a/jasna/ltx/native", "/work/jasna-b/jasna/ltx/native", "/work/jasna-a/jasna/ltx/native"):
        monkeypatch.setattr(native, "_HERE", Path(checkout))
        native._extension()
    assert built[0] != built[1] and built[0] == built[2]
    assert all(path.parent == tmp_path and path.name.startswith(f"extension-torch{torch.__version__}") for path in built)
