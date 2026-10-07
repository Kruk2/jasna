"""Native LTX block wiring that needs no GPU: launch policy, kernel variants, capability
checks, the constants handed to the native side and the sampler's choice of block function."""

from __future__ import annotations

import os
import sys
from importlib.machinery import ModuleSpec
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


def test_native_block_needs_the_mirrored_comfy_kitchen(monkeypatch):
    _cuda_capability(monkeypatch, (12, 0))
    monkeypatch.setattr(native.importlib.metadata, "version", lambda package: "9.9.9")
    assert "comfy-kitchen 0.2.36" in native.unavailable_reason(torch.device("cuda"), _block_specs("int8"))


def test_native_block_without_a_toolchain_keeps_the_compiled_path(monkeypatch):
    from torch.utils import cpp_extension

    _cuda_capability(monkeypatch, (12, 0))
    monkeypatch.setattr(native, "_prebuilt_extension", lambda: None)
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", None)
    assert "nvcc" in native.unavailable_reason(torch.device("cuda"), _block_specs("int8"))


def test_extension_build_finds_ninja_from_its_package_when_off_path(monkeypatch, tmp_path):
    from torch.utils import cpp_extension

    ninja = ModuleType("ninja")
    ninja.__spec__ = ModuleSpec("ninja", None)
    ninja.BIN_DIR = str(tmp_path / "ninja-bin")
    monkeypatch.setitem(sys.modules, "ninja", ninja)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", "/cuda")
    monkeypatch.setattr(cpp_extension, "load", lambda **kwargs: os.environ["PATH"].split(os.pathsep))
    assert ninja.BIN_DIR in native._build_extension(tmp_path / "build")


def test_release_arch_list_covers_every_sm_with_tuned_int8_kernels():
    archs = {tuple(int(part) for part in arch.split(".")) for arch in native.cuda_arch_list().split(";")}
    assert archs == {*kernels.INT8_MM_TILES, *kernels.CUBLAS_INT8_MM}


def test_native_block_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("JASNA_LTX_NATIVE", "0")
    assert native.load(torch.device("cuda"), _block_specs("int8")) is None
    monkeypatch.delenv("JASNA_LTX_NATIVE")
    assert "NVIDIA GPU" in native.unavailable_reason(torch.device("cpu"), _block_specs("int8"))


def test_python_hands_every_constant_to_the_native_block(monkeypatch, tmp_path):
    pytest.importorskip("comfy_kitchen")
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


def test_rtx_3090_native_block_launches_plain_triton_gemms(monkeypatch, tmp_path):
    pytest.importorskip("comfy_kitchen")
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
    monkeypatch.setattr(sampler, "compiled_block_forward", lambda: pytest.fail("block forward compiled"))
    assert sampler._block_functions(torch.device("cuda"), {}) == (native_forward, T.prompt_key_value, "native")


def test_sampler_compiles_blocks_when_native_cannot_run(monkeypatch):
    enabled = []
    monkeypatch.setattr(native, "load", lambda device, specs: None)
    monkeypatch.setattr(T, "enable_int8_attention", lambda: enabled.append(True))
    monkeypatch.setattr(sampler, "compiled_block_forward", lambda: "compiled block")
    monkeypatch.setattr(sampler, "compile_static", lambda function, mode: ("compiled", function))
    assert sampler._block_functions(torch.device("cuda"), {}) == (
        "compiled block", ("compiled", T.prompt_key_value), "torch.compile")
    assert enabled


def test_native_package_imports_without_the_protection_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "jasna.protection", None)
    assert native.load(torch.device("cpu"), {}) is None
