"""Native LTX block wiring that needs no GPU: launch policy, kernel variants, capability
checks, the constants handed to the native side and the sampler's choice of block function."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tomllib
from importlib.machinery import ModuleSpec
from pathlib import Path, PurePosixPath
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

pytest.importorskip("triton")

from jasna.ltx import kernel_policy, kernels, native, sampler
from jasna.ltx import transformer as T
from jasna.ltx.model_files import PlaceholderHandle, PlaceholderTensors
from jasna.ltx.native import aot


NVIDIA_RELEASE = sorted({*kernel_policy.INT8_MM_TILES, *kernel_policy.CUBLAS_INT8_MM})


@pytest.mark.parametrize('windows, suffix', [(False, '.so'), (True, '.pyd')])
def test_prebuilt_extension_uses_the_platforms_python_extension(tmp_path, monkeypatch, windows, suffix):
    monkeypatch.setattr(native, 'PREBUILT', tmp_path)
    monkeypatch.setattr(native, '_IS_WINDOWS', windows)
    foreign = tmp_path / ('jasna_ltx_native.so' if windows else 'jasna_ltx_native.pyd')
    foreign.touch()
    assert native._prebuilt_extension() is None
    extension = tmp_path / ('jasna_ltx_native' + suffix)
    extension.touch()
    assert native._prebuilt_extension() == extension


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
    for capability, tiles in kernel_policy.INT8_MM_TILES.items():
        found = aot.variants(capability, dims, fp4=False)
        keys = [variant.key for variant in found]
        assert len(keys) == len(set(keys))
        gemm_tiles = {(dict(v.constexprs)["BM"], dict(v.constexprs)["BN"], dict(v.constexprs)["BK"])
                      for v in found if v.family == "mm"}
        assert gemm_tiles == {tile[:3] for tile in tiles} | {kernel_policy.SMALL_MM_TILE[:3]}
        assert any(v.family == "mm_stream_k" for v in found) == kernel_policy.has_tma(capability)
        assert not any(v.family == "mm_stream_k" and dict(v.constexprs)["BN"] == kernel_policy.SMALL_MM_TILE[1] for v in found)
    for capability in kernel_policy.CUBLAS_INT8_MM:
        found = aot.variants(capability, dims, fp4=False)
        assert {v.family for v in found} == {"rotate_quantize", "epilogue"}
        epilogue_tiles = {(dict(v.constexprs)["BM"], dict(v.constexprs)["BN"]) for v in found if v.family == "epilogue"}
        assert epilogue_tiles == {kernel_policy.CUBLAS_EPILOGUE_TILE}
    assert {v.family for v in aot.variants((12, 0), dims, fp4=True)} == {"quantize_fp4"}


def test_every_projection_has_a_variant_with_and_without_bias():
    found = aot.variants((12, 0), aot.BlockDims(dim=4096, ff_dim=16384, heads=32), fp4=False)
    down = [dict(v.constexprs) for v in found if v.family == "mm" and dict(v.constexprs)["BN"] == 128]
    assert {(flags["HAS_BIAS"], flags["RESIDUAL"]) for flags in down} == {(False, True), (True, True)}


def _cuda_capability(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: capability)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(torch.version, "hip", None)


def test_amd_gpus_take_their_tiles_by_architecture_name_not_their_reported_capability(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    monkeypatch.setattr(torch.version, "hip", "7.0")
    monkeypatch.setattr(kernel_policy, "amd_gpu_architecture", lambda device: "gfx1201")
    device = torch.device("cuda")
    assert kernel_policy.gpu_architecture(device) == "gfx1201" and kernel_policy.supports(device)
    assert kernel_policy.gemm_tile(4096, 4096, "gfx1201") == kernel_policy.AMD_INT8_MM_TILES["gfx1201"][0]
    assert kernel_policy.gemm_tile(4096, 16384, "gfx1201") == kernel_policy.AMD_INT8_MM_TILES["gfx1201"][1]
    assert kernel_policy.gemm_tile(32, 4096, "gfx1201") == kernel_policy.SMALL_MM_TILE
    assert not kernel_policy.has_tma("gfx1201")
    monkeypatch.setattr(kernel_policy, "amd_gpu_architecture", lambda device: "gfx90a")
    assert not kernel_policy.supports(device)


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
            expected = sage_attention._select_cta_k(head_dim, rows, has_mask=False)
            assert kernel_policy.attention_key_tile((12, 0), head_dim, rows) == expected
            assert kernel_policy.attention_key_tile("gfx1201", head_dim, rows) == kernel_policy.AMD_ATTENTION_KEY_TILE


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
    monkeypatch.setattr(native, "_ninja_path", lambda: "ninja path")
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
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", "/cuda")
    monkeypatch.setattr(native, "_attention_objects", lambda build_directory, cuda_home: [])
    monkeypatch.setattr(cpp_extension, "load", lambda **kwargs: os.environ["PATH"].split(os.pathsep))
    assert ninja.BIN_DIR in native._build_extension(tmp_path / "build")
    assert os.environ["PATH"] == str(tmp_path / "empty")


def test_extension_links_the_vendored_attention_objects(monkeypatch, tmp_path):
    from torch.utils import cpp_extension

    objects = [tmp_path / "quant_qk_int8.o", tmp_path / "sage_attn_launcher.o"]
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(native, "_ninja_path", lambda: "ninja path")
    monkeypatch.setattr(native, "_IS_WINDOWS", False)
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", "/cuda")
    monkeypatch.setattr(native, "_attention_objects", lambda build_directory, cuda_home: objects)
    monkeypatch.setattr(cpp_extension, "load", lambda **kwargs: kwargs)
    built = native._build_extension(tmp_path / "build")
    assert built["extra_ldflags"][-2:] == [str(path) for path in objects]
    assert built["sources"] == [str(native._HERE / source) for source in native._SOURCES]


@pytest.mark.parametrize("hip", [None, "7.15"])
def test_release_arch_list_covers_every_sm_with_tuned_int8_kernels(monkeypatch, hip):
    monkeypatch.setattr(torch.version, "hip", hip)
    archs = {tuple(int(part) for part in arch.split(".")) for arch in native.cuda_arch_list().split(";")}
    assert archs == {*kernel_policy.INT8_MM_TILES, *kernel_policy.CUBLAS_INT8_MM}


def test_building_the_extension_restores_the_environment_it_builds_in(monkeypatch, tmp_path):
    from torch.utils import cpp_extension

    _cuda_capability(monkeypatch, (12, 0))
    monkeypatch.setenv("PATH", "user path")
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(native.shutil, "which", lambda name: name)
    monkeypatch.setattr(native, "_ninja_path", lambda: "ninja dir:user path")
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", "/cuda")
    monkeypatch.setattr(native, "_attention_objects", lambda build_directory, cuda_home: [])
    seen = []
    monkeypatch.setattr(cpp_extension, "load", lambda **kwargs: seen.append((os.environ["PATH"], os.environ["CC"])))

    native._build_extension(tmp_path / "build")

    assert seen == [("ninja dir:user path", "gcc-15")]
    assert os.environ["PATH"] == "user path" and "CC" not in os.environ


@pytest.mark.parametrize(("compiled", "triton", "level", "path"), [
    (True, True, logging.INFO, "the torch.compile path"),
    (False, False, logging.WARNING, "eager PyTorch (no Triton)"),
    (False, True, logging.WARNING, "eager PyTorch (no bf16 torch.compile"),
])
def test_native_block_off_names_the_path_that_runs_instead(monkeypatch, caplog, compiled, triton, level, path):
    monkeypatch.setenv("JASNA_LTX_NATIVE", "0")
    monkeypatch.setattr(native, "compiles", lambda device, dtype: compiled)
    monkeypatch.setattr(native, "has_triton", lambda: triton)

    with caplog.at_level(logging.INFO, logger=native.logger.name):
        assert native.load(torch.device("cuda"), _block_specs("int8")) is None

    assert [(record.levelno, path in record.getMessage()) for record in caplog.records] == [(level, True)]


def _native_load_fails(monkeypatch, frozen: bool):
    monkeypatch.setattr(native, "unavailable_reason", lambda device, specs: None)
    monkeypatch.setattr(native, "compiles", lambda device, dtype: False)
    monkeypatch.setattr(native, "has_triton", lambda: False)
    monkeypatch.setattr(native, "is_frozen", lambda: frozen)
    monkeypatch.setattr(native, "_prebuilt_extension", lambda: native.PREBUILT / "jasna_ltx_native.so")
    monkeypatch.setattr(native.policy, "gpu_architecture", lambda device: "gfx1201")
    monkeypatch.setattr(native, "_kernel_manifest", lambda architecture, dims, fp4: Path("manifest.json"))
    monkeypatch.setattr(native, "_extension", Mock(side_effect=OSError("undefined symbol")))


def test_a_source_run_whose_native_block_fails_to_load_warns_and_runs_the_torch_path(monkeypatch, caplog):
    _native_load_fails(monkeypatch, frozen=False)

    with caplog.at_level(logging.WARNING, logger=native.logger.name):
        assert native.load(torch.device("cuda"), _block_specs("int8")) is None

    assert "eager PyTorch (no Triton)" in caplog.text and "undefined symbol" in caplog.text


def test_a_release_whose_shipped_native_block_fails_to_load_is_broken(monkeypatch):
    _native_load_fails(monkeypatch, frozen=True)

    with pytest.raises(RuntimeError, match=r"does not load \(undefined symbol\)\. Reinstall Jasna\."):
        native.load(torch.device("cuda"), _block_specs("int8"))


def test_native_block_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(native, "compiles", lambda device, dtype: False)
    monkeypatch.setenv("JASNA_LTX_NATIVE", "0")
    assert native.load(torch.device("cuda"), _block_specs("int8")) is None
    monkeypatch.delenv("JASNA_LTX_NATIVE")
    assert "NVIDIA or AMD GPU" in native.unavailable_reason(torch.device("cpu"), _block_specs("int8"))


@pytest.mark.parametrize("architecture", [*NVIDIA_RELEASE, (9, 0), *sorted(kernel_policy.AMD_INT8_MM_TILES)])
def test_block_parameters_need_no_gpu_and_share_no_state(monkeypatch, architecture):
    reference_keys = native.block_parameters((12, 0), 32).keys()

    class ForbiddenCuda:
        def __getattr__(self, name):
            pytest.fail(f"block parameters accessed torch.cuda.{name}")

    monkeypatch.setattr(torch, "cuda", ForbiddenCuda())
    values = native.block_parameters(architecture, 16)
    assert values["heads"] == 16 and values.keys() == reference_keys
    expected = native.block_parameters(architecture, 16)
    values["narrow_tile"][0] = -1
    assert native.block_parameters(architecture, 16) == expected


@pytest.mark.parametrize("device, index", [(torch.device("cuda", 2), 2), (torch.device("cuda"), 7)])
def test_config_values_add_this_gpu_and_its_shipped_kernels(monkeypatch, tmp_path, device, index):
    _cuda_capability(monkeypatch, (12, 0))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(kernel_policy, "programs", lambda device: 123)
    manifest = tmp_path / "manifest.json"
    manifest.write_text('[{"key":"owned-kernel","file":"owned.cubin"}]')
    assert native.config_values(device, aot.BlockDims(4096, 16384, 32), manifest) == {
        **native.block_parameters((12, 0), 32), "device": index, "programs": 123,
        "kernels": [{"key": "owned-kernel", "file": "owned.cubin"}], "kernel_directory": str(tmp_path)}


def test_rtx_3090_native_block_launches_plain_triton_gemms(monkeypatch, tmp_path):
    _cuda_capability(monkeypatch, (8, 6))
    monkeypatch.setattr(kernel_policy, "programs", lambda device: 82)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    built = []
    module = SimpleNamespace(Config=SimpleNamespace, Block=lambda config: built.append(config) or config)
    native.NativeBlockForward(module, torch.device("cuda", 0), aot.BlockDims(4096, 16384, 32), manifest)
    config = built[0]
    assert not config.has_tma and not config.cublas_int8
    assert (tuple(config.narrow_tile), tuple(config.wide_tile)) == kernel_policy.INT8_MM_TILES[(8, 6)]
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
    monkeypatch.setattr(T, "enable_int8_attention", lambda device: pytest.fail("compiled-path attention enabled"))
    monkeypatch.setattr(sampler, "compiled_block_forward", lambda device: pytest.fail("block forward compiled"))
    assert sampler._block_functions(torch.device("cuda"), {}) == (native_forward, T.prompt_key_value, "native")


def test_sampler_compiles_blocks_when_native_cannot_run(monkeypatch):
    enabled = []
    monkeypatch.setattr(native, "load", lambda device, specs: None)
    monkeypatch.setattr(T, "enable_int8_attention", lambda device: enabled.append(True))
    monkeypatch.setattr(sampler, "compiled_block_forward", lambda device: "compiled block")
    monkeypatch.setattr(sampler, "compile_static", lambda function, mode, device, dtype: ("compiled", function))
    assert sampler._block_functions(torch.device("cuda"), {}) == (
        "compiled block", ("compiled", T.prompt_key_value), "torch.compile")
    assert enabled


def _shipped_manifest(directory: Path, profile: str, entries: list[dict]) -> Path:
    manifest = directory / "kernels" / f"{profile}-0123456789abcdef" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps(entries))
    return manifest


def _quantizer(family: str, **constexprs) -> dict:
    return {"family": family, "constexprs": constexprs}


def test_native_block_without_triton_needs_shipped_kernels(monkeypatch, tmp_path):
    _amd_architecture(monkeypatch)
    monkeypatch.setattr(native, "PREBUILT", tmp_path)
    monkeypatch.setattr(native, "_prebuilt_extension", lambda: tmp_path / "jasna_ltx_native.so")
    monkeypatch.setattr(native, "has_triton", lambda: False)
    assert "Triton" in native.unavailable_reason(torch.device("cuda"), _block_specs("int8"))
    manifest = _shipped_manifest(tmp_path, "gfx1201-int8", [
        _quantizer("rotate_quantize", K=4096, MODULATE=1), _quantizer("rotate_quantize", K=16384, MODULATE=0),
        _quantizer("gated_rotate_quantize", K=4096, HEAD_DIM=128), _quantizer("mm", BM=128, BN=256, BK=128)])
    assert native.unavailable_reason(torch.device("cuda"), _block_specs("int8")) is None
    assert native._kernel_manifest("gfx1201", aot.BlockDims(4096, 16384, 32), fp4=False) == manifest


def test_shipped_kernels_must_fit_the_models_widths_and_gpu(monkeypatch, tmp_path):
    _amd_architecture(monkeypatch, "gfx1100")
    monkeypatch.setattr(native, "PREBUILT", tmp_path)
    _shipped_manifest(tmp_path, "gfx1201-int8", [_quantizer("rotate_quantize", K=4096, MODULATE=1)])
    _shipped_manifest(tmp_path, "gfx1100-int8", [
        _quantizer("rotate_quantize", K=2048, MODULATE=1), _quantizer("rotate_quantize", K=16384, MODULATE=0),
        _quantizer("gated_rotate_quantize", K=2048, HEAD_DIM=64)])
    assert "no shipped kernels for gfx1100" in native.unavailable_reason(torch.device("cuda"), _block_specs("int8"))
    with pytest.raises(RuntimeError, match="no shipped native-block kernels"):
        native._kernel_manifest("gfx1100", aot.BlockDims(4096, 16384, 32), fp4=False)


@pytest.mark.parametrize("frozen,windows,expected", [
    (False, False, native._HERE / "prebuilt"),
    (True, True, native._HERE / "prebuilt"),
    pytest.param(True, False, Path("/opt/dist/ltx/native/prebuilt"),
                 marks=pytest.mark.skipif(sys.platform == "win32", reason="resolves a POSIX executable path")),
])
def test_frozen_linux_release_ships_the_prebuilt_block_beside_its_executable(monkeypatch, frozen, windows, expected):
    monkeypatch.setattr(native, "is_frozen", lambda: frozen)
    monkeypatch.setattr(native, "_IS_WINDOWS", windows)
    monkeypatch.setattr(native.sys, "executable", "/opt/dist/jasna")
    assert native._prebuilt_directory() == expected


def test_native_block_and_its_policy_import_without_triton():
    blocker = "import sys; sys.modules['triton'] = None; import jasna.ltx.native, jasna.ltx.kernel_policy, jasna.ltx.sampler"
    subprocess.run([sys.executable, "-c", blocker], check=True, cwd=native._HERE.parents[2])


@pytest.mark.parametrize("architecture", ["gfx1103", "gfx1150", "gfx1151"])
def test_amd_integrated_gpus_ship_tuned_kernels(monkeypatch, architecture):
    _amd_architecture(monkeypatch, architecture)
    assert architecture in native.release_architectures()
    assert kernel_policy.supports(torch.device("cuda"))


def test_gfx1152_has_no_tuned_kernels_to_ship_or_run(monkeypatch):
    _amd_architecture(monkeypatch, "gfx1152")
    assert "gfx1152" not in native.release_architectures()
    assert "no tuned kernels for gfx1152" in native.unavailable_reason(torch.device("cuda"), _block_specs("int8"))


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


@pytest.mark.parametrize("windows", [False, True])
def test_prebuild_retains_a_loaded_windows_extension_until_its_process_exits(monkeypatch, tmp_path, windows):
    monkeypatch.setattr(native, "_IS_WINDOWS", windows)
    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(native, "cuda_arch_list", lambda: "12.0")

    def build(directory):
        directory.mkdir()
        extension = directory / ("jasna_ltx_native.pyd" if windows else "jasna_ltx_native.so")
        extension.write_bytes(b"compiled extension")
        return SimpleNamespace(__file__=str(extension))

    monkeypatch.setattr(native, "_build_extension", build)
    native.prebuild_extension(tmp_path)
    assert (tmp_path / "build").exists() is windows
    assert len(list(tmp_path.glob("jasna_ltx_native.*"))) == 1


def _amd_architecture(monkeypatch, architecture="gfx1201"):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (12, 0))
    monkeypatch.setattr(torch.version, "cuda", None)
    monkeypatch.setattr(torch.version, "hip", "7.15")
    monkeypatch.setattr(kernel_policy, "amd_gpu_architecture", lambda device: architecture)


@pytest.mark.parametrize("architecture", sorted(kernel_policy.AMD_INT8_MM_TILES))
def test_amd_kernel_variants_are_the_amd_torch_paths_triton_kernels(architecture):
    found = aot.variants(architecture, aot.BlockDims(dim=4096, ff_dim=16384, heads=32), fp4=False)
    assert {v.family for v in found} == {"rotate_quantize", "gated_rotate_quantize", "mm"}
    gated = [dict(v.constexprs) for v in found if v.family == "gated_rotate_quantize"]
    assert gated == [{"K": 4096, "HEAD_DIM": 128}]
    gemm_tiles = {(dict(v.constexprs)["BM"], dict(v.constexprs)["BN"], dict(v.constexprs)["BK"], v.num_stages, v.num_warps)
                  for v in found if v.family == "mm"}
    assert gemm_tiles == {*kernel_policy.AMD_INT8_MM_TILES[architecture], kernel_policy.SMALL_MM_TILE}
    # GEMMs carry the JIT launch's options (no fp fusion under ROCm), the quantizers none
    assert all(v.options == tuple(kernels.GEMM_OPTIONS.items()) for v in found if v.family == "mm")
    assert all(v.options == () for v in found if v.family != "mm")


def test_amd_kernels_build_only_under_a_rocm_triton(monkeypatch, tmp_path):
    import triton.language as tl

    monkeypatch.setattr(kernels, "HIP", tl.constexpr(False))
    variant = aot.variants("gfx1201", aot.BlockDims(4096, 16384, 32), fp4=False)[0]
    with pytest.raises(RuntimeError, match="ROCm"):
        aot._compile(variant, "gfx1201", tmp_path)


def test_amd_kernel_profiles_are_named_by_gfx_architecture():
    assert kernel_policy.profile_name("gfx1201", fp4=False) == "gfx1201-int8"
    assert kernel_policy.profile_name((12, 0), fp4=True) == "sm120-nvfp4"


def test_amd_native_block_launches_comfy_kitchens_hip_attention_tile(monkeypatch, tmp_path):
    _amd_architecture(monkeypatch)
    monkeypatch.setattr(kernel_policy, "programs", lambda device: 64)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("[]")
    calls = []

    class Block:
        def __init__(self, config):
            self.config = config

        def forward(self, *args):
            calls.append(args)

    module = SimpleNamespace(Config=SimpleNamespace, Block=Block)
    forward = native.NativeBlockForward(module, torch.device("cuda", 0), aot.BlockDims(4096, 16384, 32), manifest)
    assert (tuple(forward.block.config.narrow_tile), tuple(forward.block.config.wide_tile)) == (
        kernel_policy.AMD_INT8_MM_TILES["gfx1201"])
    weights = {"attn1.q_norm.weight": None, "attn1.k_norm.weight": None, "attn2.q_norm.weight": None,
               "scale_shift_table": None}
    weights |= {f"{name}.qdata": None for name in T.LINEAR_NAMES} | {f"{name}.scales": None for name in T.LINEAR_NAMES}
    cond = SimpleNamespace(block_rows=None)
    rope = SimpleNamespace(cos=None, sin=None)
    forward(weights, torch.zeros(1, 8192, 1), cond, (None, None), rope)
    assert calls[0][-2:] == (kernel_policy.AMD_ATTENTION_KEY_TILE, False)


@pytest.mark.parametrize("quantization, architecture, reason", [
    ("nvfp4", "gfx1201", "NVFP4 blocks need SM 12.0 / 12.1"),
    ("int8", "gfx90a", "no tuned kernels for gfx90a"),
])
def test_amd_native_block_refuses_what_it_has_no_kernels_for(monkeypatch, quantization, architecture, reason):
    _amd_architecture(monkeypatch, architecture)
    assert reason in native.unavailable_reason(torch.device("cuda"), _block_specs(quantization))


def test_amd_source_build_needs_hipcc(monkeypatch, tmp_path):
    from torch.utils import cpp_extension

    _amd_architecture(monkeypatch)
    monkeypatch.setattr(native, "_prebuilt_extension", lambda: None)
    monkeypatch.setattr(native, "is_frozen", lambda: False)
    monkeypatch.setattr(native.shutil, "which", lambda name: name)
    monkeypatch.setattr(native, "_ninja_path", lambda: "ninja path")
    monkeypatch.setattr(cpp_extension, "ROCM_HOME", None)
    assert "hipcc" in native.unavailable_reason(torch.device("cuda"), _block_specs("int8"))
    monkeypatch.setattr(cpp_extension, "ROCM_HOME", str(tmp_path))
    assert "hipcc" in native.unavailable_reason(torch.device("cuda"), _block_specs("int8"))
    native._hipcc(tmp_path).parent.mkdir()
    native._hipcc(tmp_path).touch()
    assert native.unavailable_reason(torch.device("cuda"), _block_specs("int8")) is None


def test_hip_targets_follow_pytorch_rocm_arch_else_this_gpu(monkeypatch):
    monkeypatch.setattr(kernel_policy, "gpu_architecture", lambda device: "gfx1201")
    monkeypatch.delenv("PYTORCH_ROCM_ARCH", raising=False)
    assert native.hip_architectures() == ["gfx1201"]
    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx1100;gfx1101 gfx1201")
    assert native.hip_architectures() == ["gfx1100", "gfx1101", "gfx1201"]


def test_hip_objects_compile_glue_and_attention_once_with_their_own_flags(monkeypatch, tmp_path):
    from torch.utils import cpp_extension

    commands = []

    def compile_object(command, check):
        commands.append(command)
        Path(command[-1]).write_bytes(b"object")

    monkeypatch.setattr(native.subprocess, "run", compile_object)
    monkeypatch.setattr(native, "_IS_WINDOWS", False)
    monkeypatch.setattr(cpp_extension, "include_paths", lambda device_type: ["/torch/include"])
    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx1100;gfx1201")
    objects = native._hip_objects(tmp_path, PurePosixPath("/rocm"))
    assert [path.name for path in objects] == ["glue.o", "norm_rope_quant.o", "quant_v_int8.o", "int8_attn.o"]
    assert [command[-3] for command in commands] == [
        str(native._HERE / "glue.cu"), *(str(source) for source in native._HIP_ATTENTION_SOURCES)]
    glue, *attention = commands
    assert glue[0] == "/rocm/bin/hipcc" and {"-x", "hip", "-I/torch/include", "-DUSE_ROCM=1"} <= set(glue)
    assert "-ffast-math" not in glue and all("-ffast-math" in command for command in attention)
    assert all({"--offload-arch=gfx1100", "--offload-arch=gfx1201", "-mno-wavefrontsize64", "-fPIC"} <= set(command)
               for command in commands)
    assert native._hip_objects(tmp_path, PurePosixPath("/rocm")) == objects and len(commands) == 4

    monkeypatch.setenv("PYTORCH_ROCM_ARCH", "gfx1201")
    assert native._hip_objects(tmp_path, PurePosixPath("/rocm"))[0].parent != objects[0].parent


def test_amd_extension_builds_host_sources_only_so_nothing_is_hipified(monkeypatch, tmp_path):
    from torch.utils import cpp_extension

    objects = [tmp_path / "glue.o", tmp_path / "int8_attn.o"]
    monkeypatch.setattr(torch.version, "hip", "7.15")
    monkeypatch.setattr(native, "_ninja_path", lambda: "ninja path")
    monkeypatch.setattr(native, "_IS_WINDOWS", False)
    monkeypatch.setattr(cpp_extension, "ROCM_HOME", "/rocm")
    monkeypatch.setattr(cpp_extension, "library_paths", lambda device_type: ["/torch/lib"])
    monkeypatch.setattr(native, "_hip_objects", lambda build_directory, rocm_home: objects)
    monkeypatch.setattr(cpp_extension, "load", lambda **kwargs: kwargs)
    built = native._build_extension(tmp_path / "build")
    assert built["sources"] == [str(native._HERE / "module.cpp"), str(native._HERE / "block.cpp")]
    assert built["extra_ldflags"] == ["-L/torch/lib", "-lamdhip64", "-lc10_hip", "-ltorch_hip", *map(str, objects)]
    assert built["extra_include_paths"] == [str(Path("/rocm") / "include")]
