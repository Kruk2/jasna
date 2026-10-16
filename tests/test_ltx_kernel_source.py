"""CPU-only checks of the production kernel loader with an owned fake CUDA driver."""

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import uuid

import pytest


NATIVE = Path(__file__).resolve().parents[1] / "jasna/ltx/native"


def _kernel_body(source):
    start = source.index("class Kernel {")
    end = source.index("\n};", start) + len("\n};")
    return source[start:end]


SUPPORT = r'''
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#define TORCH_CHECK(condition, ...) do { if (!(condition)) throw std::runtime_error("check"); } while (0)
using CUmodule = void*;
using CUfunction = void*;
using CUresult = int;
using CUdevice = int;
constexpr int CUDA_SUCCESS = 0;
constexpr int CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 1;
constexpr int CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 2;
constexpr int CU_FUNC_CACHE_PREFER_SHARED = 3;
constexpr int CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 4;
static int path_loads, image_loads, unloads, live_modules, calls, fail_at, shared_value;
static bool unload_failure;
static int python_reads, contexts;
static std::vector<unsigned char> seen_image;
static int fail() { return ++calls == fail_at ? 1 : 0; }
int cuGetErrorString(int, const char** s) { *s = "owned failure"; return 0; }
int cuModuleLoad(CUmodule* m, const char* path) {
    ++path_loads;
    if (std::string(path) != "owned/kernel.cubin") throw std::runtime_error("wrong path");
    if (fail()) return 1;
    *m = reinterpret_cast<void*>(1); ++live_modules; return 0;
}
int cuModuleLoadData(CUmodule* m, const void* data) {
    ++image_loads;
    const auto* p = static_cast<const unsigned char*>(data);
    seen_image.assign(p, p + 4);
    if (fail()) return 1;
    *m = reinterpret_cast<void*>(1); ++live_modules; return 0;
}
int cuModuleUnload(CUmodule m) { if (!m) throw std::runtime_error("double unload"); ++unloads; --live_modules; return unload_failure ? 1 : 0; }
int cuModuleGetFunction(CUfunction* f, CUmodule, const char*) { if (fail()) return 1; *f = reinterpret_cast<void*>(2); return 0; }
int cuCtxGetDevice(CUdevice* p) { *p = 0; return fail(); }
int cuDeviceGetAttribute(int* p, int, CUdevice) { *p = 98304; return fail(); }
int cuFuncGetAttribute(int* p, int, CUfunction) { *p = 0; return fail(); }
int cuFuncSetCacheConfig(CUfunction, int) { return fail(); }
int cuFuncSetAttribute(CUfunction, int, int) { return fail(); }
void check_cu(CUresult code, const char*) { if (code) throw std::runtime_error("original CUDA error"); }
'''


LEGACY_SUPPORT = r'''
namespace py {
struct value;
using list = std::vector<value>;
struct tuple { value operator[](size_t) const; };
struct value {
    std::string name;
    template<class T> T cast() const {
        ++python_reads;
        if constexpr (std::is_same_v<T, std::string>) return name == "cubin" ? "kernel.cubin" : "kernel";
        else if constexpr (std::is_same_v<T, int>) return name == "shared" ? shared_value : 4;
        else if constexpr (std::is_same_v<T, std::vector<std::string>>) return {"*bf16"};
        else if constexpr (std::is_same_v<T, std::vector<bool>>) return {false};
        else return {};
    }
};
value tuple::operator[](size_t) const { return {}; }
struct dict { value operator[](const char* key) const { return {key}; } };
}
struct TensorDescriptorLayout { int swizzle, elem_size, elem_type; std::vector<int64_t> block; };
'''


@pytest.fixture(scope="module")
def kernel_fixture(record_testsuite_property):
    directory = Path(tempfile.gettempdir()) / f"owned-ltx-kernel-source-{uuid.uuid4().hex}"
    directory.mkdir()
    production = (NATIVE / "block.cpp").read_text()
    modern = (NATIVE / "block_kernels.h").is_file()
    if modern:
        owner_start = production.index("class ModuleOwner {")
        owner_end = production.index("\n};", owner_start) + len("\n};")
        config_header = (NATIVE / "block.h").read_text()
        config_start = config_header.index("struct Config {")
        config_end = config_header.index("\n};", config_start) + len("\n};")
        impl_start = production.index("class Block::Impl {")
        impl_end = production.index("  at::Tensor forward(", impl_start)
        loading_impl = production[impl_start:impl_end].replace("class Block::Impl {", "class LoadingImpl {")
        loading_impl += "  Config config_;\n  std::map<std::string, std::unique_ptr<Kernel>> kernels_;\n};\n"
        loading_impl = loading_impl.replace("Impl(", "LoadingImpl(")
        body = '#include "block_kernels.h"\nusing namespace jasna_ltx;\n' + production[owner_start:owner_end] + "\n" + _kernel_body(production)
        body += LEGACY_SUPPORT.replace("struct TensorDescriptorLayout { int swizzle, elem_size, elem_type; std::vector<int64_t> block; };", "")
        body += r'''
namespace pybind11 = py;
namespace c10 { using DeviceIndex = int; namespace cuda { struct CUDAGuard { explicit CUDAGuard(int) {} }; } }
int cudaFree(void*) { ++contexts; return 0; }
#define C10_CUDA_CHECK(code) check_cu(code, "context")
'''
        body += config_header[config_start:config_end] + "\n" + loading_impl
        construction = r'''
            KernelSpec spec;
            spec.family = "family"; spec.function = "kernel"; spec.shared = shared_value; spec.warps = 4;
            spec.args = {"*bf16"}; spec.divisible = {false};
            if (mode == "empty") { Kernel k(spec, KernelSource::from_image({})); }
            else if (mode == "missing") { spec.function.clear(); Kernel k(spec, KernelSource::from_image({1,2,3,4})); }
            else if (mode == "mismatch") { spec.divisible.clear(); Kernel k(spec, KernelSource::from_image({1,2,3,4})); }
            else if (mode == "image" || mode == "image_unload_error") {
                unload_failure = mode == "image_unload_error";
                Kernel k(spec, KernelSource::from_image({1,2,3,4}));
            }
            else if (mode == "owned") {
                std::vector<unsigned char> bytes = {1,2,3,4};
                auto source = KernelSource::from_image(bytes);
                bytes.assign(100, 0);
                if (source.image() != std::vector<unsigned char>({1,2,3,4})) throw std::runtime_error("not owned");
                Kernel k(spec, std::move(source));
            }
            else if (mode == "moved_source") {
                auto original = KernelSource::from_image({1,2,3,4});
                auto held = std::move(original);
                Kernel k(spec, std::move(original));
            }
            else if (mode == "batch_empty") { validate_kernel_images({}); }
            else if (mode == "duplicate") { validate_kernel_images({{spec, {1,2,3,4}}, {spec, {1,2,3,4}}}); }
            else if (mode == "distinct") {
                auto other = spec; other.constexprs["K"] = 4096;
                validate_kernel_images({{spec, {1,2,3,4}}, {other, {1,2,3,4}}});
            }
            else if (mode == "batch_bad_later") { validate_kernel_images({{spec, {1,2,3,4}}, {spec, {}}}); }
            else if (mode == "warps0") { spec.warps = 0; validate_kernel_images({{spec,{1,2,3,4}}}); }
            else if (mode == "warps33") { spec.warps = 33; validate_kernel_images({{spec,{1,2,3,4}}}); }
            else if (mode == "negative_shared") { spec.shared = -1; validate_kernel_images({{spec,{1,2,3,4}}}); }
            else if (mode == "empty_arg") { spec.args[0].clear(); validate_kernel_images({{spec,{1,2,3,4}}}); }
            else if (mode == "nul_arg") { spec.args[0] = std::string("a\0b",3); validate_kernel_images({{spec,{1,2,3,4}}}); }
            else if (mode == "empty_args") { spec.args.clear(); spec.divisible.clear(); validate_kernel_images({{spec,{1,2,3,4}}}); }
            else if (mode == "missing_descriptor") { spec.args[0] = "nvTmaDesc"; validate_kernel_images({{spec,{1,2,3,4}}}); }
            else if (mode == "extra_descriptor") { spec.tensordesc.push_back({0,2,0,{16,32}}); validate_kernel_images({{spec,{1,2,3,4}}}); }
            else if (mode == "bad_descriptor") { spec.args[0] = "nvTmaDesc"; spec.tensordesc.push_back({0,2,0,{16}}); validate_kernel_images({{spec,{1,2,3,4}}}); }
            else if (mode == "zero_descriptor") { spec.args[0] = "nvTmaDesc"; spec.tensordesc.push_back({0,2,0,{16,0}}); validate_kernel_images({{spec,{1,2,3,4}}}); }
            else if (mode == "bad_family") { spec.family = "f|a=2"; validate_kernel_images({{spec,{1,2,3,4}}}); }
            else if (mode == "bad_constant") { spec.constexprs["a|b"] = 1; validate_kernel_images({{spec,{1,2,3,4}}}); }
            else if (mode == "path_nul") { KernelSource::from_path(std::string("a\0b",3)); }
            else if (mode == "path_empty") { KernelSource::from_path({}); }
            else if (mode == "impl_image" || mode == "impl_empty" || mode == "impl_duplicate") {
                Config config;
                config.narrow_tile = config.wide_tile = config.small_tile = {1,2,3,4,5};
                config.epilogue_tile = {1,2};
                config.kernel_directory = "forbidden";
                config.kernels.push_back({"forbidden"});
                std::vector<KernelImage> images;
                if (mode != "impl_empty") images.push_back({spec, {1,2,3,4}});
                if (mode == "impl_duplicate") images.push_back({spec, {1,2,3,4}});
                LoadingImpl loaded(config, std::move(images));
                if (python_reads) throw std::runtime_error("native image consulted Python");
            }
            else if (mode == "impl_path") {
                Config config;
                config.narrow_tile = config.wide_tile = config.small_tile = {1,2,3,4,5};
                config.epilogue_tile = {1,2}; config.kernel_directory = "owned";
                config.kernels.push_back({});
                LoadingImpl loaded(config);
                if (!python_reads) throw std::runtime_error("path adapter did not read config");
            }
            else { Kernel k(spec, KernelSource::from_path("owned/kernel.cubin")); }
        '''
    else:
        body = LEGACY_SUPPORT + _kernel_body(production)
        construction = 'Kernel k(py::dict{}, "owned");'
    harness = SUPPORT + body + r'''
int main(int argc, char** argv) {
    const std::string mode = argv[1];
    fail_at = std::stoi(argv[2]); shared_value = std::stoi(argv[3]);
    bool rejected = false;
    std::string error;
    try {
    ''' + construction + r'''
    } catch (const std::exception& e) { rejected = true; error = e.what(); }
    std::cout << "rejected=" << rejected << " path=" << path_loads << " image=" << image_loads
              << " unload=" << unloads << " live=" << live_modules << " bytes=" << seen_image.size()
              << " original=" << (error == "original CUDA error") << "\n";
    if (mode.compare(0,5,"impl_") == 0) std::cout << "python=" << python_reads << " contexts=" << contexts << "\n";
}
'''
    source = directory / "fixture.cpp"
    source.write_text(harness)
    compiler = Path(os.environ.get("CXX") or shutil.which("cl") or shutil.which("clang++") or shutil.which("g++") or "")
    if not compiler.is_file():
        pytest.skip("C++17 compiler unavailable")
    binary = directory / ("fixture.exe" if os.name == "nt" else "fixture")
    if compiler.name.casefold() in ("cl", "cl.exe"):
        command = [str(compiler), "/nologo", "/std:c++17", "/EHsc", "/O2", f"/I{NATIVE}", str(source), f"/Fe{binary}"]
    else:
        command = [str(compiler), "-std=c++17", "-O2", f"-I{NATIVE}", str(source), "-o", str(binary)]
    result = subprocess.run(command, cwd=directory, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = {"command": command, "compiler_sha256": hashlib.sha256(compiler.read_bytes()).hexdigest(),
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, NATIVE / "block.cpp", NATIVE / "block.h"]}}
    if modern:
        evidence["inputs"][str(NATIVE / "block_kernels.h")] = hashlib.sha256((NATIVE / "block_kernels.h").read_bytes()).hexdigest()
    (directory / "compile-evidence.json").write_text(json.dumps(evidence, indent=2))
    record_testsuite_property("kernel_fixture_directory", str(directory))
    record_testsuite_property("kernel_fixture_binary_sha256", evidence["binary_sha256"])
    return binary


def _run(binary, mode, fail_at=0, shared=0):
    result = subprocess.run([str(binary), mode, str(fail_at), str(shared)], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stdout + result.stderr
    return {key: int(value) for key, value in (pair.split("=") for pair in result.stdout.strip().split())}


@pytest.mark.parametrize("failure", range(2, 8))
def test_constructor_failure_unloads_module_once_and_preserves_error(kernel_fixture, failure):
    result = _run(kernel_fixture, "path", failure, 65536)
    assert result["rejected"] == result["original"] == 1
    assert result["unload"] == 1
    assert result["live"] == 0


def test_path_route_unchanged(kernel_fixture):
    assert _run(kernel_fixture, "path") == {"rejected": 0, "path": 1, "image": 0, "unload": 1, "live": 0, "bytes": 0, "original": 0}


def test_image_route_never_opens_path(kernel_fixture):
    result = _run(kernel_fixture, "image")
    assert result == {"rejected": 0, "path": 0, "image": 1, "unload": 1, "live": 0, "bytes": 4, "original": 0}


@pytest.mark.parametrize("failure", range(1, 8))
def test_image_setup_failure_one_shot_cleanup(kernel_fixture, failure):
    result = _run(kernel_fixture, "image", failure, 65536)
    assert result["rejected"] == result["original"] == 1
    assert result["path"] == 0
    assert result["unload"] == (0 if failure == 1 else 1)
    assert result["live"] == 0


def test_owned_image_survives_caller_buffer_changes(kernel_fixture):
    result = _run(kernel_fixture, "owned")
    assert result["rejected"] == result["path"] == result["live"] == 0
    assert result["image"] == 1 and result["bytes"] == 4


def test_unload_failure_does_not_mask_constructor_error_or_retry(kernel_fixture):
    result = _run(kernel_fixture, "image_unload_error", 2)
    assert result["original"] == result["rejected"] == result["unload"] == 1
    assert result["path"] == result["live"] == 0


@pytest.mark.parametrize("mode", ["batch_empty", "duplicate", "batch_bad_later", "warps0", "warps33",
    "negative_shared", "empty_arg", "nul_arg", "empty_args", "missing_descriptor", "extra_descriptor",
    "bad_descriptor", "zero_descriptor", "bad_family", "bad_constant", "path_nul", "path_empty", "moved_source"])
def test_invalid_batch_or_metadata_precedes_cuda(kernel_fixture, mode):
    result = _run(kernel_fixture, mode)
    assert result["rejected"] == 1
    assert result["path"] == result["image"] == result["live"] == result["unload"] == 0


def test_distinct_constexpr_identities_allowed(kernel_fixture):
    result = _run(kernel_fixture, "distinct")
    assert result["rejected"] == result["path"] == result["image"] == result["live"] == 0


def test_shared_memory_limit_failure_releases_loaded_module(kernel_fixture):
    result = _run(kernel_fixture, "image", shared=131072)
    assert result["rejected"] == result["unload"] == 1
    assert result["live"] == result["path"] == 0


@pytest.mark.parametrize("mode", ["empty", "missing", "mismatch"])
def test_invalid_image_rejected_before_driver(kernel_fixture, mode):
    result = _run(kernel_fixture, mode)
    assert result["rejected"] == 1
    assert result["path"] == result["image"] == result["live"] == 0


def test_explicit_native_api_present():
    assert "Block(const Config& config, std::vector<KernelImage> kernels)" in (NATIVE / "block.h").read_text()


def test_native_image_batch_never_reads_python_config():
    source = (NATIVE / "block.cpp").read_text()
    start = source.index("static std::vector<KernelInput> image_kernels")
    end = source.index("\n  Impl(", start)
    route = source[start:end]
    assert "validate_kernel_images(kernels)" in route
    assert "KernelSource::from_image" in route
    assert "kernel_directory" not in route and "config.kernels" not in route and "from_path" not in route
    assert "KernelSource source_;" in _kernel_body(source)


def test_inference_and_launch_body_not_duplicated():
    source = (NATIVE / "block.cpp").read_text()
    assert source.count("class Kernel {") == source.count("class Launch {") == 1
    assert source.count("at::Tensor forward(") == 1


def test_actual_image_impl_does_not_consult_python_or_path(kernel_fixture):
    result = _run(kernel_fixture, "impl_image")
    assert result["rejected"] == result["python"] == result["path"] == result["live"] == 0
    assert result["image"] == result["unload"] == result["contexts"] == 1


@pytest.mark.parametrize("mode", ["impl_empty", "impl_duplicate"])
def test_actual_invalid_image_batch_rejected_before_context(kernel_fixture, mode):
    result = _run(kernel_fixture, mode)
    assert result["rejected"] == 1
    assert result["python"] == result["path"] == result["image"] == result["contexts"] == result["live"] == 0


def test_actual_plaintext_impl_adapter_remains_used(kernel_fixture):
    result = _run(kernel_fixture, "impl_path")
    assert result["rejected"] == result["image"] == result["live"] == 0
    assert result["python"] > 0
    assert result["path"] == result["unload"] == result["contexts"] == 1
