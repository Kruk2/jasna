// The CUDA and HIP spellings of what the native block takes from PyTorch's GPU headers, the GPU
// driver and the device headers, so block.cpp and glue.cu compile unchanged with nvcc and with
// hipcc / a ROCm host compiler (USE_ROCM, set by PyTorch's ROCm extension flags). ROCm PyTorch
// keeps the CUDA names (at::cuda, c10::cuda, C10_CUDA_CHECK) under its HIP header paths.
#pragma once

#ifdef USE_ROCM
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPGuard.h>
#include <hip/hip_runtime_api.h>
#else
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#ifdef USE_ROCM
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#else
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#endif
#endif

namespace jasna_ltx::gpu {

#ifdef USE_ROCM
using Module = hipModule_t;
using Function = hipFunction_t;
using Result = hipError_t;
using Stream = hipStream_t;
constexpr Result kSuccess = hipSuccess;

inline const char* error_string(Result result) { return hipGetErrorString(result); }
inline Result module_load_data(Module* module, const void* image) { return hipModuleLoadData(module, image); }
inline Result module_load(Module* module, const char* path) { return hipModuleLoad(module, path); }
inline Result module_get_function(Function* function, Module module, const char* name) {
  return hipModuleGetFunction(function, module, name);
}
inline Result module_unload(Module module) { return hipModuleUnload(module); }
inline Result launch(Function function, unsigned x, unsigned y, unsigned z, unsigned threads, unsigned shared,
                     Stream stream, void** params) {
  return hipModuleLaunchKernel(function, x, y, z, threads, 1, 1, shared, stream, params, nullptr);
}
// Makes the device's primary context current for the module API.
inline void bind_primary_context() { C10_CUDA_CHECK(hipFree(nullptr)); }
#else
using Module = CUmodule;
using Function = CUfunction;
using Result = CUresult;
using Stream = cudaStream_t;
constexpr Result kSuccess = CUDA_SUCCESS;

inline const char* error_string(Result result) {
  const char* message = nullptr;
  cuGetErrorString(result, &message);
  return message ? message : "unknown CUDA error";
}
inline Result module_load_data(Module* module, const void* image) { return cuModuleLoadData(module, image); }
inline Result module_load(Module* module, const char* path) { return cuModuleLoad(module, path); }
inline Result module_get_function(Function* function, Module module, const char* name) {
  return cuModuleGetFunction(function, module, name);
}
inline Result module_unload(Module module) { return cuModuleUnload(module); }
inline Result launch(Function function, unsigned x, unsigned y, unsigned z, unsigned threads, unsigned shared,
                     Stream stream, void** params) {
  return cuLaunchKernel(function, x, y, z, threads, 1, 1, shared, stream, params, nullptr);
}
inline void bind_primary_context() { C10_CUDA_CHECK(cudaFree(nullptr)); }
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#ifdef USE_ROCM
using bf16 = __hip_bfloat16;
__device__ __forceinline__ bf16 to_bf16(float x) { return __float2bfloat16(x); }  // round to nearest even
// HIP's *_sync shuffles demand a 64-bit lane mask; the kernels shuffle whole 32-lane waves.
__device__ __forceinline__ float shuffle_xor(float value, int offset) { return __shfl_xor(value, offset); }
#else
using bf16 = __nv_bfloat16;
__device__ __forceinline__ bf16 to_bf16(float x) { return __float2bfloat16_rn(x); }
__device__ __forceinline__ float shuffle_xor(float value, int offset) {
  return __shfl_xor_sync(0xffffffffu, value, offset);
}
#endif
#endif

}  // namespace jasna_ltx::gpu
