/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Modified for Jasna: only the helpers of the vendored INT8 attention are kept (see NOTICE).
 */
#ifndef COMFY_FLOAT_UTILS_CUH_
#define COMFY_FLOAT_UTILS_CUH_

#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace comfy {

__device__ __forceinline__ float warp_reduce_fmax(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
  return v;
}

__device__ __forceinline__ int8_t float_to_int8_rn(float value) {
  int32_t quantized;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;"
               : "=r"(quantized)
               : "f"(value));
  return static_cast<int8_t>(quantized);
}

__device__ __forceinline__ int8_t quant_int8_rcp(float v,
                                                 float inv_scale) {
  return float_to_int8_rn(v * inv_scale);
}

// ptr must be 4-byte aligned. Callers must preserve that alignment at every
// row offset, for example by requiring the row stride to be a multiple of 4.
__device__ __forceinline__ void store4_i8(int8_t *ptr, int8_t a, int8_t b,
                                          int8_t c, int8_t d) {
  *reinterpret_cast<int32_t *>(ptr) =
      (uint32_t)(uint8_t)a | ((uint32_t)(uint8_t)b << 8) |
      ((uint32_t)(uint8_t)c << 16) | ((uint32_t)(uint8_t)d << 24);
}

// Vectorized half-precision loads
#pragma nv_diag_suppress 1056
template<typename IType>
    __forceinline__ __device__ const IType* load_f16x8(const IType* val) {
    float4 vals = *reinterpret_cast<const float4*>(val);
    return reinterpret_cast<const IType*>(&vals);
}
#pragma nv_diag_default 1056

} // namespace comfy

#endif // COMFY_FLOAT_UTILS_CUH_
