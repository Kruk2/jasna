// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// Modified for Jasna: bf16, unmasked paths only; launchers in jasna_ltx::sage_attention (see NOTICE).
//
// Fused V -> signed INT8 quantization with per-channel scaling.
// Output layout [B,H,D,padded_N] with INT8 MMA 16-element permutation and
// per-row scale.
//
// One thread block per (b, h, d_tile) with D_TILE=8 d-channels. Short K uses
// 128 threads so it does not launch idle warps; longer K uses 512 threads.
// Threads cooperate along N with vectorized 128-bit loads, warp-shuffle absmax
// reduction, and reverse Pass 2 iteration for L2 cache reuse. Pass 1 uses 4×
// manual unroll for memory-level parallelism at low occupancy.
//
// Pass 2 iterates over linear source indices (coalesced 128-bit reads) and
// applies the inverse permutation to compute the destination write index,
// avoiding the scattered-read pattern of iterating over destination indices.

#include "float_utils.cuh"
#include "launchers.h"

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

constexpr int kDTile = 8;

// INT8 MMA 16-element inverse permutation.
// Forward: fwd(j) maps bit pattern {b3,b2,b1,b0} → {b1,b3,b2,b0}.
// Inverse: inv(s) maps {b3,b2,b1,b0} → {b2,b0,b3,b1}.
__device__ __forceinline__ int inv_perm16(int w) {
  return (w & 1) | (((w >> 3) & 1) << 1) | (((w >> 1) & 1) << 2) |
         (((w >> 2) & 1) << 3);
}

template <typename T>
__device__ __forceinline__ void load_tile(const T *ptr, float *out_vals) {
  if constexpr (std::is_same_v<T, float>) {
    float4 v0 = *reinterpret_cast<const float4 *>(ptr);
    float4 v1 = *reinterpret_cast<const float4 *>(ptr + 4);
    out_vals[0] = v0.x;
    out_vals[1] = v0.y;
    out_vals[2] = v0.z;
    out_vals[3] = v0.w;
    out_vals[4] = v1.x;
    out_vals[5] = v1.y;
    out_vals[6] = v1.z;
    out_vals[7] = v1.w;
  } else {
    const T *loaded = comfy::load_f16x8(ptr);
#pragma unroll
    for (int i = 0; i < kDTile; ++i) {
      out_vals[i] = static_cast<float>(loaded[i]);
    }
  }
}

template <typename T, int THREADS>
__global__ void
quant_v_int8_kernel(const T *__restrict__ v, int8_t *__restrict__ out,
                   float *__restrict__ scale_out, int N, int padded_N, int H,
                   int D, int64_t sb, int64_t sh, int64_t sn) {
  const int d_tiles = D / kDTile;
  const int d_tile = blockIdx.x % d_tiles;
  const int bh = blockIdx.x / d_tiles;
  const int h = bh % H;
  const int b = bh / H;
  const int d0 = d_tile * kDTile;

  const T *base = v + b * sb + h * sh + d0;
  constexpr int WARPS = THREADS / 32;

  // ── Pass 1: per-channel absmax (threads cooperate over N) ──────────
  float mx[kDTile];
#pragma unroll
  for (int i = 0; i < kDTile; ++i)
    mx[i] = 0.f;

  // 4× unrolled: issue 4 independent 128-bit loads per iteration
  // so the memory controller can overlap them (critical at ≤50% occupancy).
  int n = threadIdx.x;
  const int N_body = N - 3 * THREADS;
  for (; n < N_body; n += 4 * THREADS) {
    float t0[kDTile], t1[kDTile], t2[kDTile], t3[kDTile];
    load_tile(base + (int64_t)n * sn, t0);
    load_tile(base + (int64_t)(n + THREADS) * sn, t1);
    load_tile(base + (int64_t)(n + 2 * THREADS) * sn, t2);
    load_tile(base + (int64_t)(n + 3 * THREADS) * sn, t3);
#pragma unroll
    for (int di = 0; di < kDTile; ++di) {
      float a = fabsf(t0[di]);
      float b = fabsf(t1[di]);
      float c = fabsf(t2[di]);
      float d = fabsf(t3[di]);
      mx[di] = fmaxf(mx[di], fmaxf(fmaxf(a, b), fmaxf(c, d)));
    }
  }
  for (; n < N; n += THREADS) {
    float tmp[kDTile];
    load_tile(base + (int64_t)n * sn, tmp);
#pragma unroll
    for (int di = 0; di < kDTile; ++di)
      mx[di] = fmaxf(mx[di], fabsf(tmp[di]));
  }

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

#pragma unroll
  for (int di = 0; di < kDTile; ++di)
    mx[di] = comfy::warp_reduce_fmax(mx[di]);

  // Cross-warp reduction
  __shared__ float warp_mx[kDTile][WARPS];
  __shared__ float inv_sc_sh[kDTile];

  if (lane == 0) {
#pragma unroll
    for (int di = 0; di < kDTile; ++di)
      warp_mx[di][warp] = mx[di];
  }
  __syncthreads();

  if (threadIdx.x < kDTile) {
    float val = 0.f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w)
      val = fmaxf(val, warp_mx[threadIdx.x][w]);

    float sc = fmaxf(val * (1.f / 127.f), 1e-12f);
    scale_out[(b * H + h) * D + d0 + threadIdx.x] = sc;
    inv_sc_sh[threadIdx.x] = 1.f / sc;
  }
  __syncthreads();

  float inv_sc[kDTile];
#pragma unroll
  for (int di = 0; di < kDTile; ++di)
    inv_sc[di] = inv_sc_sh[di];

  // ── Pass 2: quantize + permute (reverse for L2 reuse) ──────────────
  // Iterate over LINEAR source indices (coalesced reads from v) and compute
  // the permuted destination index for the INT8 MMA layout.
  const int64_t out_row = static_cast<int64_t>((b * H + h) * D + d0);

  for (int src = N - 1 - threadIdx.x; src >= 0; src -= THREADS) {
    const int w = src & 15;
    const int dst = (src & ~15) | inv_perm16(w);

    float tmp[kDTile];
    load_tile(base + (int64_t)src * sn, tmp);
#pragma unroll
    for (int di = 0; di < kDTile; ++di) {
      out[(out_row + di) * padded_N + dst] =
          comfy::float_to_int8_rn(tmp[di] * inv_sc[di]);
    }
  }

  // Zero-fill padding region [N, padded_N) with the same inverse permutation
  // so that the layout matches the eager reference for partial 16-element groups.
  for (int src = N + threadIdx.x; src < padded_N; src += THREADS) {
    const int w = src & 15;
    const int dst = (src & ~15) | inv_perm16(w);
#pragma unroll
    for (int di = 0; di < kDTile; ++di)
      out[(out_row + di) * padded_N + dst] = 0;
  }
}

} // namespace

namespace jasna_ltx::sage_attention {

void launch_quant_v_int8_kernel(const void *v, void *out, void *scale,
                                int B, int H, int N, int D,
                                int padded_N, int64_t sb, int64_t sh,
                                int64_t sn, int input_dtype_code,
                                cudaStream_t stream) {
  if (input_dtype_code != kBFloat16) {
    throw std::runtime_error("quant_v_int8: only bf16 inputs are compiled");
  }
  // An extent of one forms no offset, and PyTorch rewrites such a stride to 1 on
  // the way through DLPack (ATen/DLConvertor.cpp, gh-83069) on 2.9 and older, so
  // policing it would reject a tensor the caller has no way to hand over
  // differently.
  const size_t element_size = input_dtype_code == 0 ? sizeof(float) : sizeof(half);
  const auto stride_ok = [element_size](int64_t stride, int extent) {
    return extent < 2 || (static_cast<size_t>(stride) * element_size) % 16 == 0;
  };
  if (reinterpret_cast<uintptr_t>(v) % 16 != 0 || !stride_ok(sb, B) ||
      !stride_ok(sh, H) || !stride_ok(sn, N)) {
    throw std::runtime_error(
        "quant_v_int8: V base pointer and B/H/N strides must be 16-byte aligned");
  }
  if (D <= 0 || D % kDTile != 0) {
    throw std::runtime_error("quant_v_int8: head_dim must be a positive multiple of 8");
  }

  const int blocks = B * H * (D / kDTile);

  {
    using T = nv_bfloat16;
    if (N <= 256) {
      quant_v_int8_kernel<T, 128><<<blocks, 128, 0, stream>>>(
          static_cast<const T *>(v), static_cast<int8_t *>(out),
          static_cast<float *>(scale), N, padded_N, H, D, sb, sh, sn);
    } else {
      quant_v_int8_kernel<T, 512><<<blocks, 512, 0, stream>>>(
          static_cast<const T *>(v), static_cast<int8_t *>(out),
          static_cast<float *>(scale), N, padded_N, H, D, sb, sh, sn);
    }
  }

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("quant_v_int8 kernel launch failed: ") +
                             cudaGetErrorString(error));
  }
}

} // namespace jasna_ltx::sage_attention
