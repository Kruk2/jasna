// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. Derived from SageAttention
// (https://github.com/thu-ml/SageAttention) commit
// d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5. DLPack-compatible launcher for the
// Pure integer attention kernel: signed INT8 Q/K/V, unsigned INT8 softmax
// probabilities, INT32 tensor-core P*V accumulation, and FP32 online-softmax
// state. V scaling is fused and LSE is not returned.
// Modified for Jasna: bf16, unmasked paths only; launchers in jasna_ltx::sage_attention (see NOTICE).

#include "qk_int_sv_i8_cuda.cuh"
#include "launchers.h"
#include <math_constants.h>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace {

template <int HEAD_DIM, int CTA_K, MaskMode mask_mode, typename DTypeOut,
          bool fuse_fp32_probabilities = true, int CTA_Q = 128,
          typename Offset = uint32_t>
void launch_impl(int8_t *q, int8_t *k, int8_t *v, DTypeOut *o, float *q_scale,
                 float *k_scale, float *v_scale, const void *mask,
                 int64_t mask_stride_b, int64_t mask_stride_h,
                 int64_t mask_stride_q, int64_t mask_stride_k,
                 int mask_dtype_code, int qo_len, int kv_len,
                 int num_qo_heads, int num_kv_groups, Offset stride_bz_q,
                 int stride_seq_q, Offset stride_h_q, Offset stride_bz_k,
                 int stride_seq_k, Offset stride_h_k, Offset stride_bz_v,
                 Offset stride_h_v, int stride_d_v, Offset stride_bz_o,
                 int stride_seq_o, Offset stride_h_o, float sm_scale,
                 int batch_size, cudaStream_t stream, const float *mask_tile_bias) {
  // Tiling constants — must match sage_attention.py and dlpack_bindings.cpp.
  // D>=128 otherwise needs too many live FP32 output accumulators per thread.
  // A 16-row warp tile halves that accumulator set.
  constexpr int WARP_Q = HEAD_DIM >= 128 ? 16 : 32;
  constexpr int WARP_K = CTA_K;

  size_t smem_max =
      std::max(static_cast<size_t>(CTA_Q * HEAD_DIM * sizeof(int8_t) +
                                   CTA_K * HEAD_DIM * sizeof(int8_t) +
                                   CTA_K * HEAD_DIM * sizeof(int8_t)),
               static_cast<size_t>(CTA_Q * HEAD_DIM * sizeof(half)));

  auto kernel = qk_int_sv_i8_attn_kernel<
      CTA_Q, CTA_K, WARP_Q, WARP_K, HEAD_DIM, DataType::kInt8,
      QuantGranularity::kPerThread, QuantGranularity::kPerThread, float, false,
      DTypeOut, ComputeUnit::kCudaCore, mask_mode, false, true, false, false,
      fuse_fp32_probabilities, Offset>;

  cudaError_t error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(smem_max));
  if (error != cudaSuccess) {
    throw std::runtime_error(
        "sage_attn failed to request " + std::to_string(smem_max) +
        " bytes of dynamic shared memory: " + cudaGetErrorString(error));
  }

  dim3 grid(div_ceil(qo_len, CTA_Q), num_qo_heads, batch_size);
  dim3 block(32, (CTA_Q / WARP_Q) * (CTA_K / WARP_K));

  kernel<<<grid, block, smem_max, stream>>>(
      q, k, v, o, nullptr, q_scale, k_scale, v_scale, nullptr, mask,
      mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k,
      mask_dtype_code, qo_len, kv_len, num_kv_groups, stride_bz_q,
      stride_seq_q, stride_h_q, stride_bz_k, stride_seq_k, stride_h_k,
      stride_bz_v, stride_h_v, stride_d_v, stride_bz_o, stride_seq_o,
      stride_h_o, sm_scale, mask_tile_bias);

  error = cudaGetLastError();
  if (error != cudaSuccess) {
    throw std::runtime_error(std::string("sage_attn kernel launch failed: ") +
                             cudaGetErrorString(error));
  }
}

// Compute capability (major * 10 + minor), cached per device.
int current_device_sm() {
  static int cached[64] = {};
  int device = 0;
  cudaError_t error = cudaGetDevice(&device);
  const bool cacheable = error == cudaSuccess && device >= 0 && device < 64;
  if (cacheable && cached[device] != 0) return cached[device];
  int major = 0, minor = 0;
  if (error == cudaSuccess)
    error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  if (error == cudaSuccess)
    error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  if (error != cudaSuccess)
    throw std::runtime_error(std::string("sage_attn device query failed: ") +
                             cudaGetErrorString(error));
  if (cacheable) cached[device] = major * 10 + minor;
  return major * 10 + minor;
}

bool is_ampere(int sm) { return sm >= 80 && sm < 89; }

} // anonymous namespace

namespace jasna_ltx::sage_attention {

void launch_sage_attn_kernel(
    const void *q, const void *k, const void *v, void *o, const void *q_scale,
    const void *k_scale, const void *v_scale, const void *mask,
    int64_t mask_stride_b, int64_t mask_stride_h, int64_t mask_stride_q,
    int64_t mask_stride_k, int mask_dtype_code, int cta_k, int batch_size,
    int qo_len,
    int kv_len, int num_qo_heads, int num_kv_heads, int head_dim,
    int64_t stride_bz_q, int stride_seq_q, int64_t stride_h_q, int64_t stride_bz_k,
    int stride_seq_k, int64_t stride_h_k, int64_t stride_bz_v, int64_t stride_h_v,
    int stride_d_v, int64_t stride_bz_o, int stride_seq_o, int64_t stride_h_o,
    float sm_scale, int output_dtype_code, cudaStream_t stream,
    const float *mask_tile_bias) {
  if (mask != nullptr || mask_tile_bias != nullptr || output_dtype_code != kBFloat16) {
    throw std::runtime_error("sage_attn: only unmasked attention with bf16 outputs is compiled");
  }
  if (cta_k != 64 && cta_k != 128) {
    throw std::runtime_error("sage_attn: cta_k must be 64 or 128");
  }
  if (cta_k == 128 && (head_dim == 64 || (mask != nullptr && mask_tile_bias == nullptr))) {
    throw std::runtime_error(
        "sage_attn: cta_k 128 requires unmasked head_dim 128/256 or a prepared key mask");
  }
  // Only the per-head tile iterators require 32-bit offsets. Batch/head bases
  // can use 64-bit arithmetic, so the head count must not reduce this limit.
  if (static_cast<int64_t>(qo_len) * stride_seq_q > INT_MAX ||
      static_cast<int64_t>(kv_len) * stride_seq_k > INT_MAX ||
      static_cast<int64_t>(head_dim) * stride_d_v > INT_MAX ||
      static_cast<int64_t>(qo_len) * stride_seq_o > INT_MAX) {
    throw std::overflow_error(
        "sage_attn: within-head offsets exceed int32 range; reduce sequence length");
  }
  int num_kv_groups = num_qo_heads / num_kv_heads;
  // The existing kernel uses unsigned offsets. Keep it for the full uint32
  // range. Check the largest index products, not the total tensor size:
  // pointer additions are already 64-bit, but each index * stride can wrap.
  // A singleton dimension never uses its stride.
  const bool wide_offsets =
      static_cast<int64_t>(batch_size - 1) *
          std::max({stride_bz_q, stride_bz_k, stride_bz_v, stride_bz_o}) > UINT32_MAX ||
      static_cast<int64_t>(num_qo_heads - 1) * std::max(stride_h_q, stride_h_o) > UINT32_MAX ||
      static_cast<int64_t>(num_kv_heads - 1) * std::max(stride_h_k, stride_h_v) > UINT32_MAX;

  // Upstream kernel uses non-const pointers; cast away const from the
  // extern "C" boundary (kernel does not modify inputs).
  auto q_ = const_cast<int8_t *>(static_cast<const int8_t *>(q));
  auto k_ = const_cast<int8_t *>(static_cast<const int8_t *>(k));
  auto v_ = const_cast<int8_t *>(static_cast<const int8_t *>(v));
  auto qs_ = const_cast<float *>(static_cast<const float *>(q_scale));
  auto ks_ = const_cast<float *>(static_cast<const float *>(k_scale));
  auto vs_ = const_cast<float *>(static_cast<const float *>(v_scale));

#define LAUNCH_IMPL_Q(HD, CK, MM, DT, FUSE_FP32, CQ, OFFSET)                                     \
  launch_impl<HD, CK, MM, DT, FUSE_FP32, CQ, OFFSET>(                                    \
                          q_, k_, v_, static_cast<DT *>(o), qs_, ks_, vs_,     \
                          mask, mask_stride_b, mask_stride_h, mask_stride_q,   \
                          mask_stride_k, mask_dtype_code, qo_len, kv_len,      \
                          num_qo_heads, num_kv_groups,                         \
                          stride_bz_q, stride_seq_q, stride_h_q, stride_bz_k,  \
                          stride_seq_k, stride_h_k, stride_bz_v, stride_h_v,   \
                          stride_d_v, stride_bz_o, stride_seq_o, stride_h_o,   \
                          sm_scale, batch_size, stream, mask_tile_bias)

// Keep the ordinary launch's argument packing and call path unchanged.
#define LAUNCH_Q(HD, CK, MM, DT, FUSE_FP32, CQ)                         \
  do {                                                               \
    if (wide_offsets) {                                              \
      LAUNCH_IMPL_Q(HD, CK, MM, DT, FUSE_FP32, CQ, uint64_t);           \
    } else {                                                         \
      LAUNCH_IMPL_Q(HD, CK, MM, DT, FUSE_FP32, CQ, uint32_t);           \
    }                                                                \
  } while (false)

#define LAUNCH(HD, CK, MM, DT, FUSE_FP32) \
  LAUNCH_Q(HD, CK, MM, DT, FUSE_FP32, 128)

#define LAUNCH_CTA(HD, MM, DT, FUSE_FP32)                                     \
  if constexpr (HD == 64) {                                                   \
    LAUNCH(HD, 64, MM, DT, FUSE_FP32);                                        \
  } else if (cta_k == 128) {                                                   \
    LAUNCH(HD, 128, MM, DT, FUSE_FP32);                                       \
  } else {                                                                     \
    LAUNCH(HD, 64, MM, DT, FUSE_FP32);                                        \
  }

#define DISPATCH_MASK(HD)                                                      \
  if (kv_len <= 512 && cta_k == 64) {                                          \
    LAUNCH(HD, 64, MaskMode::kNone, nv_bfloat16, false);                       \
  } else {                                                                     \
    LAUNCH_CTA(HD, MaskMode::kNone, nv_bfloat16, true);                        \
  }

  // Ampere: 64-row tiles fit two CTAs per SM, 2-4% faster on a 3080 up to 38k keys. Longer
  // sequences lose: each CTA re-reads K/V, which costs power (6-22% slower on a 300 W A6000).
  if (mask == nullptr && head_dim == 128 && cta_k == 128 && kv_len <= 40960 &&
      is_ampere(current_device_sm())) {
    LAUNCH_Q(128, 128, MaskMode::kNone, nv_bfloat16, true, 64);
    return;
  }

  // Keep smaller unmasked query tiles limited to Blackwell image shapes.
  // Ada retains 128-query tiles: warmed ComfyUI workloads regress with the
  // smaller tiles under stock-clock thermal throttling.
  if (mask == nullptr && head_dim == 128 && cta_k == 128 &&
      qo_len >= 4096 && qo_len <= 4608 && kv_len >= 4096 && kv_len <= 4608 &&
      num_qo_heads <= 32) {
    int device = 0, major = 0, multiprocessors = 0;
    cudaError_t error = cudaGetDevice(&device);
    if (error == cudaSuccess)
      error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    if (error == cudaSuccess && major == 12 && num_qo_heads >= 30)
      error = cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device);
    if (error != cudaSuccess)
      throw std::runtime_error(std::string("sage_attn device query failed: ") +
                               cudaGetErrorString(error));
    if (major == 12) {
      bool smaller_tile = true;
      if (num_qo_heads >= 30) {
        // At 30-32 heads, use smaller tiles only when they save an SM-sized
        // round of query work. Ming's 4422 queries otherwise lose to 128-row
        // tiles, while Qwen's 4096/4608-query shapes retain the smaller tile.
        // Keep the existing selection for smaller head counts.
        const int64_t batch_heads = static_cast<int64_t>(batch_size) * num_qo_heads;
        const int64_t blocks_64 = batch_heads * div_ceil(qo_len, 64);
        const int64_t blocks_128 = batch_heads * div_ceil(qo_len, 128);
        const int64_t rounds_64 = (blocks_64 + multiprocessors - 1) / multiprocessors;
        const int64_t rounds_128 = (blocks_128 + multiprocessors - 1) / multiprocessors;
        smaller_tile = rounds_64 < 2 * rounds_128;
      }
      if (smaller_tile) {
        LAUNCH_Q(128, 128, MaskMode::kNone, nv_bfloat16, true, 64);
        return;
      }
    }
  }

  if (head_dim == 64) {
    DISPATCH_MASK(64);
  } else if (head_dim == 128) {
    DISPATCH_MASK(128);
  } else if (head_dim == 256) {
    DISPATCH_MASK(256);
  } else {
    throw std::runtime_error("sage_attn: unsupported head_dim " +
                             std::to_string(head_dim));
  }

#undef LAUNCH
#undef LAUNCH_Q
#undef LAUNCH_IMPL_Q
#undef LAUNCH_CTA
#undef DISPATCH_MASK
}

} // namespace jasna_ltx::sage_attention
