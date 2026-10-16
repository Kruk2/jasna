// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2024 by SageAttention team.
// SPDX-FileContributor: Modified by NVIDIA CORPORATION & AFFILIATES, 2025.
// Derived from SageAttention (https://github.com/thu-ml/SageAttention)
// commit d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5.
// Modifications: removed torch/extension.h dependency, flattened include paths.

/*
 * Copyright (c) 2024 by SageAttention team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <climits>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>

#include "cp_async.cuh"
#include "mma.cuh"
#include "numeric_conversion.cuh"
#include "permuted_smem.cuh"

#define WARP_SIZE 32

// Shift the online-softmax maximum so exp2(score - max) spans the complete
// unsigned INT8 range. P uses U8 because softmax probabilities are nonnegative;
// Q, K, and V remain signed INT8.
#define S_U8_OFFSET 7.9943534f

#define div_ceil(M, N) (((M) + (N) - 1) / (N))

enum class MaskMode {
  kNone = 0,
  kCausal = 1,
  kCustom = 2,
  kCustomKey = 3,
  kPreparedKey = 4,
};

enum class DataType {
  kHalf,
  kInt8,
  kInt4,
  kE4M3,
  kE5M2,
};

enum class QuantGranularity {
  kPerTensor = 0,
  kPerBlock = 1,
  kPerWarp = 2,
  kPerThread = 3,
};

enum class ComputeUnit {
  kTensorCore,
  kCudaCore,
};

__device__ __forceinline__ uint32_t get_warp_id() { return threadIdx.y; }

__device__ __forceinline__ uint32_t get_lane_id() { return threadIdx.x; }

template <uint32_t num_warps_q, uint32_t num_warps_k>
__device__ __forceinline__ uint32_t get_warp_idx_q() {
  return get_warp_id() / num_warps_k;
}

template <uint32_t num_warps_q, uint32_t num_warps_k>
__device__ __forceinline__ uint32_t get_warp_idx_k() {
  return get_warp_id() % num_warps_k;
}

template <uint32_t global_to_shared_line_lanes,
          uint32_t global_to_shared_copy_lines_per_warp_per_iter,
          uint32_t smem_iters_row, uint32_t smem_iters_col,
          SwizzleMode swizzle_mode, uint32_t stride, uint32_t CTA, typename T>
__device__ __forceinline__ void
load_global_to_share(T **lane_ptr, uint32_t &smem_offset,
                     const uint32_t &gmem_stride,
                     const smem_t<swizzle_mode, stride> &smem) {
  static_assert(global_to_shared_copy_lines_per_warp_per_iter *
                    global_to_shared_line_lanes ==
                WARP_SIZE);
  static_assert(std::is_same<T, half>::value || std::is_same<T, int8_t>::value);

  constexpr uint32_t pack_size = std::is_same<T, half>::value ? 8 : 16;

#pragma unroll
  for (uint32_t i = 0; i < smem_iters_col; i++) {
#pragma unroll
    for (uint32_t j = 0; j < smem_iters_row; j++) {
      smem.load_128b_async(smem_offset, *lane_ptr);
      *lane_ptr += (global_to_shared_line_lanes * pack_size);
      smem_offset = smem.advance_offset_by_column<global_to_shared_line_lanes>(
          smem_offset);
    }

    smem_offset = smem.advance_offset_by_row<
        global_to_shared_copy_lines_per_warp_per_iter>(
        smem_offset - (smem_iters_row * global_to_shared_line_lanes));
    *lane_ptr +=
        ((global_to_shared_copy_lines_per_warp_per_iter * gmem_stride) -
         (smem_iters_row * global_to_shared_line_lanes * pack_size));
  }
  smem_offset -=
      (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter * stride);
  *lane_ptr +=
      (CTA - smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter) *
      gmem_stride;
}

// with predicate
template <uint32_t global_to_shared_line_lanes,
          uint32_t global_to_shared_copy_lines_per_warp_per_iter,
          uint32_t smem_iters_row, uint32_t smem_iters_col,
          SwizzleMode swizzle_mode, uint32_t stride, uint32_t CTA, typename T>
__device__ __forceinline__ void
load_global_to_share(T **lane_ptr, uint32_t &smem_offset,
                     const uint32_t &gmem_stride,
                     const smem_t<swizzle_mode, stride> &smem,
                     uint32_t base_idx, uint32_t max_len) {
  static_assert(global_to_shared_copy_lines_per_warp_per_iter *
                    global_to_shared_line_lanes ==
                WARP_SIZE);
  static_assert(std::is_same<T, half>::value || std::is_same<T, int8_t>::value);

  constexpr uint32_t pack_size = std::is_same<T, half>::value ? 8 : 16;

#pragma unroll
  for (uint32_t i = 0; i < smem_iters_col; i++) {
#pragma unroll
    for (uint32_t j = 0; j < smem_iters_row; j++) {
      smem.load_128b_async<cp_async::SharedMemFillMode::kNoFill>(
          smem_offset, *lane_ptr, base_idx < max_len);
      *lane_ptr += (global_to_shared_line_lanes * pack_size);
      smem_offset = smem.advance_offset_by_column<global_to_shared_line_lanes>(
          smem_offset);
    }

    smem_offset = smem.advance_offset_by_row<
        global_to_shared_copy_lines_per_warp_per_iter>(
        smem_offset - (smem_iters_row * global_to_shared_line_lanes));
    *lane_ptr +=
        ((global_to_shared_copy_lines_per_warp_per_iter * gmem_stride) -
         (smem_iters_row * global_to_shared_line_lanes * pack_size));
    base_idx += global_to_shared_copy_lines_per_warp_per_iter;
  }
  smem_offset -=
      (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter * stride);
  *lane_ptr +=
      (CTA - smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter) *
      gmem_stride;
}

template <uint32_t global_to_shared_line_lanes,
          uint32_t global_to_shared_copy_lines_per_warp_per_iter,
          uint32_t smem_iters_row, uint32_t smem_iters_col,
          SwizzleMode swizzle_mode, uint32_t stride, uint32_t CTA>
__device__ __forceinline__ void
load_int8_V_global_to_share(int8_t **lane_ptr, uint32_t &smem_offset,
                           const uint32_t &gmem_stride,
                           const smem_t<swizzle_mode, stride> &smem) {
  static_assert(global_to_shared_copy_lines_per_warp_per_iter *
                    global_to_shared_line_lanes ==
                WARP_SIZE);
  constexpr uint32_t pack_size_int8 = 16;

#pragma unroll
  for (uint32_t i = 0; i < smem_iters_col; i++) {
#pragma unroll
    for (uint32_t j = 0; j < smem_iters_row; j++) {
      smem.load_128b_async(smem_offset, *lane_ptr);
      *lane_ptr += (global_to_shared_line_lanes * pack_size_int8);
      smem_offset = smem.advance_offset_by_column<global_to_shared_line_lanes>(
          smem_offset);
    }

    smem_offset = smem.advance_offset_by_row<
        global_to_shared_copy_lines_per_warp_per_iter>(
        smem_offset - (smem_iters_row * global_to_shared_line_lanes));
    *lane_ptr +=
        ((global_to_shared_copy_lines_per_warp_per_iter * gmem_stride) -
         (smem_iters_row * global_to_shared_line_lanes * pack_size_int8));
  }
  smem_offset -=
      (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter * stride);
  // for QK: *lane_ptr += (CTA - smem_iters_col *
  // global_to_shared_copy_lines_per_warp_per_iter) * gmem_stride;
  *lane_ptr += CTA; // ! prevent underflow
  *lane_ptr -=
      (smem_iters_col * global_to_shared_copy_lines_per_warp_per_iter) *
      gmem_stride;
}

#pragma nv_diag_suppress 174
template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q,
          uint32_t num_tiles_k, uint32_t num_tiles_qk_inner,
          SwizzleMode swizzle_mode, uint32_t stride, DataType DTypeQK>
__device__ __forceinline__ void
compute_int_qk(const smem_t<swizzle_mode, stride> &smem_Q,
               const smem_t<swizzle_mode, stride> &smem_K,
               int32_t RS[][num_tiles_k][8], uint32_t &offset_Q,
               uint32_t &offset_K) {
  static_assert(DTypeQK == DataType::kInt8 || DTypeQK == DataType::kInt4);

  uint32_t RQ[num_tiles_q][4];
  uint32_t RK[4];

  // the first iteration, mma mode is kInit
#pragma unroll
  for (uint32_t iter = 0; iter < 1; iter++) {
    // load RQ
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
      smem_Q.ldmatrix_m8n8x4(offset_Q, RQ[fq]);
      offset_Q = smem_Q.advance_offset_by_row<16>(offset_Q);
    }
    // ! using permutation invariance
    offset_Q = smem_Q.advance_offset_by_column<2>(
        offset_Q - (num_tiles_q * 16 * stride), iter);

#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
      // load RK
      smem_K.ldmatrix_m8n8x4(offset_K, RK);
      offset_K = smem_K.advance_offset_by_row<16>(offset_K);

      // mma
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        if constexpr (DTypeQK == DataType::kInt8) {
          mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInit>(
              RS[fq][fk], RQ[fq], RK);
        } else if constexpr (DTypeQK == DataType::kInt4) {
          mma::mma_sync_m16n16k64_row_col_s4s4s32<mma::MMAMode::kInit>(
              RS[fq][fk], RQ[fq], RK);
        }
      }
    }
    offset_K = smem_K.advance_offset_by_column<2>(
        offset_K - (num_tiles_k * 16 * stride), iter);
  }

  // following iteration, mma mode is kInplace
#pragma unroll
  for (uint32_t iter = 1; iter < num_tiles_qk_inner; iter++) {
    // load RQ
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
      smem_Q.ldmatrix_m8n8x4(offset_Q, RQ[fq]);
      offset_Q = smem_Q.advance_offset_by_row<16>(offset_Q);
    }
    offset_Q = smem_Q.advance_offset_by_column<2>(
        offset_Q - (num_tiles_q * 16 * stride), iter);

#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
      // load RK
      smem_K.ldmatrix_m8n8x4(offset_K, RK);
      offset_K = smem_K.advance_offset_by_row<16>(offset_K);

      // mma
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        if constexpr (DTypeQK == DataType::kInt8) {
          mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInplaceUpdate>(
              RS[fq][fk], RQ[fq], RK);
        } else if constexpr (DTypeQK == DataType::kInt4) {
          mma::mma_sync_m16n16k64_row_col_s4s4s32<mma::MMAMode::kInplaceUpdate>(
              RS[fq][fk], RQ[fq], RK);
        }
      }
    }
    offset_K = smem_K.advance_offset_by_column<2>(
        offset_K - (num_tiles_k * 16 * stride), iter);
  }

  offset_Q -= (2 * num_tiles_qk_inner);
  offset_K -= (2 * num_tiles_qk_inner);
}
template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q,
          uint32_t num_tiles_k, uint32_t num_tiles_qk_inner,
          SwizzleMode swizzle_mode, uint32_t stride, DataType DTypeQK>
__device__ __forceinline__ void compute_int_qk_cached(
    const smem_t<swizzle_mode, stride> &smem_K, int32_t RS[][num_tiles_k][8],
    uint32_t RQ[][num_tiles_qk_inner][4], uint32_t &offset_K) {
  uint32_t RK[4];

  // the first iteration, mma mode is kInit
#pragma unroll
  for (uint32_t iter = 0; iter < 1; iter++) {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
      // load RK
      smem_K.ldmatrix_m8n8x4(offset_K, RK);
      offset_K = smem_K.advance_offset_by_row<16>(offset_K);

      // mma
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        if constexpr (DTypeQK == DataType::kInt8) {
          mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInit>(
              RS[fq][fk], RQ[fq][iter], RK);
        } else if constexpr (DTypeQK == DataType::kInt4) {
          mma::mma_sync_m16n16k64_row_col_s4s4s32<mma::MMAMode::kInit>(
              RS[fq][fk], RQ[fq][iter], RK);
        }
      }
    }
    offset_K = smem_K.advance_offset_by_column<2>(
        offset_K - (num_tiles_k * 16 * stride), iter);
  }

  // following iteration, mma mode is kInplace
#pragma unroll
  for (uint32_t iter = 1; iter < num_tiles_qk_inner; iter++) {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
      // load RK
      smem_K.ldmatrix_m8n8x4(offset_K, RK);
      offset_K = smem_K.advance_offset_by_row<16>(offset_K);

      // mma
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        if constexpr (DTypeQK == DataType::kInt8) {
          mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInplaceUpdate>(
              RS[fq][fk], RQ[fq][iter], RK);
        } else if constexpr (DTypeQK == DataType::kInt4) {
          mma::mma_sync_m16n16k64_row_col_s4s4s32<mma::MMAMode::kInplaceUpdate>(
              RS[fq][fk], RQ[fq][iter], RK);
        }
      }
    }
    offset_K = smem_K.advance_offset_by_column<2>(
        offset_K - (num_tiles_k * 16 * stride), iter);
  }

  offset_K -= (2 * num_tiles_qk_inner);
}
#pragma nv_diag_default 174

// for case when num_tiles_qk_inner = 1
template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q,
          uint32_t num_tiles_k, uint32_t num_tiles_qk_inner,
          SwizzleMode swizzle_mode, uint32_t stride, DataType DTypeQK>
__device__ __forceinline__ void
compute_int_qk(const smem_t<swizzle_mode, stride> &smem_K,
               int32_t RS[][num_tiles_k][8], uint32_t RQ[][4],
               uint32_t offset_K) {
  static_assert(DTypeQK == DataType::kInt8 || DTypeQK == DataType::kInt4);
  static_assert(num_tiles_qk_inner == 1);

  uint32_t RK[4];

  // mma mode is kInit
#pragma unroll
  for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
    // load RK
    smem_K.ldmatrix_m8n8x4(offset_K, RK);
    offset_K = smem_K.advance_offset_by_row<16>(offset_K);

    // mma
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
      if constexpr (DTypeQK == DataType::kInt8) {
        mma::mma_sync_m16n16k32_row_col_s8s8s32<mma::MMAMode::kInit>(
            RS[fq][fk], RQ[fq], RK);
      } else if constexpr (DTypeQK == DataType::kInt4) {
        mma::mma_sync_m16n16k64_row_col_s4s4s32<mma::MMAMode::kInit>(
            RS[fq][fk], RQ[fq], RK);
      }
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void
apply_causal_mask(const uint32_t &Q_idx_lane_base,
                  const uint32_t &K_idx_lane_base,
                  float RS[][num_tiles_k][8],
                  const float mask_value = -50000.0f) {
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++) {
        const uint32_t q_idx = Q_idx_lane_base + fq * 16 + 8 * ((k % 4) / 2);
        const uint32_t kv_idx = K_idx_lane_base + fk * 16 + 8 * (k / 4) + k % 2;
        const bool out_of_boundary = (kv_idx > q_idx);

        if (out_of_boundary) {
          RS[fq][fk][k] = mask_value;
        }
      }
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void apply_custom_mask(
    const uint32_t &Q_idx_lane_base, const uint32_t &K_idx_lane_base,
    float RS[][num_tiles_k][8], bool valid[][2], const void *mask,
    const int64_t mask_stride_b, const int64_t mask_stride_h,
    const int64_t mask_stride_q, const int64_t mask_stride_k,
    const uint32_t batch_id, const uint32_t head_id, const uint32_t qo_len,
    const uint32_t kv_len, const int mask_dtype_code,
    const float score_scale) {
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++) {
        const uint32_t q_idx = Q_idx_lane_base + fq * 16 + 8 * ((k % 4) / 2);
        const uint32_t kv_idx = K_idx_lane_base + fk * 16 + 8 * (k / 4) + k % 2;
        const uint32_t q_group = (k % 4) / 2;
        bool keep = q_idx < qo_len && kv_idx < kv_len;
        float bias = 0.0f;

        if (keep) {
          const int64_t offset =
              static_cast<int64_t>(batch_id) * mask_stride_b +
              static_cast<int64_t>(head_id) * mask_stride_h +
              static_cast<int64_t>(q_idx) * mask_stride_q +
              static_cast<int64_t>(kv_idx) * mask_stride_k;
          if (mask_dtype_code == 3) {
            keep = static_cast<const uint8_t *>(mask)[offset] != 0;
          } else {
            if (mask_dtype_code == 0) {
              bias = static_cast<const float *>(mask)[offset];
            } else if (mask_dtype_code == 1) {
              bias = static_cast<float>(static_cast<const half *>(mask)[offset]);
            } else {
              bias = static_cast<float>(
                  static_cast<const nv_bfloat16 *>(mask)[offset]);
            }
            keep = isfinite(bias);
          }
        }

        valid[fq][q_group] = valid[fq][q_group] || keep;
        RS[fq][fk][k] = keep ? fmaf(RS[fq][fk][k], score_scale,
                                    bias * math::log2e)
                             : -50000.0f;
      }
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void apply_custom_key_mask(
    const uint32_t &K_idx_lane_base, float RS[][num_tiles_k][8],
    bool valid[][2], const void *mask, const int64_t mask_stride_b,
    const int64_t mask_stride_h, const int64_t mask_stride_k,
    const uint32_t batch_id, const uint32_t head_id, const uint32_t kv_len,
    const int mask_dtype_code, const float score_scale) {
  const int64_t head_offset =
      static_cast<int64_t>(batch_id) * mask_stride_b +
      static_cast<int64_t>(head_id) * mask_stride_h;

#pragma unroll
  for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
    bool keep[4];
    float bias[4];
#pragma unroll
    for (uint32_t value = 0; value < 4; value++) {
      const uint32_t kv_idx = K_idx_lane_base + fk * 16 +
                              (value >> 1) * 8 + (value & 1);
      keep[value] = kv_idx < kv_len;
      bias[value] = 0.0f;
      if (keep[value]) {
        const int64_t offset = head_offset +
                               static_cast<int64_t>(kv_idx) * mask_stride_k;
        if (mask_dtype_code == 3) {
          keep[value] = static_cast<const uint8_t *>(mask)[offset] != 0;
        } else {
          if (mask_dtype_code == 0) {
            bias[value] = static_cast<const float *>(mask)[offset];
          } else if (mask_dtype_code == 1) {
            bias[value] =
                static_cast<float>(static_cast<const half *>(mask)[offset]);
          } else {
            bias[value] = static_cast<float>(
                static_cast<const nv_bfloat16 *>(mask)[offset]);
          }
          keep[value] = isfinite(bias[value]);
        }
      }
    }

    const bool tile_valid = keep[0] || keep[1] || keep[2] || keep[3];
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
      valid[fq][0] = valid[fq][0] || tile_valid;
      valid[fq][1] = valid[fq][1] || tile_valid;
#pragma unroll
      for (uint32_t k = 0; k < 8; k++) {
        const uint32_t value = (k >> 2) * 2 + (k & 1);
        RS[fq][fk][k] = keep[value]
                            ? fmaf(RS[fq][fk][k], score_scale,
                                   bias[value] * math::log2e)
                            : -50000.0f;
      }
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void apply_prepared_key_mask(
    const uint32_t K_idx_lane_base, float RS[][num_tiles_k][8],
    const float *mask, const float score_scale) {
#pragma unroll
  for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
    const float2 bias_lo = *reinterpret_cast<const float2 *>(mask + K_idx_lane_base + fk * 16);
    const float2 bias_hi = *reinterpret_cast<const float2 *>(mask + K_idx_lane_base + fk * 16 + 8);
    const float bias[4] = {bias_lo.x * math::log2e, bias_lo.y * math::log2e,
                           bias_hi.x * math::log2e, bias_hi.y * math::log2e};
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++) {
        RS[fq][fk][k] = fmaf(RS[fq][fk][k], score_scale,
                             bias[(k >> 2) * 2 + (k & 1)]);
      }
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void
apply_out_of_bound_mask(const uint32_t &K_idx_lane_base,
                        float RS[][num_tiles_k][8],
                        const uint32_t &kv_len,
                        const float mask_value = -50000.0f) {
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++) {
        const uint32_t kv_idx = K_idx_lane_base + fk * 16 + 8 * (k / 4) + k % 2;
        const bool out_of_boundary = (kv_idx >= kv_len);

        if (out_of_boundary) {
          RS[fq][fk][k] = mask_value;
        }
      }
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k, uint32_t num_tiles_v,
          bool exp_offset, bool fuse_scale = false>
__device__ __forceinline__ void update_mdo(
    float RS[][num_tiles_k][8], float RO[][num_tiles_v][8], float m[][2],
    float d[][2], float pv_scale[][2], const float &sm_scale,
    const float exp_offset_value) {
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++) {
      // assign the smallest value possible
      const float m_prev = m[fq][k];
      float m_temp = -50000.0f;
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
        const float m_local = max(
            max(RS[fq][fk][k * 2 + 0], RS[fq][fk][k * 2 + 1]),
            max(RS[fq][fk][k * 2 + 4], RS[fq][fk][k * 2 + 5]));
        m_temp = max(m_temp, m_local);
      }

      if constexpr (!fuse_scale) {
        if constexpr (exp_offset) {
          m_temp = fmaf(m_temp, sm_scale, -exp_offset_value);
        } else {
          m_temp *= sm_scale;
        }
      } else if constexpr (exp_offset) {
        m_temp -= exp_offset_value;
      }

      // exchange element with the 4 threads in the row
      m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp,
                                           0x1)); // 0 exchange with 1, 2 with 3
      m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp,
                                           0x2)); // 0 exchange with 2, 1 with 3

      const float tile_m = m_temp;
      m[fq][k] = max(m_prev, tile_m);

      const float smaller_scale =
          math::ptx_exp2(-fabsf(m_prev - tile_m));
      const float o_scale = m_prev < tile_m ? smaller_scale : 1.0f;
      pv_scale[fq][k] = tile_m < m_prev ? smaller_scale : 1.0f;

      // update denominator
      d[fq][k] *= o_scale;

      // update RO
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
        RO[fq][fv][k * 2 + 0] *= o_scale;
        RO[fq][fv][k * 2 + 1] *= o_scale;
        RO[fq][fv][k * 2 + 4] *= o_scale;
        RO[fq][fv][k * 2 + 5] *= o_scale;
      }

      // raise RS to exponent
      const float negative_m = -tile_m;
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
        if constexpr (fuse_scale) {
          RS[fq][fk][k * 2 + 0] =
              math::ptx_exp2(RS[fq][fk][k * 2 + 0] + negative_m);
          RS[fq][fk][k * 2 + 1] =
              math::ptx_exp2(RS[fq][fk][k * 2 + 1] + negative_m);
          RS[fq][fk][k * 2 + 4] =
              math::ptx_exp2(RS[fq][fk][k * 2 + 4] + negative_m);
          RS[fq][fk][k * 2 + 5] =
              math::ptx_exp2(RS[fq][fk][k * 2 + 5] + negative_m);
        } else {
          RS[fq][fk][k * 2 + 0] =
              math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 0], sm_scale, negative_m));
          RS[fq][fk][k * 2 + 1] =
              math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 1], sm_scale, negative_m));
          RS[fq][fk][k * 2 + 4] =
              math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 4], sm_scale, negative_m));
          RS[fq][fk][k * 2 + 5] =
              math::ptx_exp2(fmaf(RS[fq][fk][k * 2 + 5], sm_scale, negative_m));
        }
      }
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k, typename T>
__device__ __forceinline__ void RS_32_to_16(T RS[][num_tiles_k][8],
                                            uint32_t RS_16[][num_tiles_k][4]) {
  static_assert(sizeof(T) == 4);
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
      ((half2 *)RS_16[fq][fk])[0] =
          __float22half2_rn(((float2 *)RS[fq][fk])[0]);
      ((half2 *)RS_16[fq][fk])[1] =
          __float22half2_rn(((float2 *)RS[fq][fk])[1]);
      ((half2 *)RS_16[fq][fk])[2] =
          __float22half2_rn(((float2 *)RS[fq][fk])[2]);
      ((half2 *)RS_16[fq][fk])[3] =
          __float22half2_rn(((float2 *)RS[fq][fk])[3]);
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void
RS_32_to_8(float RS[][num_tiles_k][8], uint32_t RS_8[][num_tiles_k / 2][4]) {
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
      floatx4_to_e4m3x4(RS_8[fq][fk], RS[fq][fk * 2 + 0],
                        RS[fq][fk * 2 + 0] + 4);
      floatx4_to_e4m3x4(RS_8[fq][fk] + 1, RS[fq][fk * 2 + 0] + 2,
                        RS[fq][fk * 2 + 0] + 6);
      floatx4_to_e4m3x4(RS_8[fq][fk] + 2, RS[fq][fk * 2 + 1],
                        RS[fq][fk * 2 + 1] + 4);
      floatx4_to_e4m3x4(RS_8[fq][fk] + 3, RS[fq][fk * 2 + 1] + 2,
                        RS[fq][fk * 2 + 1] + 6);
    }
  }
}

struct PackedU8RowSum {
  uint32_t probabilities;
  float denominator;
};

__device__ __forceinline__ PackedU8RowSum
pack_scaled_exp2_u8x4(const int32_t a, const int32_t b, const int32_t c,
                      const int32_t d, const float scale,
                      const float negative_m) {
  const float probability_a =
      math::ptx_exp2(fmaf(__int2float_rz(a), scale, negative_m));
  const float probability_b =
      math::ptx_exp2(fmaf(__int2float_rz(b), scale, negative_m));
  const float probability_c =
      math::ptx_exp2(fmaf(__int2float_rz(c), scale, negative_m));
  const float probability_d =
      math::ptx_exp2(fmaf(__int2float_rz(d), scale, negative_m));
  const uint32_t probabilities = mma::pack_u8x4(
      probability_a, probability_b, probability_c, probability_d);
  const uint32_t denominator = __dp4a(probabilities, 0x01010101u, 0u);
  return {probabilities, __uint2float_rn(denominator)};
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k,
          uint32_t num_tiles_v, bool add_bias = false>
__device__ __forceinline__ void update_mdo_i32_u8(
    int32_t RS[][num_tiles_k][8], float RO[][num_tiles_v][8],
    float m[][2], float d[][2], const float sm_scale, const float exp_offset_value,
    uint32_t RS_u8[][num_tiles_k / 2][4], const float tile_bias = 0.0f) {
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++) {
      const float m_prev = m[fq][k];
      int32_t m_temp_i32 = INT_MIN;
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
        const int32_t m_local =
            max(max(RS[fq][fk][k * 2], RS[fq][fk][k * 2 + 1]),
                max(RS[fq][fk][k * 2 + 4], RS[fq][fk][k * 2 + 5]));
        m_temp_i32 = max(m_temp_i32, m_local);
      }

      float m_temp = fmaf(__int2float_rz(m_temp_i32), sm_scale,
                          -exp_offset_value);
      if constexpr (add_bias) m_temp += tile_bias;
      m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x1));
      m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x2));
      const float tile_m = m_temp;
      m[fq][k] = max(m_prev, tile_m);

      const float smaller_scale =
          math::ptx_exp2(-fabsf(m_prev - tile_m));
      const float o_scale = m_prev < tile_m ? smaller_scale : 1.0f;
      const float tile_scale = tile_m < m_prev ? smaller_scale : 1.0f;
      d[fq][k] *= o_scale;
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
        RO[fq][fv][k * 2] *= o_scale;
        RO[fq][fv][k * 2 + 1] *= o_scale;
        RO[fq][fv][k * 2 + 4] *= o_scale;
        RO[fq][fv][k * 2 + 5] *= o_scale;
      }

      const float negative_m = add_bias ? tile_bias - tile_m : -tile_m;
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
        const PackedU8RowSum probabilities_0 = pack_scaled_exp2_u8x4(
            RS[fq][fk * 2][k * 2], RS[fq][fk * 2][k * 2 + 1],
            RS[fq][fk * 2][k * 2 + 4], RS[fq][fk * 2][k * 2 + 5],
            sm_scale, negative_m);
        const PackedU8RowSum probabilities_1 = pack_scaled_exp2_u8x4(
            RS[fq][fk * 2 + 1][k * 2],
            RS[fq][fk * 2 + 1][k * 2 + 1],
            RS[fq][fk * 2 + 1][k * 2 + 4],
            RS[fq][fk * 2 + 1][k * 2 + 5], sm_scale, negative_m);
        RS_u8[fq][fk][k] = probabilities_0.probabilities;
        RS_u8[fq][fk][k + 2] = probabilities_1.probabilities;
        d[fq][k] += probabilities_0.denominator * tile_scale;
        d[fq][k] += probabilities_1.denominator * tile_scale;
      }
      // QK scores are dead after packing. Reuse their registers for the
      // probability scale rather than extending the kernel's live state.
      RS[fq][0][k] = __float_as_int(tile_scale);
    }
  }
}

// Pack each group immediately after exponentiation so masked attention does
// not keep an entire tile of FP32 probabilities live through the P*V setup.
// Sum the same rounded probabilities used by P*V, as in the unmasked path.
__device__ __forceinline__ PackedU8RowSum
pack_exp2_u8x4(const float a, const float b, const float c,
              const float d, const float negative_m) {
  const float pa = math::ptx_exp2(a + negative_m);
  const float pb = math::ptx_exp2(b + negative_m);
  const float pc = math::ptx_exp2(c + negative_m);
  const float pd = math::ptx_exp2(d + negative_m);
  const uint32_t packed = mma::pack_u8x4(pa, pb, pc, pd);
  return {packed, __uint2float_rn(__dp4a(packed, 0x01010101u, 0u))};
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k,
          uint32_t num_tiles_v>
__device__ __forceinline__ void update_mdo_f32_u8(
    float RS[][num_tiles_k][8], float RO[][num_tiles_v][8],
    float m[][2], float d[][2], const float exp_offset_value,
    uint32_t RS_u8[][num_tiles_k / 2][4]) {
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++) {
      const float m_prev = m[fq][k];
      float m_temp_f32 = -50000.0f;
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
        const float m_local =
            max(max(RS[fq][fk][k * 2], RS[fq][fk][k * 2 + 1]),
                max(RS[fq][fk][k * 2 + 4], RS[fq][fk][k * 2 + 5]));
        m_temp_f32 = max(m_temp_f32, m_local);
      }

      float m_temp = m_temp_f32 - exp_offset_value;
      m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x1));
      m_temp = max(m_temp, __shfl_xor_sync(0xffffffff, m_temp, 0x2));
      const float tile_m = m_temp;
      m[fq][k] = max(m_prev, tile_m);

      const float smaller_scale =
          math::ptx_exp2(-fabsf(m_prev - tile_m));
      const float o_scale = m_prev < tile_m ? smaller_scale : 1.0f;
      const float tile_scale = tile_m < m_prev ? smaller_scale : 1.0f;
      d[fq][k] *= o_scale;
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
        RO[fq][fv][k * 2] *= o_scale;
        RO[fq][fv][k * 2 + 1] *= o_scale;
        RO[fq][fv][k * 2 + 4] *= o_scale;
        RO[fq][fv][k * 2 + 5] *= o_scale;
      }

      const float negative_m = -tile_m;
#pragma unroll
      for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
        const PackedU8RowSum probabilities_0 = pack_exp2_u8x4(
            RS[fq][fk * 2][k * 2], RS[fq][fk * 2][k * 2 + 1],
            RS[fq][fk * 2][k * 2 + 4], RS[fq][fk * 2][k * 2 + 5],
            negative_m);
        const PackedU8RowSum probabilities_1 = pack_exp2_u8x4(
            RS[fq][fk * 2 + 1][k * 2],
            RS[fq][fk * 2 + 1][k * 2 + 1],
            RS[fq][fk * 2 + 1][k * 2 + 4],
            RS[fq][fk * 2 + 1][k * 2 + 5], negative_m);
        RS_u8[fq][fk][k] = probabilities_0.probabilities;
        RS_u8[fq][fk][k + 2] = probabilities_1.probabilities;
        d[fq][k] += probabilities_0.denominator * tile_scale;
        d[fq][k] += probabilities_1.denominator * tile_scale;
      }
      // QK scores are dead after packing. Reuse their registers for the
      // probability scale rather than extending the kernel's live state.
      RS[fq][0][k] = tile_scale;
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k, typename T>
__device__ __forceinline__ void
RS_to_u8(T RS[][num_tiles_k][8], uint32_t RS_u8[][num_tiles_k / 2][4]) {
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
      RS_u8[fq][fk][0] =
          mma::pack_u8x4(RS[fq][fk * 2][0], RS[fq][fk * 2][1],
                         RS[fq][fk * 2][4], RS[fq][fk * 2][5]);
      RS_u8[fq][fk][1] =
          mma::pack_u8x4(RS[fq][fk * 2][2], RS[fq][fk * 2][3],
                         RS[fq][fk * 2][6], RS[fq][fk * 2][7]);
      RS_u8[fq][fk][2] =
          mma::pack_u8x4(RS[fq][fk * 2 + 1][0], RS[fq][fk * 2 + 1][1],
                         RS[fq][fk * 2 + 1][4], RS[fq][fk * 2 + 1][5]);
      RS_u8[fq][fk][3] =
          mma::pack_u8x4(RS[fq][fk * 2 + 1][2], RS[fq][fk * 2 + 1][3],
                         RS[fq][fk * 2 + 1][6], RS[fq][fk * 2 + 1][7]);
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void
RS_16_to_8(uint32_t RS[][num_tiles_k][4], uint32_t RS_8[][num_tiles_k / 2][4]) {
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
      halfx4_to_e4m3x4(RS_8[fq][fk], RS[fq][fk * 2 + 0],
                       RS[fq][fk * 2 + 0] + 2);
      halfx4_to_e4m3x4(RS_8[fq][fk] + 1, RS[fq][fk * 2 + 0] + 1,
                       RS[fq][fk * 2 + 0] + 3);
      halfx4_to_e4m3x4(RS_8[fq][fk] + 2, RS[fq][fk * 2 + 1],
                       RS[fq][fk * 2 + 1] + 2);
      halfx4_to_e4m3x4(RS_8[fq][fk] + 3, RS[fq][fk * 2 + 1] + 1,
                       RS[fq][fk * 2 + 1] + 3);
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void RS_8_to_16(uint32_t RS_8[][num_tiles_k / 2][4],
                                           uint32_t RS[][num_tiles_k][4]) {
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
      e4m3x4_to_halfx4(RS[fq][fk * 2 + 0], RS[fq][fk * 2 + 0] + 2,
                       RS_8[fq][fk]);
      e4m3x4_to_halfx4(RS[fq][fk * 2 + 0] + 1, RS[fq][fk * 2 + 0] + 3,
                       RS_8[fq][fk] + 1);
      e4m3x4_to_halfx4(RS[fq][fk * 2 + 1], RS[fq][fk * 2 + 1] + 2,
                       RS_8[fq][fk] + 2);
      e4m3x4_to_halfx4(RS[fq][fk * 2 + 1] + 1, RS[fq][fk * 2 + 1] + 3,
                       RS_8[fq][fk] + 3);
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void accumulate_d(
    float RS[][num_tiles_k][8], float d[][2], float pv_scale[][2]) {
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
      const float tile_sum_0 =
          (RS[fq][fk][0] + RS[fq][fk][1]) +
          (RS[fq][fk][4] + RS[fq][fk][5]);
      const float tile_sum_1 =
          (RS[fq][fk][2] + RS[fq][fk][3]) +
          (RS[fq][fk][6] + RS[fq][fk][7]);
      d[fq][0] += tile_sum_0 * pv_scale[fq][0];
      d[fq][1] += tile_sum_1 * pv_scale[fq][1];
    }
  }
}

template <uint32_t num_tiles_q, uint32_t num_tiles_k>
__device__ __forceinline__ void
accumulate_d_f8(uint32_t RS[][num_tiles_k / 2][4], float d[][2]) {
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
      mma::rowsum_f8f8f32(d[fq], RS[fq][fk]);
    }
  }
}

template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q,
          uint32_t num_tiles_k, uint32_t num_tiles_v, SwizzleMode swizzle_mode,
          uint32_t stride, typename DTypeSVAccum>
__device__ __forceinline__ void
compute_fp16_sv(const smem_t<swizzle_mode, stride> &smem_V,
                uint32_t RS_f16[][num_tiles_k][4],
                DTypeSVAccum RO[][num_tiles_v][8], float d[][2]) {
  uint32_t smem_V_row_base =
      get_warp_idx_k<num_warps_q, num_warps_k>() * (num_tiles_k * 16) +
      get_lane_id() % 16;
  uint32_t smem_V_col_base = get_lane_id() / 16;
#pragma unroll
  for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      // load RV
      uint32_t RV[4];
      uint32_t offset_V = (smem_V).get_permuted_offset(
          smem_V_row_base + fk * 16, smem_V_col_base + fv * 2);
      smem_V.ldmatrix_m8n8x4_trans(offset_V, RV);
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        if constexpr (std::is_same<DTypeSVAccum, float>::value) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32(RO[fq][fv], RS_f16[fq][fk],
                                                    RV);
        } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
          mma::mma_sync_m16n16k16_row_col_f16f16f16((uint32_t *)RO[fq][fv],
                                                    RS_f16[fq][fk], RV);
        }
      }
    }
  }
}

#pragma nv_diag_suppress 174
template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q,
          uint32_t num_tiles_k, uint32_t num_tiles_v, SwizzleMode swizzle_mode,
          uint32_t stride, uint32_t RS_width = 4, typename T,
          typename DTypeSVAccum>
__device__ __forceinline__ void
compute_fp16_sv_permuted(const smem_t<swizzle_mode, stride> &smem_V,
                         T RS_f16[][num_tiles_k][RS_width],
                         DTypeSVAccum RO[][num_tiles_v][8], float d[][2],
                         uint32_t &offset_V) {
  static_assert(sizeof(T) == 4);

  // ! be sure you know what you are doing
#pragma unroll
  for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      // load RV
      uint32_t RV[4];
      smem_V.ldmatrix_m8n8x4_trans(offset_V, RV);
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        if constexpr (std::is_same<DTypeSVAccum, float>::value) {
          mma::mma_sync_m16n16k16_row_col_f16f16f32(
              RO[fq][fv], (uint32_t *)(RS_f16[fq][fk]), RV);
        } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
          mma::mma_sync_m16n16k16_row_col_f16f16f16(
              (uint32_t *)RO[fq][fv], (uint32_t *)(RS_f16[fq][fk]), RV);
        }
      }

      offset_V = smem_V.advance_offset_by_column<2>(offset_V, fv);
    }
    offset_V = smem_V.advance_offset_by_row<16>(offset_V - (2 * num_tiles_v));
  }

  // make offset_V their original value
  offset_V -= (16 * num_tiles_k * stride);
}

template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q,
          uint32_t num_tiles_k, uint32_t num_tiles_v, SwizzleMode swizzle_mode,
          uint32_t stride, uint32_t RS_width = 4, typename T,
          typename DTypeSVAccum>
__device__ __forceinline__ void
compute_fp16_sv_permuted_inst_buf(const smem_t<swizzle_mode, stride> &smem_V,
                                  T RS_f16[][num_tiles_k][RS_width],
                                  DTypeSVAccum RO[][num_tiles_v][8],
                                  float d[][2], uint32_t &offset_V) {
  static_assert(sizeof(T) == 4);
  static_assert(std::is_same<DTypeSVAccum, float>::value);

  uint32_t RO_inst_buf[num_tiles_q][num_tiles_v][4];

  // ! be sure you know what you are doing
#pragma unroll
  for (uint32_t fk = 0; fk < 1; fk++) {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      // load RV
      uint32_t RV[4];
      smem_V.ldmatrix_m8n8x4_trans(offset_V, RV);
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        {
          mma::mma_sync_m16n16k16_row_col_f16f16f16<mma::MMAMode::kInit>(
              (uint32_t *)RO_inst_buf[fq][fv], (uint32_t *)(RS_f16[fq][fk]),
              RV);
        }
      }

      offset_V = smem_V.advance_offset_by_column<2>(offset_V, fv);
    }
    offset_V = smem_V.advance_offset_by_row<16>(offset_V - (2 * num_tiles_v));
  }

#pragma unroll
  for (uint32_t fk = 1; fk < num_tiles_k; fk++) {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      // load RV
      uint32_t RV[4];
      smem_V.ldmatrix_m8n8x4_trans(offset_V, RV);
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        {
          mma::mma_sync_m16n16k16_row_col_f16f16f16<
              mma::MMAMode::kInplaceUpdate>((uint32_t *)RO_inst_buf[fq][fv],
                                            (uint32_t *)(RS_f16[fq][fk]), RV);
        }
      }

      offset_V = smem_V.advance_offset_by_column<2>(offset_V, fv);
    }
    offset_V = smem_V.advance_offset_by_row<16>(offset_V - (2 * num_tiles_v));
  }

  // accumulate into RO
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      RO[fq][fv][0] += __half2float(((half2 *)RO_inst_buf[fq][fv])[0].x);
      RO[fq][fv][1] += __half2float(((half2 *)RO_inst_buf[fq][fv])[0].y);
      RO[fq][fv][2] += __half2float(((half2 *)RO_inst_buf[fq][fv])[1].x);
      RO[fq][fv][3] += __half2float(((half2 *)RO_inst_buf[fq][fv])[1].y);
      RO[fq][fv][4] += __half2float(((half2 *)RO_inst_buf[fq][fv])[2].x);
      RO[fq][fv][5] += __half2float(((half2 *)RO_inst_buf[fq][fv])[2].y);
      RO[fq][fv][6] += __half2float(((half2 *)RO_inst_buf[fq][fv])[3].x);
      RO[fq][fv][7] += __half2float(((half2 *)RO_inst_buf[fq][fv])[3].y);
    }
  }

  // make offset_V their original value
  offset_V -= (16 * num_tiles_k * stride);
}
#pragma nv_diag_default 174

template <uint32_t num_tiles_q, uint32_t num_tiles_v,
          ComputeUnit compute_unit =
              ComputeUnit::kTensorCore> // compute unit for accumulate_d
__device__ __forceinline__ void normalize_d(float RO[][num_tiles_v][8],
                                            float m[][2], float d[][2]) {
  if constexpr (compute_unit == ComputeUnit::kCudaCore) {
    // accumulate_d performs partial accumulation with cuda core
    // aggregate d
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
      for (uint32_t k = 0; k < 2; k++) {
        d[fq][k] += __shfl_xor_sync(
            0xffffffff, d[fq][k], 0x1); // sum 0 and 1, 2 and 3
        d[fq][k] += __shfl_xor_sync(
            0xffffffff, d[fq][k], 0x2); // sum 0 and 2, 1 and 3
      }
    }
  }

  // divide O by d
  float d_rcp[num_tiles_q][2];
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++) {
      d_rcp[fq][k] = math::ptx_rcp(d[fq][k]);
    }
  }

#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
#pragma unroll
      for (uint32_t k = 0; k < 8; k++) {
        RO[fq][fv][k] *= d_rcp[fq][(k % 4) / 2];
      }
    }
  }
}


// Use eight-column fragments so the next Tensor Core dot product can overlap
// conversion and FP32 accumulation of the previous fragment. Each output keeps
// its four integer dot products followed by the original FP32 FMA.
template <SwizzleMode swizzle_mode, uint32_t stride>
__device__ __forceinline__ void
compute_int8_sv_pipelined_n8(const smem_t<swizzle_mode, stride> &smem_V,
                             int32_t RS_scale[][8][8], uint32_t RS_u8[][4][4],
                             float RO[][8][8]) {

  auto multiply = [&](uint32_t frag, int32_t partial[4]) {
    uint32_t RV[4][2];
#pragma unroll
    for (uint32_t fk = 0; fk < 4; ++fk) {
      const uint32_t offset = smem_V.get_permuted_offset(
          get_lane_id() % 8 + frag * 8, (get_lane_id() / 8) % 2 + fk * 2);
      smem_V.ldmatrix_m8n8x2(offset, RV[fk]);
      if (fk == 0)
        mma::mma_sync_m16n8k32_row_col_u8s8s32<mma::MMAMode::kInit>(
            partial, RS_u8[0][fk], RV[fk]);
      else
        mma::mma_sync_m16n8k32_row_col_u8s8s32<mma::MMAMode::kInplaceUpdate>(
            partial, RS_u8[0][fk], RV[fk]);
    }
  };
  auto accumulate = [&](uint32_t frag, const int32_t partial[4]) {
#pragma unroll
    for (uint32_t k = 0; k < 4; ++k) {
      float &output = RO[0][frag / 2][(frag % 2) * 4 + k];
      output = fmaf(__int2float_rn(partial[k]),
                    __int_as_float(RS_scale[0][0][k / 2]), output);
    }
  };

  int32_t partial[2][4];
  multiply(0, partial[0]);
#pragma unroll
  for (uint32_t frag = 1; frag < 16; ++frag) {
    multiply(frag, partial[frag % 2]);
    accumulate(frag - 1, partial[(frag - 1) % 2]);
  }
  accumulate(15, partial[1]);
}

template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q,
          uint32_t num_tiles_k, uint32_t num_tiles_v, SwizzleMode swizzle_mode,
          uint32_t stride, bool pipeline_pv = true, typename DTypeSVAccum>
__device__ __forceinline__ void
compute_int8_sv(const smem_t<swizzle_mode, stride> &smem_V,
                int32_t RS_scale[][num_tiles_k][8],
                uint32_t RS_u8[][num_tiles_k / 2][4],
                DTypeSVAccum RO[][num_tiles_v][8]) {
#if __CUDA_ARCH__ == 890
  if constexpr (pipeline_pv && num_warps_q == 4 && num_warps_k == 1 && num_tiles_q == 1 &&
                num_tiles_k == 8 && num_tiles_v == 8) {
    compute_int8_sv_pipelined_n8<swizzle_mode, stride>(smem_V, RS_scale, RS_u8, RO);
    return;
  }
#endif

  static_assert(std::is_same<DTypeSVAccum, float>::value);
  uint32_t smem_V_row_base = get_lane_id() % 8 + (get_lane_id() / 16) * 8;
  uint32_t smem_V_col_base = (get_lane_id() / 8) % 2;
  uint32_t offsets_V[num_tiles_k / 2];
#pragma unroll
  for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
    offsets_V[fk] =
        smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk * 2);
  }
#pragma unroll
  for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
    uint32_t RV[num_tiles_k / 2][4];
#pragma unroll
    for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
      smem_V.ldmatrix_m8n8x4(offsets_V[fk], RV[fk]);
      offsets_V[fk] = smem_V.advance_offset_by_row<16>(offsets_V[fk]);
    }
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
      int32_t partial[8];
      mma::mma_sync_m16n16k32_row_col_u8s8s32<mma::MMAMode::kInit>(
          partial, RS_u8[fq][0], RV[0]);
#pragma unroll
      for (uint32_t fk = 1; fk < num_tiles_k / 2; fk++) {
        mma::mma_sync_m16n16k32_row_col_u8s8s32<
            mma::MMAMode::kInplaceUpdate>(partial, RS_u8[fq][fk], RV[fk]);
      }
#pragma unroll
      for (uint32_t k = 0; k < 8; k++) {
        RO[fq][fv][k] = fmaf(__int2float_rn(partial[k]),
                              __int_as_float(RS_scale[fq][0][(k % 4) / 2]),
                              RO[fq][fv][k]);
      }
    }
  }
}

template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q,
          uint32_t num_tiles_k, uint32_t num_tiles_v, SwizzleMode swizzle_mode,
          uint32_t stride, typename DTypeSVAccum>
__device__ __forceinline__ void
compute_fp8_sv(const smem_t<swizzle_mode, stride> &smem_V,
               uint32_t RS_f8[][num_tiles_k / 2][4],
               DTypeSVAccum RO[][num_tiles_v][8], float d[][2]) {
  uint32_t smem_V_row_base = get_lane_id() % 8 + (get_lane_id() / 16) * 8;
  // uint32_t smem_V_col_base = get_warp_idx_k<num_warps_q, num_warps_k>() *
  // ((16 * num_tiles_k) / 16) + (get_lane_id() / 8) % 2;
  uint32_t smem_V_col_base = (get_lane_id() / 8) % 2;
#pragma unroll
  for (uint32_t fk = 0; fk < num_tiles_k / 2; fk++) {
    uint32_t offset_V =
        smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk * 2);
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      // load RV
      uint32_t RV[4];
      // uint32_t offset_V = (smem_V).get_permuted_offset(smem_V_row_base + fv *
      // 16, smem_V_col_base + fk * 2);
      smem_V.ldmatrix_m8n8x4(offset_V, RV);
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        if constexpr (std::is_same<DTypeSVAccum, float>::value) {
          mma::mma_sync_m16n16k32_row_col_f8f8f32(RO[fq][fv], RS_f8[fq][fk],
                                                  RV);
        } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
          // ! Not Implemented
        }
      }
      offset_V = smem_V.advance_offset_by_row<16>(offset_V);
    }
  }
}

template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q,
          uint32_t num_tiles_k, uint32_t num_tiles_v, SwizzleMode swizzle_mode,
          uint32_t stride, typename DTypeSVAccum>
__device__ __forceinline__ void
compute_fp8_sv_inst_buf(const smem_t<swizzle_mode, stride> &smem_V,
                        uint32_t RS_f8[][num_tiles_k / 2][4],
                        DTypeSVAccum RO[][num_tiles_v][8], float d[][2]) {
  uint32_t smem_V_row_base = get_lane_id() % 8 + (get_lane_id() / 16) * 8;
  // uint32_t smem_V_col_base = get_warp_idx_k<num_warps_q, num_warps_k>() *
  // ((16 * num_tiles_k) / 16) + (get_lane_id() / 8) % 2;
  uint32_t smem_V_col_base = (get_lane_id() / 8) % 2;

  float RO_inst_buf[num_tiles_q][num_tiles_v][8];

#pragma unroll
  for (uint32_t fk = 0; fk < 1; fk++) {
    uint32_t offset_V =
        smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk * 2);
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      // load RV
      uint32_t RV[4];
      // uint32_t offset_V = (smem_V).get_permuted_offset(smem_V_row_base + fv *
      // 16, smem_V_col_base + fk * 2);
      smem_V.ldmatrix_m8n8x4(offset_V, RV);
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        if constexpr (std::is_same<DTypeSVAccum, float>::value) {
          mma::mma_sync_m16n16k32_row_col_f8f8f32<mma::MMAMode::kInit>(
              RO_inst_buf[fq][fv], RS_f8[fq][fk], RV);
        } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
          // ! Not Implemented
        }
      }
      offset_V = smem_V.advance_offset_by_row<16>(offset_V);
    }
  }

#pragma unroll
  for (uint32_t fk = 1; fk < num_tiles_k / 2; fk++) {
    uint32_t offset_V =
        smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk * 2);
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      // load RV
      uint32_t RV[4];
      // uint32_t offset_V = (smem_V).get_permuted_offset(smem_V_row_base + fv *
      // 16, smem_V_col_base + fk * 2);
      smem_V.ldmatrix_m8n8x4(offset_V, RV);
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        if constexpr (std::is_same<DTypeSVAccum, float>::value) {
          mma::mma_sync_m16n16k32_row_col_f8f8f32<mma::MMAMode::kInplaceUpdate>(
              RO_inst_buf[fq][fv], RS_f8[fq][fk], RV);
        } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
          // ! Not Implemented
        }
      }
      offset_V = smem_V.advance_offset_by_row<16>(offset_V);
    }
  }

#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      RO[fq][fv][0] += RO_inst_buf[fq][fv][0];
      RO[fq][fv][1] += RO_inst_buf[fq][fv][1];
      RO[fq][fv][2] += RO_inst_buf[fq][fv][2];
      RO[fq][fv][3] += RO_inst_buf[fq][fv][3];
      RO[fq][fv][4] += RO_inst_buf[fq][fv][4];
      RO[fq][fv][5] += RO_inst_buf[fq][fv][5];
      RO[fq][fv][6] += RO_inst_buf[fq][fv][6];
      RO[fq][fv][7] += RO_inst_buf[fq][fv][7];
    }
  }
}

template <uint32_t num_warps_q, uint32_t num_warps_k, uint32_t num_tiles_q,
          uint32_t num_tiles_k, uint32_t num_tiles_v, SwizzleMode swizzle_mode,
          uint32_t stride, typename DTypeSVAccum>
__device__ __forceinline__ void
compute_fp8_sv_inst_buf_fp16_accu(const smem_t<swizzle_mode, stride> &smem_V,
                                  uint32_t RS_f8[][num_tiles_k / 2][4],
                                  DTypeSVAccum RO[][num_tiles_v][8],
                                  float d[][2]) {
  uint32_t smem_V_row_base = get_lane_id() % 8 + (get_lane_id() / 16) * 8;
  // uint32_t smem_V_col_base = get_warp_idx_k<num_warps_q, num_warps_k>() *
  // ((16 * num_tiles_k) / 16) + (get_lane_id() / 8) % 2;
  uint32_t smem_V_col_base = (get_lane_id() / 8) % 2;

  uint32_t RO_int32[num_tiles_q][num_tiles_v][4];

#pragma unroll
  for (uint32_t fk = 0; fk < 1; fk++) {
    uint32_t offset_V =
        smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk * 2);
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      // load RV
      uint32_t RV[4];
      // uint32_t offset_V = (smem_V).get_permuted_offset(smem_V_row_base + fv *
      // 16, smem_V_col_base + fk * 2);
      smem_V.ldmatrix_m8n8x4(offset_V, RV);
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        if constexpr (std::is_same<DTypeSVAccum, float>::value) {
          // mma::mma_sync_m16n16k32_row_col_f8f8f32<mma::MMAMode::kInit>(RO_inst_buf[fq][fv],
          // RS_f8[fq][fk], RV);
          mma::mma_sync_m16n16k32_row_col_f8f8f16<mma::MMAMode::kInit>(
              RO_int32[fq][fv], RS_f8[fq][fk], RV);
        } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
          // ! Not Implemented
        }
      }
      offset_V = smem_V.advance_offset_by_row<16>(offset_V);
    }
  }

#pragma unroll
  for (uint32_t fk = 1; fk < num_tiles_k / 2; fk++) {
    uint32_t offset_V =
        smem_V.get_permuted_offset(smem_V_row_base, smem_V_col_base + fk * 2);
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      // load RV
      uint32_t RV[4];
      // uint32_t offset_V = (smem_V).get_permuted_offset(smem_V_row_base + fv *
      // 16, smem_V_col_base + fk * 2);
      smem_V.ldmatrix_m8n8x4(offset_V, RV);
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        if constexpr (std::is_same<DTypeSVAccum, float>::value) {
          // mma::mma_sync_m16n16k32_row_col_f8f8f32<mma::MMAMode::kInplaceUpdate>(RO_inst_buf[fq][fv],
          // RS_f8[fq][fk], RV);
          mma::mma_sync_m16n16k32_row_col_f8f8f16<mma::MMAMode::kInplaceUpdate>(
              RO_int32[fq][fv], RS_f8[fq][fk], RV);
        } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
          // ! Not Implemented
        }
      }
      offset_V = smem_V.advance_offset_by_row<16>(offset_V);
    }
  }
  float RO_tmp_float[2];
#pragma unroll
  for (int i = 0; i < num_tiles_q; i++) {
#pragma unroll
    for (int j = 0; j < num_tiles_v; j++) {
#pragma unroll
      for (int k = 0; k < 4; k++) {
        unpack_half2_from_uint32_to_float(RO_tmp_float, RO_int32[i][j][k]);
        RO[i][j][k * 2 + 0] += RO_tmp_float[0];
        RO[i][j][k * 2 + 1] += RO_tmp_float[1];
      }
    }
  }

  // #pragma unroll
  //   for (uint32_t fq = 0; fq < num_tiles_q; fq++)
  //   {
  // #pragma unroll
  //     for (uint32_t fv = 0; fv < num_tiles_v; fv++)
  //     {
  //       RO[fq][fv][0] += RO_inst_buf[fq][fv][0];
  //       RO[fq][fv][1] += RO_inst_buf[fq][fv][1];
  //       RO[fq][fv][2] += RO_inst_buf[fq][fv][2];
  //       RO[fq][fv][3] += RO_inst_buf[fq][fv][3];
  //       RO[fq][fv][4] += RO_inst_buf[fq][fv][4];
  //       RO[fq][fv][5] += RO_inst_buf[fq][fv][5];
  //       RO[fq][fv][6] += RO_inst_buf[fq][fv][6];
  //       RO[fq][fv][7] += RO_inst_buf[fq][fv][7];
  //     }
  //   }
}
