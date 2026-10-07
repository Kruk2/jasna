// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright (c) 2024 by SageAttention team.
// SPDX-FileContributor: Modified by NVIDIA CORPORATION & AFFILIATES, 2025.
// Derived from SageAttention (https://github.com/thu-ml/SageAttention)
// commit d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5.
// Modifications: removed torch/extension.h dependency, flattened include paths.

#pragma once

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

#include <cstdint>
#include <math_constants.h>
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>

#include "cp_async.cuh"
#include "math.cuh"
#include "mma.cuh"
#include "permuted_smem.cuh"

#include "attn_utils.cuh"

#define PACK_SIZE_QK 16 // as if it is int8
#define PACK_SIZE_V 16  // int8
#define PACK_SIZE_O 8   // fp16

// treat as if int8 tensor core
#define MMA_QK_M 16
#define MMA_QK_N 16
#define MMA_QK_K 32

// unsigned INT8 softmax x signed INT8 V tensor core
#define MMA_SV_M 16
#define MMA_SV_N 16
#define MMA_SV_K 32

// Every warp reads the same immutable descriptors and chooses the same tile.
__device__ __forceinline__ uint32_t attention_next_active_tile(
    const float *descriptors, uint32_t start, uint32_t count) {
  for (uint32_t base = start; base < count; base += 32) {
    const uint32_t tile = base + get_lane_id();
    const bool active = tile < count && descriptors[tile] != -CUDART_INF_F;
    const uint32_t votes = __ballot_sync(0xffffffff, active);
    if (votes) return base + __ffs(votes) - 1;
  }
  return count;
}

template <bool skip_masked_tiles, uint32_t CTA_Q, uint32_t CTA_K,
          uint32_t WARP_Q, uint32_t WARP_K,
          uint32_t head_dim, DataType DTypeQK, QuantGranularity Q_GRAN,
          QuantGranularity K_GRAN, typename DTypeSVAccum = float,
          bool use_inst_buffer = false, typename DTypeOut = half,
          ComputeUnit DenominatorAccumUnit,
          MaskMode mask_mode = MaskMode::kNone, bool return_lse = false,
          bool fuse_v_scale = false, bool fuse_v_mean = false,
          bool use_pv_fp16_accu = false,
          bool fuse_fp32_probabilities = true, bool skip_masked_copies = false,
          typename Offset = uint32_t>
__device__ __forceinline__ void qk_int_sv_i8_attn_body(
    int8_t *__restrict__ Q, int8_t *__restrict__ K, int8_t *__restrict__ V,
    DTypeOut *__restrict__ O, float *__restrict__ Lse,
    float *__restrict__ Q_scale, float *__restrict__ K_scale,
    float *__restrict__ V_scale, float *__restrict__ V_mean,
    const void *__restrict__ AttnMask, const int64_t mask_stride_b,
    const int64_t mask_stride_h, const int64_t mask_stride_q,
    const int64_t mask_stride_k, const int mask_dtype_code,
    const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
    const Offset stride_bz_q, const uint32_t stride_seq_q,
    const Offset stride_h_q, const Offset stride_bz_k,
    const uint32_t stride_seq_k, const Offset stride_h_k,
    const Offset stride_bz_v, const Offset stride_h_v,
    const uint32_t stride_d_v, const Offset stride_bz_o,
    const uint32_t stride_seq_o, const Offset stride_h_o, float sm_scale,
    const float *__restrict__ MaskTileBias) {
  // compile time check
  static_assert(DTypeQK == DataType::kInt8 || DTypeQK == DataType::kInt4,
                "DTypeQK must be int8 or int4");
  static_assert(Q_GRAN == QuantGranularity::kPerBlock ||
                    Q_GRAN == QuantGranularity::kPerWarp ||
                    Q_GRAN == QuantGranularity::kPerThread,
                "Q_GRAN must be kPerBlock, kPerWarp or kPerThread");
  static_assert(K_GRAN == QuantGranularity::kPerBlock ||
                    K_GRAN == QuantGranularity::kPerWarp ||
                    K_GRAN == QuantGranularity::kPerThread,
                "K_GRAN must be kPerBlock, kPerWarp or kPerThread");
  static_assert(head_dim % 64 == 0, "head_dim must be a multiple of 64");
  static_assert(std::is_same<DTypeSVAccum, float>::value,
                "DTypeSVAccum must be float, half is WIP");
  static_assert(DenominatorAccumUnit == ComputeUnit::kCudaCore,
                "pure INT8 attention accumulates the softmax denominator on CUDA cores");
  static_assert(std::is_same<DTypeOut, half>::value ||
                    std::is_same<DTypeOut, nv_bfloat16>::value,
                "DTypeOut must be half or nv_bfloat16");
  static_assert(CTA_K % 64 == 0);
  static_assert(CTA_Q / CTA_K <= 2); // for efficient causal implementation

  constexpr uint32_t num_warps_q = CTA_Q / WARP_Q;
  constexpr uint32_t num_warps_k = CTA_K / WARP_K;
  constexpr uint32_t num_warps = num_warps_q * num_warps_k;
  constexpr uint32_t num_tiles_q = WARP_Q / MMA_QK_M;
  constexpr uint32_t num_tiles_k = WARP_K / MMA_QK_N;
  constexpr uint32_t num_tiles_qk_inner = (DTypeQK == DataType::kInt8)
                                              ? (head_dim / MMA_QK_K)
                                              : (head_dim / 2 / MMA_QK_K);
  constexpr uint32_t num_tiles_v = head_dim / MMA_SV_N;
  constexpr bool custom_mask = mask_mode == MaskMode::kCustom ||
                               mask_mode == MaskMode::kCustomKey ||
                               mask_mode == MaskMode::kPreparedKey;
  // For unmasked and causal FP32 kernels, retain raw scores until update_mdo
  // fuses score scaling, max subtraction, and conversion to the exp2 domain.
  // Custom masks keep pre-scaled scores so additive bias values retain their
  // existing semantics.
  constexpr bool pre_scale_scores = custom_mask;
#if __CUDA_ARCH__ >= 1000
  // Blackwell schedules the lower-pressure generic FP32 path better for short
  // rows. The launcher passes false at K <= 512; older architectures retain
  // the fused probability path, which benchmarks faster on Ada.
  constexpr bool use_fused_fp32_probabilities = fuse_fp32_probabilities;
#else
  constexpr bool use_fused_fp32_probabilities = true;
#endif
  constexpr uint32_t QK_SMEM_STRIDE =
      (DTypeQK == DataType::kInt8) ? (head_dim) : (head_dim / 2);
  constexpr uint32_t O_SMEM_STRIDE = head_dim;
  constexpr uint32_t V_SMEM_STRIDE = CTA_K;

  extern __shared__ int8_t smem[];

  const uint32_t lane_id = get_lane_id();
  const uint32_t warp_id = get_warp_id();

  // maximize L2 hit rate
  const uint32_t batch_id = blockIdx.z;
  const uint32_t bx = blockIdx.x;
  const uint32_t num_qo_heads = gridDim.y;
  const uint32_t head_id = blockIdx.y;

  const float *key_mask = nullptr;
  const float *key_tile_bias = nullptr;
  if constexpr (mask_mode == MaskMode::kPreparedKey) {
    const int64_t offset = batch_id * mask_stride_b + head_id * mask_stride_h;
    key_mask = static_cast<const float *>(AttnMask) + offset;
    key_tile_bias = MaskTileBias + offset;
  }

  // transfer to base 2 instead of base e with better numerical efficiency
  sm_scale *= math::log2e;

  // RS holds the fragment of S
  int32_t RS[num_tiles_q][num_tiles_k][8];
  DTypeSVAccum RO[num_tiles_q][num_tiles_v][8];
  float m[num_tiles_q][2]; // max
  float d[num_tiles_q][2]; // denominator
  bool valid[num_tiles_q][2];

  uint32_t q_scale_idx, k_scale_idx;

  if constexpr (Q_GRAN == QuantGranularity::kPerBlock) {
    const uint32_t num_block_q = gridDim.x;
    q_scale_idx =
        batch_id * num_qo_heads * num_block_q + head_id * num_block_q + bx;
  } else if constexpr (Q_GRAN == QuantGranularity::kPerWarp) {
    const uint32_t num_warp_block_q = gridDim.x * num_warps_q;
    q_scale_idx = batch_id * num_qo_heads * num_warp_block_q +
                  head_id * num_warp_block_q + bx * num_warps_q +
                  get_warp_idx_q<num_warps_q, num_warps_k>();
  } else if constexpr (Q_GRAN == QuantGranularity::kPerThread) {
    if constexpr (mask_mode == MaskMode::kPreparedKey || CTA_Q == 64) {
      // Q scales are packed in 128-row blocks even when attention uses
      // smaller query tiles. D=128 also shares one scale across two warps.
      constexpr uint32_t quant_warp_q = head_dim == 256 ? 16 : 32;
      const uint32_t groups_per_head = div_ceil(qo_len, 128) * (128 / quant_warp_q);
      const uint32_t query = bx * CTA_Q +
          get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q;
      q_scale_idx = ((batch_id * num_qo_heads + head_id) * groups_per_head +
                     query / quant_warp_q) * 8 + lane_id / 4;
    } else if constexpr (head_dim == 128 && WARP_Q == 16) {
      constexpr uint32_t quant_warps_q = CTA_Q / 32;
      const uint32_t num_warp_block_q = gridDim.x * quant_warps_q;
      q_scale_idx =
          batch_id * num_qo_heads * (num_warp_block_q * 8) +
          head_id * (num_warp_block_q * 8) + bx * (quant_warps_q * 8) +
          (get_warp_idx_q<num_warps_q, num_warps_k>() / 2) * 8 + lane_id / 4;
    } else {
      const uint32_t num_warp_block_q = gridDim.x * num_warps_q;
      q_scale_idx =
          batch_id * num_qo_heads * (num_warp_block_q * 8) +
          head_id * (num_warp_block_q * 8) + bx * (num_warps_q * 8) +
          get_warp_idx_q<num_warps_q, num_warps_k>() * 8 + lane_id / 4;
    }
  }

  if constexpr (K_GRAN == QuantGranularity::kPerBlock) {
    const uint32_t num_block_k = div_ceil(kv_len, CTA_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_block_k +
                  (head_id / num_kv_groups) * num_block_k;
  } else if constexpr (K_GRAN == QuantGranularity::kPerWarp) {
    const uint32_t num_warp_block_k =
        div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
    k_scale_idx = batch_id * (num_qo_heads / num_kv_groups) * num_warp_block_k +
                  (head_id / num_kv_groups) * num_warp_block_k +
                  get_warp_idx_k<num_warps_q, num_warps_k>();
  } else if constexpr (K_GRAN == QuantGranularity::kPerThread) {
    const uint32_t num_warp_block_k =
        div_ceil(kv_len, CTA_K) * (CTA_K / WARP_K);
    k_scale_idx =
        batch_id * (num_qo_heads / num_kv_groups) * (num_warp_block_k * 4) +
        (head_id / num_kv_groups) * (num_warp_block_k * 4) +
        get_warp_idx_k<num_warps_q, num_warps_k>() * 4 + lane_id % 4;
  }

  constexpr uint32_t k_scale_advance_offset =
      (K_GRAN == QuantGranularity::kPerBlock)  ? 1
      : (K_GRAN == QuantGranularity::kPerWarp) ? (CTA_K / WARP_K)
                                               : (CTA_K / WARP_K) * 4;

  // initialize o, m, d
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      if constexpr (std::is_same<DTypeSVAccum, float>::value) {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++) {
          RO[fq][fv][k] = 0.0f;
        }
      } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
#pragma unroll
        for (uint32_t k = 0; k < 4; k++) {
          ((int32_t *)RO[fq][fv])[k] = 0;
        }
      }
    }
  }
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t k = 0; k < 2; k++) {
      m[fq][k] = -50000.0f;
      d[fq][k] = 1.0f;
      valid[fq][k] = false;
    }
  }

// Only Ada benefits from the cached-Q pipeline in measured D128 workloads.
#if __CUDA_ARCH__ == 890
  constexpr bool cache_query =
      CTA_Q == 128 && head_dim == 128 && CTA_K == 128 && mask_mode == MaskMode::kNone;
#else
  constexpr bool cache_query = false;
#endif
  constexpr uint32_t K_smem_idx_offset = cache_query ? 0 : CTA_Q;
  constexpr uint32_t V_smem_idx_offset =
      cache_query ? 2 * CTA_K : CTA_Q + CTA_K;

  constexpr SwizzleMode swizzle_mode_QK =
      (QK_SMEM_STRIDE == 32)   ? SwizzleMode::k32B
      : (QK_SMEM_STRIDE == 64) ? SwizzleMode::k64B
                               : SwizzleMode::k128B;
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_Q(smem);
  smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> smem_K(
      smem + K_smem_idx_offset * QK_SMEM_STRIDE);
  constexpr SwizzleMode swizzle_mode_V =
      (V_SMEM_STRIDE == 64) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V> smem_V(
      smem + V_smem_idx_offset * QK_SMEM_STRIDE);
  constexpr SwizzleMode swizzle_mode_O =
      (O_SMEM_STRIDE == 32) ? SwizzleMode::k64B : SwizzleMode::k128B;
  smem_t<swizzle_mode_O, O_SMEM_STRIDE / PACK_SIZE_O> smem_O(smem);

  constexpr uint32_t global_to_shared_line_lanes_QK = (QK_SMEM_STRIDE == 32) ? 2
                                                      : (QK_SMEM_STRIDE == 64)
                                                          ? 4
                                                          : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_QK =
      (QK_SMEM_STRIDE == 32)   ? 16
      : (QK_SMEM_STRIDE == 64) ? 8
                               : 4;
  constexpr uint32_t global_to_shared_line_lanes_V =
      (V_SMEM_STRIDE == 64) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_V =
      (V_SMEM_STRIDE == 64) ? 8 : 4;
  constexpr uint32_t global_to_shared_line_lanes_O =
      (O_SMEM_STRIDE == 32) ? 4 : 8;
  constexpr uint32_t global_to_shared_copy_lines_per_warp_O =
      (O_SMEM_STRIDE == 32) ? 8 : 4;

  constexpr uint32_t QK_smem_iters_row =
      QK_SMEM_STRIDE / (global_to_shared_line_lanes_QK * PACK_SIZE_QK);
  constexpr uint32_t Q_smem_iters_col =
      CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t K_smem_iters_col =
      CTA_K / (num_warps * global_to_shared_copy_lines_per_warp_QK);
  constexpr uint32_t V_smem_iters_row =
      V_SMEM_STRIDE / (global_to_shared_line_lanes_V * PACK_SIZE_V);
  constexpr uint32_t V_smem_iters_col =
      head_dim / (num_warps * global_to_shared_copy_lines_per_warp_V);
  constexpr uint32_t O_smem_iters_row =
      O_SMEM_STRIDE / (global_to_shared_line_lanes_O * PACK_SIZE_O);
  constexpr uint32_t O_smem_iters_col =
      CTA_Q / (num_warps * global_to_shared_copy_lines_per_warp_O);

  int8_t *Q_lane_base_ptr =
      Q + batch_id * stride_bz_q + head_id * stride_h_q +
      (bx * CTA_Q + CTA_Q / num_warps * warp_id +
       lane_id / global_to_shared_line_lanes_QK) *
          stride_seq_q +
      (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  int8_t *K_lane_base_ptr =
      K + batch_id * stride_bz_k + (head_id / num_kv_groups) * stride_h_k +
      (CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK) *
          stride_seq_k +
      (lane_id % global_to_shared_line_lanes_QK) * PACK_SIZE_QK;
  int8_t *V_lane_base_ptr =
      V + batch_id * stride_bz_v + (head_id / num_kv_groups) * stride_h_v +
      head_dim / num_warps * warp_id * stride_d_v +
      lane_id / global_to_shared_line_lanes_V * stride_d_v +
      (lane_id % global_to_shared_line_lanes_V) * PACK_SIZE_V;
  uint32_t Q_smem_offset_load = smem_Q.get_permuted_offset(
      warp_id * global_to_shared_copy_lines_per_warp_QK * Q_smem_iters_col +
          lane_id / global_to_shared_line_lanes_QK,
      lane_id % global_to_shared_line_lanes_QK);
  uint32_t K_smem_offset_load = smem_K.get_permuted_offset(
      warp_id * global_to_shared_copy_lines_per_warp_QK * K_smem_iters_col +
          lane_id / global_to_shared_line_lanes_QK,
      lane_id % global_to_shared_line_lanes_QK);
  uint32_t V_smem_offset_load = smem_V.get_permuted_offset(
      warp_id * global_to_shared_copy_lines_per_warp_V * V_smem_iters_col +
          lane_id / global_to_shared_line_lanes_V,
      lane_id % global_to_shared_line_lanes_V);

  uint32_t Q_smem_offset_mma = smem_Q.get_permuted_offset(
      get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id % 16,
      lane_id / 16);
  uint32_t K_smem_offset_mma = smem_K.get_permuted_offset(
      get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + lane_id % 8 +
          (lane_id / 16) * 8,
      (lane_id / 8) % 2);
  // for causal masking
  uint32_t Q_idx_lane_base =
      bx * CTA_Q + get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q +
      lane_id / 4;
  uint32_t K_idx_lane_base =
      get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K + 2 * (lane_id % 4);

  // for loading
  uint32_t Q_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id +
                                  lane_id / global_to_shared_line_lanes_QK;
  uint32_t K_load_idx_lane_base =
      CTA_K / num_warps * warp_id + lane_id / global_to_shared_line_lanes_QK;

  const uint32_t num_iterations = div_ceil(
      mask_mode == MaskMode::kCausal ? min(kv_len, (bx + 1) * CTA_Q) : kv_len,
      CTA_K);

  if constexpr (!cache_query) {
    // load Q with predicate
    load_global_to_share<global_to_shared_line_lanes_QK,
                         global_to_shared_copy_lines_per_warp_QK,
                         QK_smem_iters_row, Q_smem_iters_col, swizzle_mode_QK,
                         QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_Q>(
        &Q_lane_base_ptr, Q_smem_offset_load, stride_seq_q, smem_Q,
        Q_load_idx_lane_base, qo_len);
    cp_async::commit_group();
    cp_async::wait_group<0>();
    __syncthreads();
  }

  // for num_tiles_qk_inner = 1, we load all Qs in register
  uint32_t RQ[num_tiles_q][4];
  if constexpr (num_tiles_qk_inner == 1) {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
      smem_Q.ldmatrix_m8n8x4(Q_smem_offset_mma, RQ[fq]);
      Q_smem_offset_mma = smem_Q.advance_offset_by_row<16>(Q_smem_offset_mma);
    }
  }

  uint32_t RQ_cached[num_tiles_q][num_tiles_qk_inner][4];
  if constexpr (cache_query) {
    const int8_t *query = Q + batch_id * stride_bz_q + head_id * stride_h_q;
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; ++fq) {
      const uint32_t row = bx * CTA_Q +
                           get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q +
                           fq * 16 + lane_id / 4;
#pragma unroll
      for (uint32_t inner = 0; inner < num_tiles_qk_inner; ++inner) {
        const uint32_t col = inner * 32 + (lane_id % 4) * 4;
        RQ_cached[fq][inner][0] =
            row < qo_len ? __ldg(reinterpret_cast<const uint32_t *>(
                               query + row * stride_seq_q + col))
                         : 0;
        RQ_cached[fq][inner][1] =
            row + 8 < qo_len ? __ldg(reinterpret_cast<const uint32_t *>(
                                   query + (row + 8) * stride_seq_q + col))
                             : 0;
        RQ_cached[fq][inner][2] =
            row < qo_len ? __ldg(reinterpret_cast<const uint32_t *>(
                               query + row * stride_seq_q + col + 16))
                         : 0;
        RQ_cached[fq][inner][3] =
            row + 8 < qo_len ? __ldg(reinterpret_cast<const uint32_t *>(
                                   query + (row + 8) * stride_seq_q + col + 16))
                             : 0;
      }
    }
  }

  if constexpr (skip_masked_copies) {
    static_assert(skip_masked_tiles && mask_mode == MaskMode::kPreparedKey &&
                  head_dim == 128);
    const float q_scale = Q_scale[q_scale_idx];
    const float original_sm_scale = sm_scale;
    auto load_key = [&](uint32_t tile) {
      int8_t *pointer = K_lane_base_ptr + uint64_t(tile) * CTA_K * stride_seq_k;
      uint32_t offset = K_smem_offset_load;
      load_global_to_share<global_to_shared_line_lanes_QK,
                           global_to_shared_copy_lines_per_warp_QK,
                           QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK,
                           QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
          &pointer, offset, stride_seq_k, smem_K,
          K_load_idx_lane_base + tile * CTA_K, kv_len);
      cp_async::commit_group();
    };
    auto load_value = [&](uint32_t tile) {
      int8_t *pointer = V_lane_base_ptr + tile * CTA_K;
      uint32_t offset = V_smem_offset_load;
      load_int8_V_global_to_share<
          global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V,
          V_smem_iters_row, V_smem_iters_col, swizzle_mode_V,
          V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(&pointer, offset, stride_d_v,
                                              smem_V);
      cp_async::commit_group();
    };
    uint32_t current =
        attention_next_active_tile(key_tile_bias, 0, num_iterations);
    if (current < num_iterations) {
      load_key(current);
      load_value(current);
    }
    while (current < num_iterations) {
      const uint32_t next =
          attention_next_active_tile(key_tile_bias, current + 1, num_iterations);
      cp_async::wait_group<1>();
      __syncthreads();
      K_idx_lane_base = get_warp_idx_k<num_warps_q, num_warps_k>() * WARP_K +
                        2 * (lane_id % 4) + current * CTA_K;
      const float dequant_scale =
          q_scale * K_scale[k_scale_idx + current * k_scale_advance_offset];
      sm_scale = original_sm_scale * dequant_scale;
      uint32_t RS_u8[num_tiles_q][num_tiles_k / 2][4];
      compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                     num_tiles_qk_inner, swizzle_mode_QK,
                     QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
          smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
      // The physical final tile keeps its original FP32 mask/probability path.
      if (current + 1 == num_iterations) {
        auto &scores =
            reinterpret_cast<float(&)[num_tiles_q][num_tiles_k][8]>(RS);
#pragma unroll
        for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
          for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
            for (uint32_t k = 0; k < 8; k++)
              scores[fq][fk][k] = __int2float_rz(RS[fq][fk][k]);
          }
        }
        apply_prepared_key_mask<num_tiles_q, num_tiles_k>(K_idx_lane_base, scores,
                                                          key_mask, sm_scale);
        update_mdo_f32_u8<num_tiles_q, num_tiles_k, num_tiles_v>(
            scores, RO, m, d, S_U8_OFFSET, RS_u8);
      } else {
        const float tile_bias = key_tile_bias[K_idx_lane_base / 128];
        // The constant-tile branch pays off at D=128. A single FP32 path
        // reduces register pressure and runs faster at D=64 and D=256.
        if (head_dim == 128 && isfinite(tile_bias)) {
          update_mdo_i32_u8<num_tiles_q, num_tiles_k, num_tiles_v, true>(
              RS, RO, m, d, sm_scale, S_U8_OFFSET, RS_u8,
              tile_bias * math::log2e);
        } else {
          auto &scores =
              reinterpret_cast<float(&)[num_tiles_q][num_tiles_k][8]>(RS);
#pragma unroll
          for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
              for (uint32_t k = 0; k < 8; k++)
                scores[fq][fk][k] = __int2float_rz(RS[fq][fk][k]);
            }
          }
          apply_prepared_key_mask<num_tiles_q, num_tiles_k>(
              K_idx_lane_base, scores, key_mask, sm_scale);
          update_mdo_f32_u8<num_tiles_q, num_tiles_k, num_tiles_v>(
              scores, RO, m, d, S_U8_OFFSET, RS_u8);
        }
      }
      __syncthreads();
      if (next < num_iterations) {
        load_key(next);
        cp_async::wait_group<1>();
      } else {
        cp_async::wait_group<0>();
      }
      __syncthreads();
      compute_int8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                      num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V,
                        mask_mode == MaskMode::kNone>(
          smem_V, RS, RS_u8, RO);
      __syncthreads();
      if (next < num_iterations)
        load_value(next);
      current = next;
    }
  } else {
    // load K with predicate
    load_global_to_share<global_to_shared_line_lanes_QK,
                         global_to_shared_copy_lines_per_warp_QK,
                         QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK,
                         QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
        &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K,
        K_load_idx_lane_base, kv_len);
    cp_async::commit_group();

    float q_scale = Q_scale[q_scale_idx];

    float original_sm_scale = sm_scale;
    float dequant_scale =
        q_scale * K_scale[k_scale_idx + 0 * k_scale_advance_offset];

    sm_scale = original_sm_scale * dequant_scale;

    // load V
    // V is padded to a complete CTA_K tile by the quantizer.
    load_int8_V_global_to_share<global_to_shared_line_lanes_V,
                               global_to_shared_copy_lines_per_warp_V,
                               V_smem_iters_row, V_smem_iters_col, swizzle_mode_V,
                               V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
        &V_lane_base_ptr, V_smem_offset_load, stride_d_v, smem_V);
    cp_async::commit_group();

    K_load_idx_lane_base += CTA_K;

#pragma unroll
    for (uint32_t iter = 1; iter < num_iterations - 1; iter++) {
      // Cached Q leaves shared space for two K tiles. Wait for both K and V.
      if constexpr (cache_query)
        cp_async::wait_group<0>();
      else
        cp_async::wait_group<1>();
      __syncthreads();

      uint32_t RS_u8[num_tiles_q][num_tiles_k / 2][4];
      // A constant -inf tile contributes zero probability and leaves the
      // running maximum, denominator and output unchanged. Keep the existing
      // asynchronous copies and barriers even when its arithmetic is skipped.
      bool active_tile = true;
      if constexpr (skip_masked_tiles &&
                    mask_mode == MaskMode::kPreparedKey && head_dim == 128) {
        active_tile = key_tile_bias[K_idx_lane_base / 128] != -CUDART_INF_F;
      }
      if (active_tile) {
        if constexpr (cache_query) {
          smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> next_K(
              smem + ((iter & 1) * CTA_K) * QK_SMEM_STRIDE);
          // load K without predicate
          load_global_to_share<global_to_shared_line_lanes_QK,
                               global_to_shared_copy_lines_per_warp_QK,
                               QK_smem_iters_row, K_smem_iters_col,
                               swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK,
                               CTA_K>(&K_lane_base_ptr, K_smem_offset_load,
                                      stride_seq_k, next_K);
          cp_async::commit_group();
        }

        // compute QK^T
        if constexpr (cache_query) {
          compute_int_qk_cached<num_warps_q, num_warps_k, num_tiles_q,
                                num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK,
                                QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
              smem_K, RS, RQ_cached, K_smem_offset_mma);
        } else if constexpr (num_tiles_qk_inner == 1) {
          compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                         num_tiles_qk_inner, swizzle_mode_QK,
                         QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(smem_K, RS, RQ,
                                                                 K_smem_offset_mma);
        } else {
          compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                         num_tiles_qk_inner, swizzle_mode_QK,
                         QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
              smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
        }
        if constexpr (mask_mode == MaskMode::kPreparedKey) {
          const float tile_bias = key_tile_bias[K_idx_lane_base / 128];
          // The constant-tile branch pays off at D=128. A single FP32 path
          // reduces register pressure and runs faster at D=64 and D=256.
          if (head_dim == 128 && isfinite(tile_bias)) {
            update_mdo_i32_u8<num_tiles_q, num_tiles_k, num_tiles_v, true>(
                RS, RO, m, d, sm_scale, S_U8_OFFSET, RS_u8, tile_bias * math::log2e);
          } else {
            auto &scores = reinterpret_cast<float (&)[num_tiles_q][num_tiles_k][8]>(RS);
#pragma unroll
            for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
              for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
                for (uint32_t k = 0; k < 8; k++)
                  scores[fq][fk][k] = __int2float_rz(RS[fq][fk][k]);
              }
            }
            apply_prepared_key_mask<num_tiles_q, num_tiles_k>(
                K_idx_lane_base, scores, key_mask, sm_scale);
            update_mdo_f32_u8<num_tiles_q, num_tiles_k, num_tiles_v>(
                scores, RO, m, d, S_U8_OFFSET, RS_u8);
          }
        } else if constexpr (use_fused_fp32_probabilities &&
                             mask_mode == MaskMode::kNone) {
          update_mdo_i32_u8<num_tiles_q, num_tiles_k, num_tiles_v>(
              RS, RO, m, d, sm_scale, S_U8_OFFSET, RS_u8);
        } else {
          float pv_scale[num_tiles_q][2];
          float RS_soft[num_tiles_q][num_tiles_k][8];
#pragma unroll
          for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
              for (uint32_t k = 0; k < 8; k++) {
                const float score = __int2float_rz(RS[fq][fk][k]);
                RS_soft[fq][fk][k] =
                    pre_scale_scores ? score * sm_scale : score;
              }
            }
          }

          if constexpr (mask_mode == MaskMode::kCustom) {
            apply_custom_mask<num_tiles_q, num_tiles_k>(
                Q_idx_lane_base, K_idx_lane_base, RS_soft, valid, AttnMask,
                mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k,
                batch_id, head_id, qo_len, kv_len, mask_dtype_code, 1.0f);
          } else if constexpr (mask_mode == MaskMode::kCustomKey) {
            apply_custom_key_mask<num_tiles_q, num_tiles_k>(
                K_idx_lane_base, RS_soft, valid, AttnMask, mask_stride_b,
                mask_stride_h, mask_stride_k, batch_id, head_id, kv_len,
                mask_dtype_code, 1.0f);
          }

          update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true,
                     pre_scale_scores>(RS_soft, RO, m, d, pv_scale, sm_scale,
                                       S_U8_OFFSET);
          RS_to_u8<num_tiles_q, num_tiles_k>(RS_soft, RS_u8);

          if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore) {
            accumulate_d<num_tiles_q, num_tiles_k>(RS_soft, d, pv_scale);
          }
#pragma unroll
          for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t k = 0; k < 2; k++)
              RS[fq][0][k] = __float_as_int(pv_scale[fq][k]);
          }
        }
      }
      K_idx_lane_base += CTA_K;

      if constexpr (!cache_query) {
        __syncthreads();

        // load K without predicate
        load_global_to_share<global_to_shared_line_lanes_QK,
                             global_to_shared_copy_lines_per_warp_QK,
                             QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK,
                             QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
            &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K);
        cp_async::commit_group();
      }

      dequant_scale =
          q_scale * K_scale[k_scale_idx + iter * k_scale_advance_offset];
      sm_scale = original_sm_scale * dequant_scale;

      if constexpr (!cache_query) {
        // ensure V is ready
        cp_async::wait_group<1>();
        __syncthreads();
      }

      if (active_tile) {
        compute_int8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                        num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V,
                        mask_mode == MaskMode::kNone>(
            smem_V, RS, RS_u8, RO);
      }
      __syncthreads();
      if constexpr (cache_query) {
        smem_K.base = reinterpret_cast<b128_t *>(smem + ((iter & 1) * CTA_K) *
                                                            QK_SMEM_STRIDE);
      }
      // load V
      load_int8_V_global_to_share<
          global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V,
          V_smem_iters_row, V_smem_iters_col, swizzle_mode_V,
          V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
          &V_lane_base_ptr, V_smem_offset_load, stride_d_v, smem_V);
      cp_async::commit_group();

      K_load_idx_lane_base += CTA_K;
    }

    // second last iter, apply causal mask
    if (num_iterations > 1) {
      // Cached Q leaves shared space for two K tiles. Wait for both K and V.
      if constexpr (cache_query)
        cp_async::wait_group<0>();
      else
        cp_async::wait_group<1>();
      __syncthreads();

      uint32_t RS_u8[num_tiles_q][num_tiles_k / 2][4];
      // A constant -inf tile contributes zero probability and leaves the
      // running maximum, denominator and output unchanged. Keep the existing
      // asynchronous copies and barriers even when its arithmetic is skipped.
      bool active_tile = true;
      if constexpr (skip_masked_tiles &&
                    mask_mode == MaskMode::kPreparedKey && head_dim == 128) {
        active_tile = key_tile_bias[K_idx_lane_base / 128] != -CUDART_INF_F;
      }
      if (active_tile) {
        if constexpr (cache_query) {
          smem_t<swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK> next_K(
              smem + (((num_iterations - 1) & 1) * CTA_K) * QK_SMEM_STRIDE);
          // load K with predicate
          load_global_to_share<global_to_shared_line_lanes_QK,
                               global_to_shared_copy_lines_per_warp_QK,
                               QK_smem_iters_row, K_smem_iters_col,
                               swizzle_mode_QK, QK_SMEM_STRIDE / PACK_SIZE_QK,
                               CTA_K>(&K_lane_base_ptr, K_smem_offset_load,
                                      stride_seq_k, next_K, K_load_idx_lane_base,
                                      kv_len);
          cp_async::commit_group();
        }

        // compute QK^T
        if constexpr (cache_query) {
          compute_int_qk_cached<num_warps_q, num_warps_k, num_tiles_q,
                                num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK,
                                QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
              smem_K, RS, RQ_cached, K_smem_offset_mma);
        } else if constexpr (num_tiles_qk_inner == 1) {
          compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                         num_tiles_qk_inner, swizzle_mode_QK,
                         QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(smem_K, RS, RQ,
                                                                 K_smem_offset_mma);
        } else {
          compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                         num_tiles_qk_inner, swizzle_mode_QK,
                         QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
              smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
        }

        if constexpr (mask_mode == MaskMode::kPreparedKey) {
          const float tile_bias = key_tile_bias[K_idx_lane_base / 128];
          if (head_dim == 128 && isfinite(tile_bias)) {
            update_mdo_i32_u8<num_tiles_q, num_tiles_k, num_tiles_v, true>(
                RS, RO, m, d, sm_scale, S_U8_OFFSET, RS_u8, tile_bias * math::log2e);
          } else {
            auto &scores = reinterpret_cast<float (&)[num_tiles_q][num_tiles_k][8]>(RS);
#pragma unroll
            for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
              for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
                for (uint32_t k = 0; k < 8; k++)
                  scores[fq][fk][k] = __int2float_rz(RS[fq][fk][k]);
              }
            }
            apply_prepared_key_mask<num_tiles_q, num_tiles_k>(
                K_idx_lane_base, scores, key_mask, sm_scale);
            update_mdo_f32_u8<num_tiles_q, num_tiles_k, num_tiles_v>(
                scores, RO, m, d, S_U8_OFFSET, RS_u8);
          }
        } else if constexpr (use_fused_fp32_probabilities &&
                             mask_mode == MaskMode::kNone) {
          update_mdo_i32_u8<num_tiles_q, num_tiles_k, num_tiles_v>(
              RS, RO, m, d, sm_scale, S_U8_OFFSET, RS_u8);
        } else {
          float pv_scale[num_tiles_q][2];
          float RS_soft[num_tiles_q][num_tiles_k][8];
#pragma unroll
          for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
              for (uint32_t k = 0; k < 8; k++) {
                const float score = __int2float_rz(RS[fq][fk][k]);
                RS_soft[fq][fk][k] =
                    pre_scale_scores ? score * sm_scale : score;
              }
            }
          }

          if constexpr (mask_mode == MaskMode::kCausal) {
            apply_causal_mask<num_tiles_q, num_tiles_k>(
                Q_idx_lane_base, K_idx_lane_base, RS_soft,
                pre_scale_scores ? -50000.0f : -1.0e30f);
          } else if constexpr (mask_mode == MaskMode::kCustom) {
            apply_custom_mask<num_tiles_q, num_tiles_k>(
                Q_idx_lane_base, K_idx_lane_base, RS_soft, valid, AttnMask,
                mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k,
                batch_id, head_id, qo_len, kv_len, mask_dtype_code, 1.0f);
          } else if constexpr (mask_mode == MaskMode::kCustomKey) {
            apply_custom_key_mask<num_tiles_q, num_tiles_k>(
                K_idx_lane_base, RS_soft, valid, AttnMask, mask_stride_b,
                mask_stride_h, mask_stride_k, batch_id, head_id, kv_len,
                mask_dtype_code, 1.0f);
          }

          update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true,
                     pre_scale_scores>(RS_soft, RO, m, d, pv_scale, sm_scale,
                                       S_U8_OFFSET);
          RS_to_u8<num_tiles_q, num_tiles_k>(RS_soft, RS_u8);

          if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore) {
            accumulate_d<num_tiles_q, num_tiles_k>(RS_soft, d, pv_scale);
          }
#pragma unroll
          for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t k = 0; k < 2; k++)
              RS[fq][0][k] = __float_as_int(pv_scale[fq][k]);
          }
        }
      }
      K_idx_lane_base += CTA_K;

      if constexpr (!cache_query) {
        __syncthreads();

        // load K with predicate
        load_global_to_share<global_to_shared_line_lanes_QK,
                             global_to_shared_copy_lines_per_warp_QK,
                             QK_smem_iters_row, K_smem_iters_col, swizzle_mode_QK,
                             QK_SMEM_STRIDE / PACK_SIZE_QK, CTA_K>(
            &K_lane_base_ptr, K_smem_offset_load, stride_seq_k, smem_K,
            K_load_idx_lane_base, kv_len);
        cp_async::commit_group();
      }

      dequant_scale =
          q_scale *
          K_scale[k_scale_idx + (num_iterations - 1) * k_scale_advance_offset];
      sm_scale = original_sm_scale * dequant_scale;

      if constexpr (!cache_query) {
        // ensure V is ready
        cp_async::wait_group<1>();
        __syncthreads();
      }

      if (active_tile) {
        compute_int8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                        num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V,
                        mask_mode == MaskMode::kNone>(
            smem_V, RS, RS_u8, RO);
      }

      __syncthreads();
      if constexpr (cache_query) {
        smem_K.base = reinterpret_cast<b128_t *>(
            smem + (((num_iterations - 1) & 1) * CTA_K) * QK_SMEM_STRIDE);
      }
      // load V
      load_int8_V_global_to_share<
          global_to_shared_line_lanes_V, global_to_shared_copy_lines_per_warp_V,
          V_smem_iters_row, V_smem_iters_col, swizzle_mode_V,
          V_SMEM_STRIDE / PACK_SIZE_V, CTA_K>(
          &V_lane_base_ptr, V_smem_offset_load, stride_d_v, smem_V);
      cp_async::commit_group();
      K_load_idx_lane_base += CTA_K;
    }

    // last iter, apply causal mask and out of bound mask
    {
      // Cached Q leaves shared space for two K tiles. Wait for both K and V.
      if constexpr (cache_query)
        cp_async::wait_group<0>();
      else
        cp_async::wait_group<1>();
      __syncthreads();

      uint32_t RS_u8[num_tiles_q][num_tiles_k / 2][4];
      // A constant -inf tile contributes zero probability and leaves the
      // running maximum, denominator and output unchanged. Keep the existing
      // asynchronous copies and barriers even when its arithmetic is skipped.
      bool active_tile = true;
      if constexpr (skip_masked_tiles &&
                    mask_mode == MaskMode::kPreparedKey && head_dim == 128) {
        active_tile = key_tile_bias[K_idx_lane_base / 128] != -CUDART_INF_F;
      }
      if (active_tile) {
        // compute QK^T
        if constexpr (cache_query) {
          compute_int_qk_cached<num_warps_q, num_warps_k, num_tiles_q,
                                num_tiles_k, num_tiles_qk_inner, swizzle_mode_QK,
                                QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
              smem_K, RS, RQ_cached, K_smem_offset_mma);
        } else if constexpr (num_tiles_qk_inner == 1) {
          compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                         num_tiles_qk_inner, swizzle_mode_QK,
                         QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(smem_K, RS, RQ,
                                                                 K_smem_offset_mma);
        } else {
          compute_int_qk<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                         num_tiles_qk_inner, swizzle_mode_QK,
                         QK_SMEM_STRIDE / PACK_SIZE_QK, DTypeQK>(
              smem_Q, smem_K, RS, Q_smem_offset_mma, K_smem_offset_mma);
        }

        if constexpr (mask_mode == MaskMode::kPreparedKey) {
          auto &scores = reinterpret_cast<float (&)[num_tiles_q][num_tiles_k][8]>(RS);
#pragma unroll
          for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
              for (uint32_t k = 0; k < 8; k++)
                scores[fq][fk][k] = __int2float_rz(RS[fq][fk][k]);
            }
          }
          apply_prepared_key_mask<num_tiles_q, num_tiles_k>(
              K_idx_lane_base, scores, key_mask, sm_scale);
          update_mdo_f32_u8<num_tiles_q, num_tiles_k, num_tiles_v>(
              scores, RO, m, d, S_U8_OFFSET, RS_u8);
        } else {
          float RS_soft[num_tiles_q][num_tiles_k][8];
          float pv_scale[num_tiles_q][2];
#pragma unroll
          for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t fk = 0; fk < num_tiles_k; fk++) {
#pragma unroll
              for (uint32_t k = 0; k < 8; k++) {
                const float score = __int2float_rz(RS[fq][fk][k]);
                RS_soft[fq][fk][k] =
                    pre_scale_scores ? score * sm_scale : score;
              }
            }
          }

          if constexpr (mask_mode == MaskMode::kCausal) {
            apply_causal_mask<num_tiles_q, num_tiles_k>(
                Q_idx_lane_base, K_idx_lane_base, RS_soft,
                pre_scale_scores ? -50000.0f : -1.0e30f);
          } else if constexpr (mask_mode == MaskMode::kCustom) {
            apply_custom_mask<num_tiles_q, num_tiles_k>(
                Q_idx_lane_base, K_idx_lane_base, RS_soft, valid, AttnMask,
                mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k, batch_id,
                head_id, qo_len, kv_len, mask_dtype_code, 1.0f);
          } else if constexpr (mask_mode == MaskMode::kCustomKey) {
            apply_custom_key_mask<num_tiles_q, num_tiles_k>(
                K_idx_lane_base, RS_soft, valid, AttnMask, mask_stride_b,
                mask_stride_h, mask_stride_k, batch_id, head_id, kv_len,
                mask_dtype_code, 1.0f);
          }
          apply_out_of_bound_mask<num_tiles_q, num_tiles_k>(
              K_idx_lane_base, RS_soft, kv_len,
              pre_scale_scores ? -50000.0f : -1.0e30f);

          update_mdo<num_tiles_q, num_tiles_k, num_tiles_v, true,
                     pre_scale_scores>(RS_soft, RO, m, d, pv_scale, sm_scale,
                                       S_U8_OFFSET);

          RS_to_u8<num_tiles_q, num_tiles_k>(RS_soft, RS_u8);

          if constexpr (DenominatorAccumUnit == ComputeUnit::kCudaCore) {
            accumulate_d<num_tiles_q, num_tiles_k>(RS_soft, d, pv_scale);
          }
#pragma unroll
          for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
            for (uint32_t k = 0; k < 2; k++)
              RS[fq][0][k] = __float_as_int(pv_scale[fq][k]);
          }
        }
      }
      K_idx_lane_base += CTA_K;

      if constexpr (!cache_query) {
        // ensure V is ready
        cp_async::wait_group<0>();
        __syncthreads();
      }

      if (active_tile) {
        compute_int8_sv<num_warps_q, num_warps_k, num_tiles_q, num_tiles_k,
                        num_tiles_v, swizzle_mode_V, V_SMEM_STRIDE / PACK_SIZE_V,
                        mask_mode == MaskMode::kNone>(
            smem_V, RS, RS_u8, RO);
      }

      __syncthreads();
    }

  }

  // TODO: thread block sync mdo state for num_warps_k > 0. Then only one thread
  // block needs to do the final saving.

  normalize_d<num_tiles_q, num_tiles_v, ComputeUnit::kCudaCore>(RO, m, d);

  if constexpr (custom_mask && mask_mode != MaskMode::kPreparedKey) {
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
      for (uint32_t k = 0; k < 2; k++) {
        int row_valid = valid[fq][k] ? 1 : 0;
        row_valid |= __shfl_xor_sync(0xffffffff, row_valid, 0x1);
        row_valid |= __shfl_xor_sync(0xffffffff, row_valid, 0x2);
        valid[fq][k] = row_valid != 0;
      }
    }
#pragma unroll
    for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
      for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++) {
          if (!valid[fq][(k % 4) / 2])
            RO[fq][fv][k] = 0.0f;
        }
      }
    }
  }

  // ! here we just implement the case for fp32 acumulation
  if constexpr (fuse_v_scale) {
    float v_scale[4];
    float *V_scale_base_ptr =
        V_scale + batch_id * (num_qo_heads / num_kv_groups) * head_dim +
        (head_id / num_kv_groups) * head_dim + (lane_id % 4) * 2;
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      ((float2 *)v_scale)[0] = *((float2 *)(V_scale_base_ptr + fv * 16));
      ((float2 *)v_scale)[1] = *((float2 *)(V_scale_base_ptr + fv * 16 + 8));
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
        for (uint32_t k = 0; k < 8; k++) {
          const float scale_value = v_scale[(k / 4) * 2 + (k % 2)];
          RO[fq][fv][k] *= scale_value;
        }
      }
    }
  }

  if constexpr (fuse_v_mean) {
    float v_mean[4];
    float *V_mean_base_ptr =
        V_mean + batch_id * (num_qo_heads / num_kv_groups) * head_dim +
        (head_id / num_kv_groups) * head_dim + (lane_id % 4) * 2;
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      ((float2 *)v_mean)[0] = *((float2 *)(V_mean_base_ptr + fv * 16));
      ((float2 *)v_mean)[1] = *((float2 *)(V_mean_base_ptr + fv * 16 + 8));
#pragma unroll
      for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
        RO[fq][fv][0] += v_mean[0];
        RO[fq][fv][1] += v_mean[1];
        RO[fq][fv][2] += v_mean[0];
        RO[fq][fv][3] += v_mean[1];
        RO[fq][fv][4] += v_mean[2];
        RO[fq][fv][5] += v_mean[3];
        RO[fq][fv][6] += v_mean[2];
        RO[fq][fv][7] += v_mean[3];
      }
    }
  }

  // save the result to shared memory
  uint32_t smem_O_row_base =
      get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q + lane_id / 4;
#pragma unroll
  for (uint32_t fq = 0; fq < num_tiles_q; fq++) {
#pragma unroll
    for (uint32_t fv = 0; fv < num_tiles_v; fv++) {
      uint32_t offset_O = smem_O.get_permuted_offset(
          smem_O_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O));

      if constexpr (std::is_same<DTypeSVAccum, float>::value) {
        // convert RO to half
        uint32_t RO_f16[4];
#pragma unroll
        for (uint32_t k = 0; k < 4; k++) {
          if constexpr (std::is_same<DTypeOut, half>::value) {
            ((half2 *)RO_f16)[k] = __float22half2_rn(((float2 *)RO[fq][fv])[k]);
          } else {
            ((nv_bfloat162 *)RO_f16)[k] =
                __float22bfloat162_rn(((float2 *)RO[fq][fv])[k]);
          }
        }

        ((uint32_t *)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[0];
        ((uint32_t *)(smem_O.base + offset_O +
                      8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] =
            RO_f16[1];

        offset_O = smem_O.get_permuted_offset(
            smem_O_row_base + fq * MMA_QK_M, fv * (MMA_SV_N / PACK_SIZE_O) + 1);
        ((uint32_t *)(smem_O.base + offset_O))[lane_id % 4] = RO_f16[2];
        ((uint32_t *)(smem_O.base + offset_O +
                      8 * (O_SMEM_STRIDE / PACK_SIZE_O)))[lane_id % 4] =
            RO_f16[3];
      } else if constexpr (std::is_same<DTypeSVAccum, half>::value) {
        // TODO: not implement
      }
    }
  }

  // ! do we need to sync here?
  __syncwarp();

  // shared memory to global memory
  DTypeOut *O_lane_ptr =
      O + batch_id * stride_bz_o + head_id * stride_h_o +
      (bx * CTA_Q + WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>() +
       lane_id / global_to_shared_line_lanes_O) *
          stride_seq_o +
      lane_id % global_to_shared_line_lanes_O * PACK_SIZE_O;
  uint32_t offset_O = smem_O.get_permuted_offset(
      get_warp_idx_q<num_warps_q, num_warps_k>() * WARP_Q +
          lane_id / global_to_shared_line_lanes_O,
      lane_id % global_to_shared_line_lanes_O);
  uint32_t O_load_idx_lane_base = bx * CTA_Q + CTA_Q / num_warps * warp_id +
                                  lane_id / global_to_shared_line_lanes_O;

#pragma unroll
  for (uint32_t i = 0; i < O_smem_iters_col; i++) {
#pragma unroll
    for (uint32_t j = 0; j < O_smem_iters_row; j++) {
      if (O_load_idx_lane_base < qo_len) {
        smem_O.store_128b(offset_O, O_lane_ptr);
      }
      O_lane_ptr += (global_to_shared_line_lanes_O * PACK_SIZE_O);
      offset_O = smem_O.advance_offset_by_column<global_to_shared_line_lanes_O>(
          offset_O);
    }

    offset_O =
        smem_O.advance_offset_by_row<global_to_shared_copy_lines_per_warp_O>(
            offset_O - (O_smem_iters_row * global_to_shared_line_lanes_O));
    O_lane_ptr +=
        ((global_to_shared_copy_lines_per_warp_O * stride_seq_o) -
         (O_smem_iters_row * global_to_shared_line_lanes_O * PACK_SIZE_O));
    O_load_idx_lane_base += global_to_shared_copy_lines_per_warp_O;
  }

  if constexpr (return_lse) {
    // ! this only works for num_tiles_q = 2
    uint32_t lse_idx = bx * CTA_Q + lane_id / 4 + 8 * (lane_id % 4) +
                       WARP_Q * get_warp_idx_q<num_warps_q, num_warps_k>();
    float *lse_lane_ptr =
        Lse + batch_id * (qo_len * num_qo_heads) + head_id * qo_len + lse_idx;
    uint32_t fq = (lane_id % 4) / 2;
    uint32_t k = (lane_id % 4) % 2;

    if (lse_idx < qo_len) {
      lse_lane_ptr[0] = math::ptx_log2(d[fq][k]) + m[fq][k];
    }
  }
}

template <uint32_t CTA_Q, uint32_t CTA_K, uint32_t WARP_Q, uint32_t WARP_K,
          uint32_t head_dim, DataType DTypeQK, QuantGranularity Q_GRAN,
          QuantGranularity K_GRAN, typename DTypeSVAccum = float,
          bool use_inst_buffer = false, typename DTypeOut = half,
          ComputeUnit DenominatorAccumUnit,
          MaskMode mask_mode = MaskMode::kNone, bool return_lse = false,
          bool fuse_v_scale = false, bool fuse_v_mean = false,
          bool use_pv_fp16_accu = false,
          bool fuse_fp32_probabilities = true, typename Offset = uint32_t>
__global__ void qk_int_sv_i8_attn_kernel(
    int8_t *__restrict__ Q, int8_t *__restrict__ K, int8_t *__restrict__ V,
    DTypeOut *__restrict__ O, float *__restrict__ Lse,
    float *__restrict__ Q_scale, float *__restrict__ K_scale,
    float *__restrict__ V_scale, float *__restrict__ V_mean,
    const void *__restrict__ AttnMask, const int64_t mask_stride_b,
    const int64_t mask_stride_h, const int64_t mask_stride_q,
    const int64_t mask_stride_k, const int mask_dtype_code,
    const uint32_t qo_len, const uint32_t kv_len, const uint32_t num_kv_groups,
    const Offset stride_bz_q, const uint32_t stride_seq_q,
    const Offset stride_h_q, const Offset stride_bz_k,
    const uint32_t stride_seq_k, const Offset stride_h_k,
    const Offset stride_bz_v, const Offset stride_h_v,
    const uint32_t stride_d_v, const Offset stride_bz_o,
    const uint32_t stride_seq_o, const Offset stride_h_o, float sm_scale,
    const float *__restrict__ MaskTileBias) {
  if constexpr (mask_mode == MaskMode::kPreparedKey && head_dim == 128) {
    const uint32_t tiles = (kv_len + 127) / 128;
    const float *tile_bias = MaskTileBias + blockIdx.z * mask_stride_b +
                            blockIdx.y * mask_stride_h;
    // Each warp reads the same immutable descriptors and agrees on the route.
    uint32_t excluded = 0;
    for (uint32_t base = 0; base < tiles; base += 32) {
      const uint32_t tile = base + get_lane_id();
      const bool masked = tile < tiles && tile_bias[tile] == -CUDART_INF_F;
      excluded += __popc(__ballot_sync(0xffffffff, masked));
    }
    if (excluded * 4 >= tiles) {
      qk_int_sv_i8_attn_body<true, CTA_Q, CTA_K, WARP_Q, WARP_K, head_dim, DTypeQK,
        Q_GRAN, K_GRAN, DTypeSVAccum, use_inst_buffer, DTypeOut,
        DenominatorAccumUnit, mask_mode, return_lse, fuse_v_scale, fuse_v_mean,
        use_pv_fp16_accu, fuse_fp32_probabilities, true, Offset>(
          Q, K, V, O, Lse, Q_scale, K_scale, V_scale, V_mean, AttnMask,
          mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k,
          mask_dtype_code, qo_len, kv_len, num_kv_groups, stride_bz_q, stride_seq_q,
          stride_h_q, stride_bz_k, stride_seq_k, stride_h_k, stride_bz_v, stride_h_v,
          stride_d_v, stride_bz_o, stride_seq_o, stride_h_o, sm_scale, MaskTileBias);
    } else if (excluded) {
      qk_int_sv_i8_attn_body<true, CTA_Q, CTA_K, WARP_Q, WARP_K, head_dim, DTypeQK,
        Q_GRAN, K_GRAN, DTypeSVAccum, use_inst_buffer, DTypeOut,
        DenominatorAccumUnit, mask_mode, return_lse, fuse_v_scale, fuse_v_mean,
        use_pv_fp16_accu, fuse_fp32_probabilities, false, Offset>(
          Q, K, V, O, Lse, Q_scale, K_scale, V_scale, V_mean, AttnMask,
          mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k,
          mask_dtype_code, qo_len, kv_len, num_kv_groups, stride_bz_q, stride_seq_q,
          stride_h_q, stride_bz_k, stride_seq_k, stride_h_k, stride_bz_v, stride_h_v,
          stride_d_v, stride_bz_o, stride_seq_o, stride_h_o, sm_scale, MaskTileBias);
    } else {
      qk_int_sv_i8_attn_body<false, CTA_Q, CTA_K, WARP_Q, WARP_K, head_dim, DTypeQK,
        Q_GRAN, K_GRAN, DTypeSVAccum, use_inst_buffer, DTypeOut,
        DenominatorAccumUnit, mask_mode, return_lse, fuse_v_scale, fuse_v_mean,
        use_pv_fp16_accu, fuse_fp32_probabilities, false, Offset>(
          Q, K, V, O, Lse, Q_scale, K_scale, V_scale, V_mean, AttnMask,
          mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k,
          mask_dtype_code, qo_len, kv_len, num_kv_groups, stride_bz_q, stride_seq_q,
          stride_h_q, stride_bz_k, stride_seq_k, stride_h_k, stride_bz_v, stride_h_v,
          stride_d_v, stride_bz_o, stride_seq_o, stride_h_o, sm_scale, MaskTileBias);
    }
  } else {
    qk_int_sv_i8_attn_body<false, CTA_Q, CTA_K, WARP_Q, WARP_K, head_dim, DTypeQK,
        Q_GRAN, K_GRAN, DTypeSVAccum, use_inst_buffer, DTypeOut,
        DenominatorAccumUnit, mask_mode, return_lse, fuse_v_scale, fuse_v_mean,
        use_pv_fp16_accu, fuse_fp32_probabilities, false, Offset>(
          Q, K, V, O, Lse, Q_scale, K_scale, V_scale, V_mean, AttnMask,
          mask_stride_b, mask_stride_h, mask_stride_q, mask_stride_k,
          mask_dtype_code, qo_len, kv_len, num_kv_groups, stride_bz_q, stride_seq_q,
          stride_h_q, stride_bz_k, stride_seq_k, stride_h_k, stride_bz_v, stride_h_v,
          stride_d_v, stride_bz_o, stride_seq_o, stride_h_o, sm_scale, MaskTileBias);
  }
}
