// SPDX-License-Identifier: Apache-2.0
// Launchers of the INT8 attention vendored from comfy-kitchen 0.2.38's HIP backend (see NOTICE),
// with the signatures of its extern "C" launchers. Only bf16 inputs and outputs and unmasked
// attention are compiled; any other dtype code or a mask throws.
#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>

namespace jasna_ltx::sage_attention_hip {

constexpr int kBFloat16 = 2;  // comfy-kitchen's dtype code for bfloat16

void launch_quant_qk_int8(const void* q, void* q_int8, void* q_scale, const void* k, void* k_int8,
                          void* k_scale, void* anchor_indices, int B, int H_q, int Lq, int Lq_padded,
                          int H_kv, int Lk, int groups_per_head, int head_dim, int64_t q_stride_b,
                          int64_t q_stride_h, int64_t q_stride_n, int64_t k_stride_b, int64_t k_stride_h,
                          int64_t k_stride_n, int input_dtype_code, int rotation, hipStream_t stream);

void launch_quant_v_int8(const void* v, void* out, void* scale, int B, int H, int N, int D, int padded_N,
                         int64_t stride_b, int64_t stride_h, int64_t stride_n, int input_dtype_code,
                         hipStream_t stream);

void launch_int8_attn(const void* q, const void* k, const void* v, void* o, const void* q_scale,
                      const void* k_scale, const void* v_scale, const void* mask, int64_t mask_stride_b,
                      int64_t mask_stride_h, int64_t mask_stride_q, int64_t mask_stride_k, int mask_dtype_code,
                      int cta_k, int batch, int qo_len, int kv_len, int qo_len_padded, int num_qo_heads,
                      int num_kv_heads, int head_dim, int k_groups_per_head, int64_t q_stride_b,
                      int64_t q_stride_h, int64_t k_stride_b, int64_t k_stride_h, int64_t v_stride_b,
                      int64_t v_stride_h, int64_t v_stride_d, int64_t o_stride_b, int64_t o_stride_h,
                      int64_t o_stride_n, float sm_scale, int output_dtype_code, hipStream_t stream);

}  // namespace jasna_ltx::sage_attention_hip
