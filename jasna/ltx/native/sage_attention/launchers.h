// SPDX-License-Identifier: Apache-2.0
// Launchers of the INT8 attention vendored from comfy-kitchen 0.2.38 (see NOTICE), with the
// signatures of its CUDA extension's launchers. Only bf16 inputs and outputs and unmasked
// attention are compiled; any other dtype code or a mask throws.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace jasna_ltx::sage_attention {

constexpr int kBFloat16 = 2;  // comfy-kitchen's dtype code for bfloat16

void launch_quant_qk_per_thread_int8(
    const void *q, void *q_int8, void *q_scale, const void *k, void *k_int8,
    void *k_scale, int B, int H_q, int Lq, int H_kv, int Lk, int C, int BLKQ,
    int WARPQ, int BLKK, int WARPK, int64_t q_stride_b, int64_t q_stride_h,
    int64_t q_stride_n, int64_t k_stride_b, int64_t k_stride_h,
    int64_t k_stride_n, int input_dtype_code, void *anchor_indices,
    cudaStream_t stream);

void launch_quant_v_int8_kernel(const void *v, void *out, void *scale,
                                int B, int H, int N, int D,
                                int padded_N, int64_t sb, int64_t sh,
                                int64_t sn, int input_dtype_code,
                                cudaStream_t stream);

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
    const float *mask_tile_bias);

}  // namespace jasna_ltx::sage_attention
