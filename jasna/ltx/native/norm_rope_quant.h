// comfy-kitchen's INT8 Q and K quantization (``sage_attention_hip/quant_qk_int8.hip``) of glue.cu's
// ``norm_rope`` output, computed straight from the projection rows on AMD, so the normalized q/k
// never round-trip through a bf16 tensor. The codes and scales equal norm_rope followed by
// ``launch_quant_qk_int8`` bit for bit, in its layouts.
#pragma once

#include <hip/hip_runtime_api.h>

#include <cstdint>

namespace jasna_ltx::sage_attention_hip {

// glue.cu norm_rope's inputs: x (rows, heads * head_dim) bf16 token-major, weight (dim,) and, unless
// null, rope tables [1, heads, tokens, head_dim / 2] with these head/token strides.
struct NormRope {
  const void* x;
  const void* weight;
  const void* cos;
  const void* sin;
  int64_t rope_stride_head, rope_stride_token;
  int rows, tokens, heads, head_dim;
  float eps;
};

// Q codes [1, heads, rows, head_dim] and one scale per row [1, heads, padded_rows].
void launch_norm_rope_quant_q(const NormRope& q, void* q_int8, void* q_scale, int padded_rows, int rotation,
                              hipStream_t stream);

// K codes [1, heads, rows, head_dim] and one scale per 16 keys [1, heads, groups]; samples holds
// comfy-kitchen's nine anchor-candidate rows (9 x dim bf16) and anchors one int per head.
void launch_norm_rope_quant_k(const NormRope& k, void* k_int8, void* k_scale, void* samples, void* anchors,
                              int groups, int rotation, hipStream_t stream);

// launch_quant_qk_int8's K half alone, for keys [1, heads, keys, head_dim] already normalized.
void launch_quant_k_int8(const void* k, void* k_int8, void* k_scale, void* anchors, int heads, int keys, int groups,
                         int head_dim, int64_t stride_h, int64_t stride_n, int rotation, hipStream_t stream);

}  // namespace jasna_ltx::sage_attention_hip
