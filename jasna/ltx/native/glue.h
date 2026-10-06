#pragma once

#include <ATen/core/Tensor.h>

namespace jasna_ltx {

// out = bf16(rms_norm(x) * (1 + scale[half]) + shift[half]) for x (rows, dim); shift/scale (2, dim),
// the first half of the rows takes row 0. With amax, also max|out| per amax_group_rows rows
// (atomic, amax zeroed by the caller).
void rms_modulate(const at::Tensor& x, const at::Tensor& shift, const at::Tensor& scale, const at::Tensor& out,
                  float* amax, int64_t amax_group_rows, double eps);

// out = rope(bf16(rms_norm(x) * weight)) per head, split halves, for x (rows, dim) token-major; the
// second half of the rows reuses the first half's positions. Without cos/sin only the norm.
void norm_rope(const at::Tensor& x, const at::Tensor& weight, const at::Tensor* cos, const at::Tensor* sin,
               const at::Tensor& out, int64_t head_dim, double eps);

// out (rows, heads * head_dim) = attention (rows, heads, head_dim, any token/head strides)
// * 2 * sigmoid(logits (rows, heads)); with amax, also max|out| (atomic).
void gate_heads(const at::Tensor& attention, const at::Tensor& logits, const at::Tensor& out, float* amax);

// out = residual + y * gate[half] for rows starting at row_offset of a block whose halves have
// half_rows rows; out may be residual.
void gated_residual(const at::Tensor& out, const at::Tensor& residual, const at::Tensor& y, const at::Tensor& gate,
                    int64_t half_rows, int64_t row_offset);

// amax = max|bf16(gelu_tanh(h))| (atomic, zeroed by the caller), the GELU of quantize_fp4(gelu=True).
void gelu_amax(const at::Tensor& h, float* amax);

// scale = amax / divisor, 1 where that is not positive (transformer.fp4_tensor_scale).
void fp4_tensor_scales(const at::Tensor& amax, const at::Tensor& scale, double divisor);

}  // namespace jasna_ltx
