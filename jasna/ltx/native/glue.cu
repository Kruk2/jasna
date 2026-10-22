// Element-wise glue between the native block's GEMMs and attention: norms, AdaLN modulation,
// RoPE, attention gates, gated residuals and NVFP4 tensor scales. Math in fp32, one bf16
// rounding per stored value, like the compiled torch path.

#include "glue.h"

#include <c10/util/Exception.h>

#include "gpu_runtime.h"

namespace jasna_ltx {
namespace {

using gpu::bf16;
constexpr int kThreads = 256;
constexpr int kVector = 8;  // bf16 per 16-byte access

struct Vec {
  float v[kVector];
};

__device__ __forceinline__ Vec load8(const bf16* p) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const bf16* values = reinterpret_cast<const bf16*>(&raw);
  Vec out;
#pragma unroll
  for (int i = 0; i < kVector; ++i) out.v[i] = __bfloat162float(values[i]);
  return out;
}

__device__ __forceinline__ void store8(bf16* p, const Vec& in) {
  uint4 raw;
  bf16* values = reinterpret_cast<bf16*>(&raw);
#pragma unroll
  for (int i = 0; i < kVector; ++i) values[i] = gpu::to_bf16(in.v[i]);
  *reinterpret_cast<uint4*>(p) = raw;
}

__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(gpu::to_bf16(x)); }

// fp16 holds the prompt attention before SM 8.0; the values passing through it are bf16-rounded,
// as ``.to(torch.float16)`` of a bf16 tensor and ``.to(torch.bfloat16)`` of the fp16 result.
__device__ __forceinline__ Vec load8(const __half* p) {
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const __half* values = reinterpret_cast<const __half*>(&raw);
  Vec out;
#pragma unroll
  for (int i = 0; i < kVector; ++i) out.v[i] = round_bf16(__half2float(values[i]));
  return out;
}

__device__ __forceinline__ void store8(__half* p, const Vec& in) {
  uint4 raw;
  __half* values = reinterpret_cast<__half*>(&raw);
#pragma unroll
  for (int i = 0; i < kVector; ++i) values[i] = __float2half_rn(round_bf16(in.v[i]));
  *reinterpret_cast<uint4*>(p) = raw;
}

__device__ __forceinline__ void atomic_max_nonnegative(float* address, float value) {
  atomicMax(reinterpret_cast<int*>(address), __float_as_int(value));
}

template <bool kMax>
__device__ float block_reduce(float value) {
  __shared__ float partial[kThreads / 32];
  __syncthreads();  // partial may still be read by a previous reduction
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    const float other = gpu::shuffle_xor(value, offset);
    value = kMax ? fmaxf(value, other) : value + other;
  }
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  if (lane == 0) partial[warp] = value;
  __syncthreads();
  value = partial[0];
  for (int i = 1; i < blockDim.x / 32; ++i) value = kMax ? fmaxf(value, partial[i]) : value + partial[i];
  return value;
}

__global__ void rms_modulate_kernel(const bf16* x, const bf16* shift, const bf16* scale, bf16* out, float* amax,
                                    int dim, int half_rows, int amax_group_rows, float eps) {
  const int row = blockIdx.x;
  const bf16* in = x + static_cast<int64_t>(row) * dim;
  float squares = 0.f;
  for (int c = threadIdx.x * kVector; c < dim; c += blockDim.x * kVector) {
    const Vec a = load8(in + c);
#pragma unroll
    for (int i = 0; i < kVector; ++i) squares += a.v[i] * a.v[i];
  }
  const float rstd = rsqrtf(block_reduce<false>(squares) / dim + eps);
  const int half = row / half_rows;
  const bf16* shift_row = shift + static_cast<int64_t>(half) * dim;
  const bf16* scale_row = scale + static_cast<int64_t>(half) * dim;
  float peak = 0.f;
  for (int c = threadIdx.x * kVector; c < dim; c += blockDim.x * kVector) {
    const Vec a = load8(in + c), s = load8(scale_row + c), h = load8(shift_row + c);
    Vec o;
#pragma unroll
    for (int i = 0; i < kVector; ++i) {
      o.v[i] = round_bf16(a.v[i] * rstd * (1.f + s.v[i]) + h.v[i]);
      peak = fmaxf(peak, fabsf(o.v[i]));
    }
    store8(out + static_cast<int64_t>(row) * dim + c, o);
  }
  if (amax != nullptr) {
    peak = block_reduce<true>(peak);
    if (threadIdx.x == 0) atomic_max_nonnegative(amax + row / amax_group_rows, peak);
  }
}

template <class Out>
__global__ void norm_rope_kernel(const bf16* x, const bf16* weight, const bf16* cos, const bf16* sin, Out* out,
                                 int dim, int head_dim, int tokens, int64_t rope_stride_head,
                                 int64_t rope_stride_token, float eps) {
  const int row = blockIdx.x;
  const bf16* in = x + static_cast<int64_t>(row) * dim;
  Out* dst = out + static_cast<int64_t>(row) * dim;
  float squares = 0.f;
  for (int c = threadIdx.x * kVector; c < dim; c += blockDim.x * kVector) {
    const Vec a = load8(in + c);
#pragma unroll
    for (int i = 0; i < kVector; ++i) squares += a.v[i] * a.v[i];
  }
  const float rstd = rsqrtf(block_reduce<false>(squares) / dim + eps);
  const int half_dim = head_dim / 2;
  const int groups_per_head = half_dim / kVector;
  const int token = row % tokens;
  for (int g = threadIdx.x; g < dim / head_dim * groups_per_head; g += blockDim.x) {
    const int head = g / groups_per_head;
    const int j = (g % groups_per_head) * kVector;
    const int first = head * head_dim + j, second = first + half_dim;
    const Vec x1 = load8(in + first), x2 = load8(in + second);
    const Vec w1 = load8(weight + first), w2 = load8(weight + second);
    Vec y1, y2;
#pragma unroll
    for (int i = 0; i < kVector; ++i) {
      y1.v[i] = round_bf16(x1.v[i] * rstd * w1.v[i]);
      y2.v[i] = round_bf16(x2.v[i] * rstd * w2.v[i]);
    }
    if (cos != nullptr) {
      const int64_t at = head * rope_stride_head + token * rope_stride_token + j;
      const Vec c = load8(cos + at), s = load8(sin + at);
      Vec o1, o2;
#pragma unroll
      for (int i = 0; i < kVector; ++i) {
        o1.v[i] = y1.v[i] * c.v[i] - s.v[i] * y2.v[i];
        o2.v[i] = y2.v[i] * c.v[i] + s.v[i] * y1.v[i];
      }
      y1 = o1;
      y2 = o2;
    }
    store8(dst + first, y1);
    store8(dst + second, y2);
  }
}

template <class In>
__global__ void gate_heads_kernel(const In* attention, int64_t stride_token, int64_t stride_head, const bf16* logits,
                                  bf16* out, int64_t rows, int heads, int head_dim, float* amax) {
  const int dim = heads * head_dim;
  float peak = 0.f;
  for (int64_t e = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kVector; e < rows * dim;
       e += static_cast<int64_t>(gridDim.x) * blockDim.x * kVector) {
    const int64_t row = e / dim;
    const int column = e % dim, head = column / head_dim;
    const Vec a = load8(attention + row * stride_token + head * stride_head + column % head_dim);
    const float gate = 2.f * (1.f / (1.f + expf(-__bfloat162float(logits[row * heads + head]))));
    Vec o;
#pragma unroll
    for (int i = 0; i < kVector; ++i) {
      o.v[i] = round_bf16(a.v[i] * gate);
      peak = fmaxf(peak, fabsf(o.v[i]));
    }
    store8(out + e, o);
  }
  if (amax != nullptr) {
    peak = block_reduce<true>(peak);
    if (threadIdx.x == 0) atomic_max_nonnegative(amax, peak);
  }
}

__global__ void gated_residual_kernel(bf16* out, const bf16* residual, const bf16* y, const bf16* gate, int64_t rows,
                                      int dim, int64_t half_rows, int64_t row_offset) {
  for (int64_t e = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kVector; e < rows * dim;
       e += static_cast<int64_t>(gridDim.x) * blockDim.x * kVector) {
    const int64_t row = e / dim;
    const int column = e % dim;
    const Vec r = load8(residual + e), v = load8(y + e);
    const Vec g = load8(gate + (row + row_offset) / half_rows * dim + column);
    Vec o;
#pragma unroll
    for (int i = 0; i < kVector; ++i) o.v[i] = r.v[i] + v.v[i] * g.v[i];
    store8(out + e, o);
  }
}

__global__ void gelu_amax_kernel(const bf16* h, int64_t count, float* amax) {
  float peak = 0.f;
  for (int64_t e = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) * kVector; e < count;
       e += static_cast<int64_t>(gridDim.x) * blockDim.x * kVector) {
    const Vec a = load8(h + e);
#pragma unroll
    for (int i = 0; i < kVector; ++i) {
      const float v = a.v[i];
      const float gelu = 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
      peak = fmaxf(peak, fabsf(round_bf16(gelu)));
    }
  }
  peak = block_reduce<true>(peak);
  if (threadIdx.x == 0) atomic_max_nonnegative(amax, peak);
}

__global__ void fp4_tensor_scales_kernel(const float* amax, float* scale, int count, float divisor) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    const float value = amax[i] / divisor;
    scale[i] = value > 0.f ? value : 1.f;
  }
}

void check_bf16(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda() && t.scalar_type() == at::kBFloat16, name, " must be a CUDA bf16 tensor");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0, name, " must be 16-byte aligned");
}

void check_rows(const at::Tensor& t, const char* name) {
  check_bf16(t, name);
  TORCH_CHECK(t.dim() == 2 && t.is_contiguous() && t.size(1) % kVector == 0, name,
              " must be a contiguous (rows, k) tensor with k a multiple of ", kVector);
}

int elementwise_blocks(int64_t elements) {
  const int64_t wanted = (elements / kVector + kThreads - 1) / kThreads;
  const int64_t cap = static_cast<int64_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * 8;
  return static_cast<int>(std::max<int64_t>(1, std::min(wanted, cap)));
}

}  // namespace

void rms_modulate(const at::Tensor& x, const at::Tensor& shift, const at::Tensor& scale, const at::Tensor& out,
                  float* amax, int64_t amax_group_rows, double eps) {
  check_rows(x, "x");
  check_rows(out, "out");
  check_bf16(shift, "shift");
  check_bf16(scale, "scale");
  const int64_t rows = x.size(0), dim = x.size(1);
  TORCH_CHECK(rows % 2 == 0 && shift.is_contiguous() && scale.is_contiguous() && shift.numel() == 2 * dim &&
              scale.numel() == 2 * dim, "rms_modulate needs an even row count and (2, dim) shift/scale rows");
  rms_modulate_kernel<<<rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(x.data_ptr()), reinterpret_cast<const bf16*>(shift.data_ptr()),
      reinterpret_cast<const bf16*>(scale.data_ptr()), reinterpret_cast<bf16*>(out.data_ptr()), amax,
      static_cast<int>(dim), static_cast<int>(rows / 2), static_cast<int>(amax_group_rows), static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void norm_rope(const at::Tensor& x, const at::Tensor& weight, const at::Tensor* cos, const at::Tensor* sin,
               const at::Tensor& out, int64_t head_dim, double eps) {
  check_rows(x, "x");
  TORCH_CHECK(out.is_cuda() && (out.scalar_type() == at::kBFloat16 || out.scalar_type() == at::kHalf) &&
              out.sizes() == x.sizes() && out.is_contiguous() && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
              "out must be a contiguous bf16/fp16 tensor shaped like x");
  check_bf16(weight, "weight");
  const int64_t rows = x.size(0), dim = x.size(1);
  TORCH_CHECK(dim % head_dim == 0 && (head_dim / 2) % kVector == 0 && weight.is_contiguous() && weight.numel() == dim,
              "norm_rope needs heads of a multiple of ", 2 * kVector, " values and a (dim,) weight");
  int64_t stride_head = 0, stride_token = 0, tokens = rows / 2;
  if (cos != nullptr) {
    check_bf16(*cos, "cos");
    check_bf16(*sin, "sin");
    // [1, heads, tokens, head_dim / 2], innermost contiguous, both halves of the rows at the same positions
    TORCH_CHECK(cos->dim() == 4 && cos->size(1) * head_dim == dim && cos->size(2) == tokens &&
                cos->size(3) * 2 == head_dim && cos->stride(3) == 1 && cos->strides() == sin->strides() &&
                cos->stride(1) % kVector == 0 && cos->stride(2) % kVector == 0,
                "rope tables must be [1, heads, rows / 2, head_dim / 2]");
    stride_head = cos->stride(1);
    stride_token = cos->stride(2);
  }
  const auto launch = [&](auto* destination) {
    norm_rope_kernel<<<rows, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const bf16*>(x.data_ptr()), reinterpret_cast<const bf16*>(weight.data_ptr()),
        cos ? reinterpret_cast<const bf16*>(cos->data_ptr()) : nullptr,
        sin ? reinterpret_cast<const bf16*>(sin->data_ptr()) : nullptr, destination,
        static_cast<int>(dim), static_cast<int>(head_dim), static_cast<int>(std::max<int64_t>(tokens, 1)),
        stride_head, stride_token, static_cast<float>(eps));
  };
  if (out.scalar_type() == at::kHalf)
    launch(reinterpret_cast<__half*>(out.data_ptr()));
  else
    launch(reinterpret_cast<bf16*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gate_heads(const at::Tensor& attention, const at::Tensor& logits, const at::Tensor& out, float* amax) {
  TORCH_CHECK(attention.is_cuda() && (attention.scalar_type() == at::kBFloat16 || attention.scalar_type() == at::kHalf) &&
              reinterpret_cast<uintptr_t>(attention.data_ptr()) % 16 == 0,
              "attention must be a 16-byte aligned CUDA bf16/fp16 tensor");
  check_bf16(logits, "logits");
  check_rows(out, "out");
  TORCH_CHECK(attention.dim() == 3 && attention.stride(2) == 1 && attention.size(2) % kVector == 0 &&
              attention.stride(0) % kVector == 0 && attention.stride(1) % kVector == 0,
              "attention must be (rows, heads, head_dim) with a contiguous head_dim");
  const int64_t rows = attention.size(0), heads = attention.size(1), head_dim = attention.size(2);
  TORCH_CHECK(logits.dim() == 2 && logits.is_contiguous() && logits.size(0) == rows && logits.size(1) == heads &&
              out.size(0) == rows && out.size(1) == heads * head_dim, "gate logits must be (rows, heads)");
  const auto launch = [&](const auto* values) {
    gate_heads_kernel<<<elementwise_blocks(out.numel()), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
        values, attention.stride(0), attention.stride(1), reinterpret_cast<const bf16*>(logits.data_ptr()),
        reinterpret_cast<bf16*>(out.data_ptr()), rows, static_cast<int>(heads), static_cast<int>(head_dim), amax);
  };
  if (attention.scalar_type() == at::kHalf)
    launch(reinterpret_cast<const __half*>(attention.data_ptr()));
  else
    launch(reinterpret_cast<const bf16*>(attention.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gated_residual(const at::Tensor& out, const at::Tensor& residual, const at::Tensor& y, const at::Tensor& gate,
                    int64_t half_rows, int64_t row_offset) {
  check_rows(out, "out");
  check_rows(residual, "residual");
  check_rows(y, "y");
  check_bf16(gate, "gate");
  const int64_t rows = out.size(0), dim = out.size(1);
  TORCH_CHECK(residual.sizes() == out.sizes() && y.sizes() == out.sizes() && gate.is_contiguous() &&
              gate.numel() == 2 * dim, "gated_residual needs matching (rows, dim) tensors and (2, dim) gate rows");
  gated_residual_kernel<<<elementwise_blocks(out.numel()), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<bf16*>(out.data_ptr()), reinterpret_cast<const bf16*>(residual.data_ptr()),
      reinterpret_cast<const bf16*>(y.data_ptr()), reinterpret_cast<const bf16*>(gate.data_ptr()), rows,
      static_cast<int>(dim), half_rows, row_offset);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void gelu_amax(const at::Tensor& h, float* amax) {
  check_rows(h, "h");
  gelu_amax_kernel<<<elementwise_blocks(h.numel()), kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const bf16*>(h.data_ptr()), h.numel(), amax);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp4_tensor_scales(const at::Tensor& amax, const at::Tensor& scale, double divisor) {
  TORCH_CHECK(amax.scalar_type() == at::kFloat && scale.scalar_type() == at::kFloat && amax.is_contiguous() &&
              scale.is_contiguous() && amax.numel() == scale.numel(), "amax and scale must be matching fp32 vectors");
  const int count = static_cast<int>(amax.numel());
  fp4_tensor_scales_kernel<<<(count + kThreads - 1) / kThreads, kThreads, 0, at::cuda::getCurrentCUDAStream()>>>(
      amax.data_ptr<float>(), scale.data_ptr<float>(), count, static_cast<float>(divisor));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace jasna_ltx
