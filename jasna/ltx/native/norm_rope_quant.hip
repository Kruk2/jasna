// glue.cu's norm_rope fused into comfy-kitchen's INT8 Q/K quantizers (see norm_rope_quant.h).
//
// This translation unit is comfy-kitchen's quant_qk_int8.hip plus these kernels, so the vendored
// quantizers and the fused ones share one compilation with comfy-kitchen's flags (fast math): the
// rotation, scale and rounding code is comfy-kitchen's own, called the same way. The norm and RoPE
// arithmetic repeats glue.cu's in precise math with hipcc's default contraction, the block-wide
// sum of squares in glue's order and its division correctly rounded, as glue.cu compiles it.
#include "sage_attention_hip/quant_qk_int8.hip"

#include <type_traits>

#include "norm_rope_quant.h"

namespace jasna_ltx::sage_attention_hip {
namespace {

namespace sage = comfy::hip_backend::sage;
using sage::RowTile;

constexpr int kThreads = sage::kQuantWaves * 32;  // glue.cu's norm_rope block, whose reduction row_rstd repeats
constexpr int kKeyGroup = 16;

// glue.cu's float semantics for the norm and RoPE (AMDGPU has no float_control): no reassociation,
// hipcc's default contraction.
#pragma clang fp reassociate(off)
#pragma clang fp contract(fast)

// Round to nearest even bf16 in integer ops, which fast math cannot fold away (finite values as
// glue.cu's __float2bfloat16).
__device__ __forceinline__ float round_bf16(float x) {
  const uint32_t bits = __float_as_uint(x);
  return __uint_as_float((bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u);
}

// rsqrt(mean(row^2) + eps), summed as glue.cu's norm_rope_kernel and block_reduce sum; every thread
// of the block returns it.
__device__ float row_rstd(const __bf16* row, int dim, float eps) {
  __shared__ float partial[kThreads / 32];
  float squares = 0.f;
  for (int c = threadIdx.x * 8; c < dim; c += blockDim.x * 8) {
    const uint4 raw = *reinterpret_cast<const uint4*>(row + c);
    const __bf16* values = reinterpret_cast<const __bf16*>(&raw);
    float a[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) a[i] = static_cast<float>(values[i]);
#pragma unroll
    for (int i = 0; i < 8; ++i) squares += a[i] * a[i];
  }
  __syncthreads();  // partial may still be read by a previous reduction
#pragma unroll
  for (int offset = 16; offset; offset >>= 1) {
    const float other = __shfl_xor(squares, offset);
    squares = squares + other;
  }
  if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = squares;
  __syncthreads();
  float total = partial[0];
  for (int i = 1; i < blockDim.x / 32; ++i) total = total + partial[i];
  return rsqrtf(__fdiv_rn(total, static_cast<float>(dim)) + eps);
}

// row_rstd by one wave: lane l stands in for norm_rope_kernel's threads l, l + 32, ..., whose warp
// sums it forms one after another, so the sum and its rounding are the same.
__device__ float wave_row_rstd(const __bf16* row, int dim, float eps) {
  const int lane = threadIdx.x & 31;
  float total = 0.f;
#pragma unroll
  for (int warp = 0; warp < kThreads / 32; ++warp) {
    float squares = 0.f;
    for (int c = (warp * 32 + lane) * 8; c < dim; c += kThreads * 8) {
      const uint4 raw = *reinterpret_cast<const uint4*>(row + c);
      const __bf16* values = reinterpret_cast<const __bf16*>(&raw);
      float a[8];
#pragma unroll
      for (int i = 0; i < 8; ++i) a[i] = static_cast<float>(values[i]);
#pragma unroll
      for (int i = 0; i < 8; ++i) squares += a[i] * a[i];
    }
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) {
      const float other = __shfl_xor(squares, offset);
      squares = squares + other;
    }
    total = warp == 0 ? squares : total + squares;
  }
  return rsqrtf(__fdiv_rn(total, static_cast<float>(dim)) + eps);
}

// One head of norm_rope's bf16 output row as comfy-kitchen's quantizer lanes hold it: four adjacent
// channels of each 128-channel tile. The RoPE partner (half a head away) is another lane, or for
// 256-wide heads the other tile of this lane.
template <int HD>
__device__ __forceinline__ void normed_head(const NormRope& p, int row, int head, int lane_in_row, float rstd,
                                            float* vals) {
  using Tile = RowTile<HD>;
  constexpr int kHalf = HD / 2;
  const __bf16* x = static_cast<const __bf16*>(p.x) + static_cast<int64_t>(row) * p.heads * HD + head * HD;
  const __bf16* weight = static_cast<const __bf16*>(p.weight) + head * HD;
  float y[Tile::kValsPerLane];
#pragma unroll
  for (int tile = 0; tile < Tile::kChannelTiles; ++tile) {
    float xv[4], wv[4];
    sage::load4(x + tile * 128 + lane_in_row * 4, xv);
    sage::load4(weight + tile * 128 + lane_in_row * 4, wv);
#pragma unroll
    for (int e = 0; e < 4; ++e) y[tile * 4 + e] = round_bf16(xv[e] * rstd * wv[e]);
  }
  if (p.cos == nullptr) {
#pragma unroll
    for (int i = 0; i < Tile::kValsPerLane; ++i) vals[i] = y[i];
    return;
  }
  const int64_t at = head * p.rope_stride_head + (row % p.tokens) * p.rope_stride_token;
#pragma unroll
  for (int tile = 0; tile < Tile::kChannelTiles; ++tile) {
    const int channel = tile * 128 + lane_in_row * 4;
    const bool first_half = channel < kHalf;
    float cv[4], sv[4];
    sage::load4(static_cast<const __bf16*>(p.cos) + at + channel % kHalf, cv);
    sage::load4(static_cast<const __bf16*>(p.sin) + at + channel % kHalf, sv);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      float partner;
      if constexpr (kHalf >= 128)
        partner = y[(tile ^ 1) * 4 + e];
      else
        partner = __shfl_xor(y[tile * 4 + e], kHalf / 4, Tile::kLanesPerRow);
      // glue: first = y1 * cos - sin * y2, second = y2 * cos + sin * y1; the sign is exact
      const float signed_sin = first_half ? -sv[e] : sv[e];
      vals[tile * 4 + e] = round_bf16(y[tile * 4 + e] * cv[e] + signed_sin * partner);
    }
  }
}

#pragma clang fp reassociate(on)

template <int HD, int Rotation>
__global__ __launch_bounds__(kThreads) void norm_rope_quant_q_kernel(NormRope p, int8_t* q_int8, float* q_scale,
                                                                      int padded_rows) {
  using Tile = RowTile<HD>;
  const int row = blockIdx.x;
  if (row >= p.rows) {  // as comfy-kitchen: a finite scale for the padding rows of the last query tile
    for (int head = threadIdx.x; head < p.heads; head += blockDim.x)
      q_scale[static_cast<int64_t>(head) * padded_rows + row] = 1.0f;
    return;
  }
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int lane_in_row = lane % Tile::kLanesPerRow, row_in_wave = lane / Tile::kLanesPerRow;
  const float rstd = row_rstd(static_cast<const __bf16*>(p.x) + static_cast<int64_t>(row) * p.heads * HD,
                              p.heads * HD, p.eps);
  for (int first = wave * Tile::kRowsPerWave; first < p.heads; first += sage::kQuantWaves * Tile::kRowsPerWave) {
    const int head = first + row_in_wave;
    float vals[Tile::kValsPerLane];
    normed_head<HD>(p, row, head, lane_in_row, rstd, vals);
    const float mx = sage::row_reduce_fmax<Tile::kLanesPerRow>(sage::rotate_row<HD, Rotation>(vals));
    const float scale = mx / sage::kInt8Max + 1e-7f;
    if (lane_in_row == 0) q_scale[static_cast<int64_t>(head) * padded_rows + row] = scale;
    sage::store_row<HD, __bf16>(q_int8 + (static_cast<int64_t>(head) * p.rows + row) * HD, lane_in_row, vals,
                                1.0f / scale);
  }
}

// comfy-kitchen's nine anchor-candidate keys, normalized: sample s is key s * (rows - 1) / 8.
template <int HD>
__global__ __launch_bounds__(kThreads) void norm_rope_samples_kernel(NormRope p, __bf16* samples) {
  using Tile = RowTile<HD>;
  const int sample = blockIdx.x, row = sample * (p.rows - 1) / (sage::kAnchorSamples - 1), dim = p.heads * HD;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  const int lane_in_row = lane % Tile::kLanesPerRow, row_in_wave = lane / Tile::kLanesPerRow;
  const float rstd = row_rstd(static_cast<const __bf16*>(p.x) + static_cast<int64_t>(row) * dim, dim, p.eps);
  for (int first = wave * Tile::kRowsPerWave; first < p.heads; first += sage::kQuantWaves * Tile::kRowsPerWave) {
    const int head = first + row_in_wave;
    float vals[Tile::kValsPerLane];
    normed_head<HD>(p, row, head, lane_in_row, rstd, vals);
#pragma unroll
    for (int i = 0; i < Tile::kValsPerLane; ++i)
      samples[sample * dim + head * HD + (i / 4) * 128 + lane_in_row * 4 + i % 4] = static_cast<__bf16>(vals[i]);
  }
}

// comfy-kitchen's quant_k_int8_kernel (general path: the group's rows are recomputed for the store)
// with each key normalized on the fly; a block holds one 16-key group of eight heads, the blocks of
// one group next to each other so they share its rows in cache.
template <int HD, int Rotation>
__global__ __launch_bounds__(kThreads) void norm_rope_quant_k_kernel(NormRope p, const __bf16* samples,
                                                                      const int* anchors, int8_t* k_int8,
                                                                      float* k_scale, int groups) {
  using Tile = RowTile<HD>;
  constexpr int kRowPasses = kKeyGroup / Tile::kRowsPerWave;
  __shared__ float rstd[kKeyGroup];
  const int group = blockIdx.y, key0 = group * kKeyGroup, dim = p.heads * HD;
  const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
  for (int r = wave; r < kKeyGroup && key0 + r < p.rows; r += sage::kQuantWaves) {
    const float value =
        wave_row_rstd(static_cast<const __bf16*>(p.x) + static_cast<int64_t>(key0 + r) * dim, dim, p.eps);
    if (lane == 0) rstd[r] = value;
  }
  __syncthreads();
  const int lane_in_row = lane % Tile::kLanesPerRow, row_in_wave = lane / Tile::kLanesPerRow;
  const int head = blockIdx.x * sage::kQuantWaves + wave;
  if (head < p.heads) {
    const int anchor = anchors[head];
    float bias[Tile::kValsPerLane];
    if (anchor >= 0) {
      sage::load_row<HD, __bf16>(samples + anchor * dim + head * HD, lane_in_row, bias);
    } else {
#pragma unroll
      for (int i = 0; i < Tile::kValsPerLane; ++i) bias[i] = 0.0f;
    }
    float vals[Tile::kValsPerLane];
    float mx = 0.0f;
#pragma unroll
    for (int pass = 0; pass < kRowPasses; ++pass) {
      const int key = key0 + pass * Tile::kRowsPerWave + row_in_wave;
      if (key < p.rows) {
        normed_head<HD>(p, key, head, lane_in_row, rstd[key - key0], vals);
#pragma unroll
        for (int i = 0; i < Tile::kValsPerLane; ++i) vals[i] -= bias[i];
        mx = fmaxf(mx, sage::rotate_row<HD, Rotation>(vals));
      }
    }
    mx = sage::row_reduce_fmax<32>(mx);
    const float scale = mx / sage::kInt8Max + 1e-7f;
    const float inv_scale = 1.0f / scale;
    if (lane == 0) k_scale[static_cast<int64_t>(head) * groups + group] = scale;
#pragma unroll
    for (int pass = 0; pass < kRowPasses; ++pass) {
      const int key = key0 + pass * Tile::kRowsPerWave + row_in_wave;
      if (key < p.rows) {
        normed_head<HD>(p, key, head, lane_in_row, rstd[key - key0], vals);
#pragma unroll
        for (int i = 0; i < Tile::kValsPerLane; ++i) vals[i] -= bias[i];
        sage::rotate_row<HD, Rotation>(vals);
        sage::store_row<HD, __bf16>(k_int8 + (static_cast<int64_t>(head) * p.rows + key) * HD, lane_in_row, vals,
                                    inv_scale);
      }
    }
  }
}

template <int Value>
using Constant = std::integral_constant<int, Value>;

// Calls launch(head_dim, rotation) with both as constants, for the rotations comfy-kitchen's
// launch_quant_qk_int8 accepts: 4 for short keys, else the widest Hadamard of the head.
template <typename Launch>
void dispatch(int head_dim, int rotation, Launch&& launch) {
  const int wide = head_dim == 64 ? 64 : head_dim == 128 ? 128 : 129;
  if (rotation != 4 && rotation != wide)
    throw std::runtime_error("sage int8 attention: rotation " + std::to_string(rotation) +
                             " is not valid for head_dim " + std::to_string(head_dim));
  switch (head_dim) {
    case 64:
      rotation == 4 ? launch(Constant<64>{}, Constant<4>{}) : launch(Constant<64>{}, Constant<64>{});
      break;
    case 128:
      rotation == 4 ? launch(Constant<128>{}, Constant<4>{}) : launch(Constant<128>{}, Constant<128>{});
      break;
    case 256:
      rotation == 4 ? launch(Constant<256>{}, Constant<4>{}) : launch(Constant<256>{}, Constant<129>{});
      break;
    default:
      throw std::runtime_error("sage int8 attention: unsupported head_dim " + std::to_string(head_dim));
  }
}

void check_norm_rope(const NormRope& p) {
  if (p.head_dim == 64 && p.heads % 2)
    throw std::runtime_error("norm_rope_quant: 64-wide heads come in pairs, one per half wave");
  if (p.rows < 1) throw std::runtime_error("norm_rope_quant: no rows");
}

void check_launch(const char* what) {
  const hipError_t error = hipGetLastError();
  if (error != hipSuccess) throw std::runtime_error(std::string(what) + " launch failed: " + hipGetErrorString(error));
}

}  // namespace

void launch_norm_rope_quant_q(const NormRope& q, void* q_int8, void* q_scale, int padded_rows, int rotation,
                              hipStream_t stream) {
  check_norm_rope(q);
  dispatch(q.head_dim, rotation, [&](auto head_dim, auto rotation_tag) {
    norm_rope_quant_q_kernel<head_dim(), rotation_tag()><<<padded_rows, kThreads, 0, stream>>>(
        q, static_cast<int8_t*>(q_int8), static_cast<float*>(q_scale), padded_rows);
  });
  check_launch("norm_rope_quant q");
}

void launch_norm_rope_quant_k(const NormRope& k, void* k_int8, void* k_scale, void* samples, void* anchors,
                              int groups, int rotation, hipStream_t stream) {
  check_norm_rope(k);
  auto* sample_rows = static_cast<__bf16*>(samples);
  dispatch(k.head_dim, rotation, [&](auto head_dim, auto rotation_tag) {
    norm_rope_samples_kernel<head_dim()><<<sage::kAnchorSamples, kThreads, 0, stream>>>(k, sample_rows);
    sage::launch_anchor<__bf16>(sample_rows, static_cast<int*>(anchors), 1, k.heads, sage::kAnchorSamples,
                                head_dim(), 0, head_dim(), static_cast<int64_t>(k.heads) * head_dim(), stream);
    const dim3 grid((k.heads + sage::kQuantWaves - 1) / sage::kQuantWaves, groups);
    norm_rope_quant_k_kernel<head_dim(), rotation_tag()><<<grid, kThreads, 0, stream>>>(
        k, sample_rows, static_cast<const int*>(anchors), static_cast<int8_t*>(k_int8), static_cast<float*>(k_scale),
        groups);
  });
  check_launch("norm_rope_quant k");
}

void launch_quant_k_int8(const void* k, void* k_int8, void* k_scale, void* anchors, int heads, int keys, int groups,
                         int head_dim, int64_t stride_h, int64_t stride_n, int rotation, hipStream_t stream) {
  dispatch(head_dim, rotation, [&](auto hd, auto rotation_tag) {
    sage::launch_anchor<__bf16>(k, static_cast<int*>(anchors), 1, heads, keys, hd(), 0, stride_h, stride_n, stream);
    const dim3 grid((groups + sage::kKeyGroupsPerBlock - 1) / sage::kKeyGroupsPerBlock, heads, 1);
    sage::quant_k_int8_kernel<__bf16, hd(), rotation_tag()><<<grid, sage::kQuantWaves * 32, 0, stream>>>(
        static_cast<const __bf16*>(k), static_cast<int8_t*>(k_int8), static_cast<float*>(k_scale),
        static_cast<const int*>(anchors), keys, heads, groups, 0, stride_h, stride_n);
  });
  check_launch("sage int8 quantize k");
}

}  // namespace jasna_ltx::sage_attention_hip
