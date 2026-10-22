// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// INT8 Q and K quantization for the RDNA int8 attention kernel.
//
// The scale granularity differs from the CUDA backend because it follows the
// fragment layout, and RDNA's differs from the m16n8k16 one. On RDNA the score
// accumulator gives a lane one query column and eight key rows (see
// int8_attn.h), so a query scale is uniform over a lane and a key scale is
// uniform over one 16-key MMA tile:
//
//   q_scale: one float per query row,        [B, H_q, ceil(Lq / CTA_Q) * CTA_Q]
//   k_scale: one float per 16 adjacent keys, [B, H_kv, padded_Lk / 16]
//
// Per-row Q is finer than the CUDA per-thread grouping and costs nothing here;
// per-16-key K matches its effective grouping of 16 keys per scale.
//
// Layout is fixed by the caller: head_dim is 64, 128 or 256 (padded in Python),
// the last dimension is contiguous, and the int8 outputs are contiguous
// [B, H, L, D].
#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

#include "dense_mask.h"
#include "sage_common.h"

namespace comfy::hip_backend::sage {
namespace {

// A lane always owns four adjacent channels of a 128-channel tile; that is what
// the convrot butterflies assume. D64 therefore covers a row with a half-wave
// and puts two rows in a wave.
template <int HD>
struct RowTile {
    static constexpr int kLanesPerRow = (HD == 64) ? 16 : 32;
    static constexpr int kRowsPerWave = 32 / kLanesPerRow;
    static constexpr int kChannelTiles = HD / (kLanesPerRow * 4);
    static constexpr int kValsPerLane = kChannelTiles * 4;
};

// Load a lane's slice of one row into vals.
//
// Loading, biasing and rotating are separate steps because the K path needs an
// anchor row subtracted in between. Folding that into one function meant passing
// the anchor as a pointer, and a pointer to a private array is an escape the
// compiler cannot undo: the array lands in scratch even though nothing spills.
// Kept apart, both arrays stay in registers.
template <int HD, typename T>
__forceinline__ __device__ void load_row(const T* __restrict__ src, int lane_in_row,
                                         float* vals) {
    using Tile = RowTile<HD>;
#pragma unroll
    for (int tile = 0; tile < Tile::kChannelTiles; ++tile) {
        load4<T>(src + tile * 128 + lane_in_row * 4, vals + tile * 4);
    }
}

// Rotate a loaded row in place and return this lane's absmax. The caller reduces
// it over the row.
template <int HD, int Rotation>
__forceinline__ __device__ float rotate_row(float* vals) {
    using Tile = RowTile<HD>;
    if constexpr (Rotation != 0) {
#pragma unroll
        for (int tile = 0; tile < Tile::kChannelTiles; ++tile) {
            convrot<Rotation>(vals + tile * 4);
        }
    }
    float mx = 0.0f;
#pragma unroll
    for (int i = 0; i < Tile::kValsPerLane; ++i) mx = fmaxf(mx, fabsf(vals[i]));
    return mx;
}

template <int HD, typename T>
__forceinline__ __device__ void store_row(int8_t* __restrict__ dst, int lane_in_row,
                                          const float* vals, float inv_scale) {
    using Tile = RowTile<HD>;
#pragma unroll
    for (int tile = 0; tile < Tile::kChannelTiles; ++tile) {
        const int ch = tile * 128 + lane_in_row * 4;
        const float* v = vals + tile * 4;
        store4_i8(dst + ch, quant_int8_rcp(v[0], inv_scale), quant_int8_rcp(v[1], inv_scale),
                  quant_int8_rcp(v[2], inv_scale), quant_int8_rcp(v[3], inv_scale));
    }
}

// ---------------------------------------------------------------------------
// Q: one scale per query row, so a row never leaves its lane group and the
// reduction is a lane-group shuffle. Each block sweeps kRowIters batches of
// rows; one batch per block would leave every thread with a single 8-byte load
// and make the pass launch bound rather than bandwidth bound.
// ---------------------------------------------------------------------------

constexpr int kQuantWaves = 8;
constexpr int kRowIters = 8;

template <typename T, int HD, int Rotation>
__forceinline__ __device__ void quant_q_int8_block(
    const T* __restrict__ q, int8_t* __restrict__ q_out, float* __restrict__ q_scale, int Lq,
    int Lq_padded, int H_q, int64_t stride_b, int64_t stride_h, int64_t stride_n,
    int h, int b, int q_block) {

    using Tile = RowTile<HD>;
    constexpr int kRowsPerBatch = kQuantWaves * Tile::kRowsPerWave;

    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int lane_in_row = lane % Tile::kLanesPerRow;
    const int row_in_wave = lane / Tile::kLanesPerRow;

    const int64_t scale_base = (static_cast<int64_t>(b) * H_q + h) * Lq_padded;
    const T* head = q + static_cast<int64_t>(b) * stride_b + static_cast<int64_t>(h) * stride_h;
    int8_t* out_head = q_out + (static_cast<int64_t>(b) * H_q + h) * Lq * HD;

    const int block_first = q_block * (kRowsPerBatch * kRowIters);
#pragma unroll
    for (int it = 0; it < kRowIters; ++it) {
        const int row = block_first + it * kRowsPerBatch + wave * Tile::kRowsPerWave + row_in_wave;
        if (row >= Lq_padded) continue;

        // Rows past the end exist only so the attention kernel can read a scale
        // for every lane of its last tile. A finite scale keeps the dequantized
        // score finite; the score itself is masked out.
        if (row >= Lq) {
            if (lane_in_row == 0) q_scale[scale_base + row] = 1.0f;
            continue;
        }

        float vals[Tile::kValsPerLane];
        load_row<HD, T>(head + static_cast<int64_t>(row) * stride_n, lane_in_row, vals);
        const float mx = row_reduce_fmax<Tile::kLanesPerRow>(rotate_row<HD, Rotation>(vals));

        const float scale = mx / kInt8Max + 1e-7f;
        if (lane_in_row == 0) q_scale[scale_base + row] = scale;
        store_row<HD, T>(out_head + static_cast<int64_t>(row) * HD, lane_in_row, vals,
                         1.0f / scale);
    }
}

template <typename T, int HD, int Rotation>
__global__ __launch_bounds__(kQuantWaves * 32) void quant_q_int8_kernel(
    const T* __restrict__ q, int8_t* __restrict__ q_out, float* __restrict__ q_scale, int Lq,
    int Lq_padded, int H_q, int64_t stride_b, int64_t stride_h, int64_t stride_n) {

    quant_q_int8_block<T, HD, Rotation>(q, q_out, q_scale, Lq, Lq_padded, H_q,
        stride_b, stride_h, stride_n, blockIdx.y, blockIdx.z, blockIdx.x);
}

template <typename T, int HD, int Rotation>
__global__ __launch_bounds__(kQuantWaves * 32) void quant_q_mask_int8_kernel(
    const T* __restrict__ q, int8_t* __restrict__ q_out, float* __restrict__ q_scale, int Lq,
    int Lq_padded, int H_q, int64_t stride_b, int64_t stride_h, int64_t stride_n,
    int q_blocks, int batches, SageDenseMask16 mask) {

    // Separate blocks prepare the mask and quantize Q; their outputs are only
    // consumed by later launches on this stream, so neither task waits on the other.
    const int q_count = q_blocks * H_q * batches;
    if (blockIdx.x >= q_count) {
        const int kt = (mask.k + 63) / 64;
        const int qt = (mask.q + 15) / 16;
        const int tile = blockIdx.x - q_count;
        const int key_tile = tile % kt;
        const int query_tile = (tile / kt) % qt;
        const int mh = (tile / (kt * qt)) % mask.heads;
        const int mb = tile / (kt * qt * mask.heads);
        const int first = threadIdx.x * 4;
        const int query = query_tile * 16 + (first / 8) % 16;
        const int key = key_tile * 64 + (first / 256) * 16 + ((first / 128) % 2) * 8 + first % 8;
        const int64_t offset = mb * mask.sb + mh * mask.sh + query * mask.sq + key * mask.sk;
        const uint16_t exponent = mask.dtype == 2 ? 0x7f80u : 0x7c00u;
        uint16_t values[4];
        if (mask.sk == 1 && query < mask.q && key + 4 <= mask.k) {
            __builtin_memcpy(values, mask.data + offset, sizeof(values));
        } else {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                values[e] = query < mask.q && key + e < mask.k
                    ? mask.data[offset + e * mask.sk] : exponent | 0x8000u;
            }
        }
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            if ((values[e] & exponent) == exponent) values[e] = exponent | 0x8000u;
        }
        uint2 words;
        __builtin_memcpy(&words, values, sizeof(words));
        *reinterpret_cast<uint2*>(mask.packed + static_cast<int64_t>(tile) * 1024 + first) = words;
        return;
    }
    quant_q_int8_block<T, HD, Rotation>(q, q_out, q_scale, Lq, Lq_padded, H_q,
        stride_b, stride_h, stride_n, (blockIdx.x / q_blocks) % H_q,
        blockIdx.x / (q_blocks * H_q), blockIdx.x % q_blocks);
}

// ---------------------------------------------------------------------------
// K: one scale per 16 adjacent keys. A whole scale group lives in one wave, so
// the reduction never leaves it and the kernel needs no barrier at all. The
// general path rereads the group to limit register pressure. Paged decode
// retains the rotated rows, as CUDA does: with only two pages to refresh,
// avoiding the second rotation matters more than occupancy.
// ---------------------------------------------------------------------------

constexpr int kKeyGroupsPerBlock = kQuantWaves;

template <typename T, int HD, int Rotation, bool Paged = false>
__global__ __launch_bounds__(kQuantWaves * 32) void quant_k_int8_kernel(
    const T* __restrict__ k, int8_t* __restrict__ k_out, float* __restrict__ k_scale,
    const int* __restrict__ anchor_indices, int Lk, int H_kv, int groups_per_head,
    int64_t stride_b, int64_t stride_h, int64_t stride_n,
    const int64_t* position = nullptr, int pages = 0, bool initialize = false) {

    using Tile = RowTile<HD>;
    constexpr int kRowPasses = 16 / Tile::kRowsPerWave;

    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int lane_in_row = lane % Tile::kLanesPerRow;
    const int row_in_wave = lane / Tile::kLanesPerRow;

    const int group = blockIdx.x * kKeyGroupsPerBlock + wave;
    if (group >= groups_per_head) return;

    const int h = blockIdx.y;
    int b = blockIdx.z;
    const int padded_Lk = Lk;
    if constexpr (Paged) {
        const int count = initialize ? pages : 2;
        b = blockIdx.z / count;
        const int length = static_cast<int>(position[0]);
        const int page = initialize ? blockIdx.z % count :
            max(0, (length - 1) / padded_Lk - 1) + blockIdx.z % count;
        if (page >= pages || page * padded_Lk >= length) return;
        Lk = min(padded_Lk, length - page * padded_Lk);
        const int source_row = initialize ? page * padded_Lk :
            page * padded_Lk % (stride_h / stride_n);
        k += static_cast<int64_t>(source_row) * stride_n;
        const int64_t offset = (static_cast<int64_t>(b) * (pages - 1) + page) * H_kv;
        k_out += offset * padded_Lk * HD;
        k_scale += offset * groups_per_head;
    }
    const int64_t bh = static_cast<int64_t>(b) * H_kv + h;
    const T* head = k + static_cast<int64_t>(b) * stride_b + static_cast<int64_t>(h) * stride_h;
    const int key0 = group * 16;

    // A negative index means the detector found no anchor worth subtracting. The
    // subtraction stays unconditional either way, with a zero bias standing in,
    // so the anchor never has to be reached through a pointer.
    const int anchor_index = Paged ? -1 : anchor_indices[bh];
    float bias[Tile::kValsPerLane];
    if (anchor_index >= 0) {
        load_row<HD, T>(head + static_cast<int64_t>(anchor_index) * stride_n, lane_in_row, bias);
    } else {
#pragma unroll
        for (int i = 0; i < Tile::kValsPerLane; ++i) bias[i] = 0.0f;
    }

    float vals[Tile::kValsPerLane];
    float rotated[Paged ? kRowPasses : 1][Tile::kValsPerLane];
    float mx = 0.0f;
#pragma unroll
    for (int pass = 0; pass < kRowPasses; ++pass) {
        const int key = key0 + pass * Tile::kRowsPerWave + row_in_wave;
        if (key < Lk) {
            load_row<HD, T>(head + static_cast<int64_t>(key) * stride_n, lane_in_row, vals);
#pragma unroll
            for (int i = 0; i < Tile::kValsPerLane; ++i) vals[i] -= bias[i];
            mx = fmaxf(mx, rotate_row<HD, Rotation>(vals));
            if constexpr (Paged) {
#pragma unroll
                for (int i = 0; i < Tile::kValsPerLane; ++i) rotated[pass][i] = vals[i];
            }
        }
    }

    mx = row_reduce_fmax<32>(mx);
    const float scale = mx / kInt8Max + 1e-7f;
    const float inv_scale = 1.0f / scale;
    if (lane == 0) k_scale[bh * groups_per_head + group] = scale;

#pragma unroll
    for (int pass = 0; pass < kRowPasses; ++pass) {
        const int key = key0 + pass * Tile::kRowsPerWave + row_in_wave;
        if (key < Lk) {
            if constexpr (Paged) {
#pragma unroll
                for (int i = 0; i < Tile::kValsPerLane; ++i) vals[i] = rotated[pass][i];
            } else {
                load_row<HD, T>(head + static_cast<int64_t>(key) * stride_n, lane_in_row, vals);
#pragma unroll
                for (int i = 0; i < Tile::kValsPerLane; ++i) vals[i] -= bias[i];
                rotate_row<HD, Rotation>(vals);
            }
            store_row<HD, T>(k_out + (bh * padded_Lk + key) * HD, lane_in_row, vals, inv_scale);
        }
    }
}

// ---------------------------------------------------------------------------
// Model-independent K stabilization detector. Ported from the CUDA
// detect_k_anchor: sample nine evenly spaced keys per (batch, head), pick the
// sampled key that minimizes residual energy, and accept it only when it
// shrinks sampled energy without growing sampled abs-max by more than 12.5%.
// The result is an absolute key index, or -1 to leave K alone. Nothing here
// synchronizes with the host.
// ---------------------------------------------------------------------------

constexpr int kAnchorThreads = 128;
constexpr int kAnchorWaves = kAnchorThreads / 32;
constexpr int kAnchorSamples = 9;
constexpr int kAnchorMaxChannels = 256;

template <typename T>
__global__ __launch_bounds__(kAnchorThreads) void detect_k_anchor_kernel(
    const T* __restrict__ k, int* __restrict__ anchor_indices, int Lk, int C, int H_kv,
    int64_t stride_b, int64_t stride_h, int64_t stride_n) {

    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int64_t bh_off =
        static_cast<int64_t>(b) * stride_b + static_cast<int64_t>(h) * stride_h;

    __shared__ float samples[kAnchorSamples * kAnchorMaxChannels];
    __shared__ float wave_original_energy[kAnchorWaves];
    __shared__ float wave_original_max[kAnchorWaves];
    __shared__ float wave_candidate_distance[kAnchorSamples][kAnchorWaves];
    __shared__ float wave_best_energy[kAnchorWaves];
    __shared__ float wave_best_max[kAnchorWaves];
    __shared__ int selected_candidate;

    for (int index = tid; index < kAnchorSamples * C; index += kAnchorThreads) {
        const int sample = index / C;
        const int channel = index - sample * C;
        const int row = sample * (Lk - 1) / (kAnchorSamples - 1);
        samples[index] = to_float<T>(k[bh_off + static_cast<int64_t>(row) * stride_n + channel]);
    }
    __syncthreads();

    float original_energy = 0.0f;
    float original_max = 0.0f;
    float candidate_distance[kAnchorSamples];
#pragma unroll
    for (int c = 0; c < kAnchorSamples; ++c) candidate_distance[c] = 0.0f;

    for (int channel = tid; channel < C; channel += kAnchorThreads) {
        float channel_sum = 0.0f;
#pragma unroll
        for (int sample = 0; sample < kAnchorSamples; ++sample) {
            const float value = samples[sample * C + channel];
            original_energy = fmaf(value, value, original_energy);
            original_max = fmaxf(original_max, fabsf(value));
            channel_sum += value;
        }
#pragma unroll
        for (int c = 0; c < kAnchorSamples; ++c) {
            const float distance = kAnchorSamples * samples[c * C + channel] - channel_sum;
            candidate_distance[c] = fmaf(distance, distance, candidate_distance[c]);
        }
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        original_energy += __shfl_down(original_energy, off, 32);
        original_max = fmaxf(original_max, __shfl_down(original_max, off, 32));
#pragma unroll
        for (int c = 0; c < kAnchorSamples; ++c) {
            candidate_distance[c] += __shfl_down(candidate_distance[c], off, 32);
        }
    }

    if (lane == 0) {
        wave_original_energy[wave] = original_energy;
        wave_original_max[wave] = original_max;
#pragma unroll
        for (int c = 0; c < kAnchorSamples; ++c) {
            wave_candidate_distance[c][wave] = candidate_distance[c];
        }
    }
    __syncthreads();

    if (tid == 0) {
        int best_candidate = 0;
        float best_distance = 3.402823466e+38F;
#pragma unroll
        for (int c = 0; c < kAnchorSamples; ++c) {
            float distance = 0.0f;
#pragma unroll
            for (int w = 0; w < kAnchorWaves; ++w) distance += wave_candidate_distance[c][w];
            if (distance < best_distance) {
                best_candidate = c;
                best_distance = distance;
            }
        }
        selected_candidate = best_candidate;
    }
    __syncthreads();

    float best_energy = 0.0f;
    float best_max = 0.0f;
    for (int channel = tid; channel < C; channel += kAnchorThreads) {
        const float anchor = samples[selected_candidate * C + channel];
#pragma unroll
        for (int sample = 0; sample < kAnchorSamples; ++sample) {
            const float residual = samples[sample * C + channel] - anchor;
            best_energy = fmaf(residual, residual, best_energy);
            best_max = fmaxf(best_max, fabsf(residual));
        }
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        best_energy += __shfl_down(best_energy, off, 32);
        best_max = fmaxf(best_max, __shfl_down(best_max, off, 32));
    }
    if (lane == 0) {
        wave_best_energy[wave] = best_energy;
        wave_best_max[wave] = best_max;
    }
    __syncthreads();

    if (tid == 0) {
        float total_original_energy = 0.0f;
        float total_original_max = 0.0f;
        float total_best_energy = 0.0f;
        float total_best_max = 0.0f;
#pragma unroll
        for (int w = 0; w < kAnchorWaves; ++w) {
            total_original_energy += wave_original_energy[w];
            total_original_max = fmaxf(total_original_max, wave_original_max[w]);
            total_best_energy += wave_best_energy[w];
            total_best_max = fmaxf(total_best_max, wave_best_max[w]);
        }

        const bool improves_range = total_best_energy < total_original_energy &&
                                    total_best_max <= total_original_max * 1.125f;
        anchor_indices[b * H_kv + h] =
            improves_range ? selected_candidate * (Lk - 1) / (kAnchorSamples - 1) : -1;
    }
}

template <typename T, int HD, int Rotation>
void launch_typed(const void* q, void* q_int8, void* q_scale, const void* k, void* k_int8,
                  void* k_scale, const int* anchor_indices, int B, int H_q, int Lq, int Lq_padded,
                  int H_kv, int Lk, int groups_per_head, int64_t q_stride_b, int64_t q_stride_h,
                  int64_t q_stride_n, int64_t k_stride_b, int64_t k_stride_h, int64_t k_stride_n,
                  hipStream_t stream) {

    using Tile = RowTile<HD>;
    constexpr int kRowsPerBlock = kQuantWaves * Tile::kRowsPerWave * kRowIters;

    const dim3 q_grid((Lq_padded + kRowsPerBlock - 1) / kRowsPerBlock, H_q, B);
    quant_q_int8_kernel<T, HD, Rotation><<<q_grid, kQuantWaves * 32, 0, stream>>>(
        static_cast<const T*>(q), static_cast<int8_t*>(q_int8), static_cast<float*>(q_scale), Lq,
        Lq_padded, H_q, q_stride_b, q_stride_h, q_stride_n);

    const dim3 k_grid((groups_per_head + kKeyGroupsPerBlock - 1) / kKeyGroupsPerBlock, H_kv, B);
    quant_k_int8_kernel<T, HD, Rotation><<<k_grid, kQuantWaves * 32, 0, stream>>>(
        static_cast<const T*>(k), static_cast<int8_t*>(k_int8), static_cast<float*>(k_scale),
        anchor_indices, Lk, H_kv, groups_per_head, k_stride_b, k_stride_h, k_stride_n);
}

// Only the rotations the caller can actually select are instantiated: short keys
// take H4, and longer ones the widest Hadamard the head dimension allows.
template <typename T>
void launch_dispatch_hd(int head_dim, int rotation, const void* q, void* q_int8, void* q_scale,
                        const void* k, void* k_int8, void* k_scale, const int* anchor_indices,
                        int B, int H_q, int Lq, int Lq_padded, int H_kv, int Lk,
                        int groups_per_head, int64_t q_stride_b, int64_t q_stride_h,
                        int64_t q_stride_n, int64_t k_stride_b, int64_t k_stride_h,
                        int64_t k_stride_n, hipStream_t stream) {

#define SAGE_QK_CASE(HD, ROT)                                                                    \
    launch_typed<T, HD, ROT>(q, q_int8, q_scale, k, k_int8, k_scale, anchor_indices, B, H_q, Lq, \
                             Lq_padded, H_kv, Lk, groups_per_head, q_stride_b, q_stride_h,       \
                             q_stride_n, k_stride_b, k_stride_h, k_stride_n, stream)

    switch (head_dim) {
        case 64:
            if (rotation == 4) {
                SAGE_QK_CASE(64, 4);
            } else {
                SAGE_QK_CASE(64, 64);
            }
            break;
        case 128:
            if (rotation == 4) {
                SAGE_QK_CASE(128, 4);
            } else {
                SAGE_QK_CASE(128, 128);
            }
            break;
        case 256:
            if (rotation == 4) {
                SAGE_QK_CASE(256, 4);
            } else {
                SAGE_QK_CASE(256, 129);
            }
            break;
        default:
            throw std::runtime_error("sage int8 attention: unsupported head_dim " +
                                     std::to_string(head_dim));
    }

#undef SAGE_QK_CASE
}

template <typename T>
void launch_anchor(const void* k, int* anchor_indices, int B, int H_kv, int Lk, int C,
                   int64_t stride_b, int64_t stride_h, int64_t stride_n, hipStream_t stream) {
    const dim3 grid(H_kv, B);
    detect_k_anchor_kernel<T><<<grid, kAnchorThreads, 0, stream>>>(
        static_cast<const T*>(k), anchor_indices, Lk, C, H_kv, stride_b, stride_h, stride_n);
}

}  // namespace
}  // namespace comfy::hip_backend::sage

#include "launchers.h"

namespace jasna_ltx::sage_attention_hip {

using namespace comfy::hip_backend::sage;

// rotation is 4, 64, 128 or 129; the caller picks it the same way the CUDA
// backend does, and Q and K must agree or their dot products change.
// anchor_indices is one int per (batch, kv head) of device scratch.
void launch_quant_qk_int8(const void* q, void* q_int8, void* q_scale,
                                          const void* k, void* k_int8, void* k_scale,
                                          void* anchor_indices, int B, int H_q, int Lq,
                                          int Lq_padded, int H_kv, int Lk, int groups_per_head,
                                          int head_dim, int64_t q_stride_b, int64_t q_stride_h,
                                          int64_t q_stride_n, int64_t k_stride_b,
                                          int64_t k_stride_h, int64_t k_stride_n,
                                          int input_dtype_code, int rotation, hipStream_t stream) {

    if (anchor_indices == nullptr) {
        throw std::runtime_error("sage int8 attention: anchor_indices scratch is required");
    }
    if (head_dim > kAnchorMaxChannels) {
        throw std::runtime_error("sage int8 attention: head_dim exceeds the anchor sample width");
    }
    // The dispatch below picks the wide rotation for anything that is not 4, so a
    // tag the head dimension does not own would be widened silently and produce
    // numbers the CUDA backend never produces.
    const int wide_rotation = head_dim == 64 ? 64 : (head_dim == 128 ? 128 : 129);
    if (rotation != 4 && rotation != wide_rotation) {
        throw std::runtime_error("sage int8 attention: rotation " + std::to_string(rotation) +
                                 " is not valid for head_dim " + std::to_string(head_dim));
    }

    // The quantizers read four elements at a time from every row, so the base
    // pointer and all three strides have to keep that group aligned. An extent of
    // one is exempt: its index is always zero, so the stride never reaches the
    // address, and PyTorch rewrites such a stride to 1 on the way through DLPack
    // (ATen/DLConvertor.cpp, gh-83069), which no caller can prevent.
    const size_t element_size = input_dtype_code == 0 ? sizeof(float) : sizeof(__half);
    const size_t vector_size = 4 * element_size;
    const auto stride_ok = [](int64_t stride, int extent) {
        return extent < 2 || (stride > 0 && stride % 4 == 0);
    };
    const auto vector_aligned = [vector_size, &stride_ok](const void* p, int64_t sb, int64_t sh,
                                                          int64_t sn, int nb, int nh, int nn) {
        return reinterpret_cast<uintptr_t>(p) % vector_size == 0 && stride_ok(sb, nb) &&
               stride_ok(sh, nh) && stride_ok(sn, nn);
    };
    if (!vector_aligned(q, q_stride_b, q_stride_h, q_stride_n, B, H_q, Lq) ||
        !vector_aligned(k, k_stride_b, k_stride_h, k_stride_n, B, H_kv, Lk)) {
        throw std::runtime_error(
            "sage int8 attention: Q/K base pointers and B/H/N strides must preserve 4-element "
            "alignment");
    }
    // The int8 outputs are written four at a time through store4_i8, whose row
    // offsets are all multiples of four, so the base pointers carry the rest.
    if (reinterpret_cast<uintptr_t>(q_int8) % 4 != 0 ||
        reinterpret_cast<uintptr_t>(k_int8) % 4 != 0) {
        throw std::runtime_error(
            "sage int8 attention: quantized Q/K base pointers must be 4-byte aligned");
    }

    int* anchor_ptr = static_cast<int*>(anchor_indices);

#define SAGE_QK_DISPATCH_DTYPE(T)                                                                \
    do {                                                                                         \
        launch_anchor<T>(k, anchor_ptr, B, H_kv, Lk, head_dim, k_stride_b, k_stride_h,           \
                         k_stride_n, stream);                                                    \
        launch_dispatch_hd<T>(head_dim, rotation, q, q_int8, q_scale, k, k_int8, k_scale,        \
                              anchor_ptr, B, H_q, Lq, Lq_padded, H_kv, Lk, groups_per_head,      \
                              q_stride_b, q_stride_h, q_stride_n, k_stride_b, k_stride_h,        \
                              k_stride_n, stream);                                                \
    } while (0)

    if (input_dtype_code != kBFloat16) {
        throw std::runtime_error("sage int8 attention: input dtype must be bf16");
    }
    SAGE_QK_DISPATCH_DTYPE(__bf16);

#undef SAGE_QK_DISPATCH_DTYPE

    const hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        throw std::runtime_error(std::string("sage int8 quantize kernel launch failed: ") +
                                 hipGetErrorString(error));
    }
}

}  // namespace jasna_ltx::sage_attention_hip
