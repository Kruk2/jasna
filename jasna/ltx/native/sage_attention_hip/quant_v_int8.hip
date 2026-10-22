// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Fused V to signed INT8 with a per-channel scale, transposed to
// [B * H * D, padded_N] so the attention kernel reads a V fragment as a
// contiguous run of keys along one channel row.
//
// The CUDA backend permutes each group of 16 keys here to match what ldmatrix
// hands its MMA. RDNA needs no permutation: the probability fragment and the V
// fragment both index keys in natural order (see int8_attn.h), so keys are
// written where they belong and the tail is zero filled.
//
// One block per (b, h, d_tile) of 8 channels. Pass 1 reduces the per-channel
// absmax along N with 4x unrolled vector loads, pass 2 quantizes.
#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

#include "sage_common.h"

namespace comfy::hip_backend::sage {
namespace {

constexpr int kDTile = 8;

// Eight adjacent channels: 32 bytes for fp32, 16 for the 16-bit types.
template <typename T>
__forceinline__ __device__ void load8(const T* p, float* out) {
    if constexpr (std::is_same_v<T, float>) {
        const float4 lo = *reinterpret_cast<const float4*>(p);
        const float4 hi = *reinterpret_cast<const float4*>(p + 4);
        out[0] = lo.x;
        out[1] = lo.y;
        out[2] = lo.z;
        out[3] = lo.w;
        out[4] = hi.x;
        out[5] = hi.y;
        out[6] = hi.z;
        out[7] = hi.w;
    } else {
        const uint4 raw = *reinterpret_cast<const uint4*>(p);
        const T* vals = reinterpret_cast<const T*>(&raw);
#pragma unroll
        for (int i = 0; i < kDTile; ++i) out[i] = to_float(vals[i]);
    }
}

template <typename T, int Threads, bool Paged = false>
__global__ __launch_bounds__(Threads) void quant_v_int8_kernel(
    const T* __restrict__ v, int8_t* __restrict__ out, float* __restrict__ scale_out, int N,
    int padded_N, int H, int D, int64_t stride_b, int64_t stride_h, int64_t stride_n,
    int pages = 0, const int64_t* position = nullptr, bool initialize = false) {

    constexpr int kWaves = Threads / 32;

    const int d_tiles = D / kDTile;
    const int d_tile = blockIdx.x % d_tiles;
    const int bh = blockIdx.x / d_tiles;
    const int h = bh % H;
    const int b = bh / H;
    const int d0 = d_tile * kDTile;

    if constexpr (Paged) {
        const int length = static_cast<int>(position[0]);
        const int page = initialize ? blockIdx.y : max(0, (length - 1) / padded_N - 1) + blockIdx.y;
        if (page >= pages || page * padded_N >= length) return;
        N = min(padded_N, length - page * padded_N);
        const int source_row = initialize ? page * padded_N : page * padded_N % (stride_h / stride_n);
        v += static_cast<int64_t>(source_row) * stride_n;
        const int64_t offset = (static_cast<int64_t>(b) * (pages - 1) + page) * H * D;
        out += offset * padded_N;
        scale_out += offset;
    }
    const T* base = v + static_cast<int64_t>(b) * stride_b + static_cast<int64_t>(h) * stride_h +
                    d0;

    float mx[kDTile];
#pragma unroll
    for (int i = 0; i < kDTile; ++i) mx[i] = 0.0f;

    // Four independent loads per iteration so the memory system has something to
    // overlap; this block is the only one touching these channels.
    int n = threadIdx.x;
    const int n_body = N - 3 * Threads;
    for (; n < n_body; n += 4 * Threads) {
        float t0[kDTile], t1[kDTile], t2[kDTile], t3[kDTile];
        load8<T>(base + static_cast<int64_t>(n) * stride_n, t0);
        load8<T>(base + static_cast<int64_t>(n + Threads) * stride_n, t1);
        load8<T>(base + static_cast<int64_t>(n + 2 * Threads) * stride_n, t2);
        load8<T>(base + static_cast<int64_t>(n + 3 * Threads) * stride_n, t3);
#pragma unroll
        for (int i = 0; i < kDTile; ++i) {
            mx[i] = fmaxf(mx[i], fmaxf(fmaxf(fabsf(t0[i]), fabsf(t1[i])),
                                       fmaxf(fabsf(t2[i]), fabsf(t3[i]))));
        }
    }
    for (; n < N; n += Threads) {
        float t[kDTile];
        load8<T>(base + static_cast<int64_t>(n) * stride_n, t);
#pragma unroll
        for (int i = 0; i < kDTile; ++i) mx[i] = fmaxf(mx[i], fabsf(t[i]));
    }

    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;

#pragma unroll
    for (int i = 0; i < kDTile; ++i) mx[i] = row_reduce_fmax<32>(mx[i]);

    __shared__ float wave_max[kDTile][kWaves];
    __shared__ float inv_scale_shared[kDTile];
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < kDTile; ++i) wave_max[i][wave] = mx[i];
    }
    __syncthreads();

    if (threadIdx.x < kDTile) {
        float val = 0.0f;
#pragma unroll
        for (int w = 0; w < kWaves; ++w) val = fmaxf(val, wave_max[threadIdx.x][w]);
        const float sc = fmaxf(val * (1.0f / kInt8Max), 1e-12f);
        scale_out[(static_cast<int64_t>(b) * H + h) * D + d0 + threadIdx.x] = sc;
        inv_scale_shared[threadIdx.x] = 1.0f / sc;
    }
    __syncthreads();

    float inv_scale[kDTile];
#pragma unroll
    for (int i = 0; i < kDTile; ++i) inv_scale[i] = inv_scale_shared[i];

    const int64_t out_row = (static_cast<int64_t>(b) * H + h) * D + d0;

    // Descending so the tail of pass 1 is still in L2 when pass 2 starts.
    for (int src = N - 1 - static_cast<int>(threadIdx.x); src >= 0; src -= Threads) {
        float t[kDTile];
        load8<T>(base + static_cast<int64_t>(src) * stride_n, t);
#pragma unroll
        for (int i = 0; i < kDTile; ++i) {
            out[(out_row + i) * padded_N + src] = float_to_int8_rn(t[i] * inv_scale[i]);
        }
    }

    // The attention kernel reads whole CTA_K tiles, so the padding has to be a
    // real zero rather than whatever the allocator left behind.
    for (int src = N + static_cast<int>(threadIdx.x); src < padded_N; src += Threads) {
#pragma unroll
        for (int i = 0; i < kDTile; ++i) out[(out_row + i) * padded_N + src] = 0;
    }
}

template <typename T>
void launch_typed(const void* v, void* out, void* scale, int B, int H, int N, int D, int padded_N,
                  int64_t stride_b, int64_t stride_h, int64_t stride_n, hipStream_t stream) {
    const int blocks = B * H * (D / kDTile);
    if (N <= 256) {
        quant_v_int8_kernel<T, 128><<<blocks, 128, 0, stream>>>(
            static_cast<const T*>(v), static_cast<int8_t*>(out), static_cast<float*>(scale), N,
            padded_N, H, D, stride_b, stride_h, stride_n);
    } else {
        quant_v_int8_kernel<T, 512><<<blocks, 512, 0, stream>>>(
            static_cast<const T*>(v), static_cast<int8_t*>(out), static_cast<float*>(scale), N,
            padded_N, H, D, stride_b, stride_h, stride_n);
    }
}

}  // namespace
}  // namespace comfy::hip_backend::sage

#include "launchers.h"

namespace jasna_ltx::sage_attention_hip {

using namespace comfy::hip_backend::sage;

void launch_quant_v_int8(const void* v, void* out, void* scale, int B, int H,
                                         int N, int D, int padded_N, int64_t stride_b,
                                         int64_t stride_h, int64_t stride_n, int input_dtype_code,
                                         hipStream_t stream) {
    // The kernel loads V 16 bytes at a time, so the base pointer and every row
    // offset it can form have to be 16-byte aligned. An extent of one forms no
    // offset, and PyTorch rewrites such a stride to 1 on the way through DLPack
    // (ATen/DLConvertor.cpp, gh-83069), so policing it would reject a tensor the
    // caller has no way to hand over differently.
    const size_t element_size = input_dtype_code == 0 ? sizeof(float) : sizeof(__half);
    const auto stride_ok = [element_size](int64_t stride, int extent) {
        return extent < 2 || (static_cast<size_t>(stride) * element_size) % 16 == 0;
    };
    if (reinterpret_cast<uintptr_t>(v) % 16 != 0 || !stride_ok(stride_b, B) ||
        !stride_ok(stride_h, H) || !stride_ok(stride_n, N)) {
        throw std::runtime_error(
            "sage int8 attention: V base pointer and B/H/N strides must be 16-byte aligned");
    }
    if (D <= 0 || D % kDTile != 0) {
        throw std::runtime_error(
            "sage int8 attention: head_dim must be a positive multiple of 8");
    }
    // Rows are padded_N wide and the quantize pass writes columns up to N, so a
    // short pad would run every row past its end.
    if (N < 0 || padded_N < N) {
        throw std::runtime_error(
            "sage int8 attention: padded key length must be at least the key length");
    }

    if (input_dtype_code != kBFloat16) {
        throw std::runtime_error("sage int8 attention: input dtype must be bf16");
    }
    launch_typed<__bf16>(v, out, scale, B, H, N, D, padded_N, stride_b, stride_h, stride_n, stream);

    const hipError_t error = hipGetLastError();
    if (error != hipSuccess) {
        throw std::runtime_error(std::string("sage int8 V quantize kernel launch failed: ") +
                                 hipGetErrorString(error));
    }
}

}  // namespace jasna_ltx::sage_attention_hip
