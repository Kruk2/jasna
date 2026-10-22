// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Shared numerics for the RDNA int8 attention port. Everything here has a
// counterpart in the CUDA backend's sage_attention/ sources; where a constant is
// duplicated rather than derived, the two must agree or the backends produce
// different numbers for the same input.
#pragma once

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "mma.h"
#include "architecture_config.h"

namespace comfy::hip_backend::sage {

// Every kernel here indexes lanes with & 31, reduces with a width of 32 and packs
// fragments per half wave. RDNA defaults to wave32, but -mwavefrontsize64 would
// compile all of that into silently wrong scales rather than an error.
#if defined(__AMDGCN_WAVEFRONT_SIZE__)
static_assert(__AMDGCN_WAVEFRONT_SIZE__ == 32,
              "the int8 attention kernels are wave32 only");
#endif

// Softmax probabilities are unsigned int8, so the online maximum is shifted far
// enough that exp2 fills the whole range: exp2(7.9943534) rounds to 255. Mirrors
// S_U8_OFFSET in the CUDA attn_utils.cuh.
constexpr float kProbU8Offset = 7.9943534f;
constexpr float kLog2e = 1.44269504088896340736f;

// Finite, not -inf: masked scores go through fma and exp2 before anything tests
// them, and -ffast-math is on. See the fast-math note in the HIP backend README.
constexpr float kMaskedScore = -50000.0f;

constexpr float kInt8Max = 127.0f;

template <typename T>
__forceinline__ __device__ float to_float(T v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, __half>) {
        return __half2float(v);
    } else {
        return static_cast<float>(v);
    }
}

// Four adjacent channels in one instruction: 16 bytes for fp32, 8 for the 16-bit
// types. The caller guarantees the group is in bounds.
template <typename T>
__forceinline__ __device__ void load4(const T* p, float* out) {
    if constexpr (std::is_same_v<T, float>) {
        const float4 raw = *reinterpret_cast<const float4*>(p);
        out[0] = raw.x;
        out[1] = raw.y;
        out[2] = raw.z;
        out[3] = raw.w;
    } else {
        const uint2 raw = *reinterpret_cast<const uint2*>(p);
        const T* vals = reinterpret_cast<const T*>(&raw);
#pragma unroll
        for (int i = 0; i < 4; ++i) out[i] = to_float(vals[i]);
    }
}

// Round half to even and saturate, matching cvt.rni.sat.s8.f32, which is what
// the CUDA backend's float_to_int8_rn compiles to.
__forceinline__ __device__ int8_t float_to_int8_rn(float v) {
    const float r = rintf(v);
    return static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, r)));
}

// The caller hoists the reciprocal out of the per-channel loop, so the whole
// row divides once. Mirrors quant_int8_rcp in the CUDA float_utils.cuh.
__forceinline__ __device__ int8_t quant_int8_rcp(float v, float inv_scale) {
    return float_to_int8_rn(v * inv_scale);
}

// p must be 4-byte aligned. Every row offset the quantizers form is a multiple
// of four, so the requirement reduces to the base pointer, which the launcher
// checks. Mirrors the note on store4_i8 in the CUDA float_utils.cuh.
__forceinline__ __device__ void store4_i8(int8_t* p, int8_t a, int8_t b, int8_t c, int8_t d) {
    *reinterpret_cast<int32_t*>(p) =
        static_cast<uint32_t>(static_cast<uint8_t>(a)) |
        (static_cast<uint32_t>(static_cast<uint8_t>(b)) << 8) |
        (static_cast<uint32_t>(static_cast<uint8_t>(c)) << 16) |
        (static_cast<uint32_t>(static_cast<uint8_t>(d)) << 24);
}

// XOR-16 exchange between the two halves of a wave.
//
// __shfl_xor lowers to a ds_bpermute plus the lane arithmetic and exec-mask guard
// around it, and ds_bpermute runs on the LDS crossbar, so each exchange also
// plants an s_wait_dscnt that orders against the K and V tile loads in flight.
// v_permlanex16_b32 is one VALU instruction and touches no LDS state. The
// selectors are the identity pick inside each half, which reduces it to a plain
// half swap; verified lane by lane against __shfl_xor.
//
// gfx11 only. The attention kernel needs 228 of these there against gfx12's 52,
// because gfx11 also assembles the P fragment through this exchange on every key
// tile. On gfx12 the substitution measured as a wash, so that path keeps the
// portable form rather than carry an unmeasurable change.
//
// Callers must be wave-uniform. __shfl_xor carries an exec-mask guard and this
// does not, so a partner lane that is inactive reads back this lane's own value
// instead. Every call site in the attention kernel is outside any lane-dependent
// branch, which is what makes the two interchangeable.
#if defined(COMFY_MMA_GFX11)
__forceinline__ __device__ uint32_t swap_half_wave_b32(uint32_t v) {
    return static_cast<uint32_t>(__builtin_amdgcn_permlanex16(
        static_cast<int>(v), static_cast<int>(v), 0x76543210, 0xfedcba98, false, false));
}
#else
__forceinline__ __device__ uint32_t swap_half_wave_b32(uint32_t v) {
    return static_cast<uint32_t>(__shfl_xor(static_cast<int>(v), 16, 32));
}
#endif

__forceinline__ __device__ float swap_half_wave(float v) {
    return __uint_as_float(swap_half_wave_b32(__float_as_uint(v)));
}

// Width is a template parameter because the D64 quantizers put two rows in one
// wave and must not fold the other row's lanes in.

template <int Width>
__forceinline__ __device__ float row_reduce_fmax(float v) {
#pragma unroll
    for (int off = Width / 2; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor(v, off, Width));
    }
    return v;
}

// Fused block-Hadamard rotation (convrot), mirroring convrot4/64/128 in the CUDA
// quantizer. Orthogonal, so Q.K is unchanged; it only moves quantization outliers
// off single channels. Q and K must use the same block size or the scores are
// wrong. Each lane owns four adjacent channels of a 128-channel tile, so H4 is
// local, H64 crosses a half-wave and H128 the whole wave.

__forceinline__ __device__ void convrot4(float* v) {
    const float x0 = v[0], x1 = v[1], x2 = v[2], x3 = v[3];
    const float a0 = x0 + x1;
    const float a1 = x0 - x1;
    const float a2 = x2 + x3;
    const float a3 = x2 - x3;
    v[0] = (a0 + a2) * 0.5f;
    v[1] = (a1 + a3) * 0.5f;
    v[2] = (a0 - a2) * 0.5f;
    v[3] = (a1 - a3) * 0.5f;
}

__forceinline__ __device__ void convrot64(float* v) {
    convrot4(v);
    const int half_lane = threadIdx.x & 15;
#pragma unroll
    for (int bit = 1; bit < 16; bit <<= 1) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float other = __shfl_xor(v[c], bit, 16);
            v[c] = (half_lane & bit) ? other - v[c] : v[c] + other;
        }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) v[c] *= 0.25f;
}

__forceinline__ __device__ void convrot128_plain(float* v) {
    convrot4(v);
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int bit = 1; bit < 32; bit <<= 1) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float other = __shfl_xor(v[c], bit, 32);
            v[c] = (lane & bit) ? other - v[c] : v[c] + other;
        }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) v[c] *= 0.1767766952966369f;
}

// A fixed per-channel sign flip ahead of H128. It is a diagonal +-1 matrix, so
// applying it to Q and K alike leaves every dot product untouched; the literals
// must stay bit-identical to apply_convrot_sign128 in the CUDA quantizer or the
// two backends disagree. A set bit means keep the sign.
__forceinline__ __device__ void apply_convrot_sign128(float* v, int lane) {
    constexpr uint32_t signs_0 = 0x1035997bu;
    constexpr uint32_t signs_1 = 0x8087f5eeu;
    constexpr uint32_t signs_2 = 0xee2e4e1au;
    constexpr uint32_t signs_3 = 0x71132418u;
    const uint32_t signs = lane < 8 ? signs_0 : lane < 16 ? signs_1 : lane < 24 ? signs_2 : signs_3;
    const int shift = (lane & 7) * 4;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        const uint32_t flip = ((signs >> (shift + c)) & 1u) ^ 1u;
        v[c] = __uint_as_float(__float_as_uint(v[c]) ^ (flip << 31));
    }
}

__forceinline__ __device__ void convrot128(float* v) {
    apply_convrot_sign128(v, threadIdx.x & 31);
    convrot128_plain(v);
}

// 128 is the signed H128 the D128 path uses; 129 is the plain one padded D256
// keeps. Both span the whole wave, so a lane's channel group must be its own.
template <int Rotation>
__forceinline__ __device__ void convrot(float* v) {
    // No else below, so an unrecognized tag would leave the tile unrotated and
    // silently change the scores rather than fail. rotate_row guards the call on
    // Rotation != 0, so 0 never instantiates this.
    static_assert(Rotation == 4 || Rotation == 64 || Rotation == 128 || Rotation == 129,
                  "convrot supports rotation tags 4, 64, 128 and 129 only");
    if constexpr (Rotation == 128) {
        convrot128(v);
    } else if constexpr (Rotation == 129) {
        convrot128_plain(v);
    } else if constexpr (Rotation == 64) {
        convrot64(v);
    } else if constexpr (Rotation == 4) {
        convrot4(v);
    }
}

// ---------------------------------------------------------------------------
// WMMA fragment plumbing shared by the int8 attention kernels
// ---------------------------------------------------------------------------

// Round to nearest even and saturate, matching cvt.rni.sat.u8.f32 in the CUDA
// backend's pack_u8x4.
__forceinline__ __device__ uint32_t prob_to_u8(float p) {
    return static_cast<uint32_t>(fminf(255.0f, fmaxf(0.0f, rintf(p))));
}

// The eight probabilities a lane holds, packed into the P operand of the PV
// matmul. gfx12 element e is key 8 * (lane / 16) + e, which is already the
// fragment's K slice. gfx11 element e is key 2e + lane / 16, so a lane holds
// every other key and trades with its partner to rebuild the whole 16-key step.
__forceinline__ __device__ MmaInt8::Frag pack_prob_frag(const uint32_t p[8], int lane) {
    const uint32_t lo = p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
    const uint32_t hi = p[4] | (p[5] << 8) | (p[6] << 16) | (p[7] << 24);
#if !defined(COMFY_MMA_GFX11)
    // gfx12, and the no-matrix-core stub, whose Frag is this narrow too.
    (void)lane;
    MmaInt8::Frag f;
    f[0] = static_cast<int>(lo);
    f[1] = static_cast<int>(hi);
    return f;
#else
    const uint32_t partner_lo = swap_half_wave_b32(lo);
    const uint32_t partner_hi = swap_half_wave_b32(hi);
    const bool even_half = lane < 16;
    const uint32_t even0 = even_half ? lo : partner_lo;  // keys 0, 2, 4, 6
    const uint32_t odd0 = even_half ? partner_lo : lo;   // keys 1, 3, 5, 7
    const uint32_t even1 = even_half ? hi : partner_hi;  // keys 8, 10, 12, 14
    const uint32_t odd1 = even_half ? partner_hi : hi;   // keys 9, 11, 13, 15
    // v_perm_b32 selector bytes index {src0, src1} with src1 in bytes 0..3.
    MmaInt8::Frag f;
    f[0] = static_cast<int>(__builtin_amdgcn_perm(odd0, even0, 0x05010400u));
    f[1] = static_cast<int>(__builtin_amdgcn_perm(odd0, even0, 0x07030602u));
    f[2] = static_cast<int>(__builtin_amdgcn_perm(odd1, even1, 0x05010400u));
    f[3] = static_cast<int>(__builtin_amdgcn_perm(odd1, even1, 0x07030602u));
    return f;
#endif
}

// The same repack for a BF16 P operand, which the Sol-Attn routing kernel needs
// for its pooled PV. gfx12's K slice is the accumulator order again; gfx11 rebuilds
// the 16-key step from the two half-waves, exchanging four packed dwords instead of
// eight floats.
__forceinline__ __device__ MmaBf16::Frag pack_prob_frag_bf16(const float p[8], int lane) {
    MmaBf16::Frag f;
#if !defined(COMFY_MMA_GFX11)
    (void)lane;
#pragma unroll
    for (int e = 0; e < 8; ++e) f[e] = static_cast<__bf16>(p[e]);
#else
    union {
        uint32_t w[4];
        __bf16 e[8];
    } own, partner;
#pragma unroll
    for (int e = 0; e < 8; ++e) own.e[e] = static_cast<__bf16>(p[e]);
#pragma unroll
    for (int i = 0; i < 4; ++i) partner.w[i] = swap_half_wave_b32(own.w[i]);
    const bool even_half = lane < 16;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        f[2 * e] = even_half ? own.e[e] : partner.e[e];
        f[2 * e + 1] = even_half ? partner.e[e] : own.e[e];
    }
#endif
    return f;
}

// vals[e] belongs to channel d_base + acc_row(lane, e). gfx12 makes those eight
// channels contiguous, so the row goes out in one 16-byte store; gfx11
// interleaves them with the partner lane's and needs eight.
template <typename OutT>
__forceinline__ __device__ void store_o_tile(OutT* __restrict__ row, int d_base,
                                             const float* vals, int lane) {
#if defined(COMFY_MMA_GFX12)
    // Aligned because the 16-byte store below reinterprets it.
    __attribute__((aligned(16))) OutT packed[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) packed[e] = static_cast<OutT>(vals[e]);
    *reinterpret_cast<uint4*>(row + d_base + 8 * (lane / 16)) =
        *reinterpret_cast<const uint4*>(packed);
#else
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        row[d_base + acc_row(lane, e)] = static_cast<OutT>(vals[e]);
    }
#endif
}

// The read side of store_o_tile, for a handover the next kernel resumes from.
template <typename InT>
__forceinline__ __device__ void load_o_tile(const InT* __restrict__ row, int d_base, float* vals,
                                            int lane) {
#if defined(COMFY_MMA_GFX12)
    __attribute__((aligned(16))) InT packed[8];
    *reinterpret_cast<uint4*>(packed) =
        *reinterpret_cast<const uint4*>(row + d_base + 8 * (lane / 16));
#pragma unroll
    for (int e = 0; e < 8; ++e) vals[e] = static_cast<float>(packed[e]);
#else
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        vals[e] = static_cast<float>(row[d_base + acc_row(lane, e)]);
    }
#endif
}

}  // namespace comfy::hip_backend::sage
