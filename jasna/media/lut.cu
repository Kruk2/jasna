#include <stdint.h>

// Fused .cube LUT application: uint8 planar RGB in, uint8 planar RGB out.
//
// The Torch path costs about eight full-frame passes (uint8 -> fp32, domain
// normalise, clamp, build a (1,1,H,W,3) grid, 5-D grid_sample, rescale, round,
// cast). At 8K VR that is ~9 ms per encoded frame. Here each thread reads three
// bytes and writes three, so the frame is touched once.
//
// The sampling arithmetic mirrors F.grid_sample(mode="bilinear",
// padding_mode="border", align_corners=True) on a (1, 3, N, N, N) volume whose
// axes are (channel, B, G, R): with align_corners the caller's coordinate
// 2*v-1 lands on index v*(N-1), and border padding is an index clamp.

namespace {

__device__ __forceinline__ float clamp01(float v) {
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

__device__ __forceinline__ int clamp_index(int i, int size) {
    return i < 0 ? 0 : (i >= size ? size - 1 : i);
}

__device__ __forceinline__ uint8_t to_u8(float v) {
    const int code = __float2int_rn(v * 255.0f);
    return static_cast<uint8_t>(code < 0 ? 0 : (code > 255 ? 255 : code));
}

struct Axis {
    int lo;
    int hi;
    float weight_lo;
    float weight_hi;
};

// grid_sample derives the two weights from the bracketing indices rather than
// from one fraction and its complement, so weight_lo is (base + 1 - coordinate)
// and not 1 - weight_hi. On a LUT with a steep slope the difference between
// those two spellings moves 3% of samples by one code.
__device__ __forceinline__ Axis axis_at(float value, int size) {
    // The Torch path normalises to [-1, 1] and grid_sample maps that back with
    // ((coord + 1) / 2) * (size - 1). That round trip is not the same in float
    // as value * (size - 1), and skipping it moves 3% of samples by one code on
    // a steep LUT, so reproduce it rather than simplify it.
    const float normalized = value * 2.0f - 1.0f;
    const float coordinate = ((normalized + 1.0f) / 2.0f) * (float)(size - 1);
    const float base = floorf(coordinate);
    Axis axis;
    axis.lo = clamp_index((int)base, size);
    axis.hi = clamp_index((int)base + 1, size);
    axis.weight_lo = base + 1.0f - coordinate;
    axis.weight_hi = coordinate - base;
    return axis;
}

}  // namespace

extern "C" __global__ void lut3d_u8(
    const uint8_t* __restrict__ rgb,
    int64_t channel_stride,
    int64_t row_stride,
    uint8_t* __restrict__ out,
    int64_t out_channel_stride,
    int64_t out_row_stride,
    int height,
    int width,
    const float* __restrict__ volume,
    int lut_size,
    const float* __restrict__ domain_min,
    const float* __restrict__ domain_scale
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const uint8_t* pixel = rgb + (int64_t)y * row_stride + x;
    const float r = clamp01(((float)pixel[0] / 255.0f - domain_min[0]) * domain_scale[0]);
    const float g = clamp01(((float)pixel[channel_stride] / 255.0f - domain_min[1]) * domain_scale[1]);
    const float b = clamp01(((float)pixel[2 * channel_stride] / 255.0f - domain_min[2]) * domain_scale[2]);

    const Axis ar = axis_at(r, lut_size);
    const Axis ag = axis_at(g, lut_size);
    const Axis ab = axis_at(b, lut_size);

    // The volume is packed (B, G, R, channel), so one corner's three channels
    // are adjacent. Packed the other way round — the layout grid_sample needs —
    // those three reads sit N^3 floats apart and the kernel runs 2x slower.
    const int64_t slice = (int64_t)lut_size * lut_size * 3;
    const int64_t row = (int64_t)lut_size * 3;

    // Corner order and the (x * y) * z product order both follow grid_sample's
    // 3D kernel, so the accumulation rounds the same way it does.
    float accumulated[3] = {0.0f, 0.0f, 0.0f};
#pragma unroll
    for (int cb = 0; cb < 2; ++cb) {
        const int ib = cb ? ab.hi : ab.lo;
        const float wb = cb ? ab.weight_hi : ab.weight_lo;
#pragma unroll
        for (int cg = 0; cg < 2; ++cg) {
            const int ig = cg ? ag.hi : ag.lo;
            const float wg = cg ? ag.weight_hi : ag.weight_lo;
#pragma unroll
            for (int cr = 0; cr < 2; ++cr) {
                const int ir = cr ? ar.hi : ar.lo;
                const float wr = cr ? ar.weight_hi : ar.weight_lo;
                const float weight = wr * wg * wb;
                const float* corner =
                    volume + (int64_t)ib * slice + (int64_t)ig * row + (int64_t)ir * 3;
                accumulated[0] += corner[0] * weight;
                accumulated[1] += corner[1] * weight;
                accumulated[2] += corner[2] * weight;
            }
        }
    }

    uint8_t* destination = out + (int64_t)y * out_row_stride + x;
#pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
        destination[(int64_t)channel * out_channel_stride] = to_u8(accumulated[channel]);
    }
}

extern "C" __global__ void lut1d_u8(
    const uint8_t* __restrict__ rgb,
    int64_t channel_stride,
    int64_t row_stride,
    uint8_t* __restrict__ out,
    int64_t out_channel_stride,
    int64_t out_row_stride,
    int height,
    int width,
    const float* __restrict__ table,
    int lut_size,
    const float* __restrict__ domain_min,
    const float* __restrict__ domain_scale
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    const uint8_t* pixel = rgb + (int64_t)y * row_stride + x;
    uint8_t* destination = out + (int64_t)y * out_row_stride + x;

#pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
        const float value = clamp01(
            ((float)pixel[(int64_t)channel * channel_stride] / 255.0f - domain_min[channel])
            * domain_scale[channel]);
        // The Torch path floors, clamps, then takes the next index, so a value
        // sitting exactly on the last entry keeps a zero fraction.
        const float scaled = value * (float)(lut_size - 1);
        const int lo = clamp_index((int)floorf(scaled), lut_size);
        const int hi = clamp_index(lo + 1, lut_size);
        const float fraction = clamp01(scaled - (float)lo);
        const float low = table[(int64_t)lo * 3 + channel];
        const float high = table[(int64_t)hi * 3 + channel];
        destination[(int64_t)channel * out_channel_stride] =
            to_u8(fmaf(fraction, high - low, low));
    }
}
