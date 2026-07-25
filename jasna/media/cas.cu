#include <stdint.h>

// Contrast Adaptive Sharpening of a luma plane, one fused stencil pass.
//
// Implements AMD's FidelityFX CAS (MIT):
// https://github.com/GPUOpen-Effects/FidelityFX-CAS
//
// The weight curve and the headroom constant match FFmpeg's `cas` filter so a
// given strength reproduces `cas=strength=<s>:planes=1`; see
// scripts/cas_ffmpeg_oracle.py, which re-derives both and checks parity.

namespace {

template <typename Sample, int Shift, int Peak, int Headroom>
__device__ __forceinline__ void sharpen_pixel(
    const Sample* src,
    Sample* dst,
    int src_stride,
    int dst_stride,
    int height,
    int width,
    float weight_scale
) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= width || row >= height) {
        return;
    }

    // Edge pixels repeat the border sample, matching the reference filter.
    const int left = col > 0 ? col - 1 : 0;
    const int right = col + 1 < width ? col + 1 : width - 1;
    const int above = (row > 0 ? row - 1 : 0) * src_stride;
    const int middle = row * src_stride;
    const int below = (row + 1 < height ? row + 1 : height - 1) * src_stride;

    const float a = static_cast<float>(src[above + left] >> Shift);
    const float b = static_cast<float>(src[above + col] >> Shift);
    const float c = static_cast<float>(src[above + right] >> Shift);
    const float d = static_cast<float>(src[middle + left] >> Shift);
    const float e = static_cast<float>(src[middle + col] >> Shift);
    const float f = static_cast<float>(src[middle + right] >> Shift);
    const float g = static_cast<float>(src[below + left] >> Shift);
    const float h = static_cast<float>(src[below + col] >> Shift);
    const float i = static_cast<float>(src[below + right] >> Shift);

    // "Better diagonals": fold the cross-only extremes together with the full
    // 3x3 extremes, which doubles the scale that Headroom is expressed in.
    const float min_cross = fminf(fminf(fminf(b, d), fminf(f, h)), e);
    const float max_cross = fmaxf(fmaxf(fmaxf(b, d), fmaxf(f, h)), e);
    const float min_box = fminf(min_cross, fminf(fminf(a, c), fminf(g, i)));
    const float max_box = fmaxf(max_cross, fmaxf(fmaxf(a, c), fmaxf(g, i)));
    const float minima = min_cross + min_box;
    const float maxima = fmaxf(max_cross + max_box, 1.0f);

    const float amp = sqrtf(__saturatef(fminf(minima, Headroom - maxima) / maxima));
    const float weight = amp * weight_scale;
    const float denominator = 1.0f + 4.0f * weight;
    const float delta = (b + d + f + h) - 4.0f * e;
    // Full strength on a flat neighbourhood zeroes the denominator, but delta
    // is zero there too, so the centre sample is the limit.
    float value = denominator > 0.0f ? e + delta * weight / denominator : e;
    value = fminf(fmaxf(value, 0.0f), static_cast<float>(Peak));

    // The reference filter truncates rather than rounds on write.
    dst[row * dst_stride + col] = static_cast<Sample>(
        static_cast<int>(floorf(value)) << Shift);
}

}  // namespace

extern "C" __global__ void cas_luma8(
    const uint8_t* src,
    uint8_t* dst,
    int src_stride,
    int dst_stride,
    int height,
    int width,
    float weight_scale
) {
    sharpen_pixel<uint8_t, 0, 255, 512>(
        src, dst, src_stride, dst_stride, height, width, weight_scale);
}

extern "C" __global__ void cas_luma10(
    const uint16_t* src,
    uint16_t* dst,
    int src_stride,
    int dst_stride,
    int height,
    int width,
    float weight_scale
) {
    sharpen_pixel<uint16_t, 6, 1023, 2048>(
        src, dst, src_stride, dst_stride, height, width, weight_scale);
}
