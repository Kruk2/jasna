#include <stdint.h>

// Inverse of yuv_to_rgb.cu: planar RGB -> packed NV12 / P010, 4:2:0.
// One thread per 2x2 chroma quad, so the four luma samples that feed one
// chroma pair are computed once and averaged in registers.
//
// The coefficient literals are the exact float32 values Torch holds in
// rgb_to_nv12.py / rgb_to_p010.py, including the ones that only look wrong
// (127.49999237060547 is 224*0.5 rescaled by 255/224 in float32, not 127.5),
// so the only parity gap against the Torch reference is float reduction order.
// scripts/rgb_to_yuv_ffmpeg_oracle.py checks both against swscale.

namespace {

__device__ __forceinline__ float clamp_code(float value, float lo, float hi) {
    return fminf(fmaxf(value, lo), hi);
}

template <typename Sample, int Shift>
__device__ __forceinline__ void convert_quad(
    const uint8_t* rgb,
    int64_t channel_stride,
    int64_t row_stride,
    Sample* luma,
    int64_t luma_row_stride,
    Sample* chroma,
    int64_t chroma_row_stride,
    int height,
    int width,
    float yr, float yg, float yb,
    float ur, float ug, float ub,
    float vr, float vg, float vb,
    float y_offset,
    float c_offset,
    float y_min, float y_max,
    float c_min, float c_max
) {
    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    if (x0 >= width || y0 >= height) {
        return;
    }

    float u_sum = 0.0f;
    float v_sum = 0.0f;

#pragma unroll
    for (int dy = 0; dy < 2; ++dy) {
#pragma unroll
        for (int dx = 0; dx < 2; ++dx) {
            const int x = x0 + dx;
            const int y = y0 + dy;
            const uint8_t* pixel = rgb + static_cast<int64_t>(y) * row_stride + x;
            const float r = static_cast<float>(pixel[0]) / 255.0f;
            const float g = static_cast<float>(pixel[channel_stride]) / 255.0f;
            const float b = static_cast<float>(pixel[2 * channel_stride]) / 255.0f;

            u_sum += ur * r + ug * g + ub * b;
            v_sum += vr * r + vg * g + vb * b;

            const float code = clamp_code(
                rintf(yr * r + yg * g + yb * b + y_offset), y_min, y_max);
            luma[static_cast<int64_t>(y) * luma_row_stride + x] =
                static_cast<Sample>(static_cast<int>(code) << Shift);
        }
    }

    const float u = clamp_code(rintf(u_sum * 0.25f + c_offset), c_min, c_max);
    const float v = clamp_code(rintf(v_sum * 0.25f + c_offset), c_min, c_max);
    Sample* pair = chroma + static_cast<int64_t>(y0 >> 1) * chroma_row_stride + x0;
    pair[0] = static_cast<Sample>(static_cast<int>(u) << Shift);
    pair[1] = static_cast<Sample>(static_cast<int>(v) << Shift);
}

}  // namespace

#define RGB_KERNEL_ARGS(Sample) \
    const uint8_t* rgb, int64_t channel_stride, int64_t row_stride, \
    Sample* luma, int64_t luma_row_stride, \
    Sample* chroma, int64_t chroma_row_stride, \
    int height, int width

#define RGB_KERNEL_INPUTS \
    rgb, channel_stride, row_stride, luma, luma_row_stride, \
    chroma, chroma_row_stride, height, width

#define DEFINE_NV12_KERNEL(name, yr, yg, yb, ur, ug, ub, vr, vg, vb, \
                           y_off, c_off, y_min, y_max, c_min, c_max) \
extern "C" __global__ void name(RGB_KERNEL_ARGS(uint8_t)) { \
    convert_quad<uint8_t, 0>(RGB_KERNEL_INPUTS, \
        yr, yg, yb, ur, ug, ub, vr, vg, vb, \
        y_off, c_off, y_min, y_max, c_min, c_max); \
}

#define DEFINE_P010_KERNEL(name, yr, yg, yb, ur, ug, ub, vr, vg, vb, \
                           y_off, c_off, y_min, y_max, c_min, c_max) \
extern "C" __global__ void name(RGB_KERNEL_ARGS(uint16_t)) { \
    convert_quad<uint16_t, 6>(RGB_KERNEL_INPUTS, \
        yr, yg, yb, ur, ug, ub, vr, vg, vb, \
        y_off, c_off, y_min, y_max, c_min, c_max); \
}

DEFINE_NV12_KERNEL(nv12_bt601_limited,
    65.48100280761719f, 128.55299377441406f, 24.965999603271484f,
    -37.7968635559082f, -74.20313262939453f, 112.0f,
    112.0f, -93.7861099243164f, -18.21388816833496f,
    16.0f, 128.0f, 16.0f, 235.0f, 16.0f, 240.0f)
DEFINE_NV12_KERNEL(nv12_bt601_full,
    76.24500274658203f, 149.6849822998047f, 29.06999969482422f,
    -43.027679443359375f, -84.4723129272461f, 127.49999237060547f,
    127.49999237060547f, -106.76543426513672f, -20.734560012817383f,
    0.0f, 128.0f, 0.0f, 255.0f, 0.0f, 255.0f)
DEFINE_NV12_KERNEL(nv12_bt709_limited,
    46.55939865112305f, 156.62879943847656f, 15.811800003051758f,
    -25.664127349853516f, -86.33586883544922f, 112.0f,
    112.0f, -101.73027038574219f, -10.26972770690918f,
    16.0f, 128.0f, 16.0f, 235.0f, 16.0f, 240.0f)
DEFINE_NV12_KERNEL(nv12_bt709_full,
    54.21299743652344f, 182.37599182128906f, 18.410999298095703f,
    -29.215858459472656f, -98.28413391113281f, 127.49999237060547f,
    127.49999237060547f, -115.80900573730469f, -11.690983772277832f,
    0.0f, 128.0f, 0.0f, 255.0f, 0.0f, 255.0f)
DEFINE_NV12_KERNEL(nv12_bt2020_limited,
    57.53129959106445f, 148.48199462890625f, 12.986700057983398f,
    -31.27712059020996f, -80.7228775024414f, 112.0f,
    112.0f, -102.9920654296875f, -9.007935523986816f,
    16.0f, 128.0f, 16.0f, 235.0f, 16.0f, 240.0f)
DEFINE_NV12_KERNEL(nv12_bt2020_full,
    66.98849487304688f, 172.88998413085938f, 15.121500015258789f,
    -35.605648040771484f, -91.89434051513672f, 127.49999237060547f,
    127.49999237060547f, -117.24542999267578f, -10.254569053649902f,
    0.0f, 128.0f, 0.0f, 255.0f, 0.0f, 255.0f)

DEFINE_P010_KERNEL(p010_bt601_limited,
    261.92401123046875f, 514.2119750976562f, 99.86399841308594f,
    -151.1874542236328f, -296.8125305175781f, 448.0f,
    448.0f, -375.1444396972656f, -72.85555267333984f,
    64.0f, 512.0f, 64.0f, 940.0f, 64.0f, 960.0f)
DEFINE_P010_KERNEL(p010_bt601_full,
    305.87701416015625f, 600.5009765625f, 116.62199401855469f,
    -172.61692810058594f, -338.883056640625f, 511.5f,
    511.5f, -428.31781005859375f, -83.18217468261719f,
    0.0f, 512.0f, 0.0f, 1023.0f, 0.0f, 1023.0f)
DEFINE_P010_KERNEL(p010_bt709_limited,
    186.2375946044922f, 626.5151977539062f, 63.24720001220703f,
    -102.65650939941406f, -345.3434753417969f, 448.0f,
    448.0f, -406.92108154296875f, -41.07891082763672f,
    64.0f, 512.0f, 64.0f, 940.0f, 64.0f, 960.0f)
DEFINE_P010_KERNEL(p010_bt709_full,
    217.4897918701172f, 731.6495971679688f, 73.860595703125f,
    -117.2071533203125f, -394.2928161621094f, 511.5f,
    511.5f, -464.5985107421875f, -46.9014778137207f,
    0.0f, 512.0f, 0.0f, 1023.0f, 0.0f, 1023.0f)
DEFINE_P010_KERNEL(p010_bt2020_limited,
    230.1251983642578f, 593.927978515625f, 51.946800231933594f,
    -125.10848236083984f, -322.8915100097656f, 448.0f,
    448.0f, -411.96826171875f, -36.031742095947266f,
    64.0f, 512.0f, 64.0f, 940.0f, 64.0f, 960.0f)
DEFINE_P010_KERNEL(p010_bt2020_full,
    268.7420959472656f, 693.5939331054688f, 60.66389846801758f,
    -142.84149169921875f, -368.6584777832031f, 511.5f,
    511.5f, -470.361083984375f, -41.138919830322266f,
    0.0f, 512.0f, 0.0f, 1023.0f, 0.0f, 1023.0f)
