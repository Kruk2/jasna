#include <stdint.h>

namespace {

__device__ __constant__ uint8_t kBayer8[64] = {
    0, 48, 12, 60, 3, 51, 15, 63,
    32, 16, 44, 28, 35, 19, 47, 31,
    8, 56, 4, 52, 11, 59, 7, 55,
    40, 24, 36, 20, 43, 27, 39, 23,
    2, 50, 14, 62, 1, 49, 13, 61,
    34, 18, 46, 30, 33, 17, 45, 29,
    10, 58, 6, 54, 9, 57, 5, 53,
    42, 26, 38, 22, 41, 25, 37, 21,
};

template <typename Input>
__device__ __forceinline__ const Input* select_plane(
    int frame,
    const Input* p0,
    const Input* p1,
    const Input* p2,
    const Input* p3,
    const Input* p4,
    const Input* p5,
    const Input* p6,
    const Input* p7
) {
    switch (frame) {
        case 0: return p0;
        case 1: return p1;
        case 2: return p2;
        case 3: return p3;
        case 4: return p4;
        case 5: return p5;
        case 6: return p6;
        default: return p7;
    }
}

template <bool Dither10>
__device__ __forceinline__ uint8_t quantize(float value, int row, int col) {
    int quantized = __float2int_rn(value);
    if constexpr (Dither10) {
        quantized = quantized < 0 ? 0 : (quantized > 1023 ? 1023 : quantized);
        quantized = (quantized + (kBayer8[(row & 7) * 8 + (col & 7)] >> 4)) >> 2;
        quantized = quantized > 255 ? 255 : quantized;
    } else {
        quantized = quantized < 0 ? 0 : (quantized > 255 ? 255 : quantized);
    }
    return static_cast<uint8_t>(quantized);
}

template <typename Input, bool Dither10>
__device__ __forceinline__ void convert_pixel_batch(
    const Input* y0,
    const Input* y1,
    const Input* y2,
    const Input* y3,
    const Input* y4,
    const Input* y5,
    const Input* y6,
    const Input* y7,
    const Input* uv0,
    const Input* uv1,
    const Input* uv2,
    const Input* uv3,
    const Input* uv4,
    const Input* uv5,
    const Input* uv6,
    const Input* uv7,
    int y_stride,
    int uv_stride,
    uint8_t* out,
    int64_t out_batch_stride,
    int64_t out_channel_stride,
    int64_t out_row_stride,
    int batch_size,
    int height,
    int width,
    float luma_scale,
    float red_v,
    float green_u,
    float green_v,
    float blue_u,
    float red_offset,
    float green_offset,
    float blue_offset
) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t plane_size = static_cast<int64_t>(height) * width;
    if (index >= static_cast<int64_t>(batch_size) * plane_size) {
        return;
    }

    const int frame = static_cast<int>(index / plane_size);
    const int64_t pixel = index - static_cast<int64_t>(frame) * plane_size;
    const int row = static_cast<int>(pixel / width);
    const int col = static_cast<int>(pixel - static_cast<int64_t>(row) * width);
    const Input* y = select_plane(frame, y0, y1, y2, y3, y4, y5, y6, y7);
    const Input* uv = select_plane(frame, uv0, uv1, uv2, uv3, uv4, uv5, uv6, uv7);

    const float yy = static_cast<float>(y[static_cast<int64_t>(row) * y_stride + col]);
    const int64_t uv_index = static_cast<int64_t>(row >> 1) * uv_stride + (col >> 1) * 2;
    const float uu = static_cast<float>(uv[uv_index]);
    const float vv = static_cast<float>(uv[uv_index + 1]);
    const float scaled_y = luma_scale * yy;
    out += static_cast<int64_t>(frame) * out_batch_stride
        + static_cast<int64_t>(row) * out_row_stride + col;
    out[0] = quantize<Dither10>(scaled_y + red_v * vv + red_offset, row, col);
    out[out_channel_stride] = quantize<Dither10>(
        scaled_y + green_u * uu + green_v * vv + green_offset, row, col);
    out[2 * out_channel_stride] = quantize<Dither10>(
        scaled_y + blue_u * uu + blue_offset, row, col);
}

}  // namespace

#define YUV_KERNEL_ARGS(Input) \
    const Input* y0, const Input* y1, const Input* y2, const Input* y3, \
    const Input* y4, const Input* y5, const Input* y6, const Input* y7, \
    const Input* uv0, const Input* uv1, const Input* uv2, const Input* uv3, \
    const Input* uv4, const Input* uv5, const Input* uv6, const Input* uv7, \
    int y_stride, int uv_stride, uint8_t* out, int64_t out_batch_stride, \
    int64_t out_channel_stride, int64_t out_row_stride, int batch_size, int height, int width

#define YUV_KERNEL_INPUTS \
    y0, y1, y2, y3, y4, y5, y6, y7, uv0, uv1, uv2, uv3, uv4, uv5, uv6, uv7

#define DEFINE_YUV8_KERNEL(name, ls, rv, gu, gv, bu, ro, go, bo) \
extern "C" __global__ void name(YUV_KERNEL_ARGS(uint8_t)) { \
    convert_pixel_batch<uint8_t, false>( \
        YUV_KERNEL_INPUTS, y_stride, uv_stride, out, out_batch_stride, \
        out_channel_stride, out_row_stride, batch_size, height, width, \
        ls, rv, gu, gv, bu, ro, go, bo); \
}

#define DEFINE_YUV10_KERNEL(name, ls, rv, gu, gv, bu, ro, go, bo) \
extern "C" __global__ void name(YUV_KERNEL_ARGS(uint16_t)) { \
    convert_pixel_batch<uint16_t, true>( \
        YUV_KERNEL_INPUTS, y_stride, uv_stride, out, out_batch_stride, \
        out_channel_stride, out_row_stride, batch_size, height, width, \
        ls, rv, gu, gv, bu, ro, go, bo); \
}

DEFINE_YUV8_KERNEL(yuv8_bt601_limited, 1.1643835616438356f, 1.596026785714286f,
    -0.39176229009491365f, -0.81296764723777071f, 2.0172321428571429f,
    -222.92156555772999f, 135.57529499228224f, -276.83585127201565f)
DEFINE_YUV8_KERNEL(yuv8_bt601_full, 1.0f, 1.4020000000000001f,
    -0.34413628620102216f, -0.7141362862010221f, 1.772f,
    -179.45600000000002f, 135.45888926746167f, -226.816f)
DEFINE_YUV8_KERNEL(yuv8_bt709_limited, 1.1643835616438356f, 1.7927410714285714f,
    -0.21324861427372965f, -0.53290932855944406f, 2.1124017857142858f,
    -248.10099412915852f, 76.878079696344869f, -289.01756555772994f)
DEFINE_YUV8_KERNEL(yuv8_bt709_full, 1.0f, 1.5748f,
    -0.18732427293064879f, -0.46812427293064884f, 1.8555999999999999f,
    -201.5744f, 83.897413870246098f, -237.51679999999999f)
DEFINE_YUV8_KERNEL(yuv8_bt2020_limited, 1.1643835616438356f, 1.6786741071428575f,
    -0.18732610421934257f, -0.65042431850505689f, 2.1417723214285713f,
    -233.50042270058714f, 88.601917122421767f, -292.77699412915848f)
DEFINE_YUV8_KERNEL(yuv8_bt2020_full, 1.0f, 1.4746000000000001f,
    -0.16455312684365778f, -0.5713531268436578f, 1.8814f,
    -188.74880000000002f, 94.19600047197639f, -240.8192f)

DEFINE_YUV10_KERNEL(yuv10_bt601_limited, 0.018247003424657533f, 0.025011265345982144f,
    -0.0061392895644469458f, -0.012739980133643374f, 0.031611955915178569f,
    -894.30886888454017f, 543.89618343962627f, -1110.6002974559685f)
DEFINE_YUV10_KERNEL(yuv10_bt601_full, 0.015625f, 0.021906250000000002f,
    -0.0053771294718909712f, -0.01115837947189097f, 0.0276875f,
    -717.82400000000007f, 541.83555706984669f, -907.26400000000001f)
DEFINE_YUV10_KERNEL(yuv10_bt709_limited, 0.018247003424657533f, 0.028093966238839283f,
    -0.003341809626256517f, -0.0083511985771493741f, 0.033103355189732141f,
    -995.32281174168293f, 308.41676678180704f, -1159.47046888454f)
DEFINE_YUV10_KERNEL(yuv10_bt709_full, 0.015625f, 0.02460625f,
    -0.0029269417645413874f, -0.0073144417645413882f, 0.028993749999999999f,
    -806.29759999999999f, 335.58965548098439f, -950.06719999999996f)
DEFINE_YUV10_KERNEL(yuv10_bt2020_limited, 0.018247003424657533f, 0.026306427873883931f,
    -0.0029355791148343662f, -0.010192770800102224f, 0.033563619559151783f,
    -936.74875459882594f, 355.45004398524492f, -1174.5524117416828f)
DEFINE_YUV10_KERNEL(yuv10_bt2020_full, 0.015625f, 0.023040625000000002f,
    -0.0025711426069321528f, -0.0089273926069321531f, 0.029396874999999999f,
    -754.99520000000007f, 376.78400188790556f, -963.27679999999998f)
