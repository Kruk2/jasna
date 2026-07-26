#include <stdint.h>
#include <cuda_fp16.h>

// Fused detector preprocess: uint8 planar RGB -> bilinear resize -> optional
// letterbox padding -> /255 and per-channel mean/std, written as fp16 or fp32.
//
// The Torch path casts and divides at full source resolution and only then
// downscales, so at 8K VR it writes a 384 MiB fp16 tensor to produce a 5 MiB
// one. Here the frame is read once at its source resolution and only the small
// output is written.
//
// Everything is rounded the way the Torch path rounds it, including the
// intermediate quantisation to fp16 that x.to(half).div_(255) performs *before*
// interpolation, so detector inputs come out bit-identical rather than merely
// close. The sampler follows F.interpolate(mode="bilinear",
// align_corners=False): src = scale * (dst + 0.5) - 0.5, negatives clamped to
// zero, the upper tap clamped to the last row/column.

namespace {

struct Tap {
    int lo;
    int hi;
    float weight_lo;
    float weight_hi;
};

__device__ __forceinline__ Tap tap_at(int destination, float scale, int extent) {
    const float source = fmaf(scale, (float)destination + 0.5f, -0.5f);
    const float clamped = source < 0.0f ? 0.0f : source;
    const float base = floorf(clamped);
    Tap tap;
    tap.lo = (int)base;
    tap.hi = tap.lo + 1 < extent ? tap.lo + 1 : extent - 1;
    tap.weight_hi = clamped - base;
    tap.weight_lo = 1.0f - tap.weight_hi;
    return tap;
}

// Torch keeps half tensors in half between ops and computes each op in float,
// so every step is a float computation rounded back to half.
struct HalfSample {
    __device__ __forceinline__ static float quantize(float value) {
        return __half2float(__float2half_rn(value));
    }
    __device__ __forceinline__ static void store(void* out, int64_t index, float value) {
        static_cast<__half*>(out)[index] = __float2half_rn(value);
    }
};

struct FloatSample {
    __device__ __forceinline__ static float quantize(float value) { return value; }
    __device__ __forceinline__ static void store(void* out, int64_t index, float value) {
        static_cast<float*>(out)[index] = value;
    }
};

template <typename Sample>
__device__ __forceinline__ void resize_normalize(
    const uint8_t* __restrict__ src,
    int64_t src_batch_stride,
    int64_t src_channel_stride,
    int64_t src_row_stride,
    void* __restrict__ dst,
    int64_t dst_batch_stride,
    int64_t dst_channel_stride,
    int64_t dst_row_stride,
    int batch,
    int src_height,
    int src_width,
    int out_height,
    int out_width,
    int content_left,
    int content_top,
    int content_width,
    int content_height,
    const float* __restrict__ mean,
    const float* __restrict__ std,
    const float* __restrict__ fill
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int b = blockIdx.z;
    if (x >= out_width || y >= out_height || b >= batch) {
        return;
    }

    void* destination = dst;
    const int64_t base = (int64_t)b * dst_batch_stride + (int64_t)y * dst_row_stride + x;

    const int local_x = x - content_left;
    const int local_y = y - content_top;
    if (local_x < 0 || local_x >= content_width || local_y < 0 || local_y >= content_height) {
#pragma unroll
        for (int channel = 0; channel < 3; ++channel) {
            Sample::store(destination, base + (int64_t)channel * dst_channel_stride, fill[channel]);
        }
        return;
    }

    const Tap tx = tap_at(local_x, (float)src_width / (float)content_width, src_width);
    const Tap ty = tap_at(local_y, (float)src_height / (float)content_height, src_height);

    const uint8_t* plane = src + (int64_t)b * src_batch_stride;
#pragma unroll
    for (int channel = 0; channel < 3; ++channel) {
        const uint8_t* channel_plane = plane + (int64_t)channel * src_channel_stride;
        const uint8_t* row_lo = channel_plane + (int64_t)ty.lo * src_row_stride;
        const uint8_t* row_hi = channel_plane + (int64_t)ty.hi * src_row_stride;

        // The division by 255 happens before interpolation in the Torch path,
        // so for fp16 the taps are quantised here and not at the end.
        const float v00 = Sample::quantize((float)row_lo[tx.lo] / 255.0f);
        const float v01 = Sample::quantize((float)row_lo[tx.hi] / 255.0f);
        const float v10 = Sample::quantize((float)row_hi[tx.lo] / 255.0f);
        const float v11 = Sample::quantize((float)row_hi[tx.hi] / 255.0f);

        const float interpolated = Sample::quantize(
            ty.weight_lo * (tx.weight_lo * v00 + tx.weight_hi * v01)
            + ty.weight_hi * (tx.weight_lo * v10 + tx.weight_hi * v11));

        // x.new_tensor(mean) lands the constants in the input dtype before the
        // subtraction, so they have to be rounded here too or fp16 output comes
        // out one ulp off.
        const float centred = Sample::quantize(interpolated - Sample::quantize(mean[channel]));
        const float scaled = Sample::quantize(centred / Sample::quantize(std[channel]));
        Sample::store(destination, base + (int64_t)channel * dst_channel_stride, scaled);
    }
}

}  // namespace

#define RESIZE_KERNEL_ARGS \
    const uint8_t* __restrict__ src, int64_t src_batch_stride, \
    int64_t src_channel_stride, int64_t src_row_stride, \
    void* __restrict__ dst, int64_t dst_batch_stride, \
    int64_t dst_channel_stride, int64_t dst_row_stride, \
    int batch, int src_height, int src_width, int out_height, int out_width, \
    int content_left, int content_top, int content_width, int content_height, \
    const float* __restrict__ mean, const float* __restrict__ std, \
    const float* __restrict__ fill

#define RESIZE_KERNEL_INPUTS \
    src, src_batch_stride, src_channel_stride, src_row_stride, \
    dst, dst_batch_stride, dst_channel_stride, dst_row_stride, \
    batch, src_height, src_width, out_height, out_width, \
    content_left, content_top, content_width, content_height, \
    mean, std, fill

extern "C" __global__ void resize_normalize_fp16(RESIZE_KERNEL_ARGS) {
    resize_normalize<HalfSample>(RESIZE_KERNEL_INPUTS);
}

extern "C" __global__ void resize_normalize_fp32(RESIZE_KERNEL_ARGS) {
    resize_normalize<FloatSample>(RESIZE_KERNEL_INPUTS);
}
