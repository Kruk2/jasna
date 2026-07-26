#include <stdint.h>

// Fused spatial bilateral filter over a clip of [T, 3, H, W] float frames.
//
// The Torch expression walks the window in Python, so a 5x5 window is 25
// iterations of roughly seven full-tensor ops: about 175 passes over the clip,
// plus a device-to-host sync per iteration to read one spatial weight as a
// Python float. Here each thread keeps its window in registers and the clip is
// read once.
//
// Neighbour lookups reproduce F.pad(mode="reflect"): an index past an edge is
// mirrored back without repeating the edge sample itself.

namespace {

__device__ __forceinline__ int reflect(int index, int extent) {
    if (index < 0) {
        return -index;
    }
    if (index >= extent) {
        return 2 * (extent - 1) - index;
    }
    return index;
}

}  // namespace

extern "C" __global__ void bilateral_denoise_fp32(
    const float* __restrict__ frames,
    int64_t frame_stride,
    int64_t channel_stride,
    int64_t row_stride,
    float* __restrict__ out,
    int64_t out_frame_stride,
    int64_t out_channel_stride,
    int64_t out_row_stride,
    int frame_count,
    int height,
    int width,
    int kernel_size,
    float range_scale,
    const float* __restrict__ spatial_weights
) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int t = blockIdx.z;
    if (x >= width || y >= height || t >= frame_count) {
        return;
    }

    const float* frame = frames + (int64_t)t * frame_stride;
    const int64_t centre = (int64_t)y * row_stride + x;
    const float c0 = frame[centre];
    const float c1 = frame[channel_stride + centre];
    const float c2 = frame[2 * channel_stride + centre];

    const int half = kernel_size / 2;
    float accumulated0 = 0.0f;
    float accumulated1 = 0.0f;
    float accumulated2 = 0.0f;
    float weight_sum = 0.0f;

    for (int dy = 0; dy < kernel_size; ++dy) {
        const int sy = reflect(y + dy - half, height);
        for (int dx = 0; dx < kernel_size; ++dx) {
            const int sx = reflect(x + dx - half, width);
            const int64_t offset = (int64_t)sy * row_stride + sx;
            const float n0 = frame[offset];
            const float n1 = frame[channel_stride + offset];
            const float n2 = frame[2 * channel_stride + offset];

            const float d0 = c0 - n0;
            const float d1 = c1 - n1;
            const float d2 = c2 - n2;
            // Torch reduces the channel mean as ((a + b) + c) / 3.
            const float difference = ((d0 * d0 + d1 * d1) + d2 * d2) / 3.0f;
            const float weight =
                spatial_weights[dy * kernel_size + dx] * expf(difference * range_scale);

            accumulated0 += n0 * weight;
            accumulated1 += n1 * weight;
            accumulated2 += n2 * weight;
            weight_sum += weight;
        }
    }

    float* destination = out + (int64_t)t * out_frame_stride
        + (int64_t)y * out_row_stride + x;
    destination[0] = accumulated0 / weight_sum;
    destination[out_channel_stride] = accumulated1 / weight_sum;
    destination[2 * out_channel_stride] = accumulated2 / weight_sum;
}
