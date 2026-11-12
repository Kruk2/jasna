#include <stdint.h>

// The per-frame composite of LTX windows (jasna/ltx/compose.py composite_frame) in
// four launches instead of ~150 short Torch kernels: a frame's compositing is
// bound by launches, each waiting behind the window decoder's queued kernels.
//
// Bit-identical with the Torch path: every float operation is the one Torch runs,
// in its order, rounded on its own (add_rn, mul_rn, ...), so the compiler never
// contracts a multiply and an add into one FMA. The Gaussian blur of the feather
// stays two cuBLAS matrix products on the host side, so its sums keep cuBLAS's order.
//
// One frame's geometry arrives as a packed int64 table (FrameTable below), uploaded
// with the track masks in one copy. Coordinates are in the compositing region, the
// frame area every live crop grown by the feather reach spans; boxes are x1, y1, x2,
// y2, half-open.

// Single-rounding float operations the compiler must not contract into an FMA. HIP's __fmul_rn and
// friends are header functions whose plain operators keep the contract flag when inlined, so HIP
// spells them as operators under this file's contract(off).
#if defined(__HIP_PLATFORM_AMD__)
#pragma clang fp contract(off)
#endif

namespace {

#if defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float add_rn(float a, float b) { return a + b; }
__device__ __forceinline__ float sub_rn(float a, float b) { return a - b; }
__device__ __forceinline__ float mul_rn(float a, float b) { return a * b; }
__device__ __forceinline__ float div_rn(float a, float b) { return a / b; }
#else
__device__ __forceinline__ float add_rn(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float sub_rn(float a, float b) { return __fsub_rn(a, b); }
__device__ __forceinline__ float mul_rn(float a, float b) { return __fmul_rn(a, b); }
__device__ __forceinline__ float div_rn(float a, float b) { return __fdiv_rn(a, b); }
#endif

constexpr int HEADER_SLOTS = 20;
constexpr int TRACK_SLOTS = 12;
constexpr int LAYER_SLOTS = 16;
constexpr int MEDIAN_BINS = 511;
constexpr int CHANNELS = 3;
constexpr float COLOUR_MATCH_STRENGTH = 0.75f;
constexpr int TILE = 32;

struct Box {
    int x1, y1, x2, y2;
    __device__ bool holds(int x, int y) const { return x >= x1 && x < x2 && y >= y1 && y < y2; }
};

__device__ Box box_at(const int64_t* slots) {
    return Box{(int)slots[0], (int)slots[1], (int)slots[2], (int)slots[3]};
}

// Header: track count, layer count, region origin in the frame (x, y), region height and width,
// source frame (pointer, channel stride, row stride), output frame (same), ramp feather
// inverse (float bits), histogram pointer, colour offsets pointer; slots 16..19 the blend box.
struct Frame {
    int tracks, layers, x0, y0, height, width;
    const uint8_t* source;
    int64_t source_channel, source_row;
    uint8_t* out;
    int64_t out_channel, out_row;
    float ramp_inverse;
    int* histogram;
    float* offsets;
};

__device__ Frame frame_at(const int64_t* table) {
    Frame f;
    f.tracks = (int)table[0];
    f.layers = (int)table[1];
    f.x0 = (int)table[2];
    f.y0 = (int)table[3];
    f.height = (int)table[4];
    f.width = (int)table[5];
    f.source = reinterpret_cast<const uint8_t*>(table[6]);
    f.source_channel = table[7];
    f.source_row = table[8];
    f.out = reinterpret_cast<uint8_t*>(table[9]);
    f.out_channel = table[10];
    f.out_row = table[11];
    f.ramp_inverse = __int_as_float((int)table[12]);
    f.histogram = reinterpret_cast<int*>(table[13]);
    f.offsets = reinterpret_cast<float*>(table[14]);
    return f;
}

// Track: mask patch (uint8, 0 or 255, row-major over its polygon box), polygon box, feather
// box (the polygon box grown by 2*feather+1, clipped to the region), feather alpha (float,
// row-major over the feather box), dilated buffer (float, feather box rows by feather box
// width + 2*feather columns).
struct Track {
    const uint8_t* mask;
    Box polygon;
    Box feather;
    const float* alpha;
    float* dilated;
};

__device__ Track track_at(const int64_t* table, int index) {
    const int64_t* slots = table + HEADER_SLOTS + TRACK_SLOTS * index;
    Track t;
    t.mask = reinterpret_cast<const uint8_t*>(slots[0]);
    t.polygon = box_at(slots + 1);
    t.feather = box_at(slots + 5);
    t.alpha = reinterpret_cast<const float*>(slots[9]);
    t.dilated = reinterpret_cast<float*>(slots[10]);
    return t;
}

__device__ bool in_mask(const Track& t, int x, int y) {
    if (!t.polygon.holds(x, y)) {
        return false;
    }
    const int width = t.polygon.x2 - t.polygon.x1;
    return t.mask[(int64_t)(y - t.polygon.y1) * width + (x - t.polygon.x1)] != 0;
}

__device__ float feather_alpha(const Track& t, int x, int y) {
    if (!t.feather.holds(x, y)) {
        return 0.0f;
    }
    const int width = t.feather.x2 - t.feather.x1;
    return t.alpha[(int64_t)(y - t.feather.y1) * width + (x - t.feather.x1)];
}

// Layer: track index, crop box, paint box (crop and feather box overlap), the resampled crop
// before rounding (float pointer and channel/row/column strides, origin at the crop's corner),
// crossfade weight (float bits).
struct Layer {
    int track;
    Box crop;
    Box paint;
    const float* resampled;
    int64_t channel_stride, row_stride, column_stride;
    float weight;
};

__device__ Layer layer_at(const int64_t* table, int tracks, int index) {
    const int64_t* slots = table + HEADER_SLOTS + TRACK_SLOTS * tracks + LAYER_SLOTS * index;
    Layer l;
    l.track = (int)slots[0];
    l.crop = box_at(slots + 1);
    l.paint = box_at(slots + 5);
    l.resampled = reinterpret_cast<const float*>(slots[9]);
    l.channel_stride = slots[10];
    l.row_stride = slots[11];
    l.column_stride = slots[12];
    l.weight = __int_as_float((int)slots[13]);
    return l;
}

// canvas_to_source's round_().clamp_(0, 255).
__device__ float restored_value(const Layer& l, int channel, int x, int y) {
    const float value = l.resampled[channel * l.channel_stride + (int64_t)(y - l.crop.y1) * l.row_stride
                                    + (int64_t)(x - l.crop.x1) * l.column_stride];
    return fminf(fmaxf(rintf(value), 0.0f), 255.0f);
}

__device__ float source_value(const Frame& f, int channel, int x, int y) {
    return (float)f.source[channel * f.source_channel + (int64_t)(f.y0 + y) * f.source_row + (f.x0 + x)];
}

// compose.crop_edge_ramp along one axis at pixel ``position``.
__device__ float ramp_axis(int position, int low, int high, int size, float inverse) {
    const float centre = add_rn((float)position, 0.5f);
    float ramp = (centre > (float)low && centre < (float)high) ? 1.0f : 0.0f;
    if (low > 0) {
        ramp = fminf(ramp, mul_rn(sub_rn(centre, (float)low), inverse));
    }
    if (high < size) {
        ramp = fminf(ramp, mul_rn(sub_rn((float)high, centre), inverse));
    }
    return fminf(fmaxf(ramp, 0.0f), 1.0f);
}

// Outside the feather of every track: the colour ring.
__device__ bool outside_every_feather(const int64_t* table, int tracks, int x, int y) {
    for (int t = 0; t < tracks; ++t) {
        if (feather_alpha(track_at(table, t), x, y) > 0.0f) {
            return false;
        }
    }
    return true;
}

}  // namespace

// Each track's mask dilated by an ellipse of ``radius`` (compose._dilate_and_blur's
// dilate_ellipse) over its feather box, as 0/1 floats reflect-101 padded by ``radius`` columns
// on both sides: the input of the row blur. ``reach[d]`` is the farthest row offset an ellipse
// row still spans at column distance ``d`` (blending._ellipse_row_reach). Grid: column tiles,
// row tiles, tracks; dynamic shared memory (TILE + 2*radius) * (TILE + 1) bytes.
extern "C" __global__ void dilate_feather_masks(const int64_t* __restrict__ table, const int* __restrict__ reach, int radius) {
    extern __shared__ unsigned char distance[];
    const Frame f = frame_at(table);
    const int track_index = blockIdx.z;
    if (track_index >= f.tracks) {
        return;
    }
    const Track t = track_at(table, track_index);
    const Box box = t.feather;
    const int height = box.y2 - box.y1;
    const int width = box.x2 - box.x1;
    const int tile_x = blockIdx.x * TILE;
    const int tile_y = blockIdx.y * TILE;
    if (tile_x >= width || tile_y >= height) {
        return;
    }
    const int halo_rows = TILE + 2 * radius;
    const int far = radius + 1;
    const int thread = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads = blockDim.x * blockDim.y;

    // Column distance from each tile column to the nearest mask pixel of its row, for the tile
    // rows and ``radius`` rows above and below; ``far`` when none is within ``radius``.
    for (int row = thread; row < halo_rows; row += threads) {
        unsigned char* line = distance + row * (TILE + 1);
        const int y = tile_y - radius + row;
        if (y < 0 || y >= height) {
            for (int c = 0; c < TILE; ++c) {
                line[c] = (unsigned char)far;
            }
            continue;
        }
        int last = -(1 << 20);
        for (int x = max(tile_x - radius, 0); x < min(tile_x + TILE, width); ++x) {
            if (in_mask(t, box.x1 + x, box.y1 + y)) {
                last = x;
            }
            if (x >= tile_x) {
                line[x - tile_x] = (unsigned char)min(x - last, far);
            }
        }
        int next = 1 << 20;
        for (int x = min(tile_x + TILE + radius, width) - 1; x >= tile_x; --x) {
            if (in_mask(t, box.x1 + x, box.y1 + y)) {
                next = x;
            }
            if (x < tile_x + TILE) {
                line[x - tile_x] = (unsigned char)min((int)line[x - tile_x], min(next - x, far));
            }
        }
    }
    __syncthreads();

    const int padded_width = width + 2 * radius;
    for (int row = threadIdx.y; row < TILE; row += blockDim.y) {
        const int y = tile_y + row;
        const int x = tile_x + threadIdx.x;
        if (y >= height || x >= width) {
            continue;
        }
        bool dilated = false;
        for (int dy = -radius; dy <= radius && !dilated; ++dy) {
            const int d = distance[(row + radius + dy) * (TILE + 1) + threadIdx.x];
            dilated = d <= radius && reach[d] >= abs(dy);
        }
        const float value = dilated ? 1.0f : 0.0f;
        float* out = t.dilated + (int64_t)y * padded_width;
        out[radius + x] = value;
        // The reflect-101 border columns that mirror this one.
        if (x >= 1 && x <= radius) {
            out[radius - x] = value;
        }
        const int mirrored = radius + 2 * (width - 1) - x;
        if (mirrored >= radius + width && mirrored < padded_width) {
            out[mirrored] = value;
        }
    }
}

// Per layer and channel, the histogram of (region - restored crop) + 255 over the crop's pixels
// outside every track's feather: compose.masked_median's histogram. Grid: column tiles, row
// tiles, layers.
extern "C" __global__ void colour_ring_histograms(const int64_t* __restrict__ table) {
    __shared__ int bins[CHANNELS * MEDIAN_BINS];
    const Frame f = frame_at(table);
    const int layer_index = blockIdx.z;
    const Layer l = layer_at(table, f.tracks, layer_index);
    const int thread = threadIdx.y * blockDim.x + threadIdx.x;
    const int threads = blockDim.x * blockDim.y;
    for (int i = thread; i < CHANNELS * MEDIAN_BINS; i += threads) {
        bins[i] = 0;
    }
    __syncthreads();
    const int x = l.crop.x1 + blockIdx.x * TILE + threadIdx.x;
    for (int row = threadIdx.y; row < TILE; row += blockDim.y) {
        const int y = l.crop.y1 + blockIdx.y * TILE + row;
        if (x >= l.crop.x2 || y >= l.crop.y2 || !outside_every_feather(table, f.tracks, x, y)) {
            continue;
        }
        for (int c = 0; c < CHANNELS; ++c) {
            const int difference = (int)sub_rn(source_value(f, c, x, y), restored_value(l, c, x, y));
            atomicAdd(&bins[c * MEDIAN_BINS + difference + 255], 1);
        }
    }
    __syncthreads();
    int* histogram = f.histogram + layer_index * CHANNELS * MEDIAN_BINS;
    for (int i = thread; i < CHANNELS * MEDIAN_BINS; i += threads) {
        if (bins[i] != 0) {
            atomicAdd(&histogram[i], bins[i]);
        }
    }
}

// Per layer and channel, the colour offset COLOUR_MATCH_STRENGTH * median of its ring histogram,
// the mean of the two central values for an even count and 0 for an empty ring
// (compose.masked_median). Grid: layers; one thread per channel.
extern "C" __global__ void colour_offsets(const int64_t* __restrict__ table) {
    const Frame f = frame_at(table);
    const int layer_index = blockIdx.x;
    const int channel = threadIdx.x;
    if (layer_index >= f.layers || channel >= CHANNELS) {
        return;
    }
    const int* bins = f.histogram + (layer_index * CHANNELS + channel) * MEDIAN_BINS;
    long long count = 0;
    for (int i = 0; i < MEDIAN_BINS; ++i) {
        count += bins[i];
    }
    float median = 0.0f;
    if (count > 0) {
        // searchsorted(cumulative, k, right=True): the first bin whose cumulative count exceeds k.
        const long long low_rank = (count - 1) / 2;
        const long long high_rank = count / 2;
        long long cumulative = 0;
        int low = -1;
        int high = -1;
        for (int i = 0; i < MEDIAN_BINS && high < 0; ++i) {
            cumulative += bins[i];
            if (low < 0 && cumulative > low_rank) {
                low = i;
            }
            if (cumulative > high_rank) {
                high = i;
            }
        }
        median = sub_rn(mul_rn((float)(low + high), 0.5f), 255.0f);
    }
    f.offsets[layer_index * CHANNELS + channel] = mul_rn(median, COLOUR_MATCH_STRENGTH);
}

// Every layer blended over the blend box (the union of the paint boxes), normalised where the
// alphas sum past 1, over the region, rounded into the output frame: composite_frame's layer
// loop and final blend for tracks at full presence. Grid: column tiles, row tiles of the blend box.
extern "C" __global__ void blend_layers(const int64_t* __restrict__ table) {
    const Frame f = frame_at(table);
    const Box blend = box_at(table + 16);
    const int x = blend.x1 + blockIdx.x * TILE + threadIdx.x;
    for (int row = threadIdx.y; row < TILE; row += blockDim.y) {
        const int y = blend.y1 + blockIdx.y * TILE + row;
        if (x >= blend.x2 || y >= blend.y2) {
            continue;
        }
        bool in_union = false;
        for (int t = 0; t < f.tracks && !in_union; ++t) {
            in_union = in_mask(track_at(table, t), x, y);
        }
        float accumulated[CHANNELS] = {0.0f, 0.0f, 0.0f};
        float total = 0.0f;
        for (int index = 0; index < f.layers; ++index) {
            const Layer l = layer_at(table, f.tracks, index);
            if (!l.paint.holds(x, y)) {
                continue;
            }
            const Track t = track_at(table, l.track);
            const bool roi = in_mask(t, x, y);
            const float ramp = fminf(
                ramp_axis(y, l.crop.y1, l.crop.y2, f.height, f.ramp_inverse),
                ramp_axis(x, l.crop.x1, l.crop.x2, f.width, f.ramp_inverse)
            );
            const float edge = roi ? (ramp > 0.0f ? 1.0f : 0.0f) : ramp;
            float alpha = mul_rn(mul_rn(feather_alpha(t, x, y), edge), l.weight);
            alpha = mul_rn(alpha, (roi || !in_union) ? 1.0f : 0.0f);
            const float* offsets = f.offsets + index * CHANNELS;
            for (int c = 0; c < CHANNELS; ++c) {
                const float restored = add_rn(restored_value(l, c, x, y), offsets[c]);
                accumulated[c] = add_rn(accumulated[c], mul_rn(restored, alpha));
            }
            total = add_rn(total, alpha);
        }
        const bool over = total > 1.0f;
        const float kept = sub_rn(1.0f, over ? 1.0f : total);
        for (int c = 0; c < CHANNELS; ++c) {
            const float painted = over ? div_rn(accumulated[c], total) : accumulated[c];
            const float blended = add_rn(painted, mul_rn(source_value(f, c, x, y), kept));
            f.out[c * f.out_channel + (int64_t)(f.y0 + y) * f.out_row + (f.x0 + x)] =
                (uint8_t)fminf(fmaxf(rintf(blended), 0.0f), 255.0f);
        }
    }
}
