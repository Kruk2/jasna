import torch
from torch.utils.cpp_extension import load_inline
import logging

logger = logging.getLogger(__name__)


_rgb_to_p010_cuda_source = """
#include <cuda_runtime.h>
#include <stdint.h>

// Single fused kernel - reads CHW directly, writes Y and UV planes
__global__ void rgb_to_p010_fused_kernel(
    const float* __restrict__ img,  // CHW layout: [3, H, W]
    int16_t* __restrict__ out,       // P010: Y plane [H, W] + UV plane [H/2, W]
    int H, int W
) {
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = H * W;
    
    if (pixel_idx >= total_pixels) return;
    
    int plane_size = total_pixels;
    float r = img[pixel_idx];
    float g = img[plane_size + pixel_idx];
    float b = img[2 * plane_size + pixel_idx];
    
    // Y calculation
    float y = 64.0f + 876.0f * (0.2126f * r + 0.7152f * g + 0.0722f * b);
    y = fminf(fmaxf(y, 64.0f), 940.0f) * 64.0f;
    out[pixel_idx] = static_cast<int16_t>(y);
    
    // UV calculation - only for top-left pixel of each 2x2 block
    int row = pixel_idx / W;
    int col = pixel_idx % W;
    
    if ((row & 1) == 0 && (col & 1) == 0) {
        int idx01 = pixel_idx + 1;
        int idx10 = pixel_idx + W;
        int idx11 = pixel_idx + W + 1;
        
        float r_avg = (r + img[idx01] + img[idx10] + img[idx11]) * 0.25f;
        float g_avg = (g + img[plane_size + idx01] + img[plane_size + idx10] + img[plane_size + idx11]) * 0.25f;
        float b_avg = (b + img[2*plane_size + idx01] + img[2*plane_size + idx10] + img[2*plane_size + idx11]) * 0.25f;
        
        float u = 512.0f + 896.0f * (-0.114572f * r_avg - 0.385428f * g_avg + 0.500000f * b_avg);
        float v = 512.0f + 896.0f * (0.500000f * r_avg - 0.454153f * g_avg - 0.045847f * b_avg);
        
        u = fminf(fmaxf(u, 64.0f), 960.0f) * 64.0f;
        v = fminf(fmaxf(v, 64.0f), 960.0f) * 64.0f;
        
        int uv_row = row >> 1;
        int uv_col = col >> 1;
        int uv_out_idx = total_pixels + uv_row * W + uv_col * 2;
        out[uv_out_idx] = static_cast<int16_t>(u);
        out[uv_out_idx + 1] = static_cast<int16_t>(v);
    }
}

extern "C" void launch_rgb_to_p010_kernel(
    const float* img, int16_t* out, int H, int W, cudaStream_t stream
) {
    int total_pixels = H * W;
    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;
    rgb_to_p010_fused_kernel<<<blocks, threads, 0, stream>>>(img, out, H, W);
}
"""

_rgb_to_p010_cpp_source = """
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

extern "C" void launch_rgb_to_p010_kernel(
    const float* img, int16_t* out, int H, int W, cudaStream_t stream
);

torch::Tensor rgb_to_p010_cuda(torch::Tensor img_chw) {
    TORCH_CHECK(img_chw.dim() == 3, "Expected 3D tensor");
    TORCH_CHECK(img_chw.size(0) == 3, "Expected 3 channels");
    
    int H = img_chw.size(1);
    int W = img_chw.size(2);
    TORCH_CHECK(H % 2 == 0 && W % 2 == 0, "Height and width must be even");
    
    // Ensure contiguous float32
    torch::Tensor img_float;
    if (img_chw.scalar_type() == torch::kFloat && img_chw.is_contiguous()) {
        img_float = img_chw;
    } else if (img_chw.scalar_type() == torch::kFloat) {
        img_float = img_chw.contiguous();
    } else if (img_chw.scalar_type() == torch::kHalf || img_chw.scalar_type() == torch::kBFloat16) {
        img_float = img_chw.to(torch::kFloat).contiguous();
    } else {
        img_float = img_chw.to(torch::kFloat).div_(255.0f).contiguous();
    }
    
    // Output: Y plane (H*W) + UV plane (H/2 * W) as int16
    auto out = torch::empty({H + H/2, W}, img_float.options().dtype(torch::kShort));
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    
    launch_rgb_to_p010_kernel(
        img_float.data_ptr<float>(),
        out.data_ptr<int16_t>(),
        H, W, stream
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rgb_to_p010_cuda", &rgb_to_p010_cuda, "RGB to P010 conversion (CUDA)");
}
"""

_rgb_to_p010_module = None
_use_fallback = False


def _get_rgb_to_p010_kernel():
    global _rgb_to_p010_module, _use_fallback
    if _rgb_to_p010_module is None and not _use_fallback:
        try:
            _rgb_to_p010_module = load_inline(
                name='rgb_to_p010',
                cpp_sources=_rgb_to_p010_cpp_source,
                cuda_sources=_rgb_to_p010_cuda_source,
                verbose=False,
                with_cuda=True,
                extra_cflags=['/O2', '/openmp', '/std:c++20', '/DENABLE_BF16'],
                extra_cuda_cflags=['-std=c++20'],
            )
        except Exception as e:
            msg = (
                "Failed to compile rgb_to_p010 CUDA kernel. Consider installing build tools if on windows and add it to path: "
                "winget install --id Microsoft.VisualStudio.2022.BuildTools --force --override \"--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64\""
            )
            print(msg)
            logger.info("%s", msg)
            print("Error:", e)
            logger.info("Error: %s", e)
            _use_fallback = True
    return _rgb_to_p010_module


def _chw_rgb_to_p010_bt709_limited_fallback(img_chw: torch.Tensor) -> torch.Tensor:
    """Pure PyTorch fallback implementation."""
    C, H, W = img_chw.shape
    assert C == 3 and H % 2 == 0 and W % 2 == 0
    
    if img_chw.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        img_chw = img_chw.float() / 255.0
    
    R = img_chw[0].float()
    G = img_chw[1].float()
    B = img_chw[2].float()

    # 10-bit BT.709 limited range: Y: 64-940, U/V: 64-960
    Yf = 64.0 + 876.0 * (0.2126 * R + 0.7152 * G + 0.0722 * B)
    Uf = 512.0 + 896.0 * (-0.114572 * R - 0.385428 * G + 0.500000 * B)
    Vf = 512.0 + 896.0 * (0.500000 * R - 0.454153 * G - 0.045847 * B)

    # Clamp and shift left by 6 for P010 (10-bit in upper bits of 16-bit)
    Y = (Yf.round().clamp(64, 940) * 64).to(torch.int16)

    # Subsample UV (4:2:0)
    U00, U01, U10, U11 = Uf[0::2, 0::2], Uf[0::2, 1::2], Uf[1::2, 0::2], Uf[1::2, 1::2]
    V00, V01, V10, V11 = Vf[0::2, 0::2], Vf[0::2, 1::2], Vf[1::2, 0::2], Vf[1::2, 1::2]

    U_ds = (((U00 + U01 + U10 + U11) * 0.25).round().clamp(64, 960) * 64).to(torch.int16)
    V_ds = (((V00 + V01 + V10 + V11) * 0.25).round().clamp(64, 960) * 64).to(torch.int16)

    # Interleave U and V
    uv = torch.stack((U_ds, V_ds), dim=-1).reshape(U_ds.shape[0], -1)

    return torch.cat([Y, uv], dim=0).contiguous()


def chw_rgb_to_p010_bt709_limited(img_chw: torch.Tensor) -> torch.Tensor:
    """Convert CHW RGB tensor to P010 (10-bit NV12) format in BT.709 limited range.
    
    Uses JIT-compiled CUDA kernel for optimal GPU performance, with PyTorch fallback.
    P010 format: Y plane followed by interleaved UV plane, 16-bit values with 10-bit data in MSBs.
    """
    # module = _get_rgb_to_p010_kernel()
    # if module is not None:
    #     return module.rgb_to_p010_cuda(img_chw)
    return _chw_rgb_to_p010_bt709_limited_fallback(img_chw)
