import torch
import torch.nn.functional as F

# BT.709 limited-range RGB→YUV coefficients fused with scale+offset.
# Row 0: Y  = 64 + 876 * (0.2126*R + 0.7152*G + 0.0722*B)
# Row 1: U  = 512 + 896 * (-0.114572*R - 0.385428*G + 0.5*B)
# Row 2: V  = 512 + 896 * (0.5*R - 0.454153*G - 0.045847*B)
_YUV_MATRIX = torch.tensor([
    [876.0 * 0.2126,    876.0 * 0.7152,    876.0 * 0.0722],
    [896.0 * -0.114572, 896.0 * -0.385428, 896.0 * 0.500000],
    [896.0 * 0.500000,  896.0 * -0.454153, 896.0 * -0.045847],
], dtype=torch.float32)
_YUV_OFFSET = torch.tensor([64.0, 512.0, 512.0], dtype=torch.float32)

# BT.709 limited-range RGB→YUV coefficients for 8-bit NV12 (no scale offset)
# Y  = 16 + 219 * (0.2126*R + 0.7152*G + 0.0722*B)
# U  = 128 + 224 * (-0.114572*R - 0.385428*G + 0.5*B)
# V  = 128 + 224 * (0.5*R - 0.454153*G - 0.045847*B)
_YUV_MATRIX_8BIT = torch.tensor([
    [219.0 * 0.2126,    219.0 * 0.7152,    219.0 * 0.0722],
    [224.0 * -0.114572, 224.0 * -0.385428, 224.0 * 0.500000],
    [224.0 * 0.500000,  224.0 * -0.454153, 224.0 * -0.045847],
], dtype=torch.float32)
_YUV_OFFSET_8BIT = torch.tensor([16.0, 128.0, 128.0], dtype=torch.float32)

_cache: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}
_cache_8bit: dict[torch.device, tuple[torch.Tensor, torch.Tensor]] = {}


def _get_coeffs(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    cached = _cache.get(device)
    if cached is not None:
        return cached
    mat = _YUV_MATRIX.to(device=device)
    off = _YUV_OFFSET.to(device=device)
    _cache[device] = (mat, off)
    return mat, off


def _get_coeffs_8bit(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    cached = _cache_8bit.get(device)
    if cached is not None:
        return cached
    mat = _YUV_MATRIX_8BIT.to(device=device)
    off = _YUV_OFFSET_8BIT.to(device=device)
    _cache_8bit[device] = (mat, off)
    return mat, off


def chw_rgb_to_p010_bt709_limited(img_chw: torch.Tensor) -> torch.Tensor:
    C, H, W = img_chw.shape

    if img_chw.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        rgb = img_chw.float().div_(255.0)
    else:
        rgb = img_chw.float()

    mat, off = _get_coeffs(rgb.device)

    # (3, H*W) matmul → (3, H*W) → (3, H, W): produces Y, U, V planes
    yuv = mat.mm(rgb.reshape(3, -1)).reshape(3, H, W)
    yuv[0].add_(off[0])
    yuv[1].add_(off[1])
    yuv[2].add_(off[2])

    # Y plane: clamp to limited range, shift left 6 bits for P010
    Y = yuv[0].round_().clamp_(64, 940).mul_(64).to(torch.int16)

    # UV planes: subsample 4:2:0 via avg_pool2d on both channels at once
    uv_full = yuv[1:3].unsqueeze(0)  # (1, 2, H, W)
    uv_ds = F.avg_pool2d(uv_full, 2).squeeze(0)  # (2, H/2, W/2)
    uv_ds.round_().clamp_(64, 960).mul_(64)
    uv_i16 = uv_ds.to(torch.int16)

    # Interleave U and V: (H/2, W) with alternating U, V
    uv_interleaved = uv_i16.permute(1, 2, 0).reshape(H // 2, W)

    return torch.cat([Y, uv_interleaved], dim=0).contiguous()


def chw_rgb_to_nv12_bt709_limited(img_chw: torch.Tensor) -> torch.Tensor:
    """Convert CHW RGB tensor to NV12 format (8-bit) for PyNvVideoCodec encoder.
    
    NV12 format:
    - Y plane: full resolution, 8-bit
    - UV plane: half resolution, interleaved U and V, 8-bit
    """
    C, H, W = img_chw.shape

    if img_chw.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        rgb = img_chw.float().div_(255.0)
    else:
        rgb = img_chw.float()

    mat, off = _get_coeffs_8bit(rgb.device)

    # (3, H*W) matmul → (3, H*W) → (3, H, W): produces Y, U, V planes
    yuv = mat.mm(rgb.reshape(3, -1)).reshape(3, H, W)
    yuv[0].add_(off[0])
    yuv[1].add_(off[1])
    yuv[2].add_(off[2])

    # Y plane: clamp to limited range [16, 235], convert to uint8
    Y = yuv[0].round_().clamp_(16, 235).to(torch.uint8)

    # UV planes: subsample 4:2:0 via avg_pool2d on both channels at once
    uv_full = yuv[1:3].unsqueeze(0)  # (1, 2, H, W)
    uv_ds = F.avg_pool2d(uv_full, 2).squeeze(0)  # (2, H/2, W/2)
    uv_ds.round_().clamp_(16, 240)
    uv_u8 = uv_ds.to(torch.uint8)

    # Interleave U and V: (H/2, W) with alternating U, V
    uv_interleaved = uv_u8.permute(1, 2, 0).reshape(H // 2, W)

    return torch.cat([Y, uv_interleaved], dim=0).contiguous()
