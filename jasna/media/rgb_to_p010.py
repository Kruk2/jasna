import torch
import torch.nn.functional as F

# BT.709 limited-range RGB→YUV coefficients fused with scale+offset.
# Row 0: Y  = 64 + 876 * (0.2126*R + 0.7152*G + 0.0722*B)
# Row 1: U  = 512 + 896 * (-0.114572*R - 0.385428*G + 0.5*B)
# Row 2: V  = 512 + 896 * (0.5*R - 0.454153*G - 0.045847*B)
_YUV_MATRIX_BT709 = torch.tensor([
    [876.0 * 0.2126,    876.0 * 0.7152,    876.0 * 0.0722],
    [896.0 * -0.114572, 896.0 * -0.385428, 896.0 * 0.500000],
    [896.0 * 0.500000,  896.0 * -0.454153, 896.0 * -0.045847],
], dtype=torch.float32)

# BT.601 limited-range RGB→YUV coefficients fused with scale+offset.
# Row 0: Y  = 64 + 876 * (0.299*R + 0.587*G + 0.114*B)
# Row 1: U  = 512 + 896 * (-0.168736*R - 0.331264*G + 0.5*B)
# Row 2: V  = 512 + 896 * (0.5*R - 0.418688*G - 0.081312*B)
_YUV_MATRIX_BT601 = torch.tensor([
    [876.0 * 0.299000,  876.0 * 0.587000,  876.0 * 0.114000],
    [896.0 * -0.168736, 896.0 * -0.331264, 896.0 * 0.500000],
    [896.0 * 0.500000,  896.0 * -0.418688, 896.0 * -0.081312],
], dtype=torch.float32)

_YUV_OFFSET = torch.tensor([64.0, 512.0, 512.0], dtype=torch.float32)

_cache: dict[tuple[str, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_coeffs(name: str, matrix: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    key = (name, device)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    mat = matrix.to(device=device)
    off = _YUV_OFFSET.to(device=device)
    _cache[key] = (mat, off)
    return mat, off


def _chw_rgb_to_p010_limited(img_chw: torch.Tensor, name: str, matrix: torch.Tensor) -> torch.Tensor:
    C, H, W = img_chw.shape

    if img_chw.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        rgb = img_chw.float().div_(255.0)
    else:
        rgb = img_chw.float()

    mat, off = _get_coeffs(name, matrix, rgb.device)

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


def chw_rgb_to_p010_bt709_limited(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_p010_limited(img_chw, "bt709", _YUV_MATRIX_BT709)


def chw_rgb_to_p010_bt601_limited(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_p010_limited(img_chw, "bt601", _YUV_MATRIX_BT601)
