import torch
import torch.nn.functional as F

# BT.709 limited-range RGB→YUV coefficients fused with scale+offset (8-bit).
# Row 0: Y  = 16 + 219 * (0.2126*R + 0.7152*G + 0.0722*B)
# Row 1: U  = 128 + 224 * (-0.114572*R - 0.385428*G + 0.5*B)
# Row 2: V  = 128 + 224 * (0.5*R - 0.454153*G - 0.045847*B)
_YUV_MATRIX_BT709 = torch.tensor([
    [219.0 * 0.2126,    219.0 * 0.7152,    219.0 * 0.0722],
    [224.0 * -0.114572, 224.0 * -0.385428, 224.0 * 0.500000],
    [224.0 * 0.500000,  224.0 * -0.454153, 224.0 * -0.045847],
], dtype=torch.float32)

# BT.601 limited-range RGB→YUV coefficients fused with scale+offset (8-bit).
_YUV_MATRIX_BT601 = torch.tensor([
    [219.0 * 0.299000,  219.0 * 0.587000,  219.0 * 0.114000],
    [224.0 * -0.168736, 224.0 * -0.331264, 224.0 * 0.500000],
    [224.0 * 0.500000,  224.0 * -0.418688, 224.0 * -0.081312],
], dtype=torch.float32)

# BT.2020 non-constant-luminance limited-range RGB→YUV coefficients (8-bit).
_YUV_MATRIX_BT2020 = torch.tensor([
    [219.0 * 0.262700,  219.0 * 0.678000,  219.0 * 0.059300],
    [224.0 * -0.139630, 224.0 * -0.360370, 224.0 * 0.500000],
    [224.0 * 0.500000,  224.0 * -0.459786, 224.0 * -0.040214],
], dtype=torch.float32)


def _full_range_matrix(limited_matrix: torch.Tensor) -> torch.Tensor:
    matrix = limited_matrix.clone()
    matrix[0].mul_(255.0 / 219.0)
    matrix[1:3].mul_(255.0 / 224.0)
    return matrix


_YUV_MATRIX_BT709_FULL = _full_range_matrix(_YUV_MATRIX_BT709)
_YUV_MATRIX_BT601_FULL = _full_range_matrix(_YUV_MATRIX_BT601)
_YUV_MATRIX_BT2020_FULL = _full_range_matrix(_YUV_MATRIX_BT2020)

_YUV_OFFSET_LIMITED = torch.tensor([16.0, 128.0, 128.0], dtype=torch.float32)
_YUV_OFFSET_FULL = torch.tensor([0.0, 128.0, 128.0], dtype=torch.float32)

_cache: dict[tuple[str, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}


def _get_coeffs(
    name: str,
    matrix: torch.Tensor,
    offset: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (name, device)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    mat = matrix.to(device=device)
    off = offset.to(device=device)
    _cache[key] = (mat, off)
    return mat, off


def _chw_rgb_to_nv12(
    img_chw: torch.Tensor,
    name: str,
    matrix: torch.Tensor,
    *,
    full_range: bool,
) -> torch.Tensor:
    C, H, W = img_chw.shape
    if H % 2 or W % 2:
        raise ValueError(f"NV12 conversion requires even dimensions, got {H}x{W}")

    if img_chw.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        rgb = img_chw.float().div_(255.0)
    else:
        rgb = img_chw.float()

    offset = _YUV_OFFSET_FULL if full_range else _YUV_OFFSET_LIMITED
    mat, off = _get_coeffs(name, matrix, offset, rgb.device)

    # (3, H*W) matmul → (3, H*W) → (3, H, W): produces Y, U, V planes
    yuv = mat.mm(rgb.reshape(3, -1)).reshape(3, H, W)
    yuv[0].add_(off[0])
    yuv[1].add_(off[1])
    yuv[2].add_(off[2])

    # Clamp to the selected 8-bit code range.
    y_min, y_max = (0, 255) if full_range else (16, 235)
    uv_min, uv_max = (0, 255) if full_range else (16, 240)
    Y = yuv[0].round_().clamp_(y_min, y_max).to(torch.uint8)

    # UV planes: subsample 4:2:0 via avg_pool2d on both channels at once
    uv_full = yuv[1:3].unsqueeze(0)  # (1, 2, H, W)
    uv_ds = F.avg_pool2d(uv_full, 2).squeeze(0)  # (2, H/2, W/2)
    uv_ds.round_().clamp_(uv_min, uv_max)
    uv_u8 = uv_ds.to(torch.uint8)

    # Interleave U and V: (H/2, W) with alternating U, V
    uv_interleaved = uv_u8.permute(1, 2, 0).reshape(H // 2, W)

    return torch.cat([Y, uv_interleaved], dim=0).contiguous()


def chw_rgb_to_nv12_bt709_limited(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_nv12(
        img_chw, "bt709-limited", _YUV_MATRIX_BT709, full_range=False
    )


def chw_rgb_to_nv12_bt601_limited(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_nv12(
        img_chw, "bt601-limited", _YUV_MATRIX_BT601, full_range=False
    )


def chw_rgb_to_nv12_bt2020_limited(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_nv12(
        img_chw, "bt2020-limited", _YUV_MATRIX_BT2020, full_range=False
    )


def chw_rgb_to_nv12_bt709_full(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_nv12(
        img_chw, "bt709-full", _YUV_MATRIX_BT709_FULL, full_range=True
    )


def chw_rgb_to_nv12_bt601_full(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_nv12(
        img_chw, "bt601-full", _YUV_MATRIX_BT601_FULL, full_range=True
    )


def chw_rgb_to_nv12_bt2020_full(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_nv12(
        img_chw, "bt2020-full", _YUV_MATRIX_BT2020_FULL, full_range=True
    )
