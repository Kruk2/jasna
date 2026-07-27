import torch

from jasna.media.yuv_scratch import (
    YuvScratch,
    apply_matrix,
    average_quads,
    interleave_chroma,
)

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

# BT.2020 non-constant-luminance limited-range RGB→YUV coefficients.
_YUV_MATRIX_BT2020 = torch.tensor([
    [876.0 * 0.262700,  876.0 * 0.678000,  876.0 * 0.059300],
    [896.0 * -0.139630, 896.0 * -0.360370, 896.0 * 0.500000],
    [896.0 * 0.500000,  896.0 * -0.459786, 896.0 * -0.040214],
], dtype=torch.float32)


def _full_range_matrix(limited_matrix: torch.Tensor) -> torch.Tensor:
    matrix = limited_matrix.clone()
    matrix[0].mul_(1023.0 / 876.0)
    matrix[1:3].mul_(1023.0 / 896.0)
    return matrix


_YUV_MATRIX_BT709_FULL = _full_range_matrix(_YUV_MATRIX_BT709)
_YUV_MATRIX_BT601_FULL = _full_range_matrix(_YUV_MATRIX_BT601)
_YUV_MATRIX_BT2020_FULL = _full_range_matrix(_YUV_MATRIX_BT2020)

_YUV_OFFSET_LIMITED = (64.0, 512.0, 512.0)
_YUV_OFFSET_FULL = (0.0, 512.0, 512.0)


def _rows(matrix: torch.Tensor) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _chw_rgb_to_p010_into(
    img_chw: torch.Tensor,
    luma: torch.Tensor,
    chroma: torch.Tensor,
    scratch: YuvScratch,
    rows: tuple[tuple[float, float, float], ...],
    *,
    full_range: bool,
) -> None:
    C, H, W = img_chw.shape
    if H % 2 or W % 2:
        raise ValueError(f"P010 conversion requires even dimensions, got {H}x{W}")

    offsets = _YUV_OFFSET_FULL if full_range else _YUV_OFFSET_LIMITED
    yuv = scratch.yuv
    apply_matrix(img_chw, rows, offsets, yuv)

    # Clamp to the selected 10-bit code range, then store P010 values << 6.
    y_min, y_max = (0, 1023) if full_range else (64, 940)
    uv_min, uv_max = (0, 1023) if full_range else (64, 960)
    luma.copy_(yuv[0].round_().clamp_(y_min, y_max).mul_(64))

    subsampled = scratch.chroma
    average_quads(yuv[1:3], subsampled)
    subsampled.round_().clamp_(uv_min, uv_max).mul_(64)
    interleave_chroma(subsampled, chroma)


def _chw_rgb_to_p010(
    img_chw: torch.Tensor,
    matrix: torch.Tensor,
    *,
    full_range: bool,
) -> torch.Tensor:
    C, H, W = img_chw.shape
    if H % 2 or W % 2:
        raise ValueError(f"P010 conversion requires even dimensions, got {H}x{W}")
    packed = torch.empty((H + H // 2, W), dtype=torch.int16, device=img_chw.device)
    _chw_rgb_to_p010_into(
        img_chw,
        packed[:H],
        packed[H:],
        YuvScratch(H, W, img_chw.device),
        _rows(matrix),
        full_range=full_range,
    )
    return packed


def chw_rgb_to_p010_bt709_limited(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_p010(img_chw, _YUV_MATRIX_BT709, full_range=False)


def chw_rgb_to_p010_bt601_limited(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_p010(img_chw, _YUV_MATRIX_BT601, full_range=False)


def chw_rgb_to_p010_bt2020_limited(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_p010(img_chw, _YUV_MATRIX_BT2020, full_range=False)


def chw_rgb_to_p010_bt709_full(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_p010(img_chw, _YUV_MATRIX_BT709_FULL, full_range=True)


def chw_rgb_to_p010_bt601_full(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_p010(img_chw, _YUV_MATRIX_BT601_FULL, full_range=True)


def chw_rgb_to_p010_bt2020_full(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_p010(img_chw, _YUV_MATRIX_BT2020_FULL, full_range=True)


P010_VARIANTS: dict[str, tuple[tuple[tuple[float, float, float], ...], bool]] = {
    "p010_bt601_limited": (_rows(_YUV_MATRIX_BT601), False),
    "p010_bt601_full": (_rows(_YUV_MATRIX_BT601_FULL), True),
    "p010_bt709_limited": (_rows(_YUV_MATRIX_BT709), False),
    "p010_bt709_full": (_rows(_YUV_MATRIX_BT709_FULL), True),
    "p010_bt2020_limited": (_rows(_YUV_MATRIX_BT2020), False),
    "p010_bt2020_full": (_rows(_YUV_MATRIX_BT2020_FULL), True),
}
