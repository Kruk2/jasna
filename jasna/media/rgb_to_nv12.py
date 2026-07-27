import torch

from jasna.media.yuv_scratch import (
    YuvScratch,
    apply_matrix,
    average_quads,
    interleave_chroma,
)

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

_YUV_OFFSET_LIMITED = (16.0, 128.0, 128.0)
_YUV_OFFSET_FULL = (0.0, 128.0, 128.0)


def _rows(matrix: torch.Tensor) -> tuple[tuple[float, float, float], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix)


def _chw_rgb_to_nv12_into(
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
        raise ValueError(f"NV12 conversion requires even dimensions, got {H}x{W}")

    offsets = _YUV_OFFSET_FULL if full_range else _YUV_OFFSET_LIMITED
    yuv = scratch.yuv
    apply_matrix(img_chw, rows, offsets, yuv)

    # Clamp to the selected 8-bit code range.
    y_min, y_max = (0, 255) if full_range else (16, 235)
    uv_min, uv_max = (0, 255) if full_range else (16, 240)
    luma.copy_(yuv[0].round_().clamp_(y_min, y_max))

    subsampled = scratch.chroma
    average_quads(yuv[1:3], subsampled)
    subsampled.round_().clamp_(uv_min, uv_max)
    interleave_chroma(subsampled, chroma)


def _chw_rgb_to_nv12(
    img_chw: torch.Tensor,
    matrix: torch.Tensor,
    *,
    full_range: bool,
) -> torch.Tensor:
    C, H, W = img_chw.shape
    if H % 2 or W % 2:
        raise ValueError(f"NV12 conversion requires even dimensions, got {H}x{W}")
    packed = torch.empty((H + H // 2, W), dtype=torch.uint8, device=img_chw.device)
    _chw_rgb_to_nv12_into(
        img_chw,
        packed[:H],
        packed[H:],
        YuvScratch(H, W, img_chw.device),
        _rows(matrix),
        full_range=full_range,
    )
    return packed


def chw_rgb_to_nv12_bt709_limited(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_nv12(img_chw, _YUV_MATRIX_BT709, full_range=False)


def chw_rgb_to_nv12_bt601_limited(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_nv12(img_chw, _YUV_MATRIX_BT601, full_range=False)


def chw_rgb_to_nv12_bt2020_limited(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_nv12(img_chw, _YUV_MATRIX_BT2020, full_range=False)


def chw_rgb_to_nv12_bt709_full(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_nv12(img_chw, _YUV_MATRIX_BT709_FULL, full_range=True)


def chw_rgb_to_nv12_bt601_full(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_nv12(img_chw, _YUV_MATRIX_BT601_FULL, full_range=True)


def chw_rgb_to_nv12_bt2020_full(img_chw: torch.Tensor) -> torch.Tensor:
    return _chw_rgb_to_nv12(img_chw, _YUV_MATRIX_BT2020_FULL, full_range=True)


NV12_VARIANTS: dict[str, tuple[tuple[tuple[float, float, float], ...], bool]] = {
    "nv12_bt601_limited": (_rows(_YUV_MATRIX_BT601), False),
    "nv12_bt601_full": (_rows(_YUV_MATRIX_BT601_FULL), True),
    "nv12_bt709_limited": (_rows(_YUV_MATRIX_BT709), False),
    "nv12_bt709_full": (_rows(_YUV_MATRIX_BT709_FULL), True),
    "nv12_bt2020_limited": (_rows(_YUV_MATRIX_BT2020), False),
    "nv12_bt2020_full": (_rows(_YUV_MATRIX_BT2020_FULL), True),
}
