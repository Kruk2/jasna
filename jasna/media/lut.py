"""GPU-accelerated .cube color LUT parsing and application.

Supports both 1D and 3D ``.cube`` LUTs as specified by Adobe / The Foundry:
https://wwwimages2.adobe.com/content/dam/acom/en/products/speedgrade/cc/pdfs/cube-lut-specification-1.0.pdf
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


@dataclass
class CubeLut:
    """Parsed .cube LUT.

    For 3D LUTs, ``data`` has shape ``(3, N, N, N)`` — channel-first, with axes
    ordered ``(channel, B, G, R)``. This matches PyTorch's ``grid_sample`` 5D
    convention so we can pass it directly as the input volume.

    For 1D LUTs, ``data`` has shape ``(N, 3)``.
    """
    size: int
    is_3d: bool
    data: torch.Tensor
    domain_min: tuple[float, float, float]
    domain_max: tuple[float, float, float]


def parse_cube_file(path: str | Path) -> CubeLut:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return parse_cube_text(text)


def parse_cube_text(text: str) -> CubeLut:
    size_3d: int | None = None
    size_1d: int | None = None
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]
    samples: list[tuple[float, float, float]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if upper.startswith("TITLE"):
            continue
        if upper.startswith("LUT_3D_SIZE"):
            size_3d = int(line.split()[1])
            continue
        if upper.startswith("LUT_1D_SIZE"):
            size_1d = int(line.split()[1])
            continue
        if upper.startswith("DOMAIN_MIN"):
            parts = line.split()[1:4]
            domain_min = [float(p) for p in parts]
            continue
        if upper.startswith("DOMAIN_MAX"):
            parts = line.split()[1:4]
            domain_max = [float(p) for p in parts]
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            r, g, b = float(parts[0]), float(parts[1]), float(parts[2])
        except ValueError:
            continue
        samples.append((r, g, b))

    if size_3d is None and size_1d is None:
        raise ValueError("Invalid .cube file: missing LUT_3D_SIZE or LUT_1D_SIZE")

    if size_3d is not None:
        expected = size_3d ** 3
        if len(samples) != expected:
            raise ValueError(
                f"Invalid 3D .cube file: expected {expected} samples for size {size_3d}, got {len(samples)}"
            )
        # Spec: "the lookup table is given in red-fastest, then green, then blue order"
        flat = torch.tensor(samples, dtype=torch.float32)  # (N^3, 3)
        # Reshape so axes correspond to (B, G, R, channel)
        cube = flat.view(size_3d, size_3d, size_3d, 3)  # [b, g, r, c]
        # Move channel to front for grid_sample: (channel, B, G, R)
        data = cube.permute(3, 0, 1, 2).contiguous()
        return CubeLut(
            size=size_3d, is_3d=True, data=data,
            domain_min=tuple(domain_min), domain_max=tuple(domain_max),
        )

    expected = size_1d
    if len(samples) != expected:
        raise ValueError(
            f"Invalid 1D .cube file: expected {expected} samples for size {size_1d}, got {len(samples)}"
        )
    data = torch.tensor(samples, dtype=torch.float32)  # (N, 3)
    return CubeLut(
        size=size_1d, is_3d=False, data=data,
        domain_min=tuple(domain_min), domain_max=tuple(domain_max),
    )


class GpuLutApplier:
    """Applies a parsed CubeLut to RGB CHW frames on GPU.

    Frames are expected as uint8 ``(3, H, W)`` tensors. Output dtype matches
    input. Internally we trilinear-interpolate via ``F.grid_sample`` on float32
    on the LUT device, then convert back.
    """

    def __init__(self, lut: CubeLut, device: torch.device):
        self.lut = lut
        self.device = device

        dmin = torch.tensor(lut.domain_min, dtype=torch.float32, device=device).view(3, 1, 1)
        dmax = torch.tensor(lut.domain_max, dtype=torch.float32, device=device).view(3, 1, 1)
        self._domain_min = dmin
        self._domain_scale = 1.0 / (dmax - dmin)

        if lut.is_3d:
            # (3, B, G, R) → (1, 3, B, G, R)
            self._lut_volume = lut.data.to(device).unsqueeze(0)
        else:
            # (N, 3) → keep on device
            self._lut_1d = lut.data.to(device)

    def apply(self, frame_chw: torch.Tensor) -> torch.Tensor:
        if frame_chw.ndim != 3 or frame_chw.shape[0] != 3:
            raise ValueError(f"Expected (3, H, W) RGB tensor, got {tuple(frame_chw.shape)}")

        original_dtype = frame_chw.dtype
        if original_dtype == torch.uint8:
            rgb = frame_chw.to(self.device, dtype=torch.float32).div_(255.0)
        else:
            rgb = frame_chw.to(self.device, dtype=torch.float32)

        # Normalize into [0, 1] using domain.
        rgb = (rgb - self._domain_min) * self._domain_scale
        rgb.clamp_(0.0, 1.0)

        if self.lut.is_3d:
            out = self._apply_3d(rgb)
        else:
            out = self._apply_1d(rgb)

        if original_dtype == torch.uint8:
            return out.mul_(255.0).round_().clamp_(0, 255).to(torch.uint8)
        return out.to(original_dtype)

    def _apply_3d(self, rgb: torch.Tensor) -> torch.Tensor:
        _, h, w = rgb.shape
        # grid_sample wants coords in [-1, 1] in (x=r, y=g, z=b) order to index
        # the input volume which is (1, 3, B, G, R) → (depth=B, height=G, width=R).
        coords = rgb.mul(2.0).sub_(1.0)  # (3, H, W) in [-1, 1]; channel order RGB
        # (1, 1, H, W, 3) with last-dim = (x=r, y=g, z=b)
        grid = coords.permute(1, 2, 0).unsqueeze(0).unsqueeze(0)
        sampled = F.grid_sample(
            self._lut_volume, grid,
            mode="bilinear", padding_mode="border", align_corners=True,
        )  # (1, 3, 1, H, W)
        return sampled.squeeze(2).squeeze(0)  # (3, H, W)

    def _apply_1d(self, rgb: torch.Tensor) -> torch.Tensor:
        n = self.lut.size
        scaled = rgb * (n - 1)
        floor = scaled.floor().clamp_(0, n - 1).long()
        ceil = (floor + 1).clamp_(0, n - 1)
        frac = (scaled - floor.float()).clamp_(0.0, 1.0)

        out_channels = []
        for c in range(3):
            table_c = self._lut_1d[:, c]
            lo = table_c[floor[c]]
            hi = table_c[ceil[c]]
            out_channels.append(torch.lerp(lo, hi, frac[c]))
        return torch.stack(out_channels, dim=0)
