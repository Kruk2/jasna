"""
GPU fisheye remap — half-equirectangular SBS -> equidistant fisheye SBS (VR180).

Operates on the in-pipeline decoded frame tensors produced by
NvidiaVideoReader.frames(): (N, 3, H, W) uint8, RGB planar, on CUDA.

The remap geometry is static, so a normalized [-1, 1] sampling grid is built once
and reused with torch.nn.functional.grid_sample (same pattern as the GPU color-LUT
applier in jasna/media/lut.py). Bilinear sampling matches ffmpeg v360 interp=line.

Grid math verified against ffmpeg v360
(input=hequirect:output=fisheye:in_stereo=sbs:out_stereo=sbs:ih_fov=180:iv_fov=180).
The grid is kept in float32 — fp16 grid coordinates would round to ~2 px error at
8K — so frames are sampled in float32 and returned as uint8.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class FisheyeRemapper:
    def __init__(self, width: int, height: int, device: torch.device,
                 fov_deg: float = 180.0) -> None:
        self.width = int(width)
        self.height = int(height)
        self.device = device
        self._grid = self._build_grid(self.width, self.height, fov_deg).to(device)

    @staticmethod
    def _build_grid(W: int, H: int, fov_deg: float) -> torch.Tensor:
        half_fov = math.radians(fov_deg) * 0.5  # 90 deg at the circle edge
        ys = (torch.arange(H, dtype=torch.float64) + 0.5) / H
        xs = (torch.arange(W, dtype=torch.float64) + 0.5) / W
        oy, ox = torch.meshgrid(ys, xs, indexing="ij")           # (H, W)

        left = ox < 0.5
        eu = torch.where(left, ox * 2.0, (ox - 0.5) * 2.0)        # eye-local x [0,1]
        ev = oy                                                    # eye-local y [0,1]

        fx = eu * 2.0 - 1.0                                        # centered fisheye [-1,1]
        fy = ev * 2.0 - 1.0
        r = torch.sqrt(fx * fx + fy * fy)

        theta = r * half_fov                                       # equidistant: angle ~ radius
        phi = torch.atan2(fy, fx)
        sinT = torch.sin(theta)
        X = sinT * torch.cos(phi)                                  # right
        Y = sinT * torch.sin(phi)                                  # down
        Z = torch.cos(theta)                                       # forward

        lon = torch.atan2(X, Z)                                    # [-pi/2, pi/2]
        lat = torch.asin(torch.clamp(Y, -1.0, 1.0))
        su = lon / math.pi + 0.5                                   # eye-local source x [0,1]
        sv = lat / math.pi + 0.5                                   # eye-local source y [0,1]

        sx = torch.where(left, su * 0.5, 0.5 + su * 0.5)           # full SBS source x [0,1]
        sy = sv

        gx = sx * 2.0 - 1.0                                        # -> grid_sample [-1,1]
        gy = sy * 2.0 - 1.0

        # outside the fisheye circle: sample out of range -> padding_mode='zeros' -> black
        outside = r > 1.0
        gx = torch.where(outside, torch.full_like(gx, 2.0), gx)
        gy = torch.where(outside, torch.full_like(gy, 2.0), gy)

        return torch.stack((gx, gy), dim=-1).unsqueeze(0).to(torch.float32)  # (1,H,W,2)

    @torch.inference_mode()
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (3,H,W) or (N,3,H,W) uint8 CUDA RGB -> same shape/dtype, fisheye-remapped.

        Sampled one frame at a time: at 8K an fp32 grid_sample transient is ~0.8 GB
        per frame, so a whole batch at once would spike VRAM on a tool that runs
        restoration concurrently. The per-frame loop caps the transient; grid_sample
        is fast enough that the extra launches are negligible.
        """
        single = frames.ndim == 3
        if single:
            frames = frames.unsqueeze(0)
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"expected (3,H,W) or (N,3,H,W), got {tuple(frames.shape)}")
        if frames.shape[-2:] != (self.height, self.width):
            raise ValueError(
                f"frame size {tuple(frames.shape[-2:])} != grid {(self.height, self.width)}")
        out = torch.empty_like(frames)
        for i in range(frames.shape[0]):
            sampled = F.grid_sample(frames[i:i + 1].float(), self._grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=False)
            out[i] = sampled[0].round_().clamp_(0, 255).to(torch.uint8)
        return out[0] if single else out


class InverseFisheyeRemapper:
    """Equidistant fisheye SBS -> half-equirectangular SBS (VR180): the inverse of
    FisheyeRemapper. Used to reproject the restored frame back into the source
    projection so the exported file matches the original VR180 (the "reproject to
    source" option).

    This is a full-frame round trip (every pixel resampled once more). Measured
    fidelity on 8K is high: SSIM ~0.996 / PSNR ~42 dB overall, ~0.998-0.999 in the
    central viewing area; loss is concentrated at the periphery where fisheye and
    equirect sampling densities differ most.

    NOTE (option 2, not implemented): a marginally higher-quality variant would keep
    the untouched background pixel-perfect by inverse-remapping only the restored
    regions (via the restoration mask) and compositing them onto the original equirect
    frame, instead of round-tripping the whole frame. The gain over this approach is
    small (background SSIM ~0.996 -> ~1.0) and was deemed not worth the extra plumbing
    for now. If pursued, do it at the crop level in blend_buffer rather than here.

    Bilinear sampling, matching the forward remap (the bicubic gain is ~0.0003 SSIM and
    risks a thin cross-eye seam artifact, so it is not worth it).
    """

    def __init__(self, width: int, height: int, device: torch.device,
                 fov_deg: float = 180.0) -> None:
        self.width = int(width)
        self.height = int(height)
        self.device = device
        self._grid = self._build_grid(self.width, self.height, fov_deg).to(device)

    @staticmethod
    def _build_grid(W: int, H: int, fov_deg: float) -> torch.Tensor:
        half_fov = math.radians(fov_deg) * 0.5
        ys = (torch.arange(H, dtype=torch.float64) + 0.5) / H
        xs = (torch.arange(W, dtype=torch.float64) + 0.5) / W
        oy, ox = torch.meshgrid(ys, xs, indexing="ij")            # output equirect px

        left = ox < 0.5
        su = torch.where(left, ox * 2.0, (ox - 0.5) * 2.0)        # eye-local equirect x [0,1]
        sv = oy
        lon = (su - 0.5) * math.pi                                # [-pi/2, pi/2]
        lat = (sv - 0.5) * math.pi
        X = torch.cos(lat) * torch.sin(lon)
        Y = torch.sin(lat)
        Z = torch.cos(lat) * torch.cos(lon)

        theta = torch.acos(torch.clamp(Z, -1.0, 1.0))            # angle from forward
        phi = torch.atan2(Y, X)
        r = theta / half_fov                                      # equidistant: radius ~ angle
        fx = r * torch.cos(phi)
        fy = r * torch.sin(phi)
        eu = (fx + 1.0) * 0.5                                      # eye-local fisheye [0,1]
        ev = (fy + 1.0) * 0.5

        sx = torch.where(left, eu * 0.5, 0.5 + eu * 0.5)          # full SBS fisheye x [0,1]
        gx = sx * 2.0 - 1.0
        gy = ev * 2.0 - 1.0
        return torch.stack((gx, gy), dim=-1).unsqueeze(0).to(torch.float32)  # (1,H,W,2)

    @torch.inference_mode()
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """Accepts (3,H,W) or (N,3,H,W) uint8 CUDA RGB; returns the same shape."""
        single = frames.ndim == 3
        if single:
            frames = frames.unsqueeze(0)
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError(f"expected (3,H,W) or (N,3,H,W), got {tuple(frames.shape)}")
        if frames.shape[-2:] != (self.height, self.width):
            raise ValueError(
                f"frame size {tuple(frames.shape[-2:])} != grid {(self.height, self.width)}")
        out = torch.empty_like(frames)
        for i in range(frames.shape[0]):
            sampled = F.grid_sample(frames[i:i + 1].float(), self._grid, mode="bilinear",
                                    padding_mode="zeros", align_corners=False)
            out[i] = sampled[0].round_().clamp_(0, 255).to(torch.uint8)
        return out[0] if single else out
