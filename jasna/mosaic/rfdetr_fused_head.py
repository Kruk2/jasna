"""RF-DETR segmentation head on AMD with the fused, precompiled block kernel.

Drop-in for the export form of rfdetr's ``SegmentationHead``. Each
``DepthwiseConvBlock`` (depthwise conv, LayerNorm, Linear, GELU, residual: six
kernels and two permute copies in torch) is one kernel from
``scripts/build_rfdetr_hipfbs.py`` on channels-last features. The 1x1
``spatial_features_proj`` is folded into the query side, so the mask logits
are one ``baddbmm`` instead of a conv plus an einsum.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from jasna.accelerator import amd_gpu_architecture, is_amd_device
from jasna.media.cuda_kernel import TritonKernel
from jasna.mosaic.rfdetr_hip_launch import KERNEL_LAUNCH

CHANNELS = 256
LAYER_NORM_EPS = 1e-6


def _module_name(feature_size: int) -> str:
    return f"rfdetr_dwblock_{feature_size}"


def has_fused_head_kernels(weight: torch.Tensor, feature_size: int) -> bool:
    """Whether a head owning ``weight`` with ``feature_size``² mask features can run the fused kernel."""
    return (
        weight.dtype == torch.float16
        and is_amd_device(weight.device)
        and _module_name(feature_size) in KERNEL_LAUNCH.get(amd_gpu_architecture(weight.device), {})
    )


def fuse_segmentation_head(core: nn.Module, resolution: int) -> None:
    """Swap ``core``'s segmentation head for the fused one when this GPU has its kernel."""
    head = core.segmentation_head
    feature_size = resolution // head.downsample_ratio
    if has_fused_head_kernels(head.bias, feature_size):
        core.segmentation_head = FusedSegmentationHead(head, feature_size)


class FusedSegmentationHead(nn.Module):
    def __init__(self, head: nn.Module, feature_size: int):
        super().__init__()
        projection = head.spatial_features_proj
        if (
            any(
                block.gamma is not None
                or block.dwconv.weight.shape != (CHANNELS, 1, 3, 3)
                or block.norm.eps != LAYER_NORM_EPS
                for block in head.blocks
            )
            or not isinstance(projection, nn.Conv2d)
            or projection.weight.shape != (CHANNELS, CHANNELS, 1, 1)
        ):
            raise ValueError("RF-DETR segmentation head differs from the one the fused kernel was built for")
        self.downsample_ratio = head.downsample_ratio
        self.feature_size = feature_size
        self.blocks = head.blocks
        self.query_features_block = head.query_features_block
        self.query_features_proj = head.query_features_proj
        self.projection_weight = projection.weight.view(CHANNELS, CHANNELS)
        self.projection_bias = projection.bias
        self.bias = head.bias
        module_name = _module_name(feature_size)
        self._block = TritonKernel(module_name, KERNEL_LAUNCH[amd_gpu_architecture(head.bias.device)][module_name], 8)

    def forward(
        self, spatial_features: torch.Tensor, query_features: list[torch.Tensor], image_size: tuple[int, int]
    ) -> list[torch.Tensor]:
        target_size = (image_size[0] // self.downsample_ratio, image_size[1] // self.downsample_ratio)
        if target_size != (self.feature_size, self.feature_size):
            raise ValueError(f"fused segmentation head was built for {self.feature_size}² features, got {target_size}")
        features = F.interpolate(spatial_features, size=target_size, mode="bilinear", align_corners=False)
        features = features.contiguous(memory_format=torch.channels_last)
        spare = torch.empty_like(features)
        stream = torch.cuda.current_stream(features.device).cuda_stream
        grid = (self._block.grid[0], features.shape[0], 1)
        for block in self.blocks:
            self._block.launch(
                grid, stream,
                features, block.dwconv.weight, block.dwconv.bias, block.norm.weight, block.norm.bias,
                block.pwconv1.weight, block.pwconv1.bias, spare,
            )
            features, spare = spare, features

        queries = self.query_features_proj(self.query_features_block(query_features[0]))
        batch, channels, height, width = features.shape
        pixels = features.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        logits = torch.baddbmm(
            (queries @ self.projection_bias + self.bias).unsqueeze(-1),
            queries @ self.projection_weight,
            pixels.transpose(1, 2),
        )
        return [logits.view(batch, -1, height, width)]
