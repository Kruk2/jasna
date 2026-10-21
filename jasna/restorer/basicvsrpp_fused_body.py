"""BasicVSR++ propagation step on AMD with fused, precompiled kernels.

Drop-in for ``_PropagateBodyWrapper`` on the AMD split forward. The second-order deformable conv (offset/mask activation, both flows,
fp32 bilinear sampling, GEMM) and every residual-block conv (bias, ReLU and
residual add fused) run as kernels from ``scripts/build_basicvsrpp_hipfbs.py``;
the flow warps, ``conv_offset`` and the backbone input conv stay on MIOpen. The
kernels have no host sync or allocation of their own, so the body is
HIP-graph safe, but graphs are not used.
"""
from __future__ import annotations

import ctypes

import torch
import torch.nn as nn
import torch.nn.functional as F

from jasna.accelerator import amd_gpu_architecture, is_amd_device
from jasna.media.cuda_kernel import Kernel
from jasna.restorer.basicvsrpp_hip_launch import KERNEL_LAUNCH

CHANNELS = 64
DEFORM_GROUPS = 16
MAX_RESIDUE_MAGNITUDE = 10
LEAKY_SLOPE = 0.1

_CONV_RELU = "basicvsrpp_conv3x3_relu"
_CONV_RESIDUAL = "basicvsrpp_conv3x3_residual"
_DCN = "basicvsrpp_dcn"
_TRITON_SCRATCH_ARGS = 2


def has_fused_body_kernels(weight: torch.Tensor) -> bool:
    """Whether the propagation step owning ``weight`` can run the fused kernels."""
    return (
        weight.dtype == torch.float16
        and is_amd_device(weight.device)
        and amd_gpu_architecture(weight.device) in KERNEL_LAUNCH
    )


def pack_conv3x3_weight(weight: torch.Tensor) -> torch.Tensor:
    """``[out, in, 3, 3]`` → ``[9, in, out]``, the kernels' GEMM layout."""
    out_channels, in_channels = weight.shape[:2]
    return weight.permute(2, 3, 1, 0).reshape(9, in_channels, out_channels).contiguous()


class _LaunchedKernel:
    def __init__(self, module_name: str, architecture: str, pointer_count: int):
        function, self.grid, threads, shared = KERNEL_LAUNCH[architecture][module_name]
        self.block = (threads, 1, 1)
        self.scratch = (0,) * _TRITON_SCRATCH_ARGS
        self.kernel = Kernel(
            module_name, function, (ctypes.c_void_p,) * (pointer_count + _TRITON_SCRATCH_ARGS), shared,
        )

    def __call__(self, stream: int, *tensors: torch.Tensor) -> None:
        self.kernel.launch(self.grid, self.block, (*(t.data_ptr() for t in tensors), *self.scratch), stream)


class FusedPropagateBody(nn.Module):
    def __init__(self, deform_align: nn.Module, backbone: nn.Module):
        super().__init__()
        blocks = backbone.main[2]
        if (
            deform_align.deform_groups != DEFORM_GROUPS
            or deform_align.max_residue_magnitude != MAX_RESIDUE_MAGNITUDE
            or deform_align.out_channels != CHANNELS
            or backbone.main[1].negative_slope != LEAKY_SLOPE
            or any(block.res_scale != 1.0 for block in blocks)
        ):
            raise ValueError("BasicVSR++ layout differs from the one the fused kernels were built for")
        self.conv_offset = deform_align.conv_offset
        self.conv_in = backbone.main[0]
        self.dcn_weight = pack_conv3x3_weight(deform_align.weight)
        self.dcn_bias = deform_align.bias
        self._block_weights = [
            (pack_conv3x3_weight(b.conv1.weight), b.conv1.bias, pack_conv3x3_weight(b.conv2.weight), b.conv2.bias)
            for b in blocks
        ]
        architecture = amd_gpu_architecture(deform_align.weight.device)
        self._conv_relu = _LaunchedKernel(_CONV_RELU, architecture, 5)
        self._conv_residual = _LaunchedKernel(_CONV_RESIDUAL, architecture, 5)
        self._dcn = _LaunchedKernel(_DCN, architecture, 8)

    def forward(
        self,
        feat_prop: torch.Tensor,
        grid_n1: torch.Tensor,
        feat_n2: torch.Tensor,
        grid_n2: torch.Tensor,
        feat_current: torch.Tensor,
        flow_1: torch.Tensor,
        flow_2: torch.Tensor,
        backbone_prefix: torch.Tensor,
    ) -> torch.Tensor:
        stream = torch.cuda.current_stream(feat_prop.device).cuda_stream
        cond_n1 = F.grid_sample(feat_prop, grid_n1, mode="bilinear", padding_mode="zeros", align_corners=True)
        cond_n2 = F.grid_sample(feat_n2, grid_n2, mode="bilinear", padding_mode="zeros", align_corners=True)
        offset_and_mask = self.conv_offset(torch.cat([cond_n1, feat_current, cond_n2, flow_1, flow_2], dim=1))

        feat_prop_new = torch.empty_like(feat_prop)
        self._dcn(
            stream, feat_prop, feat_n2, offset_and_mask, flow_1, flow_2, self.dcn_weight, self.dcn_bias, feat_prop_new,
        )

        feat = F.leaky_relu_(self.conv_in(torch.cat([backbone_prefix, feat_prop_new], dim=1)), LEAKY_SLOPE)
        hidden = torch.empty_like(feat)
        for weight_1, bias_1, weight_2, bias_2 in self._block_weights:
            self._conv_relu(stream, feat, weight_1, bias_1, feat, hidden)
            out = torch.empty_like(feat)
            self._conv_residual(stream, hidden, weight_2, bias_2, feat, out)
            feat = out
        return feat_prop_new + feat
