# SPDX-FileCopyrightText: OpenMMLab. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND AGPL-3.0
# Code vendored from: https://github.com/open-mmlab/mmagic

import torch
import torch.nn.functional as F

# The identity grid only depends on (n, h, w, dtype, device, align_corners) -
# it never depends on the input data. flow_warp runs ~1400 times per
# BasicVSR++ forward at T=60 (every frame, both directions, second-order
# terms and SPyNet levels), and rebuilding eye+affine_grid each time was
# measured at ~25 % of the forward's device time (profiler, RX 7900 XT),
# so the grid is cached per key instead. Cached tensors are created under
# no_grad (plain tensors, safe in and out of inference_mode) and never
# mutated: callers only do grid + offset, which allocates a new tensor.
_GRID_CACHE: dict = {}


def _identity_grid(n, c, h, w, dtype, device, align_corners):
    key = (n, h, w, dtype, str(device), bool(align_corners))
    grid = _GRID_CACHE.get(key)
    if grid is None:
        with torch.no_grad():
            theta = torch.eye(2, 3, device=device, dtype=dtype).unsqueeze(0).expand(n, -1, -1)
            grid = F.affine_grid(theta, (n, c, h, w), align_corners=align_corners)
        _GRID_CACHE[key] = grid
    return grid


def flow_warp(x,
              flow,
              interpolation='bilinear',
              padding_mode='zeros',
              align_corners=True):
    """Warp an image or a feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2). The last dimension is
            a two-channel, denoting the width and height relative offsets.
            Note that the values are not normalized to [-1, 1].
        interpolation (str): Interpolation mode: 'nearest' or 'bilinear'.
            Default: 'bilinear'.
        padding_mode (str): Padding mode: 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Whether align corners. Default: True.

    Returns:
        Tensor: Warped image or feature map.
    """
    n, c, h, w = x.shape
    # Identity grid via affine_grid (has a native TRT converter, unlike
    # the arange+meshgrid approach which produces IR that TRT 2.10 can't
    # track through downstream cat operations).
    grid = _identity_grid(n, c, h, w, x.dtype, x.device, align_corners)

    # Convert pixel-space flow offsets to normalised [-1, 1] offsets
    flow_x = flow[..., 0] * (2.0 / max(w - 1, 1))
    flow_y = flow[..., 1] * (2.0 / max(h - 1, 1))
    grid_flow = grid + torch.stack((flow_x, flow_y), dim=-1)

    return F.grid_sample(
        x,
        grid_flow,
        mode=interpolation,
        padding_mode=padding_mode,
        align_corners=align_corners)
