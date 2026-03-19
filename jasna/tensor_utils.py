from __future__ import annotations

import numpy as np
import torch


def to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if tensor.device == device:
        return tensor
    if tensor.device.type == "cpu" and not tensor.is_contiguous():
        tensor = torch.from_numpy(np.ascontiguousarray(tensor.numpy()))
    out = torch.empty(tensor.shape, dtype=tensor.dtype, device=device)
    out.copy_(tensor, non_blocking=True)
    return out


def pad_batch_with_last(x: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    n = int(x.shape[0])
    bs = int(batch_size)
    if n == bs:
        return x
    pad = x[-1:].expand(bs - n, *x.shape[1:])
    return torch.cat([x, pad], dim=0)

