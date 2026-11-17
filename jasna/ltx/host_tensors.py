"""Model tensors read into host memory, and zeroed in place once released."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn

TensorSpecs = dict[str, tuple[torch.dtype, tuple[int, ...]]]


def zero_tensors(tensors: Iterable[torch.Tensor], keep: nn.Module | None = None) -> None:
    """Zero ``tensors`` in place, except meta tensors and storage ``keep`` still holds; returns once
    the GPU has zeroed them."""
    kept = {(tensor.device, tensor.untyped_storage().data_ptr())
            for tensor in keep.state_dict().values()} if keep is not None else set()
    devices = set()
    with torch.no_grad():
        for tensor in tensors:
            if tensor.is_meta or (tensor.device, tensor.untyped_storage().data_ptr()) in kept:
                continue
            tensor.zero_()
            if tensor.device.type == "cuda":
                devices.add(tensor.device)
    for device in devices:
        torch.cuda.synchronize(device)


def read_host_tensors(reader, names: Iterable[str], specs: TensorSpecs) -> dict[str, torch.Tensor]:
    """The tensors ``names`` of ``reader`` (whose ``tensor_specs`` are ``specs``) in host memory;
    a failed read zeroes what it filled."""
    tensors = {name: torch.empty(specs[name][1], dtype=specs[name][0], device="cpu") for name in names}
    try:
        reader.read_tensors_into((tuple(tensors),), iter((tensors,)))
    except BaseException:
        zero_tensors(tensors.values())
        raise
    return tensors
