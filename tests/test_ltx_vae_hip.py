"""The ahead-of-time AMD decoder kernels (``na_hip``, ``diffusion_hip``) and their routing; no GPU needed."""

import pytest
import torch

from jasna.models.ltx_vae import diffusion_hip, layout_hip, na_hip
from jasna.models.ltx_vae import transformer as T
from jasna.models.ltx_vae.hip_launch import KERNEL_LAUNCH


def test_the_launch_table_has_every_kernel_the_launchers_launch():
    launched = {*map(na_hip.module_name, na_hip.WINDOWS), diffusion_hip.SWIGLU_MODULE,
                *map(diffusion_hip.qkv_module, diffusion_hip.QKV_DIMS), layout_hip.MODULE}

    assert all(set(kernels) == launched for kernels in KERNEL_LAUNCH.values())


def test_amd_without_triton_runs_the_ahead_of_time_kernels(monkeypatch):
    monkeypatch.setattr(T, "_fused_na3d", None)
    monkeypatch.setattr(T, "_fused_residual_swiglu", None)
    monkeypatch.setattr(T, "_fused_slab_qkv", None)
    monkeypatch.setattr(T, "has_triton_package", lambda: False)
    monkeypatch.setattr(T, "is_amd_device", lambda device: True)

    T.enable_fused_kernels(torch.device("cpu"))

    assert T._fused_na3d is na_hip.na3d
    assert T._fused_residual_swiglu is diffusion_hip.residual_swiglu_
    assert T._fused_slab_qkv is diffusion_hip.slab_qkv


def test_nvidia_never_takes_the_amd_kernels(monkeypatch):
    monkeypatch.setattr(T, "_fused_na3d", None)
    monkeypatch.setattr(T, "has_triton_package", lambda: False)
    monkeypatch.setattr(T, "is_amd_device", lambda device: False)

    T.enable_fused_kernels(torch.device("cpu"))

    assert T._fused_na3d is not na_hip.na3d


def test_the_diffusion_kernels_reject_inputs_they_were_not_built_for():
    vector = torch.zeros(256, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="contiguous bf16"):
        diffusion_hip.residual_swiglu_(torch.zeros(4, 256, dtype=torch.float16), vector, vector, vector,
                                       *(torch.zeros(1024, 256, dtype=torch.bfloat16),) * 2,
                                       torch.zeros(256, 1024, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="width 256"):
        diffusion_hip.residual_swiglu_(torch.zeros(4, 512, dtype=torch.bfloat16), *(vector,) * 6)
    x = torch.zeros(1, 2, 2, 12, 256, dtype=torch.bfloat16)
    columns = torch.arange(7)
    modulation = [vector] * 3
    weights = [torch.zeros(256, 256, dtype=torch.bfloat16), vector] * 3 + [torch.zeros(32, dtype=torch.bfloat16)] * 2
    with pytest.raises(ValueError, match="width 256"):
        diffusion_hip.slab_qkv(x, x[:, :, :, :5], columns, modulation, weights, torch.zeros(32), torch.zeros(12),
                               [8, 12], 2, 5, 0.125)
    weights[6:] = [torch.zeros(64, dtype=torch.bfloat16)] * 2
    with pytest.raises(ValueError, match="width 256"):
        diffusion_hip.slab_qkv(x, x[:, :, :, :5].contiguous(), columns, [], weights, torch.zeros(32),
                               torch.zeros(12), [8, 12], 2, 5, 0.125)


def test_rejects_inputs_the_kernel_was_not_built_for():
    q = torch.zeros(1, 4, 4, 4, 1, 32, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="head dim 64"):
        na_hip.na3d(q, q, q, [3, 3, 3], 0)
    with pytest.raises(ValueError, match="head dim 64"):
        na_hip.na3d(q.half(), q.half(), q.half(), [3, 3, 3], 0)


def test_the_channels_first_copy_takes_only_channels_last_fp16():
    with pytest.raises(ValueError, match="channels-last 3D fp16"):
        layout_hip.channels_first_bf16(torch.zeros(1, 4, 2, 3, 3, dtype=torch.float16))
    with pytest.raises(ValueError, match="channels-last 3D fp16"):
        layout_hip.channels_first_bf16(torch.zeros(1, 4, 2, 3, 3).contiguous(memory_format=torch.channels_last_3d))
