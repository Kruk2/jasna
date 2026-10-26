"""The ahead-of-time AMD decoder kernels (``na_hip``, ``diffusion_hip``) and their routing; no GPU needed."""

import pytest
import torch

from jasna.accelerator import KernelBackend
from jasna.models.ltx_vae import diffusion_hip, layout_hip, na_hip
from jasna.models.ltx_vae import transformer as T
from jasna.models.ltx_vae.hip_launch import KERNEL_LAUNCH


def test_the_launch_table_has_every_kernel_the_launchers_launch():
    launched = {*map(na_hip.module_name, na_hip.WINDOWS), diffusion_hip.SWIGLU_MODULE,
                *map(diffusion_hip.qkv_module, T.FUSED_QKV_DIMS), layout_hip.MODULE}

    assert all(set(kernels) == launched for kernels in KERNEL_LAUNCH.values())


def _backend_kernels():
    from jasna.models.ltx_vae import diffusion_triton, na_triton

    return [
        (KernelBackend.NVIDIA, T.DecoderKernels(na3d=na_triton.na3d)),
        (KernelBackend.NVIDIA_PRE_SM80, T.DecoderKernels(na3d=T._fp16_eager_na3d)),
        (KernelBackend.AMD_TRITON,
         T.DecoderKernels(na_triton.na3d, diffusion_triton.residual_swiglu_, diffusion_triton.slab_qkv)),
        (KernelBackend.AMD_PREBUILT, T.DecoderKernels(na_hip.na3d, diffusion_hip.residual_swiglu_, diffusion_hip.slab_qkv)),
        (KernelBackend.CPU, T.DecoderKernels()),
    ]


@pytest.mark.parametrize(("backend", "expected"), _backend_kernels(), ids=lambda value: str(value)[:20])
def test_each_backend_runs_its_decoder_kernels(monkeypatch, backend, expected):
    monkeypatch.setattr(T, "kernel_backend", lambda device: backend)

    assert T.decoder_kernels(torch.device("cuda")) == expected


def test_a_model_runs_the_kernels_it_was_given_and_others_keep_the_torch_code():
    kernels = T.DecoderKernels(na3d=na_hip.na3d, residual_swiglu=diffusion_hip.residual_swiglu_)
    block = T.ChunkedDiffusionNABlock(64, (3, 5, 5), context_channels=16, head_dim=32)

    T.use_kernels(block, kernels)

    assert block.attn.kernels is kernels and block.mlp.kernels is kernels
    assert T.ChunkedDiffusionNABlock(64, (3, 5, 5), context_channels=16, head_dim=32).attn.kernels == T.DecoderKernels()


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
