from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn.functional as F
import torchvision
from factories import offload_bundle_targets

import jasna.media.cuda_kernel as cuda_kernel
import jasna.restorer.basicvsrpp_fused_body as fused_body
import jasna.restorer.basicvsrpp_sub_engines as sub_engines
from jasna.models.basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet
from jasna.restorer.basicvsrpp_fused_body import FusedPropagateBody, has_fused_body_kernels
from jasna.restorer.basicvsrpp_hip_launch import KERNEL_LAUNCH
from jasna.restorer.basicvsrpp_sub_engines import _PropagateBodyWrapper

MEDIA = Path(cuda_kernel.__file__).resolve().parent
FS = 64


def _use_rocm(monkeypatch, architecture: str) -> None:
    monkeypatch.setattr(torch.version, "hip", "7.15.26333")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(gcnArchName=f"{architecture}:sramecc-:xnack-"),
    )


def _weight(dtype: torch.dtype, device: str = "cuda:0") -> SimpleNamespace:
    return SimpleNamespace(dtype=dtype, device=torch.device(device))


def _net(mid_channels: int = 64) -> BasicVSRPlusPlusNet:
    torch.manual_seed(0)
    return BasicVSRPlusPlusNet(mid_channels=mid_channels, num_blocks=2, spynet_pretrained=None).eval()


def _body_inputs(device: str, dtype: torch.dtype, seed: int = 1) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    make = lambda *shape, scale=1.0: (torch.randn(*shape, generator=generator) * scale).to(device, dtype)
    grid = lambda: (torch.rand(1, FS, FS, 2, generator=generator) * 2.2 - 1.1).to(device, dtype)
    return [
        make(1, 64, FS, FS), grid(), make(1, 64, FS, FS), grid(), make(1, 64, FS, FS),
        make(1, 2, FS, FS, scale=3.0), make(1, 2, FS, FS, scale=3.0), make(1, 256, FS, FS),
    ]


@pytest.mark.parametrize("architecture", ["gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1200", "gfx1201"])
def test_rdna3_and_rdna4_run_the_fused_kernels_in_fp16(monkeypatch, architecture):
    _use_rocm(monkeypatch, architecture)

    assert has_fused_body_kernels(_weight(torch.float16))
    assert not has_fused_body_kernels(_weight(torch.float32))


def test_other_gpus_keep_the_eager_body(monkeypatch):
    _use_rocm(monkeypatch, "gfx1030")
    assert not has_fused_body_kernels(_weight(torch.float16))
    assert not has_fused_body_kernels(_weight(torch.float16, "cpu"))

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    assert not has_fused_body_kernels(_weight(torch.float16))


@pytest.mark.parametrize("module_name", sorted(KERNEL_LAUNCH["gfx1201"]))
def test_every_architecture_in_the_launch_table_ships_its_code_object(module_name):
    targets = offload_bundle_targets(MEDIA / f"{module_name}.hipfb")

    assert {target.rpartition("--")[2] for target in targets if target.startswith("hipv4-")} == set(KERNEL_LAUNCH)
    assert all(set(kernels) == set(KERNEL_LAUNCH["gfx1201"]) for kernels in KERNEL_LAUNCH.values())


def test_eager_split_uses_the_fused_body_when_the_kernels_match(monkeypatch):
    _use_rocm(monkeypatch, "gfx1201")
    monkeypatch.setattr(sub_engines, "has_fused_body_kernels", lambda weight: True)

    split = sub_engines.create_eager_split_forward(MagicMock(generator_ema=_net()))

    assert all(isinstance(body, FusedPropagateBody) for body in split._loop_body_engines.values())


def test_fused_body_rejects_a_layout_it_was_not_built_for(monkeypatch):
    _use_rocm(monkeypatch, "gfx1201")
    net = _net(mid_channels=16)

    with pytest.raises(ValueError, match="layout"):
        FusedPropagateBody(net.deform_align["forward_1"], net.backbone["forward_1"])


def _unpack(weight: torch.Tensor) -> torch.Tensor:
    _, in_channels, out_channels = weight.shape
    return weight.reshape(3, 3, in_channels, out_channels).permute(3, 2, 0, 1)


def _reference_launch(self, grid, stream, *tensors):
    if self.module_name == "basicvsrpp_dcn":
        xa, xb, om, flow_1, flow_2, weight, bias, out = tensors
        o1, o2, mask = torch.chunk(om, 3, dim=1)
        offset_1, offset_2 = torch.chunk(fused_body.MAX_RESIDUE_MAGNITUDE * torch.tanh(torch.cat((o1, o2), 1)), 2, 1)
        repeat = fused_body.DEFORM_GROUPS * 9 // 2
        offset = torch.cat(
            [offset_1 + flow_1.flip(1).repeat(1, repeat, 1, 1), offset_2 + flow_2.flip(1).repeat(1, repeat, 1, 1)], 1,
        )
        out.copy_(torchvision.ops.deform_conv2d(
            torch.cat([xa, xb], 1), offset, _unpack(weight), bias, 1, 1, 1, torch.sigmoid(mask),
        ))
        return
    x, weight, bias, residual, out = tensors
    conv = F.conv2d(x, _unpack(weight), bias, padding=1)
    out.copy_(F.relu(conv) if self.module_name == "basicvsrpp_conv3x3_relu" else residual + conv)


def test_fused_body_feeds_each_kernel_the_tensors_of_the_eager_body(monkeypatch):
    _use_rocm(monkeypatch, "gfx1201")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=0))
    monkeypatch.setattr(cuda_kernel.TritonKernel, "launch", _reference_launch)
    net = _net()
    inputs = _body_inputs("cpu", torch.float32)

    with torch.inference_mode():
        expected = _PropagateBodyWrapper(net.deform_align["forward_2"], net.backbone["forward_2"])(*inputs)
        actual = FusedPropagateBody(net.deform_align["forward_2"], net.backbone["forward_2"])(*inputs)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def _has_amd_fused_kernels() -> bool:
    return bool(torch.cuda.is_available() and torch.version.hip) and has_fused_body_kernels(
        torch.empty(0, dtype=torch.float16, device="cuda")
    )


@pytest.mark.skipif(not _has_amd_fused_kernels(), reason="needs an AMD GPU with the fused BasicVSR++ kernels")
def test_fused_body_on_the_gpu_is_at_least_as_close_to_fp32_as_the_eager_fp16_body():
    net32 = _net().cuda()
    net16 = _net().cuda().half()
    inputs = _body_inputs("cuda", torch.float16)
    direction = "forward_2"

    with torch.inference_mode():
        reference = _PropagateBodyWrapper(net32.deform_align[direction], net32.backbone[direction])(
            *[tensor.float() for tensor in inputs]
        )
        eager = _PropagateBodyWrapper(net16.deform_align[direction], net16.backbone[direction])(*inputs)
        fused = FusedPropagateBody(net16.deform_align[direction], net16.backbone[direction])(*inputs)

    rms = lambda tensor: tensor.float().pow(2).mean().sqrt().item()
    assert rms(fused - reference) <= 1.2 * rms(eager - reference)
    assert rms(fused - reference) <= 1e-2 * rms(reference)
