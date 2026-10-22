from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from factories import offload_bundle_targets

import jasna.media.cuda_kernel as cuda_kernel
from jasna.mosaic.rfdetr_fused_head import FusedSegmentationHead, fuse_segmentation_head, has_fused_head_kernels
from jasna.mosaic.rfdetr_hip_launch import KERNEL_LAUNCH

MEDIA = Path(cuda_kernel.__file__).resolve().parent
C = 256
SIZE = 144


def _use_rocm(monkeypatch, architecture: str) -> None:
    monkeypatch.setattr(torch.version, "hip", "7.15.26333")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(gcnArchName=f"{architecture}:sramecc-:xnack-"),
    )


def _weight(dtype: torch.dtype, device: str = "cuda:0") -> SimpleNamespace:
    return SimpleNamespace(dtype=dtype, device=torch.device(device))


@pytest.mark.parametrize("architecture", ["gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1200", "gfx1201"])
@pytest.mark.parametrize("feature_size", [144, 192])
def test_rdna3_and_rdna4_run_the_fused_head_in_fp16(monkeypatch, architecture, feature_size):
    _use_rocm(monkeypatch, architecture)

    assert has_fused_head_kernels(_weight(torch.float16), feature_size)
    assert not has_fused_head_kernels(_weight(torch.float32), feature_size)


def test_other_gpus_and_feature_sizes_keep_the_torch_head(monkeypatch):
    _use_rocm(monkeypatch, "gfx1201")
    assert not has_fused_head_kernels(_weight(torch.float16), 160)
    assert not has_fused_head_kernels(_weight(torch.float16, "cpu"), 144)

    _use_rocm(monkeypatch, "gfx1030")
    assert not has_fused_head_kernels(_weight(torch.float16), 144)

    monkeypatch.setattr(torch.version, "hip", None)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    assert not has_fused_head_kernels(_weight(torch.float16), 144)


@pytest.mark.parametrize("module_name", sorted(KERNEL_LAUNCH["gfx1201"]))
def test_every_architecture_in_the_launch_table_ships_its_code_object(module_name):
    targets = offload_bundle_targets(MEDIA / f"{module_name}.hipfb")

    assert {target.rpartition("--")[2] for target in targets if target.startswith("hipv4-")} == set(KERNEL_LAUNCH)
    assert all(set(kernels) == set(KERNEL_LAUNCH["gfx1201"]) for kernels in KERNEL_LAUNCH.values())


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwconv = nn.Conv2d(C, C, kernel_size=3, padding=1, groups=C)
        self.norm = nn.LayerNorm(C, eps=1e-6)
        self.pwconv1 = nn.Linear(C, C)
        self.gamma = None

    def forward(self, x):
        y = F.gelu(self.pwconv1(self.norm(self.dwconv(x).permute(0, 2, 3, 1))))
        return y.permute(0, 3, 1, 2) + x


class _Head(nn.Module):
    """rfdetr's SegmentationHead export forward, for the default layout."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.downsample_ratio = 4
        self.blocks = nn.ModuleList([_Block() for _ in range(2)])
        self.spatial_features_proj = nn.Conv2d(C, C, kernel_size=1)
        self.query_features_block = nn.Sequential(nn.LayerNorm(C), nn.Linear(C, C))
        self.query_features_proj = nn.Identity()
        self.bias = nn.Parameter(torch.full((1,), 0.25))

    def forward(self, spatial_features, query_features, image_size):
        target = (image_size[0] // self.downsample_ratio, image_size[1] // self.downsample_ratio)
        features = F.interpolate(spatial_features, size=target, mode="bilinear", align_corners=False)
        for block in self.blocks:
            features = block(features)
        queries = self.query_features_proj(self.query_features_block(query_features[0]))
        return [torch.einsum("bchw,bnc->bnhw", self.spatial_features_proj(features), queries) + self.bias]


def _reference_block_launch(self, grid, stream, x, dw_w, dw_b, ln_w, ln_b, pw_w, pw_b, out):
    assert x.is_contiguous(memory_format=torch.channels_last)
    assert grid == (self.grid[0], x.shape[0], 1)
    y = F.layer_norm(F.conv2d(x, dw_w, dw_b, padding=1, groups=C).permute(0, 2, 3, 1), (C,), ln_w, ln_b, 1e-6)
    out.copy_(F.gelu(F.linear(y, pw_w, pw_b)).permute(0, 3, 1, 2) + x)


def test_fused_head_feeds_the_kernel_each_block_and_folds_the_projection(monkeypatch):
    _use_rocm(monkeypatch, "gfx1201")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=0))
    monkeypatch.setattr(cuda_kernel.TritonKernel, "launch", _reference_block_launch)
    head = _Head().eval()
    generator = torch.Generator().manual_seed(1)
    spatial = torch.randn(2, C, SIZE // 4, SIZE // 4, generator=generator)
    queries = [torch.randn(2, 7, C, generator=generator)]

    with torch.no_grad():
        expected = head(spatial, queries, (SIZE * 4, SIZE * 4))[0]
        actual = FusedSegmentationHead(head, SIZE)(spatial, queries, (SIZE * 4, SIZE * 4))[0]

    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)


def test_fused_head_rejects_a_layout_it_was_not_built_for(monkeypatch):
    _use_rocm(monkeypatch, "gfx1201")
    head = _Head()
    head.blocks[0].gamma = nn.Parameter(torch.ones(C))

    with pytest.raises(ValueError, match="differs"):
        FusedSegmentationHead(head, SIZE)


def test_cpu_and_fp32_keep_rfdetrs_head():
    core = SimpleNamespace(segmentation_head=_Head())
    original = core.segmentation_head

    fuse_segmentation_head(core, SIZE * 4)

    assert core.segmentation_head is original


def _has_amd_fused_head() -> bool:
    return bool(torch.cuda.is_available() and torch.version.hip) and has_fused_head_kernels(
        torch.empty(0, dtype=torch.float16, device="cuda"), SIZE
    )


@pytest.mark.skipif(not _has_amd_fused_head(), reason="needs an AMD GPU with the fused RF-DETR head kernel")
@pytest.mark.parametrize("feature_size", [144, 192])
def test_fused_head_on_the_gpu_is_as_close_to_fp32_as_rfdetrs_fp16_head(feature_size):
    from rfdetr.models.heads.segmentation import SegmentationHead

    torch.manual_seed(0)
    head32 = SegmentationHead(C, 5).cuda().eval()
    head32.export()
    head16 = SegmentationHead(C, 5).cuda().eval()
    head16.load_state_dict(head32.state_dict())
    head16 = head16.half()
    head16.export()
    fused = FusedSegmentationHead(head16, feature_size)
    spatial = torch.randn(4, C, feature_size // 4, feature_size // 4, device="cuda").contiguous(
        memory_format=torch.channels_last
    )
    queries = torch.randn(4, 200, C, device="cuda")
    image = (feature_size * 4, feature_size * 4)

    with torch.no_grad():
        reference = head32(spatial, [queries], image)[0]
        torch_fp16 = head16(spatial.half(), [queries.half()], image)[0]
        fused_fp16 = fused(spatial.half(), [queries.half()], image)[0]

    rms = lambda tensor: tensor.float().pow(2).mean().sqrt().item()
    assert rms(fused_fp16 - reference) <= 1.2 * rms(torch_fp16 - reference)
    assert rms(fused_fp16 - reference) <= 1e-2 * rms(reference)
