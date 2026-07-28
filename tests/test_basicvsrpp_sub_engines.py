from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from jasna.restorer.basicvsrpp_sub_engines import (
    DIRECTIONS,
    FEATURE_SIZE,
    BasicVSRPlusPlusNetSplit,
    _PreprocessWrapper,
    _PropagateBodyWrapper,
    _SPyNetWrapper,
    _UpsampleWrapper,
    _get_inference_generator,
    _sub_engine_dir,
    all_sub_engines_exist,
    get_sub_engine_paths,
    load_sub_engines,
)


def test_sub_engine_dir_uses_stem() -> None:
    path = r"C:\weights\model_v1.2.pth"
    result = _sub_engine_dir(path)
    assert "model_v1.2_sub_engines" in result
    assert result.startswith(r"C:\weights")


def test_get_sub_engine_paths_returns_6_paths() -> None:
    paths = get_sub_engine_paths("model_weights/model.pth", fp16=True)
    assert len(paths) == 6
    for d in DIRECTIONS:
        assert f"loop_body_{d}" in paths
    assert "preprocess" in paths
    assert "upsample" in paths
    for p in paths.values():
        assert p.endswith(".engine")
        assert "fp16" in p


def test_get_sub_engine_paths_fp32() -> None:
    paths = get_sub_engine_paths("model.pth", fp16=False)
    for p in paths.values():
        assert "fp32" in p


def test_all_sub_engines_exist_false_when_missing(tmp_path: Path) -> None:
    model_path = str(tmp_path / "model.pth")
    assert not all_sub_engines_exist(model_path, fp16=True)


def test_all_sub_engines_exist_true_when_all_present(tmp_path: Path) -> None:
    model_path = str(tmp_path / "model.pth")
    paths = get_sub_engine_paths(model_path, fp16=True)
    for p in paths.values():
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text("x", encoding="utf-8")
    assert all_sub_engines_exist(model_path, fp16=True)


def test_all_sub_engines_exist_false_when_partial(tmp_path: Path) -> None:
    model_path = str(tmp_path / "model.pth")
    paths = get_sub_engine_paths(model_path, fp16=True)
    items = list(paths.values())
    for p in items[:-1]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text("x", encoding="utf-8")
    assert not all_sub_engines_exist(model_path, fp16=True)


def test_load_sub_engines_returns_none_when_missing(tmp_path: Path) -> None:
    result = load_sub_engines(str(tmp_path / "model.pth"), torch.device("cpu"), fp16=True)
    assert result is None


def _make_engine_files(model_path: str) -> dict[str, str]:
    paths = get_sub_engine_paths(model_path, fp16=True)
    for p in paths.values():
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text("x", encoding="utf-8")
    return paths


def test_warmup_capture_loop_body_graphs_toggles_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from jasna.restorer.basicvsrpp_sub_engines import _warmup_capture_loop_body_graphs

    fake_trt = MagicMock()
    monkeypatch.setitem(sys.modules, "torch_tensorrt", fake_trt)
    engines = {d: MagicMock(return_value=torch.zeros(1)) for d in DIRECTIONS}
    _warmup_capture_loop_body_graphs(engines, 16, torch.device("cpu"), torch.float32)
    for i, d in enumerate(DIRECTIONS):
        assert engines[d].call_count == 2
        prefix = engines[d].call_args.args[7]
        assert prefix.shape == (1, (1 + i) * 16, FEATURE_SIZE, FEATURE_SIZE)
    modes = [c.args[0] for c in fake_trt.runtime.set_cudagraphs_mode.call_args_list]
    assert modes == [True, False]


def test_create_split_forward_cudagraphs_env_gates_warmup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jasna.restorer import basicvsrpp_sub_engines as mod

    model_path = str(tmp_path / "model.pth")
    _make_engine_files(model_path)
    monkeypatch.setattr(
        mod, "load_torchtrt_export",
        lambda *, checkpoint_path, device: MagicMock(),
    )
    warmups: list[int] = []
    monkeypatch.setattr(
        mod, "_warmup_capture_loop_body_graphs",
        lambda *a, **k: warmups.append(1),
    )
    generator = MagicMock(mid_channels=16)
    model = MagicMock(generator_ema=generator)

    monkeypatch.setenv("JASNA_TRT_CUDAGRAPHS", "0")
    split = mod.create_split_forward(model, model_path, torch.device("cpu"), fp16=True)
    assert split is not None
    assert not warmups
    assert split._loop_body_cudagraphs is False

    monkeypatch.delenv("JASNA_TRT_CUDAGRAPHS", raising=False)
    split = mod.create_split_forward(model, model_path, torch.device("cpu"), fp16=True)
    assert split is not None
    assert warmups == [1]
    assert split._loop_body_cudagraphs is True


def test_propagate_body_wrapper_forward_shape() -> None:
    from jasna.models.basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet

    torch.manual_seed(0)
    net = BasicVSRPlusPlusNet(mid_channels=16, num_blocks=2, spynet_pretrained=None)
    net.eval()
    mid = net.mid_channels
    d = "backward_1"
    wrapper = _PropagateBodyWrapper(net.deform_align[d], net.backbone[d])
    fp = torch.randn(1, mid, FEATURE_SIZE, FEATURE_SIZE)
    g1 = torch.randn(1, FEATURE_SIZE, FEATURE_SIZE, 2)
    fn2 = torch.randn(1, mid, FEATURE_SIZE, FEATURE_SIZE)
    g2 = torch.randn(1, FEATURE_SIZE, FEATURE_SIZE, 2)
    fc = torch.randn(1, mid, FEATURE_SIZE, FEATURE_SIZE)
    f1 = torch.randn(1, 2, FEATURE_SIZE, FEATURE_SIZE)
    f2 = torch.randn(1, 2, FEATURE_SIZE, FEATURE_SIZE)
    bp = torch.randn(1, mid, FEATURE_SIZE, FEATURE_SIZE)
    with torch.inference_mode():
        out = wrapper(fp, g1, fn2, g2, fc, f1, f2, bp)
    assert out.shape == (1, mid, FEATURE_SIZE, FEATURE_SIZE)


def test_upsample_wrapper_forward_shape() -> None:
    from jasna.models.basicvsrpp.mmagic.basicvsr_plusplus_net import (
        PixelShufflePack,
        ResidualBlocksWithInputConv,
    )

    reconstruction = ResidualBlocksWithInputConv(320, 64, 2)
    upsample1 = PixelShufflePack(64, 64, 2, upsample_kernel=3)
    upsample2 = PixelShufflePack(64, 64, 2, upsample_kernel=3)
    conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
    conv_last = nn.Conv2d(64, 3, 3, 1, 1)

    wrapper = _UpsampleWrapper(reconstruction, upsample1, upsample2, conv_hr, conv_last)
    x = torch.randn(1, 320, FEATURE_SIZE, FEATURE_SIZE)
    out = wrapper(x)
    assert out.shape == (1, 3, FEATURE_SIZE * 4, FEATURE_SIZE * 4)


def test_get_inference_generator_prefers_ema() -> None:
    model = MagicMock()
    model.generator_ema = MagicMock(spec=nn.Module)
    model.generator = MagicMock(spec=nn.Module)
    assert _get_inference_generator(model) is model.generator_ema


def test_get_inference_generator_falls_back_to_generator() -> None:
    model = MagicMock()
    model.generator_ema = None
    model.generator = MagicMock(spec=nn.Module)
    assert _get_inference_generator(model) is model.generator


def test_spynet_wrapper_matches_original() -> None:
    from jasna.models.basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet

    torch.manual_seed(42)
    net = BasicVSRPlusPlusNet(mid_channels=16, num_blocks=2, spynet_pretrained=None)
    net.eval()

    wrapper = _SPyNetWrapper(net.spynet)
    wrapper.eval()

    ref_img = torch.randn(4, 3, 64, 64)
    supp_img = torch.randn(4, 3, 64, 64)

    with torch.inference_mode():
        orig = net.spynet(ref_img, supp_img)
        wrap = wrapper(ref_img, supp_img)

    assert orig.shape == wrap.shape
    assert torch.allclose(orig, wrap, atol=1e-5, rtol=1e-5), \
        f"max diff: {(orig - wrap).abs().max().item()}"


def _build_split_from_net(net):
    """Build a BasicVSRPlusPlusNetSplit using PyTorch modules as engines."""

    class _UpsamplePassthrough(nn.Module):
        def __init__(self, parent):
            super().__init__()
            self.parent = parent

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.parent.reconstruction(x)
            x = self.parent.lrelu(self.parent.upsample1(x))
            x = self.parent.lrelu(self.parent.upsample2(x))
            x = self.parent.lrelu(self.parent.conv_hr(x))
            return self.parent.conv_last(x)

    return BasicVSRPlusPlusNetSplit(
        net,
        {d: _PropagateBodyWrapper(net.deform_align[d], net.backbone[d]) for d in DIRECTIONS},
        _PreprocessWrapper(net.feat_extract, net.spynet),
        _UpsamplePassthrough(net),
    )


@pytest.mark.parametrize("T", [1, 2, 3, 60, 71])
def test_split_forward_matches_pytorch_forward(T: int) -> None:
    """Verify that BasicVSRPlusPlusNetSplit produces the same output as the
    original BasicVSRPlusPlusNet.  T=1 and T=2 exercise the short-clip
    padding path (preprocess engine min batch = 3)."""
    from jasna.models.basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet

    torch.manual_seed(42)
    net = BasicVSRPlusPlusNet(mid_channels=16, num_blocks=2, spynet_pretrained=None)
    net.eval()

    split = _build_split_from_net(net)
    split.eval()

    torch.manual_seed(7)
    lqs = torch.randn(1, T, 3, 256, 256)

    with torch.inference_mode():
        ref = net(lqs)
        out = split(lqs)

    assert ref.shape == out.shape
    assert torch.allclose(ref, out, atol=1e-5, rtol=1e-5), \
        f"T={T} max diff: {(ref - out).abs().max().item()}"


def test_upsample_runs_in_fixed_size_batches() -> None:
    """The upsample stage is per-frame, so it is called in UPSAMPLE_BATCH-sized
    batches; a b180 TensorRT engine would reserve ~3.2 GB more scratch than b30
    for the same work."""
    from jasna.restorer.basicvsrpp_sub_engines import UPSAMPLE_BATCH

    batch_sizes: list[int] = []

    class _RecordingUpsample(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_sizes.append(x.shape[0])
            return torch.zeros(x.shape[0], 3, 8, 8)

    split = BasicVSRPlusPlusNetSplit.__new__(BasicVSRPlusPlusNetSplit)
    nn.Module.__init__(split)
    split._upsample_engine = _RecordingUpsample()

    t = 2 * UPSAMPLE_BATCH + 11
    lqs = torch.randn(1, t, 3, 8, 8)
    feats = {
        "spatial": [torch.randn(1, 4, 2, 2) for _ in range(t)],
        "forward_1": [torch.randn(1, 4, 2, 2) for _ in range(t)],
    }

    out = split.upsample(lqs, feats)

    assert batch_sizes == [UPSAMPLE_BATCH, UPSAMPLE_BATCH, 11]
    assert out.shape == lqs.shape
    assert torch.equal(out, lqs)


def test_preprocess_runs_in_overlapping_batches() -> None:
    """Batches overlap by one frame so every consecutive pair still gets a flow,
    and each batch stays at the engine's minimum size."""
    from jasna.restorer.basicvsrpp_sub_engines import PREPROCESS_BATCH

    calls: list[tuple[int, int]] = []

    class _RecordingPreprocess(nn.Module):
        def forward(self, x: torch.Tensor):
            n = int(x[0, 0, 0, 0].item())
            calls.append((n, x.shape[0]))
            frames = torch.arange(n, n + x.shape[0], dtype=torch.float32)
            feats = frames.view(-1, 1, 1, 1)
            flows = frames[:-1].view(-1, 1, 1, 1)
            return feats, flows, flows.clone()

    split = BasicVSRPlusPlusNetSplit.__new__(BasicVSRPlusPlusNetSplit)
    nn.Module.__init__(split)
    split._preprocess_engine = _RecordingPreprocess()

    t = 2 * PREPROCESS_BATCH + 1
    lqs_flat = torch.arange(t, dtype=torch.float32).view(t, 1, 1, 1).expand(t, 3, 1, 1)

    feats, flows_fwd, flows_bwd = split._preprocess(lqs_flat)

    assert [start for start, _ in calls] == [0, PREPROCESS_BATCH - 1, t - 3]
    assert all(size >= split._PREPROCESS_MIN_BATCH for _, size in calls)
    assert torch.equal(feats.flatten(), torch.arange(t, dtype=torch.float32))
    assert torch.equal(flows_fwd.flatten(), torch.arange(t - 1, dtype=torch.float32))
    assert torch.equal(flows_bwd, flows_fwd)


def test_preprocess_single_batch_calls_engine_once() -> None:
    from jasna.restorer.basicvsrpp_sub_engines import PREPROCESS_BATCH

    engine = MagicMock(return_value=("f", "fwd", "bwd"))
    split = BasicVSRPlusPlusNetSplit.__new__(BasicVSRPlusPlusNetSplit)
    nn.Module.__init__(split)
    split._preprocess_engine = engine

    lqs_flat = torch.zeros(PREPROCESS_BATCH, 3, 4, 4)
    assert split._preprocess(lqs_flat) == ("f", "fwd", "bwd")
    assert engine.call_count == 1
