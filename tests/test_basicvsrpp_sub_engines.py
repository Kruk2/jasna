from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from jasna.models.basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet
from jasna.restorer.basicvsrpp_sub_engines import (
    EAGER_UPSAMPLE_BATCH,
    UPSAMPLE_BATCH,
    BasicVSRPlusPlusNetSplit,
    _SPyNetWrapper,
    _get_inference_generator,
    create_eager_split_forward,
    load_sub_engines,
)
from jasna.engine_paths import (
    all_basicvsrpp_sub_engines_exist as all_sub_engines_exist,
    basicvsrpp_sub_engine_dir as _sub_engine_dir,
    get_basicvsrpp_sub_engine_paths as get_sub_engine_paths,
)


def test_sub_engine_dir_uses_stem() -> None:
    path = r"C:\weights\model_v1.2.pth"
    result = _sub_engine_dir(path)
    assert "model_v1.2_sub_engines" in result
    assert result.startswith(r"C:\weights")


def _make_engine_files(model_path: str) -> dict[str, str]:
    paths = get_sub_engine_paths(model_path, fp16=True)
    for p in paths.values():
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        Path(p).write_text("x", encoding="utf-8")
    return paths


def test_sub_engines_exist_only_when_all_six_of_this_precision_do(tmp_path: Path) -> None:
    model_path = str(tmp_path / "model.pth")
    assert not all_sub_engines_exist(model_path, fp16=True)
    paths = list(_make_engine_files(model_path).values())
    assert len(paths) == 6 and all_sub_engines_exist(model_path, fp16=True)
    assert not all_sub_engines_exist(model_path, fp16=False)
    Path(paths[-1]).unlink()
    assert not all_sub_engines_exist(model_path, fp16=True)


def test_load_sub_engines_returns_none_when_missing(tmp_path: Path) -> None:
    result = load_sub_engines(str(tmp_path / "model.pth"), torch.device("cpu"), fp16=True)
    assert result is None


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
    assert split._upsample_batch == UPSAMPLE_BATCH


def test_inference_uses_the_ema_generator_when_there_is_one() -> None:
    ema, generator = nn.Identity(), nn.Linear(1, 1)
    assert _get_inference_generator(SimpleNamespace(generator_ema=ema, generator=generator)) is ema
    assert _get_inference_generator(SimpleNamespace(generator_ema=None, generator=generator)) is generator


def test_spynet_wrapper_matches_original() -> None:

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
    return create_eager_split_forward(MagicMock(generator_ema=net))


def _split(*, preprocess: nn.Module | None = None, upsample: nn.Module | None = None) -> BasicVSRPlusPlusNetSplit:
    generator = BasicVSRPlusPlusNet(mid_channels=16, num_blocks=1, spynet_pretrained=None)
    return BasicVSRPlusPlusNetSplit(generator, {}, preprocess, upsample, UPSAMPLE_BATCH, False)


@pytest.mark.parametrize("T", [1, 2, 3, pytest.param(60, marks=pytest.mark.slow), pytest.param(71, marks=pytest.mark.slow)])
def test_split_forward_matches_pytorch_forward(T: int) -> None:
    """Verify that BasicVSRPlusPlusNetSplit produces the same output as the
    original BasicVSRPlusPlusNet.  T=1 and T=2 exercise the short-clip
    padding path (preprocess engine min batch = 3)."""

    torch.manual_seed(42)
    net = BasicVSRPlusPlusNet(mid_channels=16, num_blocks=2, spynet_pretrained=None)
    net.eval()

    split = _build_split_from_net(net)

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

    batch_sizes: list[int] = []

    class _RecordingUpsample(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_sizes.append(x.shape[0])
            return torch.zeros(x.shape[0], 3, 8, 8)

    split = _split(upsample=_RecordingUpsample())

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

    split = _split(preprocess=_RecordingPreprocess())

    t = 2 * PREPROCESS_BATCH + 1
    lqs_flat = torch.arange(t, dtype=torch.float32).view(t, 1, 1, 1).expand(t, 3, 1, 1)

    feats, flows_fwd, flows_bwd = split._preprocess(lqs_flat)

    assert [start for start, _ in calls] == [0, PREPROCESS_BATCH - 1, t - 3]
    assert all(size >= split._PREPROCESS_MIN_BATCH for _, size in calls)
    assert torch.equal(feats.flatten(), torch.arange(t, dtype=torch.float32))
    assert torch.equal(flows_fwd.flatten(), torch.arange(t - 1, dtype=torch.float32))
    assert torch.equal(flows_bwd, flows_fwd)


def test_eager_split_forward_never_touches_torch_tensorrt(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys


    monkeypatch.setitem(sys.modules, "torch_tensorrt", None)
    monkeypatch.delenv("JASNA_TRT_CUDAGRAPHS", raising=False)
    net = BasicVSRPlusPlusNet(mid_channels=16, num_blocks=2, spynet_pretrained=None).eval()
    split = create_eager_split_forward(MagicMock(generator_ema=net))

    with torch.inference_mode():
        out = split(torch.randn(1, 4, 3, 256, 256))

    assert out.shape == (1, 4, 3, 256, 256)
    assert split._upsample_batch == EAGER_UPSAMPLE_BATCH
