import pytest
import torch
from torch import nn

import jasna.restorer.basicvsrpp_mosaic_restorer as br
import jasna.restorer.basicvsrpp_sub_engines as sub_engines
from jasna.models.basicvsrpp.mmagic.basicvsr_plusplus_net import BasicVSRPlusPlusNet


class _CaptureIdentityModel:
    def __init__(self) -> None:
        self.captured_inputs: torch.Tensor | None = None

    def __call__(self, *, inputs: torch.Tensor) -> torch.Tensor:
        self.captured_inputs = inputs.detach().clone()
        return inputs


def _make_restorer(monkeypatch, model):
    monkeypatch.setattr(br, "load_model", lambda config, checkpoint_path, device, fp16: model)
    return br.BasicvsrppMosaicRestorer(
        checkpoint_path="unused.pth",
        device=torch.device("cpu"),
        max_clip_size=30,
        use_tensorrt=False,
        fp16=False,
        config=None,
    )


def test_raw_process_feeds_the_model_one_contiguous_clip_in_unit_range(monkeypatch) -> None:
    model = _CaptureIdentityModel()
    restorer = _make_restorer(monkeypatch, model)
    frames = [torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8) for _ in range(2)]

    out = restorer.raw_process(frames)

    assert out.shape == (2, 3, 256, 256)
    clip = model.captured_inputs
    assert clip.shape == (1, 2, 3, 256, 256) and clip.dtype == torch.float32 and clip.is_contiguous()
    assert torch.equal(clip[0], torch.stack(frames).float().div(255.0))


def test_raw_process_empty_video_raises(monkeypatch) -> None:
    restorer = _make_restorer(monkeypatch, _CaptureIdentityModel())

    with pytest.raises(RuntimeError):
        restorer.raw_process([])


def test_amd_restores_through_the_eager_split_forward_when_graphs_are_off(monkeypatch) -> None:
    clips = []

    class _Split:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            clips.append(x.detach().clone())
            return x

    model = _CaptureIdentityModel()
    monkeypatch.setenv(br.AMD_GRAPHS_ENV, "0")
    monkeypatch.setattr(br, "is_amd_device", lambda device: True)
    monkeypatch.setattr(sub_engines, "create_eager_split_forward", lambda generator: _Split())
    restorer = _make_restorer(monkeypatch, model)

    restorer.raw_process([torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)])

    assert [clip.shape for clip in clips] == [(1, 1, 3, 256, 256)]
    assert model.captured_inputs is None


class _Gan(nn.Module):
    def __init__(self, generator: nn.Module) -> None:
        super().__init__()
        self.generator = generator


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_amd_graph_replay_matches_the_eager_forward(monkeypatch) -> None:
    torch.manual_seed(0)
    gan = _Gan(BasicVSRPlusPlusNet(mid_channels=16, num_blocks=1, spynet_pretrained=None)).cuda().eval()
    monkeypatch.setattr(br, "is_amd_device", lambda device: True)
    monkeypatch.setattr(br, "load_model", lambda config, checkpoint_path, device, fp16: gan)

    def restorer():
        return br.BasicvsrppMosaicRestorer(
            checkpoint_path="unused.pth", device=torch.device("cuda"), max_clip_size=6, use_tensorrt=False, fp16=False,
        )

    graphed = restorer()
    monkeypatch.setenv(br.AMD_GRAPHS_ENV, "0")
    eager = restorer()

    for length in (6, 6, 4):
        frames = [torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8, device="cuda") for _ in range(length)]
        torch.testing.assert_close(graphed.raw_process(frames), eager.raw_process(frames), atol=1e-2, rtol=0)
