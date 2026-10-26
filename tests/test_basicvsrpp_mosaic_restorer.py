import pytest
import torch

import jasna.restorer.basicvsrpp_mosaic_restorer as br
import jasna.restorer.basicvsrpp_sub_engines as sub_engines


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


def test_amd_restores_through_the_eager_split_forward(monkeypatch) -> None:
    clips = []

    class _Split:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            clips.append(x.detach().clone())
            return x

    model = _CaptureIdentityModel()
    monkeypatch.setattr(br, "is_amd_device", lambda device: True)
    monkeypatch.setattr(sub_engines, "create_eager_split_forward", lambda generator: _Split())
    restorer = _make_restorer(monkeypatch, model)

    restorer.raw_process([torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)])

    assert [clip.shape for clip in clips] == [(1, 1, 3, 256, 256)]
    assert model.captured_inputs is None
