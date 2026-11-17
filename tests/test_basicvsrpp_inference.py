import pytest
import torch

from jasna.models.basicvsrpp.basicvsrpp_gan import BasicVSRPlusPlusGanNet
from jasna.models.basicvsrpp.inference import inference_generator_state, load_model


def _weights(value: float) -> dict[str, torch.Tensor]:
    return {"conv.weight": torch.full((1,), value)}


def _prefixed(prefix: str, weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {prefix + key: value for key, value in weights.items()}


@pytest.mark.parametrize(
    ("checkpoint", "expected"),
    [
        ({**_prefixed("generator.", _weights(1.0)), **_prefixed("generator_ema.", _weights(2.0))}, 2.0),
        (_prefixed("generator.", _weights(1.0)), 1.0),
        ({"state_dict": _prefixed("generator_ema.", _weights(2.0)), "meta": {}}, 2.0),
        (_prefixed("module.generator_ema.", _weights(2.0)), 2.0),
    ],
    ids=["ema-preferred", "no-ema", "nested-state-dict", "data-parallel"],
)
def test_inference_runs_the_ema_generator_else_the_trained_one(checkpoint, expected) -> None:
    state = inference_generator_state(checkpoint)
    assert list(state) == ["conv.weight"]
    assert state["conv.weight"].item() == expected


def test_a_checkpoint_without_generator_weights_is_refused() -> None:
    with pytest.raises(ValueError, match="no BasicVSR\\+\\+ generator weights"):
        inference_generator_state({"discriminator.conv.weight": torch.zeros(1)})


def test_load_model_returns_the_checkpoints_ema_generator(tmp_path) -> None:
    torch.manual_seed(0)
    trained = BasicVSRPlusPlusGanNet(mid_channels=64, num_blocks=15, spynet_pretrained=None).state_dict()
    path = tmp_path / "model.pth"
    torch.save(_prefixed("generator_ema.", trained), path)

    generator = load_model(str(path), torch.device("cpu"), fp16=True)

    loaded = generator.state_dict()
    assert not generator.training
    assert all(torch.equal(loaded[key], value.half()) for key, value in trained.items())
