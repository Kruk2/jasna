import torch
import torch.nn.functional as F


class _CaptureIdentityModel:
    def __init__(self) -> None:
        self.captured_inputs: torch.Tensor | None = None

    def __call__(self, *, inputs: torch.Tensor) -> torch.Tensor:
        self.captured_inputs = inputs.detach().clone()
        return inputs


def test_restore_identity_returns_resized_rgb_uint8(monkeypatch) -> None:
    import jasna.restorer.basicvsrpp_mosaic_restorer as br

    model = _CaptureIdentityModel()
    monkeypatch.setattr(br, "load_model", lambda config, checkpoint_path, device, fp16: model)

    restorer = br.BasicvsrppMosaicRestorer(
        checkpoint_path="unused.pth",
        device=torch.device("cpu"),
        max_clip_size=30,
        use_tensorrt=True,
        fp16=False,
        config=None,
    )

    frame = torch.tensor(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=torch.uint8,
    )  # (H, W, C) RGB

    out = restorer.restore([frame])
    assert len(out) == 1
    assert out[0].shape == (br.INFERENCE_SIZE, br.INFERENCE_SIZE, 3)
    assert out[0].dtype == torch.uint8

    expected = (
        F.interpolate(
            frame.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32).div(255.0),
            size=(br.INFERENCE_SIZE, br.INFERENCE_SIZE),
            mode="bilinear",
            align_corners=False,
        )
        .mul(255.0)
        .round()
        .clamp(0, 255)
        .to(dtype=torch.uint8)
        .squeeze(0)
        .permute(1, 2, 0)
    )
    assert torch.equal(out[0], expected)

    assert model.captured_inputs is not None
    assert model.captured_inputs.shape == (1, 1, 3, br.INFERENCE_SIZE, br.INFERENCE_SIZE)
    assert model.captured_inputs.dtype == torch.float32
    assert model.captured_inputs.device.type == "cpu"

    expected_inputs = F.interpolate(
        frame.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32).div(255.0),
        size=(br.INFERENCE_SIZE, br.INFERENCE_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    assert torch.equal(model.captured_inputs[0], expected_inputs)


def test_restore_multiple_frames_varied_sizes(monkeypatch) -> None:
    import jasna.restorer.basicvsrpp_mosaic_restorer as br

    model = _CaptureIdentityModel()
    monkeypatch.setattr(br, "load_model", lambda config, checkpoint_path, device, fp16: model)

    restorer = br.BasicvsrppMosaicRestorer(
        checkpoint_path="unused.pth",
        device=torch.device("cpu"),
        max_clip_size=30,
        use_tensorrt=True,
        fp16=False,
        config=None,
    )

    f0 = torch.randint(0, 256, (7, 11, 3), dtype=torch.uint8)
    f1 = torch.randint(0, 256, (19, 5, 3), dtype=torch.uint8)
    out = restorer.restore([f0, f1])

    assert len(out) == 2
    assert out[0].shape == (br.INFERENCE_SIZE, br.INFERENCE_SIZE, 3)
    assert out[1].shape == (br.INFERENCE_SIZE, br.INFERENCE_SIZE, 3)

    assert model.captured_inputs is not None
    assert model.captured_inputs.shape[:2] == (1, 2)


def test_restore_empty_video_raises(monkeypatch) -> None:
    import pytest
    import jasna.restorer.basicvsrpp_mosaic_restorer as br

    monkeypatch.setattr(br, "load_model", lambda config, checkpoint_path, device, fp16: _CaptureIdentityModel())
    restorer = br.BasicvsrppMosaicRestorer(
        checkpoint_path="unused.pth",
        device=torch.device("cpu"),
        max_clip_size=30,
        use_tensorrt=True,
        fp16=False,
        config=None,
    )

    with pytest.raises(RuntimeError):
        restorer.restore([])


def test_init_sets_device_dtype_and_loads_model(monkeypatch) -> None:
    import jasna.restorer.basicvsrpp_mosaic_restorer as br

    captured: dict[str, object] = {}

    def fake_load_model(config, checkpoint_path, device, fp16):
        captured["config"] = config
        captured["checkpoint_path"] = checkpoint_path
        captured["device"] = device
        captured["fp16"] = fp16
        return _CaptureIdentityModel()

    monkeypatch.setattr(br, "load_model", fake_load_model)

    restorer = br.BasicvsrppMosaicRestorer(
        checkpoint_path="ckpt.pth",
        device=torch.device("cpu"),
        max_clip_size=30,
        use_tensorrt=True,
        fp16=True,
        config={"x": 1},
    )

    assert restorer.device.type == "cpu"
    assert restorer.dtype == torch.float16
    assert isinstance(restorer.model, _CaptureIdentityModel)

    assert captured["checkpoint_path"] == "ckpt.pth"
    assert captured["device"] == torch.device("cpu")
    assert captured["fp16"] is True
    assert captured["config"] == {"x": 1}


def test_restore_fails_on_invalid_frame_rank(monkeypatch) -> None:
    import pytest
    import jasna.restorer.basicvsrpp_mosaic_restorer as br

    monkeypatch.setattr(br, "load_model", lambda config, checkpoint_path, device, fp16: _CaptureIdentityModel())
    restorer = br.BasicvsrppMosaicRestorer(
        checkpoint_path="unused.pth",
        device=torch.device("cpu"),
        max_clip_size=30,
        use_tensorrt=True,
        fp16=False,
        config=None,
    )

    frame_hw = torch.zeros((10, 10), dtype=torch.uint8)
    with pytest.raises(RuntimeError):
        restorer.restore([frame_hw])


def test_engine_padding_uses_mirror_repeat_sequence(monkeypatch) -> None:
    import jasna.restorer.basicvsrpp_mosaic_restorer as br

    class _CaptureEngine:
        def __init__(self) -> None:
            self.captured_inputs: torch.Tensor | None = None

        def __call__(self, *, inputs: torch.Tensor) -> torch.Tensor:
            self.captured_inputs = inputs.detach().clone()
            return inputs

    monkeypatch.setattr(br, "load_model", lambda config, checkpoint_path, device, fp16: _CaptureIdentityModel())

    restorer = br.BasicvsrppMosaicRestorer(
        checkpoint_path="unused.pth",
        device=torch.device("cpu"),
        max_clip_size=5,
        use_tensorrt=True,
        fp16=False,
        config=None,
    )

    engine = _CaptureEngine()
    restorer._engine_main = engine  # type: ignore[attr-defined]
    restorer._engine_main_len = 5  # type: ignore[attr-defined]

    x = torch.tensor([[[10, 0, 0]]], dtype=torch.uint8)  # (H=1, W=1, C=3)
    y = torch.tensor([[[20, 0, 0]]], dtype=torch.uint8)
    z = torch.tensor([[[30, 0, 0]]], dtype=torch.uint8)

    restorer.restore([x, y, z])
    assert engine.captured_inputs is not None
    assert engine.captured_inputs.shape[:2] == (1, 5)

    # Expected pad indices for [X,Y,Z] to length 5:
    # X Y Z Y X
    got = engine.captured_inputs[0, :, 0, 0, 0].cpu()  # red channel, first pixel
    expected = torch.tensor([10, 20, 30, 20, 10], dtype=torch.float32) / 255.0
    assert torch.allclose(got, expected, atol=0, rtol=0)
