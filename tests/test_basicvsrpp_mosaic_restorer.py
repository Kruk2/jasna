import torch


class _CaptureIdentityModel:
    def __init__(self) -> None:
        self.captured_inputs: torch.Tensor | None = None

    def __call__(self, *, inputs: torch.Tensor) -> torch.Tensor:
        self.captured_inputs = inputs.detach().clone()
        return inputs


def _make_restorer(monkeypatch, model, *, use_tensorrt=False, fp16=False, config=None):
    import jasna.restorer.basicvsrpp_mosaic_restorer as br

    monkeypatch.setattr(br, "load_model", lambda config, checkpoint_path, device, fp16: model)
    return br.BasicvsrppMosaicRestorer(
        checkpoint_path="unused.pth",
        device=torch.device("cpu"),
        max_clip_size=30,
        use_tensorrt=use_tensorrt,
        fp16=fp16,
        config=config,
    )


def test_raw_process_normalizes_frames_to_unit_range(monkeypatch) -> None:
    model = _CaptureIdentityModel()
    restorer = _make_restorer(monkeypatch, model)

    frame = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)

    out = restorer.raw_process([frame])

    assert out.shape == (1, 3, 256, 256)
    assert model.captured_inputs.shape == (1, 1, 3, 256, 256)
    assert model.captured_inputs.dtype == torch.float32
    assert torch.equal(model.captured_inputs[0, 0], frame.to(torch.float32).div(255.0))

def test_raw_process_stacks_frames_into_one_clip(monkeypatch) -> None:
    model = _CaptureIdentityModel()
    restorer = _make_restorer(monkeypatch, model)

    frames = [torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8) for _ in range(2)]
    restorer.raw_process(frames)

    assert model.captured_inputs.shape[:2] == (1, 2)

def test_raw_process_empty_video_raises(monkeypatch) -> None:
    import pytest

    restorer = _make_restorer(monkeypatch, _CaptureIdentityModel())

    with pytest.raises(RuntimeError):
        restorer.raw_process([])

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
        use_tensorrt=False,
        fp16=True,
        config={"x": 1},
    )

    assert restorer.device.type == "cpu"
    assert restorer.input_dtype == torch.float16
    assert isinstance(restorer.model, _CaptureIdentityModel)

    assert captured["checkpoint_path"] == "ckpt.pth"
    assert captured["device"] == torch.device("cpu")
    assert captured["fp16"] is True
    assert captured["config"] == {"x": 1}


def test_raw_process_produces_contiguous_nchw_input(monkeypatch) -> None:
    import jasna.restorer.basicvsrpp_mosaic_restorer as br

    model = _CaptureIdentityModel()
    restorer = _make_restorer(monkeypatch, model)

    frames = [torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8) for _ in range(3)]
    restorer.raw_process(frames)

    assert model.captured_inputs is not None
    inp = model.captured_inputs.squeeze(0)
    assert inp.is_contiguous(), f"model input must be contiguous NCHW, got stride {inp.stride()}"


def test_split_forward_path_used_when_available(monkeypatch) -> None:
    import jasna.restorer.basicvsrpp_mosaic_restorer as br

    captured: list[torch.Tensor] = []

    class _FakeSplit:
        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            captured.append(x.detach().clone())
            return x

    model = _CaptureIdentityModel()
    restorer = _make_restorer(monkeypatch, model)
    restorer._split_forward = _FakeSplit()

    frame = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
    restorer.raw_process([frame])

    assert len(captured) == 1
    assert captured[0].shape == (1, 1, 3, 256, 256)
    assert model.captured_inputs is None


def test_amd_migraphx_b1_dispatch_is_used_and_closed(monkeypatch) -> None:
    from contextlib import contextmanager

    import jasna.restorer.basicvsrpp_migraphx_b1 as b1
    import jasna.restorer.basicvsrpp_mosaic_restorer as br

    entered = []

    class _FakeB1:
        directory = "verified-artifacts"

        @contextmanager
        def dispatch(self):
            entered.append("enter")
            try:
                yield
            finally:
                entered.append("exit")

        def close(self):
            entered.append("close")

    model = _CaptureIdentityModel()
    monkeypatch.setattr(
        br,
        "load_model",
        lambda config, checkpoint_path, device, fp16: model,
    )
    monkeypatch.setattr(b1, "basicvsrpp_migraphx_b1_enabled", lambda *a, **k: True)
    monkeypatch.setattr(b1, "load_basicvsrpp_b1_migraphx", lambda *a, **k: _FakeB1())
    restorer = br.BasicvsrppMosaicRestorer(
        checkpoint_path="unused.pth",
        device=torch.device("cpu"),
        max_clip_size=30,
        use_tensorrt=False,
        fp16=False,
    )

    frame = torch.zeros((3, 256, 256), dtype=torch.uint8)
    restorer.raw_process([frame])
    restorer.close()

    assert entered == ["enter", "exit", "close"]
