import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import pytest

from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel, compile_rfdetr_engine


class TestRfDetrPostprocess:
    def test_basic_single_detection(self):
        B, Q, C = 1, 4, 1
        pred_boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2],
                                     [0.1, 0.1, 0.1, 0.1],
                                     [0.9, 0.9, 0.1, 0.1],
                                     [0.0, 0.0, 0.0, 0.0]]])
        pred_logits = torch.tensor([[[5.0], [-5.0], [-5.0], [-5.0]]])
        pred_masks = torch.ones((B, Q, 8, 8))

        boxes_list, masks_list = RfDetrMosaicDetectionModel._postprocess(
            pred_boxes=pred_boxes,
            pred_logits=pred_logits,
            pred_masks=pred_masks,
            target_hw=(100, 200),
            score_threshold=0.5,
            max_select=4,
        )

        assert len(boxes_list) == 1
        assert len(masks_list) == 1
        assert boxes_list[0].shape[0] == 1
        assert boxes_list[0].shape[1] == 4
        box = boxes_list[0][0]
        assert box[0] == pytest.approx(0.4 * 200, abs=1)
        assert box[1] == pytest.approx(0.4 * 100, abs=1)

    def test_no_detections_above_threshold(self):
        B, Q, C = 1, 4, 1
        pred_boxes = torch.rand((B, Q, 4))
        pred_logits = torch.full((B, Q, C), -10.0)
        pred_masks = torch.ones((B, Q, 8, 8))

        boxes_list, masks_list = RfDetrMosaicDetectionModel._postprocess(
            pred_boxes=pred_boxes,
            pred_logits=pred_logits,
            pred_masks=pred_masks,
            target_hw=(100, 100),
            score_threshold=0.5,
            max_select=4,
        )

        assert len(boxes_list) == 1
        assert boxes_list[0].shape[0] == 0
        assert masks_list[0].shape[0] == 0

    def test_batch_of_two(self):
        B, Q, C = 2, 4, 1
        pred_boxes = torch.tensor([
            [[0.5, 0.5, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            [[0.3, 0.3, 0.1, 0.1], [0.7, 0.7, 0.1, 0.1], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ])
        pred_logits = torch.tensor([
            [[5.0], [-10.0], [-10.0], [-10.0]],
            [[5.0], [5.0], [-10.0], [-10.0]],
        ])
        pred_masks = torch.ones((B, Q, 8, 8))

        boxes_list, masks_list = RfDetrMosaicDetectionModel._postprocess(
            pred_boxes=pred_boxes,
            pred_logits=pred_logits,
            pred_masks=pred_masks,
            target_hw=(100, 100),
            score_threshold=0.5,
            max_select=4,
        )

        assert len(boxes_list) == 2
        assert boxes_list[0].shape[0] == 1
        assert boxes_list[1].shape[0] == 2

    def test_max_select_limits_detections(self):
        B, Q, C = 1, 10, 1
        pred_boxes = torch.rand((B, Q, 4)) * 0.5 + 0.25
        pred_logits = torch.full((B, Q, C), 5.0)
        pred_masks = torch.ones((B, Q, 8, 8))

        boxes_list, masks_list = RfDetrMosaicDetectionModel._postprocess(
            pred_boxes=pred_boxes,
            pred_logits=pred_logits,
            pred_masks=pred_masks,
            target_hw=(100, 100),
            score_threshold=0.5,
            max_select=3,
        )

        assert boxes_list[0].shape[0] <= 3

    def test_masks_are_boolean(self):
        B, Q, C = 1, 2, 1
        pred_boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0]]])
        pred_logits = torch.tensor([[[5.0], [-10.0]]])
        pred_masks = torch.randn((B, Q, 8, 8))

        _, masks_list = RfDetrMosaicDetectionModel._postprocess(
            pred_boxes=pred_boxes,
            pred_logits=pred_logits,
            pred_masks=pred_masks,
            target_hw=(100, 100),
            score_threshold=0.5,
            max_select=2,
        )

        assert masks_list[0].dtype == torch.bool

    def test_boxes_scaled_to_target_hw(self):
        B, Q, C = 1, 1, 1
        pred_boxes = torch.tensor([[[0.5, 0.5, 1.0, 1.0]]])
        pred_logits = torch.tensor([[[10.0]]])
        pred_masks = torch.ones((B, Q, 8, 8))

        boxes_list, _ = RfDetrMosaicDetectionModel._postprocess(
            pred_boxes=pred_boxes,
            pred_logits=pred_logits,
            pred_masks=pred_masks,
            target_hw=(480, 640),
            score_threshold=0.1,
            max_select=1,
        )

        box = boxes_list[0][0]
        assert box[0] == pytest.approx(0.0, abs=1)
        assert box[1] == pytest.approx(0.0, abs=1)
        assert box[2] == pytest.approx(640.0, abs=1)
        assert box[3] == pytest.approx(480.0, abs=1)


def _boolean_mask_postprocess(*, pred_boxes, pred_logits, pred_masks, target_hw, score_threshold, max_select):
    b, q, c = pred_logits.shape
    k = min(max_select, q)
    topk_values, topk_indexes = torch.topk(pred_logits.sigmoid().view(b, -1), k, dim=1)
    topk_boxes = topk_indexes // c
    x_c, y_c, w, h = pred_boxes.unbind(-1)
    boxes = torch.stack((x_c - 0.5 * w, y_c - 0.5 * h, x_c + 0.5 * w, y_c + 0.5 * h), dim=-1)
    boxes = boxes.gather(1, topk_boxes.unsqueeze(-1).expand(b, k, 4)) * pred_boxes.new_tensor(
        (target_hw[1], target_hw[0], target_hw[1], target_hw[0])
    )
    hm, wm = pred_masks.shape[-2:]
    masks = pred_masks.gather(1, topk_boxes[:, :, None, None].expand(b, k, hm, wm)) > 0.0
    valid = topk_values > score_threshold
    boxes_cpu = boxes.float().cpu().numpy()
    valid_cpu = valid.cpu().numpy()
    return [boxes_cpu[i][valid_cpu[i]] for i in range(b)], [masks[i][valid[i]] for i in range(b)]


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _DEVICES)
def test_postprocess_matches_boolean_mask_selection(device):
    generator = torch.Generator().manual_seed(3)
    b, q, c = 5, 200, 2
    logits = torch.randn(b, q, c, generator=generator) * 3
    logits[1] = -20.0
    logits[2, :40] = torch.linspace(1.0, 5.0, 80).view(40, 2)
    kwargs = dict(
        pred_boxes=torch.rand(b, q, 4, generator=generator).to(device),
        pred_logits=logits.to(device),
        pred_masks=torch.randn(b, q, 16, 16, generator=generator).half().to(device),
        target_hw=(1080, 1920),
        score_threshold=0.35,
        max_select=16,
    )

    boxes, masks = RfDetrMosaicDetectionModel._postprocess(**kwargs)
    expected_boxes, expected_masks = _boolean_mask_postprocess(**kwargs)

    assert [len(frame) for frame in boxes] == [len(frame) for frame in expected_boxes]
    assert len(boxes[1]) == 0 and len(boxes[2]) == 16
    for got, want in zip(boxes, expected_boxes):
        np.testing.assert_array_equal(got, want)
    for got, want in zip(masks, expected_masks):
        assert torch.equal(got, want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_postprocess_waits_for_the_gpu_once():
    b, q = 4, 200
    kwargs = dict(
        pred_boxes=torch.rand(b, q, 4, device="cuda"),
        pred_logits=torch.randn(b, q, 2, device="cuda"),
        pred_masks=torch.randn(b, q, 16, 16, device="cuda"),
        target_hw=(1080, 1920),
        score_threshold=0.35,
        max_select=16,
    )
    RfDetrMosaicDetectionModel._postprocess(**kwargs)
    torch.cuda.synchronize()

    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            RfDetrMosaicDetectionModel._postprocess(**kwargs)
    finally:
        torch.cuda.set_sync_debug_mode("default")

    assert len([w for w in caught if "called a synchronizing" in str(w.message)]) == 1


def _build_rfdetr_model():
    mock_runner = MagicMock()
    mock_runner.input_dtype = torch.float16
    mock_runner.output_names = ["pred_boxes", "pred_logits", "pred_masks"]
    mock_runner.outputs = {
        "pred_boxes": torch.zeros(1, 100, 4),
        "pred_logits": torch.zeros(1, 100, 1),
        "pred_masks": torch.zeros(1, 100, 8, 8),
    }

    engine_path = MagicMock()
    engine_path.exists.return_value = True

    with (
        patch("jasna.mosaic.rfdetr.is_nvidia_device", return_value=True),
        patch("jasna.mosaic.rfdetr.get_onnx_tensorrt_engine_path", return_value=engine_path),
        patch("jasna.mosaic.rfdetr.TrtRunner", return_value=mock_runner),
    ):
        model = RfDetrMosaicDetectionModel(
            weights_path=Path("model.onnx"),
            batch_size=2,
            device=torch.device("cpu"),
            resolution=768,
            dynamic_batch=True,
            fp16=False,
        )
    return model, mock_runner


class TestRfDetrInit:
    def test_basic_init(self):
        model, runner = _build_rfdetr_model()
        assert model.batch_size == 2
        assert model.resolution == 768
        assert model.boxes_out == "pred_boxes"
        assert model.logits_out == "pred_logits"
        assert model.masks_out == "pred_masks"

    def test_amd_init_uses_torch_runner(self, monkeypatch, tmp_path):
        import jasna.mosaic.rfdetr as module
        import jasna.mosaic.rfdetr_torch_runner as torch_runner_module

        weights_path = tmp_path / "rfdetr-v6.pt"
        runner = MagicMock()
        runner.input_names = ["input"]
        runner.input_dtypes = {"input": torch.float32}
        runner.output_names = ["dets", "labels", "masks"]
        runner.outputs = {
            "dets": MagicMock(ndim=3, shape=(1, 100, 4)),
            "labels": MagicMock(ndim=3, shape=(1, 100, 3)),
            "masks": MagicMock(ndim=4, shape=(1, 100, 8, 8)),
        }
        constructed = MagicMock(return_value=runner)
        monkeypatch.setattr(module, "is_amd_device", lambda _device: True)
        monkeypatch.setattr(module, "is_nvidia_device", lambda _device: False)
        monkeypatch.setattr(torch_runner_module, "RfDetrTorchRunner", constructed)

        model = RfDetrMosaicDetectionModel(
            weights_path=weights_path,
            batch_size=1,
            device=torch.device("cpu"),
            resolution=576,
            dynamic_batch=True,
            torch_variant="medium",
            fp16=True,
        )

        assert model.engine_path == weights_path
        assert constructed.call_args.kwargs["variant"] == "medium"

    def test_amd_init_requires_torch_variant(self, monkeypatch, tmp_path):
        import jasna.mosaic.rfdetr as module

        monkeypatch.setattr(module, "is_amd_device", lambda _device: True)
        with pytest.raises(RuntimeError, match="torch variant"):
            RfDetrMosaicDetectionModel(
                weights_path=tmp_path / "rfdetr-v6.pt",
                batch_size=1,
                device=torch.device("cpu"),
                resolution=576,
                dynamic_batch=True,
                fp16=True,
            )


class TestRfDetrPreprocess:
    def test_output_shape_and_dtype(self):
        model, _ = _build_rfdetr_model()
        model.input_dtype = torch.float32
        frames = torch.randint(0, 256, (2, 3, 100, 200), dtype=torch.uint8)
        out = model._preprocess(frames)
        assert out.shape == (2, 3, 768, 768)
        assert out.dtype == torch.float32


class TestRfDetrCall:
    def test_call_returns_detections(self):
        model, mock_runner = _build_rfdetr_model()
        model.input_dtype = torch.float32

        pred_boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]] + [[0.0, 0.0, 0.0, 0.0]] * 99] * 2)
        pred_logits = torch.tensor([[[5.0]] + [[-10.0]] * 99] * 2)
        pred_masks = torch.ones(2, 100, 8, 8)

        mock_runner.infer.return_value = {
            "pred_boxes": pred_boxes,
            "pred_logits": pred_logits,
            "pred_masks": pred_masks,
        }

        frames = torch.randint(0, 256, (2, 3, 100, 200), dtype=torch.uint8)
        det = model(frames, target_hw=(480, 640))

        assert len(det.boxes_xyxy) == 2
        assert len(det.masks) == 2
        assert det.boxes_xyxy[0].shape[0] == 1
        mock_runner.infer.assert_called_once()

    def test_fixed_batch_model_pads_and_chunks(self):
        model, mock_runner = _build_rfdetr_model()
        model.dynamic_batch = False
        model.input_dtype = torch.float32
        shared_outputs = {
            "pred_boxes": torch.zeros(2, 1, 4),
            "pred_logits": torch.zeros(2, 1, 1),
            "pred_masks": torch.zeros(2, 1, 8, 8),
        }
        call_count = 0

        def infer(inputs):
            nonlocal call_count
            call_count += 1
            batch = next(iter(inputs.values()))
            assert batch.shape[0] == 2
            for output in shared_outputs.values():
                output.fill_(call_count)
            return shared_outputs

        mock_runner.infer.side_effect = infer
        frames = torch.randint(0, 256, (5, 3, 32, 32), dtype=torch.uint8)

        outputs = model._infer(model._preprocess(frames))

        assert outputs["pred_boxes"][:, 0, 0].tolist() == [1, 1, 2, 2, 3]
        assert mock_runner.infer.call_count == 3

    def test_dynamic_batch_model_does_not_pad(self):
        model, mock_runner = _build_rfdetr_model()
        model.input_dtype = torch.float32
        mock_runner.infer.return_value = {
            "pred_boxes": torch.zeros(1, 1, 4),
            "pred_logits": torch.full((1, 1, 1), -10.0),
            "pred_masks": torch.zeros(1, 1, 8, 8),
        }
        frames = torch.randint(0, 256, (1, 3, 32, 32), dtype=torch.uint8)

        model(frames, target_hw=(32, 32))

        fed = next(iter(mock_runner.infer.call_args.args[0].values()))
        assert fed.shape[0] == 1


class TestCompileRfdetrEngine:
    def test_delegates_to_compile_onnx(self, nvidia_vendor):
        with patch("jasna.trt.compile_onnx_to_tensorrt_engine", return_value=Path("out.engine")) as mock_compile:
            result = compile_rfdetr_engine(
                Path("model.onnx"),
                torch.device("cuda:0"),
                batch_size=4,
                resolution=576,
                dynamic_batch=True,
                fp16=True,
            )
            mock_compile.assert_called_once_with(
                Path("model.onnx"), torch.device("cuda:0"),
                batch_size=4, fp16=True, workspace_gb=20, dynamic_batch=True,
            )
            assert result == Path("out.engine")

    def test_amd_is_a_noop_returning_the_weights_path(self, monkeypatch, tmp_path):
        import jasna.mosaic.rfdetr as module

        weights = tmp_path / "rfdetr-v6.pt"
        monkeypatch.setattr(module, "is_amd_device", lambda _device: True)

        result = compile_rfdetr_engine(
            weights,
            torch.device("cpu"),
            batch_size=4,
            resolution=576,
            dynamic_batch=True,
            fp16=True,
        )

        # AMD runs the checkpoint through the rfdetr torch model; nothing to build.
        assert result == weights


class _BrightnessRunner:
    """Scores each frame by its mean input: bright frames are a detection, dark ones are not."""

    input_names = ["input"]
    input_dtypes = {"input": torch.float32}
    output_names = ["dets", "labels", "masks"]
    outputs = {
        "dets": MagicMock(ndim=3, shape=(1, 1, 4)),
        "labels": MagicMock(ndim=3, shape=(1, 1, 1)),
        "masks": MagicMock(ndim=4, shape=(1, 1, 8, 8)),
    }

    def infer(self, inputs):
        x = inputs["input"]
        b = x.shape[0]
        return {
            "dets": torch.full((b, 1, 4), 0.5, device=x.device),
            "labels": x.mean(dim=(1, 2, 3)).view(b, 1, 1) * 10,
            "masks": x[:, :1, :8, :8].clone(),
        }


def _amd_model(monkeypatch, device: torch.device) -> RfDetrMosaicDetectionModel:
    import jasna.mosaic.rfdetr as module
    import jasna.mosaic.rfdetr_torch_runner as torch_runner_module

    monkeypatch.setattr(module, "is_amd_device", lambda _device: True)
    monkeypatch.setattr(torch_runner_module, "RfDetrTorchRunner", lambda *a, **k: _BrightnessRunner())
    return RfDetrMosaicDetectionModel(
        weights_path=Path("rfdetr-v6.pt"),
        batch_size=4,
        device=device,
        resolution=64,
        dynamic_batch=True,
        torch_variant="medium",
        score_threshold=0.5,
    )


class TestRfDetrOwnStream:
    def test_amd_detects_on_its_stream_after_the_callers_work(self, monkeypatch):
        import jasna.mosaic.rfdetr as module
        from jasna.mosaic.detections import Detections

        events = []
        caller, model_stream = MagicMock(name="caller"), MagicMock(name="model")
        caller.wait_stream.side_effect = lambda stream: events.append(("caller waits", stream))
        model_stream.wait_stream.side_effect = lambda stream: events.append(("model waits", stream))
        monkeypatch.setattr(module, "new_stream", lambda _device: model_stream)
        monkeypatch.setattr(module, "current_stream", lambda _device: caller)

        class _Context:
            def __init__(self, stream):
                self.stream = stream

            def __enter__(self):
                events.append(("enter", self.stream))

            def __exit__(self, *exc):
                events.append(("exit", self.stream))

        monkeypatch.setattr(module, "stream_context", _Context)
        model = _amd_model(monkeypatch, torch.device("cpu"))
        mask = MagicMock()
        monkeypatch.setattr(
            model, "_detect", lambda frames, target_hw: events.append("detect") or Detections([np.zeros((1, 4))], [mask])
        )

        model(torch.zeros(1, 3, 8, 8, dtype=torch.uint8), target_hw=(8, 8))

        assert events == [
            ("model waits", caller), ("enter", model_stream), "detect", ("exit", model_stream),
            ("caller waits", model_stream),
        ]
        mask.record_stream.assert_called_once_with(caller)

    def test_nvidia_detects_on_the_callers_stream(self):
        model, _ = _build_rfdetr_model()
        assert model._stream is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
    def test_amd_stream_sees_the_frames_the_caller_is_still_writing(self, monkeypatch):
        model = _amd_model(monkeypatch, torch.device("cuda"))
        frames = torch.zeros(4, 3, 64, 64, dtype=torch.uint8, device="cuda")
        model(frames, target_hw=(64, 64))
        torch.cuda.synchronize()

        torch.cuda._sleep(200_000_000)
        frames.fill_(255)
        detections = model(frames, target_hw=(64, 64))
        mask_pixels = [int(masks.sum()) for masks in detections.masks]

        assert [len(boxes) for boxes in detections.boxes_xyxy] == [1, 1, 1, 1]
        assert mask_pixels == [64, 64, 64, 64]
