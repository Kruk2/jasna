import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import pytest

from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel, compile_rfdetr_engine


_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("device", _DEVICES)
def test_postprocess_keeps_the_most_confident_queries_as_pixel_boxes_and_masks(device):
    no = -10.0
    logits = torch.tensor([
        [[no, 5.0], [no, no], [no, no], [no, no]],
        [[no, no], [5.0, no], [no, 4.0], [3.0, 3.5]],  # four confident scores, three kept
    ])
    boxes_cxcywh = torch.tensor([[0.5, 0.5, 1.0, 1.0], [0.25, 0.25, 0.5, 0.5],
                                 [0.5, 0.5, 0.2, 0.2], [0.75, 0.75, 0.5, 0.5]]).expand(2, 4, 4)
    masks = torch.tensor([-1.5, -0.5, 0.5, 1.5]).view(1, 4, 1, 1).expand(2, 4, 2, 2)

    boxes, kept_masks = RfDetrMosaicDetectionModel._postprocess(
        pred_boxes=boxes_cxcywh.to(device), pred_logits=logits.to(device), pred_masks=masks.half().to(device),
        target_hw=(480, 640), score_threshold=0.5, max_select=3,
    )

    np.testing.assert_allclose(boxes[0], [[0, 0, 640, 480]], atol=1e-3)
    np.testing.assert_allclose(boxes[1], [[0, 0, 320, 240], [256, 192, 384, 288], [320, 240, 640, 480]], atol=1e-3)
    assert torch.equal(kept_masks[0].cpu(), torch.zeros(1, 2, 2, dtype=torch.bool))
    assert torch.equal(kept_masks[1].cpu(), torch.tensor([False, True, True]).view(3, 1, 1).expand(3, 2, 2))


@pytest.mark.parametrize("device", _DEVICES)
def test_postprocess_drops_confident_empty_boxes(device):
    logits = torch.tensor([[[5.0], [4.0], [3.0]]])
    boxes_cxcywh = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0], [0.25, 0.25, 0.5, 0.5]]])
    masks = torch.tensor([-1.0, 1.0, 1.0]).view(1, 3, 1, 1)

    boxes, kept_masks = RfDetrMosaicDetectionModel._postprocess(
        pred_boxes=boxes_cxcywh.to(device), pred_logits=logits.to(device), pred_masks=masks.to(device),
        target_hw=(480, 640), score_threshold=0.5, max_select=3,
    )

    np.testing.assert_allclose(boxes[0], [[256, 192, 384, 288], [0, 0, 320, 240]], atol=1e-3)
    assert kept_masks[0].cpu().view(-1).tolist() == [False, True]


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
    def test_frames_are_resized_and_normalized_with_imagenet_statistics(self):
        model, _ = _build_rfdetr_model()
        model.input_dtype = torch.float32
        out = model._preprocess(torch.full((2, 3, 100, 200), 255, dtype=torch.uint8))
        assert out.shape == (2, 3, 768, 768) and out.dtype == torch.float32
        white = (1 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
        torch.testing.assert_close(out[:, :, 0, 0], white.expand(2, 3))

    def test_normalization_tensors_are_reused_per_dtype(self):
        model, _ = _build_rfdetr_model()
        mean, std = model._normalization(torch.zeros(1, 3, 4, 4))
        again_mean, again_std = model._normalization(torch.zeros(2, 3, 8, 8))
        assert again_mean is mean and again_std is std
        half_mean, _ = model._normalization(torch.zeros(1, 3, 4, 4, dtype=torch.float16))
        assert half_mean is not mean and half_mean.dtype == torch.float16


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
