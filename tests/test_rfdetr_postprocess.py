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


def _build_rfdetr_model():
    mock_runner = MagicMock()
    mock_runner.input_dtype = torch.float16
    mock_runner.output_names = ["pred_boxes", "pred_logits", "pred_masks"]
    mock_runner.outputs = {
        "pred_boxes": torch.zeros(1, 100, 4),
        "pred_logits": torch.zeros(1, 100, 1),
        "pred_masks": torch.zeros(1, 100, 8, 8),
    }

    with (
        patch("jasna.mosaic.rfdetr.compile_onnx_to_tensorrt_engine", return_value=Path("model.engine")),
        patch("jasna.mosaic.rfdetr.TrtRunner", return_value=mock_runner),
    ):
        model = RfDetrMosaicDetectionModel(
            onnx_path=Path("model.onnx"),
            batch_size=2,
            device=torch.device("cpu"),
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


class TestCompileRfdetrEngine:
    def test_delegates_to_compile_onnx(self):
        with patch("jasna.mosaic.rfdetr.compile_onnx_to_tensorrt_engine", return_value=Path("out.engine")) as mock_compile:
            result = compile_rfdetr_engine(Path("model.onnx"), torch.device("cuda:0"), batch_size=4, fp16=True)
            mock_compile.assert_called_once_with(Path("model.onnx"), torch.device("cuda:0"), batch_size=4, fp16=True, workspace_gb=20)
            assert result == Path("out.engine")
