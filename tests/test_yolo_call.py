from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import pytest

from jasna.mosaic.yolo import YoloMosaicDetectionModel


def _build_yolo_model(*, batch_size=2, imgsz=640):
    mock_runner = MagicMock()
    mock_runner.input_dtype = torch.float32

    pred = torch.zeros(batch_size, 4 + 1 + 32, 100)
    proto = torch.zeros(batch_size, 32, imgsz // 4, imgsz // 4)
    mock_runner.infer.return_value = {"pred": pred, "proto": proto}
    mock_runner.outputs = {"pred": pred, "proto": proto}

    with (
        patch("jasna.mosaic.yolo.compile_yolo_to_tensorrt_engine", return_value=Path("model.engine")),
        patch("jasna.mosaic.yolo.TrtRunner", return_value=mock_runner),
    ):
        model = YoloMosaicDetectionModel(
            model_path=Path("model.pt"),
            batch_size=batch_size,
            device=torch.device("cuda:0"),
            imgsz=imgsz,
        )
    return model, mock_runner


class TestYoloInit:
    def test_basic_init_trt(self):
        model, runner = _build_yolo_model()
        assert model.batch_size == 2
        assert model.imgsz == 640
        assert model.runner is runner
        assert model.stride == 32

    def test_score_threshold_out_of_range(self):
        with pytest.raises(ValueError, match="score_threshold"):
            with (
                patch("jasna.mosaic.yolo.compile_yolo_to_tensorrt_engine"),
                patch("jasna.mosaic.yolo.TrtRunner"),
            ):
                YoloMosaicDetectionModel(
                    model_path=Path("model.pt"),
                    batch_size=1,
                    device=torch.device("cuda:0"),
                    score_threshold=1.5,
                )

    def test_iou_threshold_out_of_range(self):
        with pytest.raises(ValueError, match="iou_threshold"):
            with (
                patch("jasna.mosaic.yolo.compile_yolo_to_tensorrt_engine"),
                patch("jasna.mosaic.yolo.TrtRunner"),
            ):
                YoloMosaicDetectionModel(
                    model_path=Path("model.pt"),
                    batch_size=1,
                    device=torch.device("cuda:0"),
                    iou_threshold=-0.1,
                )

    def test_max_det_zero_raises(self):
        with pytest.raises(ValueError, match="max_det"):
            with (
                patch("jasna.mosaic.yolo.compile_yolo_to_tensorrt_engine"),
                patch("jasna.mosaic.yolo.TrtRunner"),
            ):
                YoloMosaicDetectionModel(
                    model_path=Path("model.pt"),
                    batch_size=1,
                    device=torch.device("cuda:0"),
                    max_det=0,
                )


class TestYoloCall:
    def test_call_no_detections(self):
        model, mock_runner = _build_yolo_model(batch_size=1, imgsz=640)

        pred = torch.zeros(1, 4 + 1 + 32, 100)
        pred[:, 4, :] = -10.0
        proto = torch.zeros(1, 32, 160, 160)
        mock_runner.infer.return_value = {"pred": pred, "proto": proto}

        frames = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8, device="cuda:0")
        det = model(frames, target_hw=(480, 640))

        assert len(det.boxes_xyxy) == 1
        assert det.boxes_xyxy[0].shape[0] == 0
        assert len(det.masks) == 1
        mock_runner.infer.assert_called_once()

    def test_call_with_detections(self):
        model, mock_runner = _build_yolo_model(batch_size=1, imgsz=640)

        pred = torch.zeros(1, 4 + 1 + 32, 100)
        pred[0, 0, 0] = 100.0   # x1
        pred[0, 1, 0] = 100.0   # y1
        pred[0, 2, 0] = 200.0   # x2
        pred[0, 3, 0] = 200.0   # y2
        pred[0, 4, 0] = 10.0    # high confidence
        pred[0, 5:, 0] = 0.5    # mask coefficients

        proto = torch.ones(1, 32, 160, 160)
        mock_runner.infer.return_value = {"pred": pred, "proto": proto}

        frames = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8, device="cuda:0")
        det = model(frames, target_hw=(480, 640))

        assert len(det.boxes_xyxy) == 1
        assert len(det.masks) == 1
