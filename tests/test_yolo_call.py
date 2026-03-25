from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import pytest

from jasna.mosaic.yolo import YoloMosaicDetectionModel


def _mock_engine_path():
    p = MagicMock()
    p.exists.return_value = True
    p.suffix = ".engine"
    return p


def _build_yolo_model(*, batch_size=2, imgsz=640):
    mock_runner = MagicMock()
    mock_runner.input_dtype = torch.float32

    pred = torch.zeros(batch_size, 4 + 1 + 32, 100)
    proto = torch.zeros(batch_size, 32, imgsz // 4, imgsz // 4)
    mock_runner.infer.return_value = {"pred": pred, "proto": proto}
    mock_runner.outputs = {"pred": pred, "proto": proto}

    with (
        patch("jasna.mosaic.yolo.get_yolo_tensorrt_engine_path", return_value=_mock_engine_path()),
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

    def test_trt_runner_called_with_input_shapes(self):
        mock_runner_cls = MagicMock()
        mock_runner_cls.return_value.input_dtype = torch.float32
        engine = _mock_engine_path()

        with (
            patch("jasna.mosaic.yolo.get_yolo_tensorrt_engine_path", return_value=engine),
            patch("jasna.mosaic.yolo.TrtRunner", mock_runner_cls),
        ):
            YoloMosaicDetectionModel(
                model_path=Path("model.pt"),
                batch_size=2,
                device=torch.device("cuda:0"),
                imgsz=640,
            )

        mock_runner_cls.assert_called_once_with(
            engine,
            input_shapes=(2, 3, 640, 640),
            device=torch.device("cuda:0"),
        )

    def test_score_threshold_out_of_range(self):
        with pytest.raises(ValueError, match="score_threshold"):
            with (
                patch("jasna.mosaic.yolo.get_yolo_tensorrt_engine_path"),
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
                patch("jasna.mosaic.yolo.get_yolo_tensorrt_engine_path"),
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
                patch("jasna.mosaic.yolo.get_yolo_tensorrt_engine_path"),
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


def _build_yolo_autobackend(*, batch_size=1, imgsz=640):
    mock_instance = MagicMock()
    mock_instance.fp16 = False
    mock_instance.names = {0: "mosaic"}
    mock_instance.end2end = False
    mock_instance.stride = torch.tensor([32.0])
    mock_instance.eval = MagicMock(return_value=mock_instance)

    with patch("ultralytics.nn.autobackend.AutoBackend", return_value=mock_instance, create=True):
        model = YoloMosaicDetectionModel(
            model_path=Path("model.pt"),
            batch_size=batch_size,
            device=torch.device("cpu"),
            imgsz=imgsz,
            fp16=False,
        )
    return model, mock_instance


class TestYoloAutoBackendInit:
    def test_autobackend_path(self):
        model, mock_ab = _build_yolo_autobackend()
        assert model.runner is None
        assert model.stride == 32
        assert model.end2end is False
        assert model.input_dtype == torch.float32
        mock_ab.eval.assert_called_once()

    def test_autobackend_stride_list(self):
        mock_ab = MagicMock()
        mock_ab.fp16 = False
        mock_ab.names = {}
        mock_ab.end2end = False
        mock_ab.stride = [8, 16, 32]
        mock_ab.eval = MagicMock(return_value=mock_ab)

        with patch("ultralytics.nn.autobackend.AutoBackend", return_value=mock_ab, create=True):
            model = YoloMosaicDetectionModel(
                model_path=Path("model.pt"),
                batch_size=1,
                device=torch.device("cpu"),
                imgsz=640,
                fp16=False,
            )
        assert model.stride == 32

    def test_autobackend_stride_int(self):
        mock_ab = MagicMock()
        mock_ab.fp16 = False
        mock_ab.names = {}
        mock_ab.end2end = False
        mock_ab.stride = 16
        mock_ab.eval = MagicMock(return_value=mock_ab)

        with patch("ultralytics.nn.autobackend.AutoBackend", return_value=mock_ab, create=True):
            model = YoloMosaicDetectionModel(
                model_path=Path("model.pt"),
                batch_size=1,
                device=torch.device("cpu"),
                imgsz=640,
                fp16=False,
            )
        assert model.stride == 16


class TestYoloAutoBackendCall:
    def test_call_non_trt(self):
        model, mock_ab = _build_yolo_autobackend(batch_size=1, imgsz=640)

        pred = torch.zeros(1, 100, 4 + 1 + 32)
        proto = torch.zeros(1, 32, 160, 160)
        mock_ab.return_value = (pred, proto)

        frames = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8)
        det = model(frames, target_hw=(480, 640))

        assert len(det.boxes_xyxy) == 1
        assert det.boxes_xyxy[0].shape[0] == 0

    def test_protos_transpose_nhwc_to_nchw(self):
        model, mock_ab = _build_yolo_autobackend(batch_size=1, imgsz=640)

        pred = torch.zeros(1, 100, 4 + 1 + 32)
        proto = torch.zeros(1, 160, 160, 32)  # NHWC layout, last dim is channel-like
        mock_ab.return_value = (pred, proto)

        frames = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8)
        det = model(frames, target_hw=(480, 640))
        assert len(det.boxes_xyxy) == 1

    def test_pred_raw_transpose_when_shape1_gt_shape2(self):
        model, mock_ab = _build_yolo_autobackend(batch_size=1, imgsz=640)

        pred = torch.zeros(1, 4 + 1 + 32, 100)  # shape[1] > shape[2], needs transpose
        proto = torch.zeros(1, 32, 160, 160)
        mock_ab.return_value = (pred, proto)

        frames = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8)
        det = model(frames, target_hw=(480, 640))
        assert len(det.boxes_xyxy) == 1
