from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import pytest

from jasna.mosaic.yolo import YoloMosaicDetectionModel, _batched_nms_keep

tensorrt_gpu_only = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None, reason="the TensorRT YOLO path runs on NVIDIA only"
)


def _mock_engine_path():
    p = MagicMock()
    p.exists.return_value = True
    p.suffix = ".engine"
    return p


def _build_yolo_model(*, batch_size=2, imgsz=640):
    mock_runner = MagicMock()
    mock_runner.input_names = ["images"]
    mock_runner.input_dtypes = {"images": torch.float32}

    pred = torch.zeros(batch_size, 4 + 1 + 32, 100)
    proto = torch.zeros(batch_size, 32, imgsz // 4, imgsz // 4)
    outs = {"pred": pred, "proto": proto}
    mock_runner.infer.return_value = outs
    mock_runner.outputs = outs

    with (
        patch("jasna.mosaic.yolo.get_yolo_tensorrt_engine_path", return_value=_mock_engine_path()),
        patch("torch.cuda.get_device_capability", return_value=(8, 9)),
        patch("jasna.trt.trt_runner.TrtRunner", return_value=mock_runner),
    ):
        model = YoloMosaicDetectionModel(
            model_path=Path("model.pt"),
            batch_size=batch_size,
            device=torch.device("cuda:0"),
            imgsz=imgsz,
        )
    return model, mock_runner


def _centered_detection(anchors: int = 100) -> torch.Tensor:
    """Raw (1, 4 + 1 + 32, anchors) prediction with one confident 100x100 box centred in a 640x640 input."""
    pred = torch.zeros(1, 4 + 1 + 32, anchors)
    pred[0, :4, 0] = torch.tensor([320.0, 320.0, 100.0, 100.0])
    pred[0, 4, 0] = 10.0
    pred[0, 5:, 0] = 0.5
    return pred


# The centred box of ``_centered_detection`` in a 480x640 frame letterboxed by 80 rows at the top.
_CENTERED_BOX_IN_480x640 = [270.0, 190.0, 370.0, 290.0]


@pytest.mark.parametrize(
    "invalid",
    [
        {"score_threshold": 1.5},
        {"iou_threshold": -0.1},
        {"max_det": 0},
        {"imgsz": 0},
        {"batch_size": 0},
    ],
    ids=["score_threshold", "iou_threshold", "max_det", "imgsz", "batch_size"],
)
def test_yolo_rejects_invalid_settings_before_loading_the_model(invalid):
    settings = {"model_path": Path("missing.pt"), "batch_size": 1, "device": torch.device("cpu"), **invalid}
    with pytest.raises(ValueError, match=next(iter(invalid))):
        YoloMosaicDetectionModel(**settings)


class TestYoloCall:
    def test_scan_nms_is_class_aware(self):
        boxes = torch.tensor(
            [[[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 4.0, 4.0]]]
        )
        keep = _batched_nms_keep(
            boxes,
            torch.tensor([[0.9, 0.8, 0.7]]),
            torch.tensor([[0, 0, 1]]),
            score_threshold=0.05,
            iou_threshold=0.7,
            max_det=2,
        )
        assert keep.tolist() == [[True, False, True]]

    def test_scan_mask_is_cropped_to_predicted_box(self):
        model, mock_ab = _build_yolo_autobackend(imgsz=4)
        pred = torch.zeros(1, 6, 8)
        pred[0, :4, 0] = torch.tensor([2.0, 2.0, 2.0, 2.0])
        pred[0, 4, 0] = 0.9
        pred[0, 5, 0] = 100.0
        mock_ab.return_value = (pred, torch.ones(1, 1, 4, 4))

        scores, masks = model.scan_scores_masks(
            torch.zeros(1, 3, 4, 4, dtype=torch.uint8),
            mask_hw=(4, 4),
        )

        assert scores.tolist() == pytest.approx([0.9])
        assert masks[0].sum().item() == 4
        assert masks[0, 1:3, 1:3].all()

    @tensorrt_gpu_only
    @pytest.mark.usefixtures("nvidia_vendor")
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

    @tensorrt_gpu_only
    @pytest.mark.usefixtures("nvidia_vendor")
    def test_call_with_detections(self):
        model, mock_runner = _build_yolo_model(batch_size=1, imgsz=640)
        mock_runner.infer.return_value = {"pred": _centered_detection(), "proto": torch.ones(1, 32, 160, 160)}

        frames = torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8, device="cuda:0")
        det = model(frames, target_hw=(480, 640))

        np.testing.assert_allclose(det.boxes_xyxy[0], [_CENTERED_BOX_IN_480x640], atol=1.0)
        assert det.masks[0].shape[0] == 1 and det.masks[0].any()


def _build_yolo_autobackend(*, batch_size=1, imgsz=640, stride=torch.tensor([32.0])):
    mock_instance = MagicMock()
    mock_instance.fp16 = False
    mock_instance.names = {0: "mosaic"}
    mock_instance.end2end = False
    mock_instance.stride = stride
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


@pytest.mark.parametrize(
    ("stride", "expected"), [(torch.tensor([8.0, 32.0]), 32), ([8, 16, 32], 32), (16, 16)], ids=["tensor", "list", "int"]
)
def test_autobackend_stride_is_the_largest_model_stride(stride, expected):
    model, _ = _build_yolo_autobackend(stride=stride)
    assert model.stride == expected


class TestYoloAutoBackendCall:
    def test_call_without_detections_returns_empty_boxes_and_masks(self):
        model, mock_ab = _build_yolo_autobackend(batch_size=1, imgsz=640)
        mock_ab.return_value = (torch.zeros(1, 100, 4 + 1 + 32), torch.zeros(1, 32, 160, 160))

        det = model(torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8), target_hw=(480, 640))

        assert det.boxes_xyxy[0].shape == (0, 4)
        assert det.masks[0].shape == (0, 192, 256)
        assert det.masks[0].dtype == torch.bool

    @pytest.mark.parametrize(
        ("pred_layout", "proto_layout"),
        [("channels-first", "nchw"), ("anchors-first", "nchw"), ("channels-first", "nhwc")],
    )
    def test_call_reads_every_output_layout(self, pred_layout, proto_layout):
        model, mock_ab = _build_yolo_autobackend(batch_size=1, imgsz=640)
        pred = _centered_detection()
        if pred_layout == "anchors-first":
            pred = pred.permute(0, 2, 1).contiguous()
        proto = torch.ones(1, 32, 160, 160)
        if proto_layout == "nhwc":
            proto = proto.permute(0, 2, 3, 1).contiguous()
        mock_ab.return_value = (pred, proto)

        det = model(torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8), target_hw=(480, 640))

        np.testing.assert_allclose(det.boxes_xyxy[0], [_CENTERED_BOX_IN_480x640], atol=1.0)
        assert det.masks[0].shape[0] == 1 and det.masks[0].any()

    def test_call_accepts_nested_segmentation_output_with_metadata_dict(self):
        model, mock_ab = _build_yolo_autobackend(batch_size=1, imgsz=640)
        pred = torch.zeros(1, 4 + 2 + 32, 100)
        proto = torch.zeros(1, 32, 160, 160)
        mock_ab.return_value = (
            (pred, proto),
            {"boxes": None, "mask_coefficient": None, "proto": proto},
        )

        det = model(
            torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8),
            target_hw=(480, 640),
        )

        assert len(det.boxes_xyxy) == 1
        assert det.boxes_xyxy[0].shape[0] == 0

    def test_scan_accepts_nested_segmentation_output_with_metadata_dict(self):
        model, mock_ab = _build_yolo_autobackend(batch_size=1, imgsz=640)
        pred = torch.zeros(1, 4 + 2 + 32, 100)
        proto = torch.zeros(1, 32, 160, 160)
        mock_ab.return_value = (
            (pred, proto),
            {"boxes": None, "mask_coefficient": None, "proto": proto},
        )

        scores, masks = model.scan_scores_masks(
            torch.randint(0, 256, (1, 3, 480, 640), dtype=torch.uint8),
            mask_hw=(90, 160),
        )

        assert scores.shape == (1,)
        assert masks.shape == (1, 90, 160)
