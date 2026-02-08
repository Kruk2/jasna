import pytest
import torch

from jasna.mosaic.yolo import YoloMosaicDetectionModel


def test_yolo_init_rejects_invalid_imgsz() -> None:
    with pytest.raises(ValueError):
        YoloMosaicDetectionModel(
            model_path="dummy.pt",  # ValueError should happen before file access
            stream=None,  # type: ignore[arg-type]
            batch_size=1,
            device=torch.device("cpu"),
            imgsz=0,
        )

