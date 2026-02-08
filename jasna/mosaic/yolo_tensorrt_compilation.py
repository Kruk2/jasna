from __future__ import annotations

import logging
import warnings
from pathlib import Path


def get_yolo_tensorrt_engine_path(model_path: str | Path, *, fp16: bool) -> Path:
    model_path = Path(model_path)
    if model_path.suffix.lower() != ".pt":
        raise ValueError(f"Expected a .pt YOLO checkpoint, got: {model_path}")
    from jasna.trt import get_onnx_tensorrt_engine_path

    onnx_path = model_path.with_suffix(".onnx")
    return get_onnx_tensorrt_engine_path(onnx_path, batch_size=None, fp16=bool(fp16))


def compile_yolo_to_tensorrt_engine(
    model_path: str | Path,
    *,
    batch: int,
    fp16: bool,
    imgsz: int | tuple[int, int],
) -> Path:
    model_path = Path(model_path)
    if int(batch) <= 0:
        raise ValueError(f"batch must be > 0, got {batch}")
    if isinstance(imgsz, int):
        if int(imgsz) <= 0:
            raise ValueError(f"imgsz must be > 0, got {imgsz}")
    else:
        h, w = int(imgsz[0]), int(imgsz[1])
        if h <= 0 or w <= 0:
            raise ValueError(f"imgsz must be > 0, got {imgsz}")

    if not model_path.exists():
        raise FileNotFoundError(str(model_path))

    engine_path = get_yolo_tensorrt_engine_path(model_path, fp16=bool(fp16))
    if engine_path.exists():
        return engine_path

    log = logging.getLogger(__name__)
    msg = (
        f"Compiling YOLO TensorRT engine for {model_path} (this can take a few minutes). "
        f"Output: {engine_path} (batch={int(batch)}, fp16={bool(fp16)}, imgsz={imgsz})"
    )
    print(msg)
    log.info("%s", msg)

    from ultralytics import YOLO

    from jasna.trt import compile_onnx_to_tensorrt_engine

    model = YOLO(str(model_path), verbose=False, task="segment")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"To copy construct from a tensor, it is recommended to use sourceTensor\.detach\(\)\.clone\(\).*",
            category=UserWarning,
            module=r"ultralytics\.engine\.exporter",
        )
        exported = model.export(
            format="onnx",
            imgsz=int(imgsz) if isinstance(imgsz, int) else tuple(int(x) for x in imgsz),
            dynamic=False,
            nms=False,
            batch=int(batch),
            half=bool(fp16),
        )
    del model

    exported_path = Path(str(exported)) if exported is not None else None
    if exported_path is None or not exported_path.exists():
        raise FileNotFoundError(str(exported_path) if exported_path is not None else "YOLO ONNX export returned None")
    if exported_path.suffix.lower() != ".onnx":
        raise RuntimeError(f"Expected ONNX export, got: {exported_path}")

    out = compile_onnx_to_tensorrt_engine(exported_path, batch_size=None, fp16=bool(fp16))
    if out != engine_path:
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        out.replace(engine_path)

    if not engine_path.exists():
        raise FileNotFoundError(str(engine_path))
    return engine_path

