from __future__ import annotations

import time
from pathlib import Path

import torch

from jasna.benchmark.harness import run_repeatedly
from jasna.media.probe import get_video_meta_data
from jasna.media.video_decoder import NvidiaVideoReader
from jasna.mosaic.detection_registry import (
    DEFAULT_DETECTION_MODEL_NAME,
    build_detection_model,
    detection_model_weights_path,
    recommended_score_threshold,
)

LADA_YOLO_FAST_MODEL = "lada-yolo-v4"


def _run_single(
    model_name: str,
    *,
    device: torch.device,
    batch_size: int,
    fp16: bool,
    video_path: Path,
    score_threshold: float,
) -> tuple[float, dict]:
    path = video_path.resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))

    metadata = get_video_meta_data(str(path))
    model_path = detection_model_weights_path(model_name)
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))

    detection_model = build_detection_model(
        model_name,
        model_path,
        batch_size=batch_size,
        device=device,
        score_threshold=score_threshold,
        fp16=fp16,
    )

    target_hw = (int(metadata.video_height), int(metadata.video_width))
    total_frames = 0
    total_detections = 0

    with (
        NvidiaVideoReader(
            str(path),
            batch_size=batch_size,
            device=device,
            metadata=metadata,
        ) as reader,
        torch.inference_mode(),
    ):
        start = time.perf_counter()
        for frames, pts_list in reader.frames():
            effective_bs = len(pts_list)
            if effective_bs == 0:
                continue

            detections = detection_model(
                frames[:effective_bs],
                target_hw=target_hw,
            )

            total_frames += effective_bs
            for i in range(effective_bs):
                total_detections += len(detections.boxes_xyxy[i])

        torch.cuda.synchronize()
        duration = time.perf_counter() - start

    return duration, {
        "video": str(path),
        "model": model_name,
        "frames": total_frames,
        "total_detections": total_detections,
    }


def _benchmark_detection_speed(
    model_name: str,
    *,
    device: torch.device,
    batch_size: int,
    fp16: bool,
    benchmark_videos: list[Path],
    detection_score_threshold: float | None,
    **_: object,
) -> dict[str, tuple[float, float]]:
    results: dict[str, tuple[float, float]] = {}
    score_threshold = (
        recommended_score_threshold(model_name)
        if detection_score_threshold is None
        else float(detection_score_threshold)
    )
    for video_path in benchmark_videos:
        path = video_path.resolve()
        if not path.exists():
            continue
        median_duration, result = run_repeatedly(
            lambda vp=path: _run_single(
                model_name,
                device=device,
                batch_size=batch_size,
                fp16=fp16,
                video_path=vp,
                score_threshold=score_threshold,
            ),
            runs=3,
        )
        fps = result["frames"] / median_duration if median_duration > 0 else 0.0
        results[path.name] = (median_duration, fps)
    return results


def benchmark_rfdetr_detection_speed(**kwargs) -> dict[str, tuple[float, float]]:
    return _benchmark_detection_speed(DEFAULT_DETECTION_MODEL_NAME, **kwargs)


def benchmark_lada_yolo_detection_speed(**kwargs) -> dict[str, tuple[float, float]]:
    return _benchmark_detection_speed(LADA_YOLO_FAST_MODEL, **kwargs)
