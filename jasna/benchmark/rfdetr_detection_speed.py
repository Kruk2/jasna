from __future__ import annotations

import time
from pathlib import Path

import torch

from jasna.benchmark.harness import run_repeatedly
from jasna.media import get_video_meta_data
from jasna.media.video_decoder import NvidiaVideoReader
from jasna.mosaic.detection_registry import DEFAULT_DETECTION_MODEL_NAME, detection_model_weights_path
from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel
from jasna.tensor_utils import pad_batch_with_last


def _run_single(
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
    model_path = detection_model_weights_path(DEFAULT_DETECTION_MODEL_NAME)
    if not model_path.exists():
        raise FileNotFoundError(str(model_path))

    stream = torch.cuda.Stream(device=device)
    detection_model = RfDetrMosaicDetectionModel(
        onnx_path=model_path,
        stream=stream,
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
            stream=stream,
            metadata=metadata,
        ) as reader,
        torch.inference_mode(),
        torch.cuda.stream(stream),
    ):
        start = time.perf_counter()
        for frames, pts_list in reader.frames():
            effective_bs = len(pts_list)
            if effective_bs == 0:
                continue

            frames_eff = frames[:effective_bs]
            frames_in = pad_batch_with_last(frames_eff, batch_size=batch_size)
            detections = detection_model(frames_in, target_hw=target_hw)

            total_frames += effective_bs
            for i in range(effective_bs):
                total_detections += len(detections.boxes_xyxy[i])

        torch.cuda.synchronize()
        duration = time.perf_counter() - start

    return duration, {
        "video": str(path),
        "model": DEFAULT_DETECTION_MODEL_NAME,
        "frames": total_frames,
        "total_detections": total_detections,
    }


def benchmark_rfdetr_detection_speed(
    *,
    device: torch.device,
    batch_size: int,
    fp16: bool,
    benchmark_videos: list[Path],
    detection_score_threshold: float,
    **_: object,
) -> dict[str, tuple[float, float]]:
    results: dict[str, tuple[float, float]] = {}
    for video_path in benchmark_videos:
        path = video_path.resolve()
        if not path.exists():
            continue
        median_duration, result = run_repeatedly(
            lambda vp=path: _run_single(
                device=device,
                batch_size=batch_size,
                fp16=fp16,
                video_path=vp,
                score_threshold=detection_score_threshold,
            ),
            runs=3,
        )
        fps = result["frames"] / median_duration if median_duration > 0 else 0.0
        results[path.name] = (median_duration, fps)
    return results
