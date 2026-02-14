from __future__ import annotations

import statistics
import time
from pathlib import Path

import torch

from jasna.restorer.basicvrspp_tenorrt_compilation import basicvsrpp_startup_policy
from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer

CLIP_LENGTH = 60
FRAME_H = 256
FRAME_W = 256
RUNS = 200


def benchmark_basicvsrpp_restoration(
    *,
    device: torch.device,
    fp16: bool,
    restoration_model_path: Path | None,
    compile_basicvsrpp: bool,
    **_: object,
) -> None:
    if restoration_model_path is None or not restoration_model_path.resolve().exists():
        return
    path = restoration_model_path.resolve()

    use_tensorrt = basicvsrpp_startup_policy(
        restoration_model_path=str(path),
        max_clip_size=CLIP_LENGTH,
        device=device,
        fp16=fp16,
        compile_basicvsrpp=compile_basicvsrpp,
    )
    restorer = BasicvsrppMosaicRestorer(
        checkpoint_path=str(path),
        device=device,
        max_clip_size=CLIP_LENGTH,
        use_tensorrt=use_tensorrt,
        fp16=fp16,
    )

    durations: list[float] = []
    video = [
        torch.randint(0, 256, (FRAME_H, FRAME_W, 3), dtype=torch.uint8, device=device)
        for _ in range(CLIP_LENGTH)
    ]

    with torch.cuda.device(device), torch.inference_mode():
        for _ in range(RUNS):
            start = time.perf_counter()
            restorer.raw_process(video)
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - start)

    median_duration = statistics.median(durations)
    print("Benchmark: basicvsrpp_restoration")
    print(f"  model: basicvsrpp  clip_length: {CLIP_LENGTH}  frame_size: {FRAME_H}x{FRAME_W}  runs: {RUNS}")
    print(f"  median_duration_s: {median_duration:.2f}")
    print()
