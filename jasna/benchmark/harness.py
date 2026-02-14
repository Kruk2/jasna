from __future__ import annotations

import statistics
from typing import Callable

import torch


def run_repeatedly(
    fn: Callable[[], tuple[float, dict]],
    runs: int = 3,
) -> tuple[float, dict]:
    durations: list[float] = []
    result: dict = {}
    for _ in range(runs):
        duration, result = fn()
        durations.append(duration)
        torch.cuda.synchronize()
    return statistics.median(durations), result
