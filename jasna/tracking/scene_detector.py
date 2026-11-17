"""GPU hard-cut detector used to end mosaic tracks at scene boundaries."""

from __future__ import annotations

from collections import deque
from statistics import median

import torch

_TARGET_WIDTH = 192
_LUMA_WEIGHTS = (0.299, 0.587, 0.114)
_ADAPTIVE_RATIO = 3.0
_MIN_CONTENT = 15.0
_WINDOW_SIZE = 5
_MIN_SCENE_LEN = 2
_REPEAT_RATIO = 0.1
_MIN_REPEAT_SCORE = 0.5


class SceneCutDetector:
    """Detects hard cuts between consecutive frames of one video stream.

    Frames are stride-subsampled to ~192px wide on their own device and each
    consecutive pair is scored by mean absolute luma difference. A cut fires
    when the score spikes above ``_ADAPTIVE_RATIO`` times the median of the
    recent scores (PySceneDetect AdaptiveDetector style; the median stays
    robust against spikes from earlier cuts, while sustained fast motion
    raises it and suppresses false positives). No cut can fire until the
    score window is full, so the first ``_WINDOW_SIZE`` frames of a stream
    are never split — tracks barely exist there anyway.

    Repeated frames (a 30 fps video doubled to 60, 3:2 pulldown) score about
    zero right after each real change and would drag the median to zero, so
    ordinary motion between two real frames would fire as a cut. A pair that
    follows a real change (a non-repeat scoring at least ``_MIN_REPEAT_SCORE``)
    and scores under ``_REPEAT_RATIO`` times that change (or
    ``_MIN_REPEAT_SCORE``) is a repeat: it never fires and stays out of the
    window. Everything else is scored as before, so still shots, slow motion
    and flashes keep their old behavior.
    """

    def __init__(self) -> None:
        self._prev_small: torch.Tensor | None = None
        self._recent_scores: deque[float] = deque(maxlen=_WINDOW_SIZE)
        self._frames_since_cut = _MIN_SCENE_LEN
        self._last_change: float | None = None

    def reset(self) -> None:
        self._prev_small = None
        self._recent_scores.clear()
        self._frames_since_cut = _MIN_SCENE_LEN
        self._last_change = None

    def find_cuts(self, frames_uint8_bchw: torch.Tensor) -> set[int]:
        """Return batch offsets whose frame starts a new scene.

        Offset ``i`` means a hard cut between frame ``i - 1`` and frame ``i``
        of this batch; the previous batch's last frame counts as ``i - 1``
        for ``i == 0``.
        """
        n = int(frames_uint8_bchw.shape[0])
        if n == 0:
            return set()

        small = self._subsample(frames_uint8_bchw)
        if self._prev_small is None:
            seq = small
            first_offset = 1
        else:
            seq = torch.cat([self._prev_small, small])
            first_offset = 0
        self._prev_small = small[-1:].clone()
        if seq.shape[0] < 2:
            return set()

        weights = torch.tensor(_LUMA_WEIGHTS, device=seq.device).view(1, 3, 1, 1)
        luma = (seq * weights).sum(dim=1)
        scores = (luma[1:] - luma[:-1]).abs().mean(dim=(1, 2)).cpu()

        cuts: set[int] = set()
        for pair_idx in range(scores.shape[0]):
            score = float(scores[pair_idx])
            if self._is_repeat(score):
                self._frames_since_cut += 1
                self._last_change = None
                continue
            if self._is_cut(score):
                cuts.add(first_offset + pair_idx)
                self._frames_since_cut = 1
            else:
                self._frames_since_cut += 1
            self._recent_scores.append(score)
            self._last_change = score if score >= _MIN_REPEAT_SCORE else None
        return cuts

    def _is_repeat(self, score: float) -> bool:
        if self._last_change is None:
            return False
        return score < max(_MIN_REPEAT_SCORE, _REPEAT_RATIO * self._last_change)

    def _is_cut(self, score: float) -> bool:
        if self._frames_since_cut < _MIN_SCENE_LEN:
            return False
        if len(self._recent_scores) < _WINDOW_SIZE:
            return False
        baseline = median(self._recent_scores)
        return score > max(_MIN_CONTENT, _ADAPTIVE_RATIO * max(baseline, 1.0))

    @staticmethod
    def _subsample(frames_uint8_bchw: torch.Tensor) -> torch.Tensor:
        stride = max(1, int(frames_uint8_bchw.shape[-1]) // _TARGET_WIDTH)
        return frames_uint8_bchw[:, :, ::stride, ::stride].float()
