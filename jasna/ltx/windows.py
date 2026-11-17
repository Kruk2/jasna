"""The 121-frame model windows of a track and how neighbouring windows share frames: window starts,
the compose crossfade and the latent frames fused while denoising."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from jasna.ltx.camera import (
    WINDOW_FRAMES,
    CropFrame,
)

WINDOW_HOP = 88
"""The hop of a 33-frame overlap; jobs take theirs from ``window_hop`` of the chosen overlap."""


TEMPORAL_SCALE = 8


@dataclass(frozen=True)
class Window:
    index: int  # global across tracks; also the noise seed offset
    track_id: int
    start: int
    real_frames: int  # the model always sees WINDOW_FRAMES; the tail is padded
    crops: tuple[CropFrame, ...]  # one per real frame
    noise: int  # noise seed offset: the index, or the left window's for a right-eye twin

    @property
    def stop(self) -> int:
        return self.start + self.real_frames

    @property
    def canvas(self) -> int:
        return self.crops[0].canvas


def window_hop(overlap_frames: int) -> int:
    return WINDOW_FRAMES - overlap_frames


def window_starts(frames: int, window_hop: int) -> list[int]:
    starts = [0]
    while starts[-1] + WINDOW_FRAMES < frames:
        starts.append(starts[-1] + window_hop)
    return starts


UNFUSED_LEAD_FRAMES = 1 + TEMPORAL_SCALE
"""A later window's free latent 0 sits in front of latents fused with the earlier window, and
the decoder garbles the colour of the frames of that latent and the next one."""


def crossfade_weights(windows: Sequence[Window]) -> list[np.ndarray]:
    """Per-frame compose weights of one track's windows: across each overlap the earlier
    window alone over the later one's ``UNFUSED_LEAD_FRAMES``, then a linear ramp."""
    weights = [np.ones(window.real_frames, dtype=np.float32) for window in windows]
    for index, (left, right) in enumerate(zip(windows, windows[1:])):
        overlap = left.stop - right.start
        ramp = np.zeros(overlap, dtype=np.float32)
        ramp[UNFUSED_LEAD_FRAMES - 1 :] = np.linspace(0.0, 1.0, overlap - UNFUSED_LEAD_FRAMES + 1, dtype=np.float32)
        weights[index][len(weights[index]) - overlap :] = 1.0 - ramp
        weights[index + 1][:overlap] = ramp
    return weights


LATENT_FRAMES = 1 + (WINDOW_FRAMES - 1) // TEMPORAL_SCALE


@dataclass(frozen=True)
class SharedLatents:
    """Window k's latents ``left`` encode the same pixel frames as window k+1's latents
    ``right`` (12..15 and 1..4 at hop 88); ``ramp`` is the later window's fusion weight on each."""

    left: range
    right: range
    ramp: tuple[float, ...]


def shared_latents(window_hop: int) -> SharedLatents:
    shift = window_hop // TEMPORAL_SCALE
    left = range(shift + 1, LATENT_FRAMES)
    ramp = tuple((i + 1) / (len(left) + 1) for i in range(len(left)))
    return SharedLatents(left, range(1, LATENT_FRAMES - shift), ramp)
