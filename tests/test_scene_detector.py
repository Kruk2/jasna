from __future__ import annotations

import torch

from jasna.tracking.scene_detector import SceneCutDetector


def _solid(n: int, rgb: tuple[int, int, int], h: int = 64, w: int = 96) -> torch.Tensor:
    frame = torch.tensor(rgb, dtype=torch.uint8).view(1, 3, 1, 1)
    return frame.expand(n, 3, h, w).contiguous()


def test_detects_hard_cut_at_offset() -> None:
    frames = torch.cat([_solid(10, (40, 40, 40)), _solid(10, (200, 200, 200))])

    assert SceneCutDetector().find_cuts(frames) == {10}


def test_detects_moderate_cut() -> None:
    frames = torch.cat([_solid(8, (40, 40, 40)), _solid(8, (60, 60, 60))])

    assert SceneCutDetector().find_cuts(frames) == {8}


def test_detects_cut_across_batch_boundary() -> None:
    detector = SceneCutDetector()
    frames = torch.cat([_solid(10, (40, 40, 40)), _solid(10, (200, 200, 200))])

    cuts: set[int] = set()
    for start in range(0, 20, 3):
        batch = frames[start:start + 3]
        cuts |= {start + offset for offset in detector.find_cuts(batch)}

    assert cuts == {10}


def test_sustained_fast_motion_is_rejected_by_median_baseline() -> None:
    torch.manual_seed(0)
    base = torch.randint(0, 256, (3, 64, 96), dtype=torch.uint8)
    frames = torch.stack([base.roll(shifts=2 * i, dims=2) for i in range(16)])

    assert SceneCutDetector().find_cuts(frames) == set()


def test_cut_spike_in_window_does_not_mask_a_later_cut() -> None:
    frames = torch.cat(
        [
            _solid(8, (40, 40, 40)),
            _solid(4, (200, 200, 200)),
            _solid(8, (100, 100, 100)),
        ]
    )

    assert SceneCutDetector().find_cuts(frames) == {8, 12}


def test_gradual_brightness_change_is_not_a_cut() -> None:
    frames = torch.stack(
        [torch.full((3, 64, 96), 40 + 3 * i, dtype=torch.uint8) for i in range(30)]
    )

    assert SceneCutDetector().find_cuts(frames) == set()


def test_min_scene_len_absorbs_flash_double_fire() -> None:
    frames = torch.cat(
        [_solid(8, (30, 30, 30)), _solid(1, (250, 250, 250)), _solid(8, (30, 30, 30))]
    )

    assert SceneCutDetector().find_cuts(frames) == {8}


def test_no_cut_until_score_window_is_full() -> None:
    frames = torch.cat([_solid(3, (30, 30, 30)), _solid(5, (250, 250, 250))])

    assert SceneCutDetector().find_cuts(frames) == set()


def test_first_frame_of_next_batch_diffs_against_stored_previous_frame() -> None:
    detector = SceneCutDetector()
    detector.find_cuts(_solid(7, (30, 30, 30)))

    assert detector.find_cuts(_solid(5, (250, 250, 250))) == {0}


def test_reset_forgets_previous_frame_and_window() -> None:
    detector = SceneCutDetector()
    detector.find_cuts(_solid(7, (30, 30, 30)))
    detector.reset()

    assert detector.find_cuts(_solid(5, (250, 250, 250))) == set()


def test_empty_batch_returns_no_cuts() -> None:
    frames = torch.zeros((0, 3, 64, 96), dtype=torch.uint8)

    assert SceneCutDetector().find_cuts(frames) == set()
