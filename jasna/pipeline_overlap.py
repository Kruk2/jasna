from __future__ import annotations


def compute_overlap_and_tail_indices(*, end_frame: int, discard_margin: int) -> tuple[list[int], list[int]]:
    if discard_margin <= 0:
        return [], []

    overlap_len = 2 * int(discard_margin)
    overlap_start = int(end_frame) - overlap_len + 1
    overlap_indices = list(range(overlap_start, int(end_frame) + 1))

    tail_start = int(end_frame) - int(discard_margin) + 1
    tail_indices = list(range(tail_start, int(end_frame) + 1))

    return overlap_indices, tail_indices


def compute_keep_range(
    *,
    frame_count: int,
    is_continuation: bool,
    split_due_to_max_size: bool,
    discard_margin: int,
) -> tuple[int, int]:
    d = int(discard_margin)
    keep_start = d if (d > 0 and bool(is_continuation)) else 0
    keep_end = int(frame_count) - d if (d > 0 and bool(split_due_to_max_size)) else int(frame_count)
    return keep_start, keep_end

