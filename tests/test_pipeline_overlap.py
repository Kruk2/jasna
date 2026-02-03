import pytest

from jasna.pipeline_overlap import compute_keep_range, compute_overlap_and_tail_indices


def test_compute_overlap_and_tail_indices_zero_margin_returns_empty() -> None:
    overlap, tail = compute_overlap_and_tail_indices(end_frame=10, discard_margin=0)
    assert overlap == []
    assert tail == []


def test_compute_overlap_and_tail_indices_matches_expected_example() -> None:
    overlap, tail = compute_overlap_and_tail_indices(end_frame=180, discard_margin=30)
    assert overlap[0] == 121
    assert overlap[-1] == 180
    assert len(overlap) == 60

    assert tail[0] == 151
    assert tail[-1] == 180
    assert len(tail) == 30


@pytest.mark.parametrize(
    ("frame_count", "is_continuation", "split_due_to_max_size", "discard_margin", "expected"),
    [
        (10, False, False, 0, (0, 10)),
        (10, True, False, 0, (0, 10)),
        (10, False, True, 0, (0, 10)),
        (10, True, True, 0, (0, 10)),
        (10, False, False, 2, (0, 10)),
        (10, True, False, 2, (2, 10)),
        (10, False, True, 2, (0, 8)),
        (10, True, True, 2, (2, 8)),
    ],
)
def test_compute_keep_range(
    frame_count: int,
    is_continuation: bool,
    split_due_to_max_size: bool,
    discard_margin: int,
    expected: tuple[int, int],
) -> None:
    assert (
        compute_keep_range(
            frame_count=frame_count,
            is_continuation=is_continuation,
            split_due_to_max_size=split_due_to_max_size,
            discard_margin=discard_margin,
        )
        == expected
    )

