import pytest

from jasna.pipeline_overlap import compute_crossfade_weights, compute_keep_range, compute_overlap_and_tail_indices, compute_parent_crossfade_weights


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


@pytest.mark.parametrize(
    ("frame_count", "is_continuation", "split_due_to_max_size", "discard_margin", "blend_frames", "expected"),
    [
        # blend_frames=0 behaves like no crossfade
        (180, True, True, 15, 0, (15, 165)),
        # blend_frames extends keep range on both sides
        (180, False, True, 15, 5, (0, 170)),
        (180, True, False, 15, 5, (10, 180)),
        (180, True, True, 15, 5, (10, 170)),
        # blend_frames clamped to discard_margin
        (10, True, True, 2, 5, (0, 10)),
        # blend_frames=0 with discard_margin=0 is no-op
        (10, True, True, 0, 5, (0, 10)),
    ],
)
def test_compute_keep_range_with_blend_frames(
    frame_count: int,
    is_continuation: bool,
    split_due_to_max_size: bool,
    discard_margin: int,
    blend_frames: int,
    expected: tuple[int, int],
) -> None:
    assert (
        compute_keep_range(
            frame_count=frame_count,
            is_continuation=is_continuation,
            split_due_to_max_size=split_due_to_max_size,
            discard_margin=discard_margin,
            blend_frames=blend_frames,
        )
        == expected
    )


def test_compute_crossfade_weights_returns_empty_when_no_blend() -> None:
    assert compute_crossfade_weights(discard_margin=15, blend_frames=0) == {}
    assert compute_crossfade_weights(discard_margin=0, blend_frames=5) == {}


def test_compute_crossfade_weights_returns_ramp() -> None:
    weights = compute_crossfade_weights(discard_margin=15, blend_frames=5)
    assert len(weights) == 10
    assert min(weights.keys()) == 10  # d - bf = 15 - 5
    assert max(weights.keys()) == 19  # d + bf - 1 = 15 + 5 - 1
    # first weight near 0, last weight near 1
    assert weights[10] == pytest.approx(0.05)
    assert weights[19] == pytest.approx(0.95)
    # monotonically increasing
    vals = [weights[k] for k in sorted(weights.keys())]
    for i in range(1, len(vals)):
        assert vals[i] > vals[i - 1]


def test_compute_crossfade_weights_bf_clamped_to_discard_margin() -> None:
    weights = compute_crossfade_weights(discard_margin=3, blend_frames=10)
    assert len(weights) == 6  # 2 * min(10, 3) = 6
    assert min(weights.keys()) == 0
    assert max(weights.keys()) == 5


def test_compute_crossfade_weights_bf_one() -> None:
    weights = compute_crossfade_weights(discard_margin=5, blend_frames=1)
    assert len(weights) == 2
    assert weights[4] == pytest.approx(0.25)
    assert weights[5] == pytest.approx(0.75)


def test_compute_parent_crossfade_weights_returns_empty_when_no_blend() -> None:
    assert compute_parent_crossfade_weights(frame_count=180, discard_margin=15, blend_frames=0) == {}
    assert compute_parent_crossfade_weights(frame_count=180, discard_margin=0, blend_frames=5) == {}


def test_compute_parent_crossfade_weights_returns_descending_ramp() -> None:
    weights = compute_parent_crossfade_weights(frame_count=180, discard_margin=15, blend_frames=5)
    assert len(weights) == 10
    assert min(weights.keys()) == 160  # frame_count - d - bf = 180 - 15 - 5
    assert max(weights.keys()) == 169  # frame_count - d + bf - 1 = 180 - 15 + 5 - 1
    assert weights[160] == pytest.approx(0.95)
    assert weights[169] == pytest.approx(0.05)
    vals = [weights[k] for k in sorted(weights.keys())]
    for i in range(1, len(vals)):
        assert vals[i] < vals[i - 1]


def test_parent_and_child_crossfade_weights_sum_to_one() -> None:
    child = compute_crossfade_weights(discard_margin=15, blend_frames=5)
    parent = compute_parent_crossfade_weights(frame_count=180, discard_margin=15, blend_frames=5)
    parent_start = min(parent.keys())
    child_start = min(child.keys())
    for j in range(10):
        p = parent[parent_start + j]
        c = child[child_start + j]
        assert p + c == pytest.approx(1.0)

