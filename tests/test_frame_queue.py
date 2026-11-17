from __future__ import annotations

import threading

import pytest

from jasna.frame_queue import FrameQueue, must_wait


@pytest.mark.parametrize(
    ("current_frames", "frame_count", "max_frames", "waits"),
    [
        (10, 0, 10, False),
        (0, 100, 10, False),
        (40, 10, 50, False),
        (50, 10, 50, True),
        (1, 100, 10, True),
    ],
    ids=["sentinel-at-limit", "oversized-into-empty", "up-to-limit", "over-limit", "oversized-into-non-empty"],
)
def test_a_put_waits_only_past_the_frame_limit(current_frames, frame_count, max_frames, waits):
    assert must_wait(current_frames, frame_count, max_frames) is waits


def test_items_come_out_in_order_and_free_their_frames():
    q = FrameQueue(max_frames=100)
    q.put("a", frame_count=10)
    q.put("b", frame_count=20)
    assert (q.qsize(), q.current_frames) == (2, 30)
    assert q.get() == "a"
    assert q.current_frames == 20
    assert q.get() == "b"
    assert (q.qsize(), q.current_frames) == (0, 0)


def test_a_get_wakes_a_waiting_put():
    q = FrameQueue(max_frames=50)
    q.put("a", frame_count=50)
    putter = threading.Thread(target=q.put, args=("b",), kwargs={"frame_count": 10})
    putter.start()

    assert q.get() == "a"
    putter.join()
    assert q.get() == "b"


def test_join_returns_after_every_task_is_done():
    q = FrameQueue(max_frames=100)
    q.put("a", frame_count=10)
    q.put("b", frame_count=20)
    for _ in range(2):
        q.get()
        q.task_done()

    q.join()
    assert q.qsize() == 0
