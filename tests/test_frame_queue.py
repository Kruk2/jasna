from __future__ import annotations

import threading
import time
from queue import Empty, Full

import pytest

from jasna.frame_queue import FrameQueue, FrameQueueCancelled


class TestFrameQueue:
    def test_put_get_basic(self):
        q = FrameQueue(max_frames=100)
        q.put("a", frame_count=10)
        q.put("b", frame_count=20)
        assert q.qsize() == 2
        assert q.current_frames == 30
        assert q.get() == "a"
        assert q.current_frames == 20
        assert q.get() == "b"
        assert q.current_frames == 0
        assert q.empty()

    def test_sentinel_always_passes(self):
        q = FrameQueue(max_frames=10)
        q.put("big", frame_count=10)
        assert q.current_frames == 10

        done = threading.Event()

        def put_sentinel():
            q.put(None, frame_count=0)
            done.set()

        t = threading.Thread(target=put_sentinel)
        t.start()
        t.join(timeout=1.0)
        assert done.is_set(), "sentinel should not block even when queue is at max"
        assert q.qsize() == 2
        q.get()
        q.get()

    def test_blocks_when_over_max(self):
        q = FrameQueue(max_frames=50)
        q.put("a", frame_count=50)

        blocked = threading.Event()
        released = threading.Event()

        def put_blocked():
            blocked.set()
            q.put("b", frame_count=10)
            released.set()

        t = threading.Thread(target=put_blocked)
        t.start()
        time.sleep(0.05)
        assert blocked.is_set()
        assert not released.is_set(), "should be blocked — queue at 50/50"

        q.get()
        t.join(timeout=1.0)
        assert released.is_set()
        assert q.get() == "b"

    def test_oversized_item_accepted_when_empty(self):
        q = FrameQueue(max_frames=10)
        q.put("huge", frame_count=100)
        assert q.current_frames == 100
        assert q.get() == "huge"
        assert q.current_frames == 0

    def test_oversized_item_blocks_when_not_empty(self):
        q = FrameQueue(max_frames=10)
        q.put("a", frame_count=1)

        blocked = threading.Event()
        released = threading.Event()

        def put_oversized():
            blocked.set()
            q.put("huge", frame_count=100)
            released.set()

        t = threading.Thread(target=put_oversized)
        t.start()
        time.sleep(0.05)
        assert blocked.is_set()
        assert not released.is_set(), "oversized should block when queue is non-empty"

        q.get()
        t.join(timeout=1.0)
        assert released.is_set()
        assert q.get() == "huge"

    def test_task_done_and_join(self):
        q = FrameQueue(max_frames=100)
        q.put("a", frame_count=10)
        q.put("b", frame_count=20)

        joined = threading.Event()

        def joiner():
            q.join()
            joined.set()

        t = threading.Thread(target=joiner)
        t.start()
        time.sleep(0.05)
        assert not joined.is_set()

        q.get()
        q.task_done()
        time.sleep(0.05)
        assert not joined.is_set()

        q.get()
        q.task_done()
        t.join(timeout=1.0)
        assert joined.is_set()

    def test_many_small_clips_fit(self):
        q = FrameQueue(max_frames=100)
        for i in range(50):
            q.put(f"clip_{i}", frame_count=2)
        assert q.qsize() == 50
        assert q.current_frames == 100

    def test_qsize_and_empty(self):
        q = FrameQueue(max_frames=100)
        assert q.empty()
        assert q.qsize() == 0
        q.put("x", frame_count=5)
        assert not q.empty()
        assert q.qsize() == 1
        q.get()
        assert q.empty()
        assert q.qsize() == 0

    def test_zero_frame_items_dont_block(self):
        q = FrameQueue(max_frames=10)
        for i in range(100):
            q.put(f"item_{i}", frame_count=0)
        assert q.qsize() == 100
        assert q.current_frames == 0

    def test_put_timeout_when_full(self):
        q = FrameQueue(max_frames=1)
        q.put("occupied", frame_count=1)

        with pytest.raises(Full):
            q.put("blocked", frame_count=1, timeout=0.01)

        assert q.qsize() == 1
        assert q.get() == "occupied"

    def test_put_observes_cancel_when_full(self):
        q = FrameQueue(max_frames=1)
        q.put("occupied", frame_count=1)
        cancel_event = threading.Event()
        started = threading.Event()
        done = threading.Event()
        outcome: list[BaseException] = []

        def blocked_put():
            started.set()
            try:
                q.put("blocked", frame_count=1, cancel_event=cancel_event)
            except BaseException as error:
                outcome.append(error)
            finally:
                done.set()

        thread = threading.Thread(target=blocked_put)
        thread.start()
        assert started.wait(timeout=0.5)
        cancel_event.set()

        assert done.wait(timeout=0.5)
        thread.join(timeout=0.5)
        assert not thread.is_alive()
        assert len(outcome) == 1
        assert isinstance(outcome[0], FrameQueueCancelled)
        assert q.qsize() == 1

    def test_get_observes_cancel_while_idle(self):
        q = FrameQueue(max_frames=1)
        cancel_event = threading.Event()
        started = threading.Event()
        done = threading.Event()
        outcome: list[BaseException] = []

        def idle_get():
            started.set()
            try:
                q.get(cancel_event=cancel_event)
            except BaseException as error:
                outcome.append(error)
            finally:
                done.set()

        thread = threading.Thread(target=idle_get)
        thread.start()
        assert started.wait(timeout=0.5)
        cancel_event.set()

        assert done.wait(timeout=0.5)
        thread.join(timeout=0.5)
        assert not thread.is_alive()
        assert len(outcome) == 1
        assert isinstance(outcome[0], FrameQueueCancelled)

    def test_join_timeout_and_discard_pending_balance_tasks(self):
        q = FrameQueue(max_frames=4)
        q.put("in-flight", frame_count=1)
        q.put("abandoned", frame_count=1)

        assert q.join(timeout=0.01) is False
        assert q.get() == "in-flight"
        assert q.discard_pending() == 1
        q.task_done()

        assert q.join(timeout=0.1) is True
        assert q.empty()
        assert q.current_frames == 0

    def test_join_observes_cancel(self):
        q = FrameQueue(max_frames=1)
        q.put("pending", frame_count=1)
        cancel_event = threading.Event()
        cancel_event.set()

        with pytest.raises(FrameQueueCancelled):
            q.join(cancel_event=cancel_event)
