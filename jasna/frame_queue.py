from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from queue import Empty, Full
from typing import Any


CancellationProbe = threading.Event | Callable[[], bool]


class FrameQueueCancelled(RuntimeError):
    """A queue operation stopped because its optional cancellation probe fired."""


class FrameQueue:
    _CANCEL_POLL_INTERVAL = 0.05

    def __init__(self, max_frames: int):
        self._deque: deque[tuple[Any, int]] = deque()
        self._cond = threading.Condition()
        self._max_frames = max_frames
        self._current_frames = 0
        self._unfinished_tasks = 0

    @staticmethod
    def _is_cancelled(cancel_event: CancellationProbe | None) -> bool:
        if cancel_event is None:
            return False
        if callable(cancel_event):
            return bool(cancel_event())
        return cancel_event.is_set()

    @staticmethod
    def _deadline(timeout: float | None) -> float | None:
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be non-negative")
        return None if timeout is None else time.monotonic() + timeout

    def _wait_timeout(
        self,
        deadline: float | None,
        cancel_event: CancellationProbe | None,
    ) -> float | None:
        if deadline is None:
            return self._CANCEL_POLL_INTERVAL if cancel_event is not None else None
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return 0.0
        if cancel_event is not None:
            return min(remaining, self._CANCEL_POLL_INTERVAL)
        return remaining

    def put(
        self,
        item: Any,
        frame_count: int = 0,
        *,
        timeout: float | None = None,
        cancel_event: CancellationProbe | None = None,
    ) -> None:
        """Append an item, optionally timing out or observing cancellation.

        The default call shape deliberately retains the previous blocking
        behavior.  Positive-frame producers can opt into a bounded wait; zero
        frame sentinels remain unblocked by frame capacity.
        """
        deadline = self._deadline(timeout)
        with self._cond:
            if frame_count > 0:
                while self._current_frames > 0 and self._current_frames + frame_count > self._max_frames:
                    if self._is_cancelled(cancel_event):
                        raise FrameQueueCancelled("queue put cancelled")
                    wait_timeout = self._wait_timeout(deadline, cancel_event)
                    if wait_timeout == 0:
                        raise Full
                    self._cond.wait(timeout=wait_timeout)
                if self._is_cancelled(cancel_event):
                    raise FrameQueueCancelled("queue put cancelled")
            self._deque.append((item, frame_count))
            self._current_frames += frame_count
            self._unfinished_tasks += 1
            self._cond.notify_all()

    def get(
        self,
        timeout: float | None = None,
        *,
        cancel_event: CancellationProbe | None = None,
    ) -> Any:
        """Return the next item, optionally timing out or observing cancellation."""
        deadline = self._deadline(timeout)
        with self._cond:
            while True:
                if self._is_cancelled(cancel_event):
                    raise FrameQueueCancelled("queue get cancelled")
                if self._deque:
                    item, frame_count = self._deque.popleft()
                    self._current_frames -= frame_count
                    self._cond.notify_all()
                    return item
                wait_timeout = self._wait_timeout(deadline, cancel_event)
                if wait_timeout == 0:
                    raise Empty
                self._cond.wait(timeout=wait_timeout)

    def get_nowait(self) -> Any:
        with self._cond:
            if not self._deque:
                raise Empty
            item, frame_count = self._deque.popleft()
            self._current_frames -= frame_count
            self._cond.notify_all()
            return item

    def task_done(self) -> None:
        with self._cond:
            if self._unfinished_tasks <= 0:
                raise ValueError("task_done() called too many times")
            self._unfinished_tasks -= 1
            if self._unfinished_tasks == 0:
                self._cond.notify_all()

    def join(
        self,
        timeout: float | None = None,
        *,
        cancel_event: CancellationProbe | None = None,
    ) -> bool:
        """Wait for unfinished tasks, returning False on an explicit timeout."""
        deadline = self._deadline(timeout)
        with self._cond:
            while self._unfinished_tasks > 0:
                if self._is_cancelled(cancel_event):
                    raise FrameQueueCancelled("queue join cancelled")
                wait_timeout = self._wait_timeout(deadline, cancel_event)
                if wait_timeout == 0:
                    return False
                self._cond.wait(timeout=wait_timeout)
            return True

    def discard_pending(self) -> int:
        """Abandon queued (not in-flight) items and balance their task counts."""
        with self._cond:
            abandoned = len(self._deque)
            if abandoned:
                self._deque.clear()
                self._current_frames = 0
                self._unfinished_tasks -= abandoned
                if self._unfinished_tasks < 0:
                    raise RuntimeError("FrameQueue task accounting underflow")
            self._cond.notify_all()
            return abandoned

    def wake_all(self) -> None:
        """Wake waiters so an external cancellation probe is re-evaluated."""
        with self._cond:
            self._cond.notify_all()

    def qsize(self) -> int:
        with self._cond:
            return len(self._deque)

    def empty(self) -> bool:
        with self._cond:
            return len(self._deque) == 0

    @property
    def current_frames(self) -> int:
        with self._cond:
            return self._current_frames
