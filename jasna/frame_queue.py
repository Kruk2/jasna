from __future__ import annotations

import threading
from collections import deque
from queue import Empty
from typing import Any


class FrameQueue:
    def __init__(self, max_frames: int):
        self._deque: deque[tuple[Any, int]] = deque()
        self._cond = threading.Condition()
        self._max_frames = max_frames
        self._current_frames = 0
        self._unfinished_tasks = 0

    def put(self, item: Any, frame_count: int = 0) -> None:
        with self._cond:
            if frame_count > 0:
                while self._current_frames > 0 and self._current_frames + frame_count > self._max_frames:
                    self._cond.wait()
            self._deque.append((item, frame_count))
            self._current_frames += frame_count
            self._unfinished_tasks += 1
            self._cond.notify_all()

    def get(self, timeout: float | None = None) -> Any:
        with self._cond:
            if not self._deque:
                self._cond.wait(timeout=timeout)
            if not self._deque:
                raise Empty
            item, frame_count = self._deque.popleft()
            self._current_frames -= frame_count
            self._cond.notify_all()
            return item

    def task_done(self) -> None:
        with self._cond:
            self._unfinished_tasks -= 1
            if self._unfinished_tasks <= 0:
                self._cond.notify_all()

    def join(self) -> None:
        with self._cond:
            while self._unfinished_tasks > 0:
                self._cond.wait()

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
