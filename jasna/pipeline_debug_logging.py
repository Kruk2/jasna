from __future__ import annotations

import logging
from queue import Queue

from jasna.tracking.frame_buffer import FrameBuffer


class PipelineDebugMemoryLogger:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        frame_buffer: FrameBuffer,
        clip_queue: Queue[object],
        secondary_queue: Queue[object],
        encode_queue: Queue[object],
    ) -> None:
        self.logger = logger
        self.frame_buffer = frame_buffer
        self.clip_queue = clip_queue
        self.secondary_queue = secondary_queue
        self.encode_queue = encode_queue

    def snapshot(self, stage: str, details: str) -> None:
        if not self.logger.isEnabledFor(logging.DEBUG):
            return

        self.logger.debug(
            "[%s] %s fb=%d clip_q=%d secondary_q=%d encode_q=%d",
            stage,
            details,
            len(self.frame_buffer.frames),
            self.clip_queue.qsize(),
            self.secondary_queue.qsize(),
            self.encode_queue.qsize(),
        )
