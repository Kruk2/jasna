from __future__ import annotations

import logging

from jasna.blend_buffer import BlendBuffer
from jasna.frame_queue import FrameQueue


class PipelineDebugMemoryLogger:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        blend_buffer: BlendBuffer,
        clip_queue: FrameQueue,
        secondary_queue: FrameQueue,
        encode_queue: FrameQueue,
    ) -> None:
        self.logger = logger
        self.blend_buffer = blend_buffer
        self.clip_queue = clip_queue
        self.secondary_queue = secondary_queue
        self.encode_queue = encode_queue

    def snapshot(self, stage: str, details: str) -> None:
        if not self.logger.isEnabledFor(logging.DEBUG):
            return

        self.logger.debug(
            "[%s] %s pending=%d clip_q=%d secondary_q=%d encode_q=%d",
            stage,
            details,
            len(self.blend_buffer.pending_map),
            self.clip_queue.qsize(),
            self.secondary_queue.qsize(),
            self.encode_queue.qsize(),
        )
