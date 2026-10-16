from __future__ import annotations

import logging
import threading
import time

import torch

from jasna.media.probe import get_video_meta_data
from jasna.pipeline_threads import run_restoration_pass
from jasna.streaming import HlsStreamingServer
from jasna.streaming_encoder import StreamingEncoder, StreamingEncodeError

log = logging.getLogger(__name__)

_MAX_SEGMENTS_AHEAD = 3


class _StreamingFrameWriter:
    def __init__(
        self,
        streaming_encoder: StreamingEncoder,
        hls_server: HlsStreamingServer,
        start_segment: int,
        cancel_event: threading.Event,
    ):
        self._encoder = streaming_encoder
        self._hls_server = hls_server
        self._start_segment = start_segment
        self._frames_per_seg = hls_server.frames_per_segment()
        self._t0 = time.monotonic()
        self._cancel_event = cancel_event

    def write(self, frame: torch.Tensor, pts: int, *, apply_lut: bool = True) -> None:
        self._encoder.write_frame(frame, pts)

    def after_write(self, frames_written: int) -> None:
        self._encoder.raise_if_failed()
        if frames_written == 1:
            log.debug("[stream-blend-encode] first frame encoded: %.2fs", time.monotonic() - self._t0)
        elif frames_written % 100 == 0:
            # Guarded: a fast path (or a frozen clock) can hand back a zero
            # elapsed time, and this log line must not be what crashes a job.
            elapsed = time.monotonic() - self._t0
            log.debug(
                "[stream-blend-encode] %d frames encoded (%.1f fps)",
                frames_written, frames_written / elapsed if elapsed > 0 else 0.0,
            )

        current_seg = self._start_segment + frames_written // self._frames_per_seg
        self._hls_server.update_production(current_seg)
        self._hls_server.wait_for_demand(current_seg, _MAX_SEGMENTS_AHEAD, self._cancel_event)


def run_streaming(pipeline, hls_server: HlsStreamingServer) -> None:
    metadata = get_video_meta_data(str(pipeline.input_video))
    pipeline.validate_metadata(metadata)
    pipeline.configure_vr(metadata)
    hls_server.load_video(metadata)

    streaming_encoder = StreamingEncoder(
        segments_dir=hls_server.segments_dir,
        segment_duration=hls_server.segment_duration,
        metadata=metadata,
        source_video=str(pipeline.input_video),
        device=pipeline.device,
    )
    try:
        _streaming_loop(
            pipeline=pipeline,
            metadata=metadata,
            hls_server=hls_server,
            streaming_encoder=streaming_encoder,
        )
    finally:
        streaming_encoder.stop()


def _streaming_loop(
    *,
    pipeline,
    metadata,
    hls_server: HlsStreamingServer,
    streaming_encoder: StreamingEncoder,
) -> None:
    start_segment = hls_server.initial_start_segment
    first_pass = True

    while True:
        start_time = hls_server.segment_start_time(start_segment)
        log.info("[stream] starting pass from segment %d (t=%.1fs)", start_segment, start_time)

        pass_t0 = time.monotonic()
        hls_server.reset_demand(start_segment)
        if start_segment > 0:
            hls_server.notify_segment_requested(start_segment)
        if first_pass:
            streaming_encoder.start(start_number=start_segment)
            first_pass = False
        else:
            streaming_encoder.flush_and_restart(start_number=start_segment)

        cancel_event = threading.Event()
        seek_result = _run_streaming_pass(
            pipeline=pipeline,
            metadata=metadata,
            hls_server=hls_server,
            streaming_encoder=streaming_encoder,
            start_segment=start_segment,
            start_time=start_time,
            cancel_event=cancel_event,
        )

        log.info("[stream] pass ran for %.2fs", time.monotonic() - pass_t0)

        if hls_server.video_change.is_set():
            log.info("[stream] video change requested, exiting streaming loop")
            return

        if seek_result is None:
            streaming_encoder.stop()
            streaming_encoder.raise_if_failed()
            hls_server.mark_finished()
            log.info("[stream] pass finished, all segments produced — waiting for seek requests")
            while True:
                if hls_server.video_change.is_set():
                    log.info("[stream] video change requested, exiting streaming loop")
                    return
                target = hls_server.consume_seek()
                if target is not None:
                    log.info("[stream] seek to segment %d (t=%.1fs)", target, hls_server.segment_start_time(target))
                    start_segment = target
                    break
                time.sleep(0.1)
        else:
            log.info("[stream] seek to segment %d (t=%.1fs)", seek_result, hls_server.segment_start_time(seek_result))
            start_segment = seek_result


def _run_streaming_pass(
    *,
    pipeline,
    metadata,
    hls_server: HlsStreamingServer,
    streaming_encoder: StreamingEncoder,
    start_segment: int,
    start_time: float,
    cancel_event: threading.Event,
) -> int | None:
    frame_writer = _StreamingFrameWriter(streaming_encoder, hls_server, start_segment, cancel_event)
    seek_target: int | None = None
    encoder_error: StreamingEncodeError | None = None

    def poll() -> bool:
        nonlocal seek_target, encoder_error
        if hls_server.video_change.is_set():
            log.info("[stream] video change detected, cancelling current pass")
            return True
        try:
            streaming_encoder.raise_if_failed()
        except StreamingEncodeError as exc:
            encoder_error = exc
            return True
        target = hls_server.consume_seek_for_pass(start_segment)
        if target is not None:
            log.info("[stream] seek requested to segment %d, cancelling current pass", target)
            seek_target = target
            return True
        return False

    error = run_restoration_pass(
        pipeline,
        metadata,
        frame_writer,
        cancel_event,
        seek_ts=start_time if start_time > 0 else None,
        use_async_secondary=False,
        poll=poll,
    )
    error = encoder_error or error
    if error is not None and seek_target is None:
        raise error
    return seek_target
