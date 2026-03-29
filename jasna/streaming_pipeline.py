from __future__ import annotations

import gc
import logging
import threading
import time
from queue import Empty, Queue

import torch

from jasna.blend_buffer import BlendBuffer
from jasna.crop_buffer import CropBuffer
from jasna.frame_queue import FrameQueue
from jasna.media import UnsupportedColorspaceError, get_video_meta_data
from jasna.pipeline_items import FrameMeta, _SENTINEL
from jasna.pipeline_threads import decode_detect_loop, primary_restore_loop, secondary_restore_loop, blend_encode_loop
from jasna.streaming import HlsStreamingServer
from jasna.streaming_encoder import StreamingEncoder
from jasna.vram_offloader import VramOffloader

log = logging.getLogger(__name__)

_MAX_SEGMENTS_AHEAD = 3


class _StreamingFrameWriter:
    def __init__(
        self,
        streaming_encoder: StreamingEncoder,
        hls_server: HlsStreamingServer,
        start_segment: int,
    ):
        self._encoder = streaming_encoder
        self._hls_server = hls_server
        self._start_segment = start_segment
        self._frames_per_seg = hls_server.frames_per_segment()
        self._t0 = time.monotonic()
        self._cancel_event: threading.Event | None = None

    def set_cancel_event(self, cancel_event: threading.Event) -> None:
        self._cancel_event = cancel_event

    def write(self, frame: torch.Tensor, pts: int) -> None:
        self._encoder.write_frame(frame, pts)

    def after_write(self, frames_written: int) -> None:
        if frames_written == 1:
            log.debug("[stream-blend-encode] first frame encoded: %.2fs", time.monotonic() - self._t0)
        elif frames_written % 100 == 0:
            log.debug(
                "[stream-blend-encode] %d frames encoded (%.1f fps)",
                frames_written, frames_written / (time.monotonic() - self._t0),
            )

        current_seg = self._start_segment + frames_written // self._frames_per_seg
        self._hls_server.update_production(current_seg)
        cancel = self._cancel_event or threading.Event()
        self._hls_server.wait_for_demand(current_seg, _MAX_SEGMENTS_AHEAD, cancel)


def run_streaming(
    pipeline,
    port: int = 8765,
    segment_duration: float = 4.0,
    hls_server: HlsStreamingServer | None = None,
) -> None:
    from av.video.reformatter import Colorspace as AvColorspace
    device = pipeline.device
    metadata = get_video_meta_data(str(pipeline.input_video))
    if metadata.color_space != AvColorspace.ITU709:
        raise UnsupportedColorspaceError(
            f"Unsupported color space: {metadata.color_space!r} in {pipeline.input_video.name}. Only BT.709 is supported."
        )

    own_server = hls_server is None
    if own_server:
        hls_server = HlsStreamingServer(
            segment_duration=segment_duration,
            port=port,
        )
        hls_server.load_video(metadata)
        url = hls_server.start()
        print(f"HLS stream: {url}")
        print(f"Browser:    http://localhost:{port}/")
    else:
        hls_server.load_video(metadata)

    streaming_encoder = StreamingEncoder(
        segments_dir=hls_server.segments_dir,
        segment_duration=segment_duration,
        metadata=metadata,
        source_video=str(pipeline.input_video),
        device=device,
    )

    try:
        _streaming_loop(
            pipeline=pipeline,
            device=device,
            metadata=metadata,
            hls_server=hls_server,
            streaming_encoder=streaming_encoder,
        )
    finally:
        streaming_encoder.stop()
        if own_server:
            hls_server.stop()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats(device)


def _streaming_loop(
    *,
    pipeline,
    device: torch.device,
    metadata,
    hls_server: HlsStreamingServer,
    streaming_encoder: StreamingEncoder,
) -> None:
    start_segment = 0

    while True:
        start_time = hls_server.segment_start_time(start_segment)
        start_frame = hls_server.segment_start_frame(start_segment)
        log.info("[stream] starting pass from segment %d (frame %d, t=%.1fs)", start_segment, start_frame, start_time)

        seek_t0 = time.monotonic()
        hls_server.reset_demand(start_segment)
        if start_segment > 0:
            hls_server.notify_segment_requested(start_segment)
        streaming_encoder.flush_and_restart(start_number=start_segment) if start_segment > 0 else streaming_encoder.start(start_number=0)

        cancel_event = threading.Event()
        seek_result = _run_streaming_pass(
            pipeline=pipeline,
            device=device,
            metadata=metadata,
            hls_server=hls_server,
            streaming_encoder=streaming_encoder,
            start_segment=start_segment,
            start_frame=start_frame,
            start_time=start_time,
            cancel_event=cancel_event,
        )

        log.info("[stream] pass teardown took %.2fs", time.monotonic() - seek_t0)

        if hls_server.video_change.is_set():
            log.info("[stream] video change requested, exiting streaming loop")
            return

        if seek_result is None:
            streaming_encoder.stop()
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
    device: torch.device,
    metadata,
    hls_server: HlsStreamingServer,
    streaming_encoder: StreamingEncoder,
    start_segment: int,
    start_frame: int,
    start_time: float,
    cancel_event: threading.Event,
) -> int | None:
    secondary_workers = max(1, int(pipeline.restoration_pipeline.secondary_num_workers))

    clip_queue = FrameQueue(max_frames=pipeline.max_clip_size)
    secondary_queue = FrameQueue(max_frames=pipeline.max_clip_size * secondary_workers)
    encode_queue = FrameQueue(max_frames=pipeline.max_clip_size)
    metadata_queue: Queue[FrameMeta | object] = Queue(maxsize=pipeline.max_clip_size * 5)

    error_holder: list[BaseException] = []
    blend_buffer = BlendBuffer(device=device)
    crop_buffers: dict[int, CropBuffer] = {}
    crop_lock = threading.Lock()
    primary_idle_event = threading.Event()
    frame_shape: list[tuple[int, int]] = []

    vram_offloader = VramOffloader(
        device=device,
        blend_buffer=blend_buffer,
        crop_buffers=crop_buffers,
        crop_lock=crop_lock,
    )
    vram_offloader.set_pipeline_queues(clip_queue, secondary_queue, encode_queue, metadata_queue)

    seek_ts = start_time if start_time > 0 else None

    frame_writer = _StreamingFrameWriter(streaming_encoder, hls_server, start_segment)
    frame_writer.set_cancel_event(cancel_event)

    threads = [
        threading.Thread(
            target=lambda: decode_detect_loop(
                input_video=str(pipeline.input_video),
                batch_size=pipeline.batch_size,
                device=device,
                metadata=metadata,
                detection_model=pipeline.detection_model,
                max_clip_size=pipeline.max_clip_size,
                temporal_overlap=pipeline.temporal_overlap,
                enable_crossfade=pipeline.enable_crossfade,
                blend_buffer=blend_buffer,
                crop_buffers=crop_buffers,
                clip_queue=clip_queue,
                metadata_queue=metadata_queue,
                error_holder=error_holder,
                frame_shape=frame_shape,
                cancel_event=cancel_event,
                seek_ts=seek_ts,
            ),
            name="StreamDecodeDetect", daemon=True,
        ),
        threading.Thread(
            target=lambda: primary_restore_loop(
                device=device,
                restoration_pipeline=pipeline.restoration_pipeline,
                clip_queue=clip_queue,
                secondary_queue=secondary_queue,
                error_holder=error_holder,
                primary_idle_event=primary_idle_event,
                cancel_event=cancel_event,
            ),
            name="StreamPrimaryRestore", daemon=True,
        ),
        threading.Thread(
            target=lambda: secondary_restore_loop(
                device=device,
                restoration_pipeline=pipeline.restoration_pipeline,
                secondary_queue=secondary_queue,
                encode_queue=encode_queue,
                error_holder=error_holder,
                cancel_event=cancel_event,
            ),
            name="StreamSecondaryRestore", daemon=True,
        ),
        threading.Thread(
            target=lambda: blend_encode_loop(
                input_video=str(pipeline.input_video),
                batch_size=pipeline.batch_size,
                device=device,
                metadata=metadata,
                blend_buffer=blend_buffer,
                encode_queue=encode_queue,
                metadata_queue=metadata_queue,
                error_holder=error_holder,
                frame_writer=frame_writer,
                cancel_event=cancel_event,
                seek_ts=seek_ts,
                vram_offloader=vram_offloader,
            ),
            name="StreamBlendEncode", daemon=True,
        ),
    ]
    vram_offloader.start()
    for t in threads:
        t.start()

    seek_target: int | None = None
    while any(t.is_alive() for t in threads):
        if hls_server.video_change.is_set():
            log.info("[stream] video change detected, cancelling current pass")
            cancel_event.set()
            break
        target = hls_server.consume_seek()
        if target is not None:
            log.info("[stream] seek requested to segment %d, cancelling current pass", target)
            seek_target = target
            cancel_event.set()
            break
        time.sleep(0.1)

    log.debug("[stream] monitoring loop exited, waiting for threads to join (seek_target=%s)", seek_target)
    alive = [(t.name, t.is_alive()) for t in threads]
    log.debug("[stream] thread states: %s", alive)

    all_queues = [clip_queue, secondary_queue, encode_queue, metadata_queue]

    def _drain_all_queues():
        for q in all_queues:
            try:
                while True:
                    q.get_nowait()
            except Empty:
                pass

    t_join = time.monotonic()
    for t in threads:
        while t.is_alive():
            _drain_all_queues()
            t.join(timeout=0.1)
        log.debug("[stream] %s joined in %.2fs", t.name, time.monotonic() - t_join)
    log.debug("[stream] all threads joined in %.2fs", time.monotonic() - t_join)
    vram_offloader.stop()

    if error_holder:
        log.debug("[stream] errors collected: %s", [type(e).__name__ for e in error_holder])

    del clip_queue, secondary_queue, encode_queue, metadata_queue
    del blend_buffer, crop_buffers

    if error_holder and seek_target is None:
        raise error_holder[0]

    return seek_target
