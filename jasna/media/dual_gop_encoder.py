"""Validated Linux AMD HEVC temporal dual-session writer.

The restoration pipeline remains single-producer. Complete closed GOPs are
alternated between two persistent AMF encoders through bounded pinned-host
NV12/P010 queues, then validated and assembled in source display order.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from fractions import Fraction
import logging
from pathlib import Path
import queue
import shutil
import sys
import tempfile
import threading

import av
import torch

from jasna.accelerator import (
    AcceleratorVendor,
    current_stream,
    set_device,
    stream_context,
)
from jasna.media.splice import (
    concatenate_fragments,
    mux_fragments_final_output,
    validate_hevc_fragment_parameter_sets,
)
from jasna.media.video_encoder import (
    NvidiaVideoEncoder,
    _amf_host_native_output_eligible,
    _amf_host_zero_copy_override,
)


log = logging.getLogger(__name__)

AMD_DUAL_GOP_QUEUE_DEPTH = 8
AMD_DUAL_GOP_SIZE = 250
AMD_DUAL_GOP_AMF_ASYNC_DEPTH = 4
# Each worker can own one full input queue plus the AMF input window.  Keep one
# additional producer slot so the producer can prepare the next frame before a
# queue put blocks.  Reusing these exact pinned allocations avoids the ROCm
# host allocator retaining a growing series of 48/96-MiB blocks on long runs.
AMD_DUAL_GOP_PINNED_POOL_SIZE = (
    2 * (AMD_DUAL_GOP_QUEUE_DEPTH + AMD_DUAL_GOP_AMF_ASYNC_DEPTH) + 1
)


def _source_requirements(
    metadata,
    *,
    smart_fragment: bool,
) -> dict[str, bool]:
    source_codec = str(getattr(metadata, "codec_name", "")).strip().lower()
    source_pixel_format = str(
        getattr(metadata, "pixel_format", "") or ""
    ).strip().lower()
    source_profile = str(
        getattr(metadata, "profile", "") or ""
    ).strip().lower()
    forced_hevc_experiment = _amf_host_zero_copy_override() is True
    ten_bit = bool(getattr(metadata, "is_10bit", False))
    supported_main10_input = (
        ten_bit
        and source_pixel_format in {"p010le", "yuv420p10le"}
    )
    supported_main8_input = (
        not ten_bit
        and source_pixel_format in {"nv12", "yuv420p"}
    )
    expected_output_format = "p010le" if ten_bit else "nv12"
    width = int(getattr(metadata, "video_width", 0))
    height = int(getattr(metadata, "video_height", 0))
    positive_even_geometry = (
        width > 0
        and height > 0
        and width % 2 == 0
        and height % 2 == 0
    )
    validated_output_geometry = _amf_host_native_output_eligible(
        width=width,
        height=height,
        ten_bit=ten_bit,
        frame_format=expected_output_format,
    )
    smart_hevc_profile = (
        source_profile in ({"", "main 10", "main10"} if ten_bit else {"", "main"})
    )
    return {
        # Full-video output is wholly re-encoded and therefore depends on the
        # requested HEVC output contract, not on the source codec/profile.
        # Smart Render also copies source HEVC packets, so it keeps the stricter
        # source compatibility gate.
        "HEVC-compatible Smart Render source": (
            not smart_fragment
            or (
                source_codec in {"hevc", "h265"}
                and smart_hevc_profile
            )
        ),
        "validated output geometry": (
            validated_output_geometry
            or (forced_hevc_experiment and positive_even_geometry)
        ),
        "NV12/P010-compatible 4:2:0 input": (
            supported_main10_input or supported_main8_input
        ),
        "positive source video bitrate": (
            int(getattr(metadata, "video_bitrate", 0)) > 0
        ),
    }


def amd_dual_gop_metadata_eligible(
    metadata,
    *,
    smart_fragment: bool = False,
) -> bool:
    """Return whether metadata maps to a validated dual-GOP HEVC output."""

    return all(
        _source_requirements(
            metadata,
            smart_fragment=smart_fragment,
        ).values()
    )


def validate_amd_dual_gop_source(
    metadata,
    *,
    smart_fragment: bool = False,
) -> None:
    """Reject requests outside the real-video acceptance matrix."""

    missing = [
        name
        for name, satisfied in _source_requirements(
            metadata,
            smart_fragment=smart_fragment,
        ).items()
        if not satisfied
    ]
    if missing:
        raise ValueError(
            "AMD dual-GOP encoding is unavailable for this source; missing: "
            + ", ".join(missing)
        )


def use_dual_gop_writer(
    encoder: NvidiaVideoEncoder,
    *,
    enabled: bool,
) -> bool:
    """Fail closed unless the request matches a validated product shape."""

    if not enabled:
        return False
    metadata = encoder.metadata
    options = encoder.encoder_options
    requirements = {
        "Linux": sys.platform == "linux",
        "AMD": encoder.vendor is AcceleratorVendor.AMD,
        "HEVC output": encoder.codec == "hevc",
        "NV12/P010 output matching the source bit depth": (
            str(getattr(encoder.spec, "frame_format", "")).lower()
            in {"nv12", "p010le"}
            and bool(encoder.spec.ten_bit)
            == bool(getattr(metadata, "is_10bit", False))
        ),
        **_source_requirements(
            metadata,
            smart_fragment=bool(encoder.smart_fragment),
        ),
        "host-native AMF input": bool(
            getattr(encoder, "_amf_host_zero_copy", False)
        ),
        "non-fMP4 output": not bool(encoder.fmp4),
        "source frame rate": (
            Fraction(encoder.output_fps)
            == Fraction(metadata.video_fps_exact)
        ),
        "automatic source-rate VBR Peak": (
            bool(encoder.auto_source_rate)
            and str(options.get("rc", "")).lower() in {"vbr_peak", "2"}
            and int(options.get("maxrate", 0)) > 0
            and int(options.get("bufsize", 0)) > 0
            and int(getattr(encoder, "_target_bit_rate", 0) or 0) > 0
            and str(options.get("preanalysis", "")) == "0"
        ),
        f"fixed GOP {AMD_DUAL_GOP_SIZE}": (
            int(options.get("g", AMD_DUAL_GOP_SIZE)) == AMD_DUAL_GOP_SIZE
        ),
        "no B-frames": int(options.get("bf", 0)) == 0,
        f"fixed host async depth {AMD_DUAL_GOP_AMF_ASYNC_DEPTH}": (
            str(options.get("async_depth", ""))
            == str(AMD_DUAL_GOP_AMF_ASYNC_DEPTH)
            and str(options.get("host_zero_copy", "")) == "1"
        ),
    }
    missing = [name for name, satisfied in requirements.items() if not satisfied]
    if missing:
        raise RuntimeError(
            "AMD dual-GOP encoding is unavailable for this request; missing: "
            + ", ".join(missing)
        )
    return True


@dataclass(frozen=True)
class _StartGroup:
    index: int
    normalized: Path
    pts_origin: int


@dataclass(frozen=True)
class _CompletedGroup:
    index: int
    path: Path
    first_pts: int
    last_pts: int


@dataclass(frozen=True)
class _Frame:
    frame: av.VideoFrame
    pts: int
    host_yuv: torch.Tensor


@dataclass(frozen=True)
class _EndGroup:
    pass


@dataclass(frozen=True)
class _Stop:
    pass


@dataclass(frozen=True)
class _Abort:
    pass


class _PinnedHostFramePool:
    """Bound the large host allocations shared by both encoder workers."""

    def __init__(self, *, shape: tuple[int, int], dtype: torch.dtype) -> None:
        self.shape = shape
        self.dtype = dtype
        self.capacity = AMD_DUAL_GOP_PINNED_POOL_SIZE
        self.available: queue.Queue[torch.Tensor] = queue.Queue(
            maxsize=self.capacity
        )
        self._lock = threading.Lock()
        self.allocated = 0
        self.in_use = 0
        self.peak_in_use = 0
        self.peak_allocated = 0
        self._closed = False

    def acquire(self, failed: threading.Event) -> torch.Tensor:
        while True:
            allocate = False
            with self._lock:
                if self._closed:
                    raise RuntimeError("pinned host frame pool is closed")
                try:
                    host_yuv = self.available.get_nowait()
                except queue.Empty:
                    host_yuv = None
                    allocate = self.allocated < self.capacity
                    if allocate:
                        self.allocated += 1
                else:
                    self.in_use += 1
                    self.peak_in_use = max(self.peak_in_use, self.in_use)
                    return host_yuv
            if not allocate:
                with self._lock:
                    if failed.is_set():
                        raise RuntimeError(
                            "dual GOP encoder failed while waiting for a "
                            "reusable pinned host frame"
                        )
                try:
                    host_yuv = self.available.get(timeout=0.1)
                except queue.Empty:
                    continue
            else:
                try:
                    host_yuv = torch.empty(
                        self.shape,
                        dtype=self.dtype,
                        pin_memory=True,
                    )
                except BaseException:
                    with self._lock:
                        self.allocated -= 1
                    raise
                with self._lock:
                    self.peak_allocated = max(
                        self.peak_allocated,
                        self.allocated,
                    )
            with self._lock:
                if self._closed:
                    if allocate:
                        self.allocated -= 1
                    raise RuntimeError("pinned host frame pool is closed")
                self.in_use += 1
                self.peak_in_use = max(self.peak_in_use, self.in_use)
            return host_yuv

    def release(self, host_yuv: torch.Tensor) -> None:
        with self._lock:
            if self.in_use <= 0:
                raise RuntimeError("pinned host frame pool release underflow")
            self.in_use -= 1
            if self._closed:
                # Drop the last Python owner immediately instead of leaving a
                # large pinned block in the pool until a later GC pass.
                self.allocated -= 1
                return
            try:
                self.available.put_nowait(host_yuv)
            except queue.Full as exc:  # pragma: no cover - invariant guard
                raise RuntimeError("pinned host frame pool overflow") from exc

    def close(self) -> None:
        """Drop all idle pinned owners at a deterministic session boundary."""

        with self._lock:
            if self._closed:
                return
            if self.in_use:
                raise RuntimeError(
                    "cannot close pinned host frame pool with "
                    f"{self.in_use} frame(s) still in use"
                )
            self._closed = True
            released = 0
            while True:
                try:
                    self.available.get_nowait()
                except queue.Empty:
                    break
                released += 1
            self.allocated = max(0, self.allocated - released)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed


class _EncoderWorker(threading.Thread):
    def __init__(
        self,
        *,
        worker_index: int,
        template: NvidiaVideoEncoder,
        work_dir: Path,
        failed: threading.Event,
        host_pool: _PinnedHostFramePool,
    ) -> None:
        super().__init__(name=f"AmdDualGopEncoder{worker_index}", daemon=True)
        self.worker_index = worker_index
        self.template = template
        self.work_dir = work_dir
        # Matroska rounds timestamps to milliseconds.  That is harmless for a
        # short sample but can exceed the strict one-millisecond source PTS
        # tolerance after many alternating GOPs.  NUT preserves the source
        # 1/60000 time base until the final MPEG-TS normalization.
        self.combined_output = work_dir / f"encoder-{worker_index}.nut"
        self.items: queue.Queue[object] = queue.Queue(
            maxsize=AMD_DUAL_GOP_QUEUE_DEPTH
        )
        self.failed = failed
        self.host_pool = host_pool
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.fragments: list[_CompletedGroup] = []
        self.groups: list[tuple[_StartGroup, int, int]] = []
        self.pending_host_yuv: dict[int, torch.Tensor] = {}

    def run(self) -> None:
        encoder: NvidiaVideoEncoder | None = None
        try:
            set_device(self.template.device)
            encoder = self._session_encoder()
            encoder.__enter__()
            # Allocate both AMF sessions before the restoration model reaches
            # its peak. Lazy session creation caused rejected 24-GiB spikes.
            encoder.out_stream.codec_context.open()
            self.ready.set()
            self._run(encoder)
            completed_encoder = encoder
            encoder = None
            completed_encoder.__exit__(None, None, None)
            self._release_pending_host_frames()
            # An abort can race with a worker consuming the queued stop
            # marker.  Closing that encoder is still required to release its
            # native state, but splitting a partially drained NUT afterwards
            # only manufactures secondary "extra packets" failures and can
            # retain more packet/surface owners during pressure recovery.
            if self.failed.is_set():
                return
            self._split_groups()
        except BaseException as exc:
            self.error = exc
            self.failed.set()
            log.exception("[dual-gop-encoder-%d] crashed", self.worker_index)
        finally:
            self.ready.set()
            if encoder is not None:
                try:
                    encoder.__exit__(RuntimeError, RuntimeError("aborted"), None)
                except BaseException:
                    log.warning(
                        "Could not cleanly abort dual GOP encoder %d",
                        self.worker_index,
                        exc_info=True,
                    )
            self._release_pending_host_frames()
            # Promptly recycle queued 48/96-MiB pinned owners after failure.
            while True:
                try:
                    item = self.items.get_nowait()
                except queue.Empty:
                    break
                else:
                    self._release_queued_item(item)
                    self.items.task_done()

    def _release_queued_item(self, item: object) -> None:
        if isinstance(item, _Frame):
            self.host_pool.release(item.host_yuv)

    def _release_pending_host_frames(self) -> None:
        pending = list(self.pending_host_yuv.values())
        self.pending_host_yuv.clear()
        for host_yuv in pending:
            self.host_pool.release(host_yuv)

    def _release_packet_host_frame(self, packet: av.Packet) -> None:
        if packet.pts is None:
            raise RuntimeError(
                f"persistent encoder {self.worker_index} returned a packet "
                "without PTS"
            )
        host_yuv = self.pending_host_yuv.pop(int(packet.pts), None)
        if host_yuv is None:
            raise RuntimeError(
                f"persistent encoder {self.worker_index} returned unexpected "
                f"packet PTS {packet.pts}"
            )
        self.host_pool.release(host_yuv)

    def _session_encoder(self) -> NvidiaVideoEncoder:
        encoder = copy.copy(self.template)
        encoder.file = str(self.combined_output)
        encoder.output_path = self.combined_output
        encoder.mux_audio = False
        encoder.pts_origin = 0
        encoder.smart_fragment = True
        encoder.fmp4 = False
        encoder.encoder_options = dict(self.template.encoder_options)
        encoder.encoder_options.update(encoder.spec.smart_fragment_options)
        return encoder

    def _run(self, encoder: NvidiaVideoEncoder) -> None:
        current: _StartGroup | None = None
        frame_count = 0
        group_session_origin = 0
        next_session_pts = 0
        last_session_pts: int | None = None
        first_pts: int | None = None
        last_pts: int | None = None
        frame_step = (
            Fraction(1, 1)
            / Fraction(self.template.output_fps)
            / Fraction(self.template.metadata.time_base)
        )
        while True:
            item = self.items.get()
            try:
                if isinstance(item, _Stop):
                    if current is not None:
                        raise RuntimeError("dual GOP worker stopped mid-fragment")
                    return
                if isinstance(item, _Abort):
                    return
                if isinstance(item, _StartGroup):
                    if current is not None:
                        raise RuntimeError("dual GOP fragment overlap in one worker")
                    current = item
                    group_session_origin = next_session_pts
                    frame_count = 0
                    last_session_pts = None
                    first_pts = None
                    last_pts = None
                    continue
                if isinstance(item, _Frame):
                    if current is None:
                        raise RuntimeError("dual GOP frame arrived outside a fragment")
                    if first_pts is None:
                        first_pts = item.pts
                    last_pts = item.pts
                    last_session_pts = (
                        group_session_origin + item.pts - current.pts_origin
                    )
                    if last_session_pts in self.pending_host_yuv:
                        self.host_pool.release(item.host_yuv)
                        raise RuntimeError(
                            f"persistent encoder {self.worker_index} received "
                            f"duplicate session PTS {last_session_pts}"
                        )
                    self.pending_host_yuv[last_session_pts] = item.host_yuv
                    item.frame.pts = last_session_pts
                    item.frame.time_base = encoder.metadata.time_base
                    for packet in encoder.out_stream.encode(item.frame):
                        self._release_packet_host_frame(packet)
                        encoder._mux_video(packet)
                    frame_count += 1
                    # The DLPack frame is an owner of the pinned tensor in
                    # addition to ``pending_host_yuv``.  Do not keep the last
                    # frame alive across the next queue wait; AMF's packet PTS
                    # is the sole lifetime lease for the host allocation.
                    del item
                    continue
                if isinstance(item, _EndGroup):
                    if current is None or frame_count == 0:
                        raise RuntimeError("dual GOP fragment ended without frames")
                    if last_session_pts is None or first_pts is None or last_pts is None:
                        raise RuntimeError("dual GOP fragment has no final timestamp")
                    if first_pts != current.pts_origin:
                        raise RuntimeError(
                            f"dual GOP fragment {current.index} started at {first_pts}, "
                            f"expected {current.pts_origin}"
                        )
                    self.groups.append((current, frame_count, last_pts))
                    next_session_pts = last_session_pts + round(frame_step)
                    log.info(
                        "Dual AMF GOP %d submitted to persistent encoder %d: "
                        "%d frames, pts=%s..%s",
                        current.index,
                        self.worker_index,
                        frame_count,
                        first_pts,
                        last_pts,
                    )
                    current = None
                    frame_count = 0
                    last_session_pts = None
                    first_pts = None
                    last_pts = None
                    continue
                raise RuntimeError(f"unknown dual GOP work item: {type(item)!r}")
            finally:
                self.items.task_done()

    def _split_groups(self) -> None:
        if not self.groups:
            self.combined_output.unlink(missing_ok=True)
            return

        with av.open(str(self.combined_output)) as source:
            in_stream = source.streams.video[0]
            packets = (
                packet
                for packet in source.demux(in_stream)
                if packet.size > 0 and packet.pts is not None
            )
            for group, expected_frames, last_source_pts in self.groups:
                first_pts: int | None = None
                first_dts: int | None = None
                written = 0
                group.normalized.unlink(missing_ok=True)
                with av.open(
                    str(group.normalized),
                    "w",
                    format="mpegts",
                    container_options={"mpegts_copyts": "1", "muxdelay": "0"},
                ) as destination:
                    out_stream = destination.add_stream_from_template(in_stream)
                    while written < expected_frames:
                        try:
                            packet = next(packets)
                        except StopIteration as exc:
                            raise RuntimeError(
                                f"persistent encoder {self.worker_index} ended "
                                f"inside GOP {group.index} at "
                                f"{written}/{expected_frames}"
                            ) from exc
                        if written == 0:
                            if not packet.is_keyframe:
                                raise RuntimeError(
                                    f"persistent encoder GOP {group.index} does "
                                    "not begin at a keyframe"
                                )
                            first_pts = int(packet.pts)
                            first_dts = int(
                                packet.dts if packet.dts is not None else packet.pts
                            )
                        packet.pts = int(packet.pts) - int(first_pts)
                        if packet.dts is not None:
                            packet.dts = int(packet.dts) - int(first_dts)
                        packet.stream = out_stream
                        destination.mux(packet)
                        written += 1
                self.fragments.append(
                    _CompletedGroup(
                        index=group.index,
                        path=group.normalized,
                        first_pts=group.pts_origin,
                        last_pts=last_source_pts,
                    )
                )
            try:
                extra = next(packets)
            except StopIteration:
                extra = None
            if extra is not None:
                raise RuntimeError(
                    f"persistent encoder {self.worker_index} produced extra packets"
                )
        self.combined_output.unlink(missing_ok=True)

    def abort(self) -> None:
        while True:
            try:
                item = self.items.get_nowait()
            except queue.Empty:
                break
            else:
                self._release_queued_item(item)
                self.items.task_done()
        if self.is_alive():
            try:
                self.items.put_nowait(_Abort())
            except queue.Full:  # pragma: no cover - drained immediately above
                pass


class AmdDualGopFrameWriter:
    """Prepare once, then encode alternate closed GOPs on two AMF sessions."""

    def __init__(self, template: NvidiaVideoEncoder) -> None:
        use_dual_gop_writer(template, enabled=True)
        self.template = template
        self.gop_frames = int(template.encoder_options.get("g", 0))
        if self.gop_frames != AMD_DUAL_GOP_SIZE:
            raise RuntimeError(
                f"dual GOP writer requires GOP {AMD_DUAL_GOP_SIZE}"
            )
        output = template.output_path
        output.parent.mkdir(parents=True, exist_ok=True)
        self.work_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output.stem}.dual-gop-",
                dir=output.parent,
            )
        )
        self.failed = threading.Event()
        height = template.metadata.video_height
        width = template.metadata.video_width
        dtype = torch.uint16 if template.spec.ten_bit else torch.uint8
        self.host_pool = _PinnedHostFramePool(
            shape=(height + height // 2, width),
            dtype=dtype,
        )
        self.workers = [
            _EncoderWorker(
                worker_index=index,
                template=template,
                work_dir=self.work_dir,
                failed=self.failed,
                host_pool=self.host_pool,
            )
            for index in range(2)
        ]
        # Mark a partially constructed writer as closed while the native
        # sessions are being opened.  If either session fails, the caller may
        # still hold the object through an exception path, but no producer
        # write is allowed and all started workers/pinned owners must be
        # isolated before the original AMF error is re-raised.
        self.closed = True
        self._workers_stopped = False
        try:
            for worker in self.workers:
                worker.start()
                if not worker.ready.wait(timeout=180):
                    raise RuntimeError(
                        f"persistent dual GOP encoder {worker.worker_index} "
                        "did not initialize"
                    )
                if worker.error is not None:
                    raise RuntimeError(
                        f"persistent dual GOP encoder {worker.worker_index} failed "
                        f"during initialization: {worker.error!r}"
                    ) from worker.error
        except BaseException:
            self.failed.set()
            workers_stopped = False
            try:
                workers_stopped = self._abort_workers()
            except BaseException:
                log.warning(
                    "Could not abort all dual GOP workers after initialization failure",
                    exc_info=True,
                )
            if workers_stopped:
                self._close_host_pool(strict=False)
                self._release_template_runtime()
                self._trim_host_allocator()
            else:
                # Do not mutate the shallow-copied encoder template or trim
                # the allocator while a native worker may still be executing.
                # The isolated child is the hard recovery boundary for this
                # case; keeping these owners attached avoids a use-after-free
                # in a driver thread that has not acknowledged abort yet.
                log.error(
                    "Dual GOP initialization worker remained alive; deferring "
                    "template/pinned-pool teardown to isolated process exit"
                )
            log.error(
                "Preserving failed dual GOP initialization workspace for diagnosis: %s",
                self.work_dir,
            )
            raise
        self.frame_count = 0
        self.group_index = -1
        self.group_worker: _EncoderWorker | None = None
        self.closed = False

        template.stream = current_stream(template.device)
        template._packed = torch.empty(
            (height + height // 2, width),
            dtype=template._converter.sample_dtype,
            device=template.device,
        )
        template._cas_luma = (
            torch.empty_like(template._packed[:height])
            if template._cas is not None
            else None
        )
        log.info(
            "Parallel dual AMF GOP writer enabled: gop=%d, queue=%d, "
            "shared pinned %s pool max=%d, two persistent encoders, one producer",
            self.gop_frames,
            AMD_DUAL_GOP_QUEUE_DEPTH,
            "P010" if template.spec.ten_bit else "NV12",
            AMD_DUAL_GOP_PINNED_POOL_SIZE,
        )

    def _errors(self) -> list[BaseException]:
        return [worker.error for worker in self.workers if worker.error is not None]

    @staticmethod
    def _trim_host_allocator() -> None:
        """Release cached pinned blocks after every native session boundary."""

        try:
            # ROCm's pinned allocator can retain freed blocks even after the
            # Python queue drops its owners.  These calls are intentionally
            # best-effort: an AMF/native exception remains the primary cause.
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except (AttributeError, RuntimeError):
            log.debug(
                "Could not trim the pinned host allocator",
                exc_info=True,
            )

    def _close_host_pool(self, *, strict: bool) -> None:
        """Close the shared pool only after worker leases have drained."""

        close_pool = getattr(self.host_pool, "close", None)
        if not callable(close_pool):
            return
        try:
            close_pool()
        except BaseException:
            if strict:
                raise
            log.warning(
                "Could not close dual GOP pinned host frame pool cleanly",
                exc_info=True,
            )
            return

    def _put(self, worker: _EncoderWorker, item: object) -> None:
        while True:
            if self.failed.is_set():
                raise RuntimeError(
                    f"dual GOP encoder failed: "
                    f"{[repr(error) for error in self._errors()]}"
                )
            try:
                worker.items.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _prepare(
        self,
        frame: torch.Tensor,
        *,
        apply_lut: bool,
    ) -> tuple[av.VideoFrame, torch.Tensor]:
        template = self.template
        height = template.metadata.video_height
        with stream_context(template.stream):
            if apply_lut and template._lut_applier is not None:
                frame = template._lut_applier.apply(frame)
            packed = template._to_yuv(frame, height)
        template.stream.synchronize()
        host_yuv = self.host_pool.acquire(self.failed)
        try:
            host_yuv.copy_(
                packed.view(torch.uint16) if template.spec.ten_bit else packed,
                non_blocking=False,
            )
            video_frame = av.VideoFrame.from_dlpack(
                [host_yuv[:height], host_yuv[height:]],
                format=template.spec.frame_format,
            )
        except BaseException:
            self.host_pool.release(host_yuv)
            raise
        return video_frame, host_yuv

    def write(self, frame: torch.Tensor, pts: int, *, apply_lut: bool = True) -> None:
        if self.closed:
            raise RuntimeError("dual GOP writer is already closed")
        within_group = self.frame_count % self.gop_frames
        if within_group == 0:
            self.group_index += 1
            self.group_worker = self.workers[self.group_index % len(self.workers)]
            normalized = self.work_dir / f"gop-{self.group_index:06d}.ts"
            self._put(
                self.group_worker,
                _StartGroup(self.group_index, normalized, int(pts)),
            )
        prepared, host_yuv = self._prepare(frame, apply_lut=apply_lut)
        try:
            self._put(
                self.group_worker,
                _Frame(prepared, int(pts), host_yuv),
            )
        except BaseException:
            self.host_pool.release(host_yuv)
            raise
        self.frame_count += 1
        if self.frame_count % self.gop_frames == 0:
            self._put(self.group_worker, _EndGroup())
            self.group_worker = None

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        succeeded = False
        try:
            if self.frame_count == 0:
                raise RuntimeError("dual GOP writer received no frames")
            if self.group_worker is not None:
                self._put(self.group_worker, _EndGroup())
                self.group_worker = None
            for worker in self.workers:
                self._put(worker, _Stop())
            for worker in self.workers:
                worker.join(timeout=1800)
                if worker.is_alive():
                    raise RuntimeError(
                        f"dual GOP encoder {worker.worker_index} did not stop"
                    )
            self._workers_stopped = True
            if self.host_pool.in_use:
                raise RuntimeError(
                    "dual GOP pinned host frame pool did not fully drain: "
                    f"{self.host_pool.in_use} still in use"
                )
            log.info(
                "Dual AMF pinned host frame pool: allocated=%d/%d, "
                "peak in-use=%d, peak allocated=%d",
                self.host_pool.allocated,
                self.host_pool.capacity,
                self.host_pool.peak_in_use,
                getattr(self.host_pool, "peak_allocated", 0),
            )
            try:
                host_stats = torch.cuda.host_memory_stats()
            except (AttributeError, RuntimeError):
                log.debug(
                    "Could not read pinned host allocator statistics",
                    exc_info=True,
                )
            else:
                log.info(
                    "Pinned host allocator: allocated=%.1f MiB, "
                    "active=%.1f MiB, peak allocated=%.1f MiB, "
                    "peak active=%.1f MiB",
                    host_stats.get("allocated_bytes.current", 0) / (1024**2),
                    host_stats.get("active_bytes.current", 0) / (1024**2),
                    host_stats.get("allocated_bytes.peak", 0) / (1024**2),
                    host_stats.get("active_bytes.peak", 0) / (1024**2),
                )
            errors = self._errors()
            if errors:
                raise RuntimeError(
                    f"dual GOP encoders failed: "
                    f"{[repr(error) for error in errors]}"
                ) from errors[0]
            fragments = sorted(
                (
                    fragment
                    for worker in self.workers
                    for fragment in worker.fragments
                ),
                key=lambda fragment: fragment.index,
            )
            if len(fragments) != self.group_index + 1:
                raise RuntimeError(
                    f"dual GOP output is incomplete: got {len(fragments)} "
                    f"of {self.group_index + 1} fragments"
                )
            frame_step = (
                Fraction(1, 1)
                / Fraction(self.template.output_fps)
                / Fraction(self.template.metadata.time_base)
            )
            paths = _mux_paths_with_source_durations(
                fragments,
                time_base=Fraction(self.template.metadata.time_base),
                frame_step=frame_step,
            )
            validate_hevc_fragment_parameter_sets(
                [(path, "render") for path, _duration in paths]
            )
            log.info(
                "Dual AMF GOP fragment validation passed: %d independent "
                "VPS/SPS/PPS access points",
                len(paths),
            )
            if self.template.smart_fragment:
                concatenate_fragments(
                    paths,
                    manifest=self.work_dir / "fragments.ffconcat",
                    destination=self.template.output_path,
                    codec="hevc",
                )
            else:
                mux_fragments_final_output(
                    paths,
                    Path(self.template.metadata.video_file),
                    self.template.output_path,
                    manifest=self.work_dir / "fragments.ffconcat",
                    codec="hevc",
                )
            succeeded = True
        except BaseException:
            try:
                self._abort_workers()
            except BaseException:
                # Preserve the first encode/mux failure.  Worker teardown
                # diagnostics remain in the log and the finally block still
                # performs pool/template cleanup.
                log.warning(
                    "Could not abort all dual GOP workers after close failure",
                    exc_info=True,
                )
            raise
        finally:
            pool_error: BaseException | None = None
            workers_stopped = bool(getattr(self, "_workers_stopped", False))
            if workers_stopped:
                try:
                    self._close_host_pool(strict=succeeded)
                except BaseException as error:
                    # Always detach template references even when a successful
                    # output encountered a teardown fault.  The output is kept
                    # for diagnosis and the teardown error is raised afterwards.
                    pool_error = error
                    log.error(
                        "Could not close dual GOP pinned host frame pool after a "
                        "successful encode",
                        exc_info=True,
                    )
                finally:
                    self._release_template_runtime()
                    self._trim_host_allocator()
            else:
                log.error(
                    "Skipping dual GOP pinned-pool/template teardown because a "
                    "native worker did not stop; isolated process must reap it"
                )
            if succeeded and pool_error is None:
                shutil.rmtree(self.work_dir, ignore_errors=True)
            else:
                log.error(
                    "Preserving failed dual GOP workspace for diagnosis: %s",
                    self.work_dir,
                )
            if pool_error is not None:
                raise pool_error

    def abort(self) -> None:
        if self.closed:
            return
        self.closed = True
        # Make the abort visible to workers before draining their queues.  A
        # worker that has already consumed _Stop must skip post-close packet
        # splitting; the isolated process remains the hard native recovery
        # boundary when a driver call does not acknowledge the abort.
        failed = getattr(self, "failed", None)
        if failed is not None:
            failed.set()
        try:
            self._abort_workers()
        finally:
            # A still-running native worker may retain a host owner.  Do not
            # mutate shared template state or trim the allocator until every
            # worker has acknowledged abort; the isolated process is the hard
            # recovery boundary when a native call remains stuck.
            if bool(getattr(self, "_workers_stopped", False)):
                self._close_host_pool(strict=False)
                self._release_template_runtime()
                self._trim_host_allocator()
            else:
                log.error(
                    "Skipping dual GOP abort teardown because a native worker "
                    "did not stop; isolated process must reap it"
                )
            log.warning(
                "Preserving aborted dual GOP workspace for diagnosis: %s",
                self.work_dir,
            )

    def _abort_workers(self) -> bool:
        all_stopped = True
        for worker in self.workers:
            worker.abort()
        for worker in self.workers:
            # Initialization is sequential.  If encoder 0 fails before encoder
            # 1 has been started, joining the untouched Thread would raise
            # ``RuntimeError: cannot join thread before it is started`` and
            # hide the real AMF initialization failure.
            if worker.ident is None:
                continue
            worker.join(timeout=30)
            if worker.is_alive():
                all_stopped = False
                log.error(
                    "Dual GOP encoder %d did not stop during abort",
                    worker.worker_index,
                )
        self._workers_stopped = all_stopped
        return all_stopped

    def _release_template_runtime(self) -> None:
        self.template._packed = None
        self.template._cas_luma = None
        self.template._converter = None
        self.template._lut_applier = None
        self.template._cas = None


def _mux_paths_with_source_durations(
    fragments: list[_CompletedGroup],
    *,
    time_base: Fraction,
    frame_step: Fraction,
) -> list[tuple[Path, float]]:
    """Preserve source gaps, including a gap exactly on a GOP boundary."""

    if not fragments:
        raise RuntimeError("dual GOP output contains no fragments")
    paths: list[tuple[Path, float]] = []
    for position, fragment in enumerate(fragments):
        if fragment.index != position:
            raise RuntimeError(
                f"dual GOP fragment order is incomplete at {position}: "
                f"got {fragment.index}"
            )
        if position + 1 < len(fragments):
            duration_pts = fragments[position + 1].first_pts - fragment.first_pts
        else:
            duration_pts = (
                fragment.last_pts - fragment.first_pts + round(frame_step)
            )
        if duration_pts <= 0:
            raise RuntimeError(
                f"dual GOP fragment {fragment.index} has non-positive source "
                f"duration {duration_pts}"
            )
        paths.append((fragment.path, float(Fraction(duration_pts) * time_base)))
    return paths
