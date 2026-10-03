import ctypes
import importlib
import json
import logging
import os
import sys
import threading
from fractions import Fraction
from typing import Iterator

import av
import torch
from av.codec.hwaccel import HWAccel
from av.video.reformatter import ColorRange as AvColorRange, VideoReformatter

from jasna.accelerator import (
    AcceleratorVendor,
    current_stream,
    new_stream,
    stream_context,
    vendor_for_device,
)
from jasna.media.container_utils import demux_video
from jasna.media.probe import VideoMetadata, resolve_video_start_pts
from jasna.media.cuda_kernel import create_stream, destroy_stream
from jasna.media.yuv_to_rgb import YuvToRgbConverter
from jasna.media.windows_d3d11_hip_resident import (
    WINDOWS_D3D11_HIP_RESIDENT_BACKEND,
)
from jasna.native_worker import run_amf_decoder_open_with_watchdog
from jasna.gpu_context_errors import (
    NativeGpuContextUnusableError,
    is_windows_amf_host_transfer_failure,
)

log = logging.getLogger(__name__)

CORRUPT_PACKET_TOLERANCE = 10

# Decode backend selection through `JASNA_DECODE_BACKEND`:
# - "auto":    NVIDIA tries VALI first and falls back to PyAV hwaccel, then PyAV
#              software, when VALI cannot open or decode the first frame. AMD
#              keeps its AMF -> software escalation. Windows AMD explicitly
#              uses software decode plus ROCm upload for HEVC Main10 and AV1.
#              Linux AMD uses the fixed-context AMF Vulkan/HIP route for the
#              documented H.264/HEVC formats and the stable dma-buf identity
#              cache for documented AV1 Main NV12/P010 formats.
# - "vali":    VALI only; any failure raises (NVIDIA only).
# - "pyav-hw": skip VALI, use the PyAV hwaccel path with its software fallback.
# - "pyav-sw": force FFmpeg software decoding with GPU upload on every vendor.
# - "amf-interop": explicit Linux AMD diagnostic backend.  It accepts only the
#                  documented H.264/HEVC/AV1 AMF Vulkan surface scope and copies
#                  directly to HIP, or raises. Linux AMD auto selects the proven
#                  private-deferred route for eligible H.264/HEVC sources and
#                  the stable dma-buf cache for eligible AV1 sources.
# - "amf-d3d11-hip-resident": explicit Windows AMD Main8/NV12 backend. It is
#                  accepted only with a pipeline-owned coordinator; no global
#                  or auto selection may silently construct one.
DECODE_BACKEND = "auto"
DECODE_BACKEND_ENV = "JASNA_DECODE_BACKEND"
_DECODE_BACKENDS = (
    "auto",
    "vali",
    "pyav-hw",
    "pyav-sw",
    "amf-interop",
    WINDOWS_D3D11_HIP_RESIDENT_BACKEND,
)

_AMF_INTEROP_MODULE = "_jasna_amf_surface_probe"
AMF_INTEROP_RESOURCE_CACHE_ENV = "JASNA_AMF_INTEROP_RESOURCE_CACHE"
AMF_INTEROP_DECODE_COPY_STREAM_ENV = "JASNA_AMF_INTEROP_DECODE_COPY_STREAM"
AMF_INTEROP_SURFACE_POOL_SIZE_ENV = "JASNA_AMF_INTEROP_SURFACE_POOL_SIZE"
AMF_INTEROP_8K_B4_SURFACE_POOL_SIZE = 16
AMF_INTEROP_STATS_PREFIX = "AMF interop transport stats reader="
_AMF_INTEROP_READER_BATCH_SIZES = frozenset({1, 2, 4, 8})
_AMF_INTEROP_DECODE_COPY_STREAMS = frozenset({"null", "private-deferred"})
# Two product readers start together (decode/detect plus blend/encode).  The
# Linux AMF runtime can deadlock in its process-global device-host setup when
# both threads enter CodecContext.open() at once.  Creation is a one-time
# session operation, so serialize only this narrow boundary; frame decode and
# Vulkan/HIP copies remain concurrent after both sessions have opened.
_AMF_DECODER_OPEN_LOCK = threading.Lock()

# PyAV's avcodec_find_decoder returns libdav1d for AV1, which carries no NVDEC
# hwaccel config, so av.open silently decodes AV1 in software. Force the native
# FFmpeg av1 decoder, which does carry the CUDA hwaccel config. Keyed by codec
# name whose default PyAV decoder lacks NVDEC (only AV1 today).
_NVDEC_DECODER_OVERRIDES = {"av1": "av1"}
_NVDEC_MIN_CODED_SIZE = {"av1": (128, 128)}


class VideoDecodeError(RuntimeError):
    pass


def _requires_windows_amd_software_decode(
    metadata: VideoMetadata,
    vendor: AcceleratorVendor,
) -> bool:
    """Return whether Windows AMD auto mode must bypass PyAV AMF.

    AMF host frames are the established route for H.264 and 8-bit HEVC.  On
    Windows, PyAV cannot reliably transfer HEVC Main10/P010 AMF frames, and
    AV1 AMF is unreliable across frame transfer and shutdown.  The selected
    software path still normalizes and uploads YUV frames to ROCm; it is not a
    CPU-only output fallback.  Explicit ``pyav-hw`` remains a diagnostic AMF
    entry point and intentionally bypasses this auto-only policy.
    """

    if sys.platform != "win32" or vendor is not AcceleratorVendor.AMD:
        return False
    codec_name = str(metadata.codec_name).casefold()
    return codec_name == "av1" or (
        codec_name == "hevc" and bool(metadata.is_10bit)
    )


def _requires_single_slice_pyav_threads(
    metadata: VideoMetadata,
    vendor: AcceleratorVendor,
) -> bool:
    """Limit the known Windows AMD HEVC Main10 software decoder to one slice."""

    return (
        sys.platform == "win32"
        and vendor is AcceleratorVendor.AMD
        and str(metadata.codec_name).casefold() == "hevc"
        and bool(metadata.is_10bit)
    )


def _decode_backend() -> str:
    backend = os.environ.get(DECODE_BACKEND_ENV, "auto")
    if backend not in _DECODE_BACKENDS:
        raise ValueError(
            f"Unknown decode backend {backend!r} from {DECODE_BACKEND_ENV}, "
            f"expected {_DECODE_BACKENDS}"
        )
    return backend


def _cuda_hwaccel(device: torch.device) -> HWAccel:
    hwaccel = HWAccel(
        "cuda",
        device=str(device.index or 0),
        allow_software_fallback=True,
        is_hw_owned=True,
    )
    # Reuse torch's current primary context without changing its scheduling flags.
    hwaccel.options["primary_ctx"] = "0"
    hwaccel.options["current_ctx"] = "1"
    return hwaccel


def _amf_interop_resource_cache_enabled(*, default: bool = False) -> bool:
    """Parse the Linux AV1 stable dma-buf identity-cache switch.

    The product default is supplied by the backend selection point so this
    parser stays independently testable.  The cache is accepted only by the
    Linux AMD AV1 fixed-context route; other codecs fail closed if an operator
    tries to force it.
    """

    raw_value = os.environ.get(AMF_INTEROP_RESOURCE_CACHE_ENV)
    if raw_value is None:
        return bool(default)
    value = raw_value.strip().casefold()
    if value in {"", "0", "false", "no", "off"}:
        return False
    if value in {"1", "true", "yes", "on"}:
        return True
    raise ValueError(
        f"Invalid {AMF_INTEROP_RESOURCE_CACHE_ENV} value {value!r}; "
        "expected 0/1, false/true, no/yes, or off/on"
    )


def _amf_interop_decode_copy_stream(*, default: str = "null") -> str:
    """Return the explicit AMF decode-copy synchronization mode.

    ``null`` preserves the already-proven source-release implementation.  The
    event-pool route is deliberately unavailable unless an operator both opts
    into the AMF backend and asks for ``private-deferred`` explicitly.
    """

    if default not in _AMF_INTEROP_DECODE_COPY_STREAMS:
        raise ValueError(f"Invalid AMF interop decode-copy default {default!r}")
    raw_value = os.environ.get(AMF_INTEROP_DECODE_COPY_STREAM_ENV)
    if raw_value is None:
        return default
    value = raw_value.strip().casefold()
    if value in _AMF_INTEROP_DECODE_COPY_STREAMS:
        return value
    raise ValueError(
        f"Invalid {AMF_INTEROP_DECODE_COPY_STREAM_ENV} value {value!r}; "
        "expected 'null' or 'private-deferred'"
    )


def _auto_amf_interop_surface_pool_size(
    metadata: VideoMetadata,
    *,
    batch_size: int,
    vendor: AcceleratorVendor,
) -> int | None:
    """Return the real-video-accepted pool only for its exact product scope."""

    profile = str(getattr(metadata, "profile", "") or "").casefold().strip()
    pixel_format = str(
        getattr(metadata, "pixel_format", "") or ""
    ).casefold().strip()
    main10_p010 = (
        profile in {"main 10", "main10"}
        or (not profile and pixel_format == "p010le")
    )
    if (
        sys.platform == "linux"
        and vendor is AcceleratorVendor.AMD
        and str(metadata.codec_name).casefold() == "hevc"
        and main10_p010
        and bool(metadata.is_10bit)
        and pixel_format in {"p010le", "yuv420p10le"}
        and int(metadata.video_width) == 8192
        and int(metadata.video_height) == 4096
        and int(batch_size) == 4
    ):
        return AMF_INTEROP_8K_B4_SURFACE_POOL_SIZE
    return None


def _amf_interop_surface_pool_size(
    metadata: VideoMetadata | None = None,
    *,
    batch_size: int | None = None,
    vendor: AcceleratorVendor | None = None,
) -> int | None:
    """Return an explicit override or the narrowly accepted product pool.

    Values below 16 are rejected because 0 and 8 starve 8K/B4.  Without an
    override, pool 16 is selected only for the real-video-validated Linux AMD
    HEVC Main10/P010 8192x4096 batch-4 route; every other format and platform
    retains the runtime default.
    """

    raw_value = os.environ.get(AMF_INTEROP_SURFACE_POOL_SIZE_ENV)
    if raw_value is None or not raw_value.strip():
        if metadata is None or batch_size is None or vendor is None:
            return None
        return _auto_amf_interop_surface_pool_size(
            metadata,
            batch_size=batch_size,
            vendor=vendor,
        )
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid {AMF_INTEROP_SURFACE_POOL_SIZE_ENV} value {raw_value!r}; "
            "expected an integer >= 16"
        ) from exc
    if value < 16:
        raise ValueError(
            f"Invalid {AMF_INTEROP_SURFACE_POOL_SIZE_ENV} value {value!r}; "
            "0 and 8 are known to starve 8K/B4, so candidates must be >= 16"
        )
    return value


def _amf_interop_stream_handle(stream: object) -> int:
    """Return a non-null PyTorch CUDA/HIP stream handle for native interop."""

    try:
        handle = int(getattr(stream, "cuda_stream"))
    except (AttributeError, OverflowError, TypeError, ValueError) as exc:
        raise VideoDecodeError(
            "The private-deferred AMF decode-copy mode requires a dedicated "
            "PyTorch CUDA/HIP consumer stream handle"
        ) from exc
    if handle <= 0:
        raise VideoDecodeError(
            "The private-deferred AMF decode-copy mode requires a non-null "
            "PyTorch CUDA/HIP consumer stream handle"
        )
    return handle


def _verify_amf_interop_private_deferred_stream_dependency(
    bridge,
    device: torch.device,
) -> object:
    """Fail closed unless HIP can order a private producer and Torch consumer.

    This is an open-time capability probe.  It may synchronize its own probe
    event once, but production copies are prohibited from synchronizing either
    the private producer stream or the device per frame.
    """

    verifier = getattr(bridge, "verify_private_deferred_stream_dependency", None)
    if not callable(verifier):
        raise VideoDecodeError(
            f"{AMF_INTEROP_DECODE_COPY_STREAM_ENV}=private-deferred requires bridge "
            "entry point verify_private_deferred_stream_dependency"
        )
    probe_stream = new_stream(device)
    consumer_stream_handle = _amf_interop_stream_handle(probe_stream)
    try:
        result = verifier(int(device.index or 0), consumer_stream_handle)
    except BaseException as exc:
        raise VideoDecodeError(
            "The private-deferred AMF decode-copy mode could not verify its "
            "Torch HIP stream dependency"
        ) from exc
    if not isinstance(result, dict):
        raise VideoDecodeError(
            "The private-deferred AMF decode-copy dependency probe returned "
            f"invalid telemetry: {result!r}"
        )
    expected = {
        "mode": "private-deferred",
        "consumer_stream_handle": consumer_stream_handle,
        "stream_create_calls": 1,
        "stream_synchronize_calls": 0,
        "device_wait_calls": 1,
        "event_create_calls": 2,
        "event_record_calls": 2,
        "event_synchronize_calls": 1,
        "event_destroy_calls": 2,
    }
    if any(result.get(name) != value for name, value in expected.items()):
        raise VideoDecodeError(
            "The private-deferred AMF decode-copy dependency probe did not "
            f"confirm its device-wait contract: {result}"
        )
    # Keep the exact non-default Torch stream whose native handle was proven by
    # the bridge.  The legacy/default HIP stream legitimately has handle zero,
    # so rediscovering ``current_stream`` later would make an otherwise valid
    # reader fail on a normal, idle PyTorch thread.
    return probe_stream


def _amf_interop_format_supported(metadata: VideoMetadata) -> bool:
    """Return whether metadata is inside the explicit native core scope."""

    codec = str(metadata.codec_name).casefold()
    profile = str(getattr(metadata, "profile", "") or "").casefold()
    pixel_format = str(getattr(metadata, "pixel_format", "") or "").casefold()
    is_10bit = bool(metadata.is_10bit)
    if codec == "h264":
        # ffprobe reports the source decoder's common yuv420p label; native
        # AMF output is checked separately at the first returned frame.
        return profile in {"main", "high"} and not is_10bit
    if codec == "av1":
        if profile != "main":
            return False
        if is_10bit:
            return pixel_format in {"yuv420p10le", "p010le"}
        return pixel_format in {"yuv420p", "nv12"}
    if codec != "hevc":
        return False
    if not is_10bit and profile == "main":
        return True
    if is_10bit and profile in {"main 10", "main10"}:
        return True
    # Bundled ffprobe metadata from older source runs may omit an HEVC profile.
    # Infer only when its pixel label itself is sufficiently specific.
    if not profile and not is_10bit and pixel_format == "nv12":
        return True
    if not profile and is_10bit and pixel_format == "p010le":
        return True
    return False


def auto_amf_interop_eligible(
    metadata: VideoMetadata,
    vendor: AcceleratorVendor,
) -> bool:
    """Return whether shared ``auto`` selects the proven native AMF route.

    Backend consumers may use this predicate to size Linux AMD-specific
    scheduling without copying the codec/profile/platform policy. Decoder
    selection itself remains owned by :class:`VideoReader`.
    """

    return (
        sys.platform == "linux"
        and vendor is AcceleratorVendor.AMD
        and str(metadata.codec_name).casefold() in {"h264", "hevc", "av1"}
        and _amf_interop_format_supported(metadata)
    )


def _amf_interop_cache_eligible(metadata: VideoMetadata) -> bool:
    """Keep the promoted cache inside its independently validated AV1 scope."""

    return (
        str(metadata.codec_name).casefold() == "av1"
        and _amf_interop_format_supported(metadata)
    )


def _load_amf_interop_bridge():
    """Load only an ABI-matched native AMF/Vulkan/HIP extension."""

    try:
        bridge = importlib.import_module(_AMF_INTEROP_MODULE)
    except (ImportError, OSError, ValueError) as exc:
        raise VideoDecodeError(
            "The explicit amf-interop backend requires the ABI-matched "
            f"{_AMF_INTEROP_MODULE} extension from the unified PyAV/FFmpeg runtime: {exc}"
        ) from exc
    required = (
        "inspect_amf_surface",
        "copy_amf_surface_to_hip",
        "get_transport_stats",
        "reset_transport_stats",
        "AmfVulkanHipInteropSession",
    )
    missing = [name for name in required if not callable(getattr(bridge, name, None))]
    if missing:
        raise VideoDecodeError(
            "The explicit amf-interop bridge is missing required entry points: "
            + ", ".join(missing)
        )
    return bridge



def _create_blocking_cuda_stream(device: torch.device) -> tuple[int, torch.cuda.ExternalStream]:
    # cuStreamCreate needs a current CUDA context; threads other than the one
    # torch initialized on have none until torch binds them to the device.
    torch.cuda.set_device(device)
    handle = create_stream()
    return handle, torch.cuda.ExternalStream(handle, device=device)


class _ValiFrameSource:
    """NVDEC decoding through python_vali, converted by the shared CUDA kernel.

    Construction is the whole VALI viability check: it opens the decoder and
    decodes the first frame, so any container/codec VALI cannot handle raises
    here, before the caller commits to this backend. Corrupt packets after that
    are skipped inside the VALI fork (up to its internal consecutive-packet
    tolerance) and surface only as recovery messages; exceeding the tolerance
    raises VideoDecodeError like the PyAV path.
    """

    def __init__(
        self,
        file: str,
        batch_size: int,
        device: torch.device,
        metadata: VideoMetadata,
        frame_stride: int,
    ):
        import python_vali as vali

        if not hasattr(vali.PyDecoder, "DecodeSingleSurfaceAsyncDetailed"):
            raise VideoDecodeError(
                "python_vali build lacks DecodeSingleSurfaceAsyncDetailed; "
                "the corruption-tolerant fork is required"
            )
        self._vali = vali
        self.file = file
        self.batch_size = batch_size
        self.device = device
        self.metadata = metadata
        self.frame_stride = frame_stride
        self.decoder = None
        self.surface = None
        self._raw_stream, self.stream = _create_blocking_cuda_stream(device)
        try:
            self.decoder = vali.PyDecoder(
                file,
                {},
                gpu_id=device.index or 0,
                stream=self.stream.cuda_stream,
            )
            if not self.decoder.IsAccelerated:
                raise VideoDecodeError(f"VALI decoder for {file} is not hardware accelerated")
            fmt = self.decoder.Format
            if fmt == vali.PixelFormat.NV12:
                self.is_10bit = False
            elif fmt == vali.PixelFormat.P10:
                self.is_10bit = True
            else:
                raise VideoDecodeError(
                    f"VALI surface format {fmt.name} of {file} has no CUDA conversion"
                )
            self.width = self.decoder.Width
            self.height = self.decoder.Height
            self.surface = vali.Surface.Make(
                format=fmt, width=self.width, height=self.height, gpu_id=device.index or 0
            )
            plane = self.surface.Planes[0]
            self._pitch = plane.Pitch
            self._y_ptr = plane.GpuMem
            self._uv_ptr = plane.GpuMem + plane.Pitch * self.height
            self._pkt_data = vali.PacketData()
            self._first_pts = self._decode_next(None)
        except BaseException:
            self.close()
            raise

    def _decode_next(self, seek_ctx) -> int | None:
        success, details = self.decoder.DecodeSingleSurfaceAsyncDetailed(
            self.surface, self._pkt_data, seek_ctx
        )
        if success:
            if details.message:
                log.warning("Recovered video corruption in %s: %s", self.file, details.message)
            return self._pkt_data.pts
        if details.info == self._vali.TaskExecInfo.END_OF_STREAM:
            return None
        raise VideoDecodeError(
            f"Failed to decode {self.file} ({details.info.name}): "
            f"{details.message or details.info.name}"
        )

    def frames(self, seek_ts: float | None) -> Iterator[tuple[torch.Tensor, list[int]]]:
        converter = YuvToRgbConverter(
            self.height,
            self.width,
            self.metadata.color_space,
            self.metadata.color_range == AvColorRange.JPEG,
            self.is_10bit,
            self.device,
        )
        # VALI seeks in absolute stream seconds while the reader contract is
        # seconds past the stream start, so shift by the start offset. VALI's
        # own StartTime is unreliable (wrong scale), use the probed metadata.
        seek_ctx = None
        if seek_ts is not None:
            start_seconds = float(
                self.metadata.start_pts * self.metadata.time_base
            )
            seek_ctx = self._vali.SeekContext(seek_ts=seek_ts + start_seconds)
        pending_pts = self._first_pts if seek_ctx is None else None
        exhausted = seek_ctx is None and self._first_pts is None
        self._first_pts = None
        frame_index = 0
        while not exhausted:
            batch = torch.empty(
                (self.batch_size, 3, self.height, self.width),
                device=self.device,
                dtype=torch.uint8,
            )
            pts: list[int] = []
            while len(pts) < self.batch_size:
                if pending_pts is not None:
                    frame_pts = pending_pts
                    pending_pts = None
                else:
                    frame_pts = self._decode_next(seek_ctx)
                    seek_ctx = None
                    if frame_pts is None:
                        exhausted = True
                        break
                selected = frame_index % self.frame_stride == 0
                frame_index += 1
                if not selected:
                    continue
                converter.convert_surface_into(
                    self._y_ptr,
                    self._uv_ptr,
                    self._pitch,
                    batch[len(pts)],
                    self.stream.cuda_stream,
                )
                pts.append(frame_pts)
            if pts:
                self.stream.synchronize()
                yield batch[: len(pts)], pts

    def close(self) -> None:
        self.decoder = None
        self.surface = None
        if self._raw_stream is None:
            return
        raw_stream, self._raw_stream = self._raw_stream, None
        destroy_stream(raw_stream)


class _AmfInteropTransportAudit:
    """Per-reader proof that explicit AMF interop stays native and balanced.

    The ordinary explicit route synchronizes the null stream before releasing a
    source.  The opt-in private-deferred route instead keeps the source and its
    per-frame external-memory import alive until a consumer-stream event says
    the queued wait has passed.  Both contracts are intentionally audited here
    instead of letting a bridge telemetry regression become a silent lifetime
    change.
    """

    _FORBIDDEN_TRANSPORT_COUNTERS = (
        "hip_non_d2d_copy_calls",
        "host_frame_transfers",
        "cpu_map_calls",
        "staging_copy_calls",
        "d2h_copy_calls",
        "av_hwframe_transfer_data_calls",
        "failed_bridge_copies",
        "vulkan_export_fd_close_failures",
    )
    _DEFERRED_SESSION_COUNTERS = (
        "vulkan_memory_exports",
        "hip_external_memory_imports",
        "hip_mapped_buffer_acquires",
        "hip_mapped_buffer_releases",
        "hip_external_memory_destroys",
        "decode_private_deferred_source_release_hip_stream_create_calls",
        "decode_private_deferred_source_release_hip_stream_create_failures",
        "decode_private_deferred_source_release_hip_stream_destroy_calls",
        "decode_private_deferred_source_release_hip_stream_destroy_failures",
        "decode_private_deferred_source_release_hip_async_copy_calls",
        "decode_private_deferred_source_release_hip_stream_synchronize_calls",
        "decode_private_deferred_source_release_error_stream_synchronize_calls",
        "decode_private_deferred_source_release_hip_event_create_calls",
        "decode_private_deferred_source_release_hip_event_create_failures",
        "decode_private_deferred_source_release_hip_event_record_calls",
        "decode_private_deferred_source_release_hip_event_record_failures",
        "decode_private_deferred_source_release_hip_event_query_calls",
        "decode_private_deferred_source_release_hip_event_query_not_ready",
        "decode_private_deferred_source_release_hip_event_synchronize_calls",
        "decode_private_deferred_source_release_hip_event_synchronize_failures",
        "decode_private_deferred_source_release_hip_event_destroy_calls",
        "decode_private_deferred_source_release_hip_event_destroy_failures",
        "decode_private_deferred_source_release_device_wait_calls",
        "decode_private_deferred_source_release_device_wait_failures",
        "decode_private_deferred_source_release_source_acquires",
        "decode_private_deferred_source_release_source_releases",
        "decode_private_deferred_source_release_forced_drains",
        "decode_private_deferred_source_release_close_drains",
        "decode_private_deferred_source_release_max_in_flight",
        "decode_private_deferred_source_release_failures",
        "last_decode_private_deferred_source_release_hip_stream_handle",
        "decode_private_deferred_source_release_in_flight",
    )

    def __init__(
        self,
        *,
        inspect_amf_surface,
        copy_amf_surface_to_hip,
        get_transport_stats,
        identity_session,
        device: torch.device,
        decode_copy_stream: str = "null",
        resource_cache: bool = False,
    ) -> None:
        if decode_copy_stream not in _AMF_INTEROP_DECODE_COPY_STREAMS:
            raise ValueError(
                "decode_copy_stream must be 'null' or 'private-deferred', got "
                f"{decode_copy_stream!r}"
            )
        self._inspect_amf_surface = inspect_amf_surface
        self._copy_amf_surface_to_hip = copy_amf_surface_to_hip
        self._get_transport_stats = get_transport_stats
        self._identity_session = identity_session
        self._decode_copy_stream = decode_copy_stream
        self._resource_cache = bool(resource_cache)
        if self._resource_cache and decode_copy_stream != "null":
            raise ValueError("resource_cache requires the null decode-copy stream")
        self._device = int(device.index or 0)
        self._identity: tuple[int, int, int, int] | None = None
        self._closed = False
        self._stats = {
            "copy_to_hip_calls": 0,
            "copy_to_hip_successes": 0,
            "copy_to_hip_failures": 0,
            "vulkan_memory_exports": 0,
            "vulkan_export_fd_close_calls": 0,
            "vulkan_export_fd_close_failures": 0,
            "last_vulkan_export_fd_close_errno": 0,
            "hip_external_memory_imports": 0,
            "hip_mapped_buffer_acquires": 0,
            "hip_mapped_buffer_releases": 0,
            "hip_external_memory_destroys": 0,
            "hip_d2d_plane_copies": 0,
            "decode_source_release_hip_stream_synchronize_calls": 0,
            "fixed_context_session_create_calls": 1,
            "fixed_context_session_close_calls": 0,
            "fixed_context_session_close_failures": 0,
            "resource_cache_session_create_calls": 1 if self._resource_cache else 0,
            "resource_cache_session_close_calls": 0,
            "resource_cache_session_close_failures": 0,
            "resource_cache_hits": 0,
            "resource_cache_misses": 0,
            **{name: 0 for name in self._DEFERRED_SESSION_COUNTERS},
        }

    @property
    def decode_copy_stream(self) -> str:
        return self._decode_copy_stream

    @property
    def resource_cache(self) -> bool:
        return self._resource_cache

    @classmethod
    def _reject_forbidden_transport(cls, values, *, source: str) -> None:
        if not isinstance(values, dict):
            raise VideoDecodeError(
                f"amf-interop {source} telemetry is not a dictionary: {values!r}"
            )
        nonzero = []
        for name in cls._FORBIDDEN_TRANSPORT_COUNTERS:
            try:
                value = int(values.get(name, 0))
            except (OverflowError, TypeError, ValueError) as exc:
                raise VideoDecodeError(
                    f"amf-interop {source} telemetry has invalid {name}: {values!r}"
                ) from exc
            if value != 0:
                nonzero.append(f"{name}={value}")
        if nonzero:
            raise VideoDecodeError(
                "amf-interop rejected a non-native transport operation from "
                f"{source}: " + ", ".join(nonzero)
            )

    @staticmethod
    def _integer(values: dict, name: str, *, default: int | None = None) -> int:
        value = values.get(name, default)
        try:
            return int(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise VideoDecodeError(
                f"amf-interop bridge returned invalid {name}: {values!r}"
            ) from exc

    def inspect_frame(self, frame) -> dict:
        try:
            info = self._inspect_amf_surface(frame)
        except BaseException as exc:
            raise VideoDecodeError(
                f"amf-interop rejected a non-native AMF frame: {exc}"
            ) from exc
        if not isinstance(info, dict):
            raise VideoDecodeError(
                f"amf-interop surface inspection returned invalid metadata: {info!r}"
            )
        memory_type = str(info.get("memory_type", "")).casefold()
        vulkan = info.get("vulkan")
        if memory_type != "vulkan" or not isinstance(vulkan, dict):
            raise VideoDecodeError(
                "amf-interop requires an AMF Vulkan external-memory surface; "
                f"inspection returned {info!r}"
            )
        fixed_context = info.get("fixed_context")
        if not isinstance(fixed_context, dict):
            fixed_context = {}
        frames_context = self._integer(fixed_context, "frames_context", default=0)
        amf_context = self._integer(fixed_context, "amf_context", default=0)
        vulkan_device = self._integer(
            fixed_context,
            "vulkan_device",
            default=vulkan.get("device", 0),
        )
        memory = self._integer(vulkan, "memory", default=0)
        if (
            frames_context <= 0
            or amf_context <= 0
            or vulkan_device <= 0
            or memory <= 0
        ):
            raise VideoDecodeError(
                "amf-interop requires a non-null AMF context, Vulkan device, and "
                f"external memory handle; inspection returned {info!r}"
            )
        identity = (frames_context, amf_context, vulkan_device, self._device)
        if self._identity is None:
            self._identity = identity
        elif self._identity != identity:
            raise VideoDecodeError(
                "amf-interop fixed identity changed (AMF/Vulkan/HIP device or context) within "
                f"one reader: expected {self._identity}, got {identity}"
            )
        return info

    def _validate_copy_result(
        self,
        result: dict,
        *,
        consumer_stream_handle: int | None,
    ) -> None:
        self._reject_forbidden_transport(result, source="copy result")
        process_stats = self._get_transport_stats()
        self._reject_forbidden_transport(process_stats, source="bridge counters")
        fd_close_calls = self._integer(result, "vulkan_export_fd_close_calls")
        fd_close_result = self._integer(result, "vulkan_export_fd_close_result")
        fd_close_errno = self._integer(result, "vulkan_export_fd_close_errno")
        self._stats["vulkan_export_fd_close_calls"] += fd_close_calls
        if fd_close_result != 0:
            self._stats["vulkan_export_fd_close_failures"] += 1
            self._stats["last_vulkan_export_fd_close_errno"] = fd_close_errno
        common = (
            self._integer(result, "hip_result") == 0
            and self._integer(result, "d2d_plane_copies", default=2) == 2
            and result.get("fixed_context_bound") is True
            and fd_close_calls == 1
            and fd_close_result == 0
            and fd_close_errno == 0
        )
        if self._decode_copy_stream == "null":
            synchronization = (
                "null-stream-cache-retained"
                if self._resource_cache
                else "null-stream-source-release"
            )
            valid = (
                common
                and (
                    self._resource_cache
                    or (
                        self._integer(result, "hip_free_result") == 0
                        and self._integer(result, "hip_destroy_result") == 0
                    )
                )
                and self._integer(
                    result,
                    "decode_source_release_hip_stream_synchronize_calls",
                )
                == 1
                and self._integer(
                    result,
                    "decode_source_release_hip_stream_synchronize_result",
                )
                == 0
                and result.get("copy_synchronization") == synchronization
                and (
                    not self._resource_cache
                    or bool(result.get("cache_hit"))
                    != bool(result.get("cache_miss"))
                )
            )
        else:
            returned_consumer = self._integer(result, "consumer_stream_handle", default=0)
            valid = (
                common
                and consumer_stream_handle is not None
                and consumer_stream_handle > 0
                and returned_consumer == consumer_stream_handle
                and self._integer(
                    result,
                    "decode_source_release_hip_stream_synchronize_calls",
                    default=0,
                )
                == 0
                and self._integer(
                    result,
                    "decode_null_stream_source_release_hip_stream_synchronize_calls",
                    default=0,
                )
                == 0
                and self._integer(
                    result,
                    "decode_private_deferred_source_release_hip_async_copy_calls",
                )
                == 2
                and self._integer(
                    result,
                    "decode_private_deferred_source_release_hip_stream_synchronize_calls",
                    default=0,
                )
                == 0
                and self._integer(
                    result,
                    "decode_private_deferred_source_release_device_wait_calls",
                )
                == 1
                and self._integer(
                    result,
                    "decode_private_deferred_source_release_hip_event_record_calls",
                )
                == 2
                and self._integer(
                    result,
                    "decode_private_deferred_source_release_source_acquires",
                )
                == 1
                and self._integer(
                    result,
                    "decode_private_deferred_source_release_hip_event_destroy_calls",
                    default=0,
                )
                == 0
                and result.get("copy_synchronization") == "private-deferred-device-wait"
            )
        if not valid:
            raise VideoDecodeError(
                "amf-interop bridge did not prove its AMF-source-release D2D "
                f"contract: {result}"
            )

    def _accumulate_deferred_result(self, result: dict) -> None:
        # Imports are per source; releases and final pool teardown are session
        # totals and replace these provisional counters in ``snapshot``.
        self._stats["vulkan_memory_exports"] += 1
        self._stats["hip_external_memory_imports"] += 1
        self._stats["hip_mapped_buffer_acquires"] += 1
        for name in self._DEFERRED_SESSION_COUNTERS:
            if name in result and name not in {
                "decode_private_deferred_source_release_max_in_flight",
                "last_decode_private_deferred_source_release_hip_stream_handle",
                "decode_private_deferred_source_release_in_flight",
            }:
                self._stats[name] += self._integer(result, name)
        for name in (
            "decode_private_deferred_source_release_max_in_flight",
            "last_decode_private_deferred_source_release_hip_stream_handle",
            "decode_private_deferred_source_release_in_flight",
        ):
            if name in result:
                self._stats[name] = self._integer(result, name)

    def copy_to_hip(
        self,
        frame,
        destination: int,
        destination_size: int,
        *,
        consumer_stream_handle: int | None = None,
    ) -> dict:
        self._stats["copy_to_hip_calls"] += 1
        self.inspect_frame(frame)
        if self._decode_copy_stream == "private-deferred":
            if consumer_stream_handle is None or int(consumer_stream_handle) <= 0:
                self._stats["copy_to_hip_failures"] += 1
                raise VideoDecodeError(
                    "private-deferred AMF copies require a non-null Torch consumer "
                    "stream handle"
                )
            copy_args = (
                frame,
                int(destination),
                int(destination_size),
                self._device,
                int(consumer_stream_handle),
            )
        else:
            if consumer_stream_handle is not None:
                self._stats["copy_to_hip_failures"] += 1
                raise VideoDecodeError(
                    "null-stream AMF copies must not receive a deferred consumer stream"
                )
            copy_args = (frame, int(destination), int(destination_size), self._device)
        try:
            result = self._copy_amf_surface_to_hip(*copy_args)
        except BaseException:
            self._stats["copy_to_hip_failures"] += 1
            raise
        if not isinstance(result, dict):
            self._stats["copy_to_hip_failures"] += 1
            raise VideoDecodeError(
                f"amf-interop bridge returned invalid copy telemetry: {result!r}"
            )
        try:
            self._validate_copy_result(
                result,
                consumer_stream_handle=consumer_stream_handle,
            )
        except VideoDecodeError:
            self._stats["copy_to_hip_failures"] += 1
            raise
        self._stats["copy_to_hip_successes"] += 1
        self._stats["hip_d2d_plane_copies"] += 2
        if self._decode_copy_stream == "null":
            self._stats["vulkan_memory_exports"] += 1
            if self._resource_cache:
                if bool(result.get("cache_hit")):
                    self._stats["resource_cache_hits"] += 1
                else:
                    self._stats["resource_cache_misses"] += 1
                    self._stats["hip_external_memory_imports"] += 1
                    self._stats["hip_mapped_buffer_acquires"] += 1
            else:
                self._stats["hip_external_memory_imports"] += 1
                self._stats["hip_mapped_buffer_acquires"] += 1
                self._stats["hip_mapped_buffer_releases"] += 1
                self._stats["hip_external_memory_destroys"] += 1
            self._stats["decode_source_release_hip_stream_synchronize_calls"] += 1
        else:
            self._accumulate_deferred_result(result)
        return result

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._identity_session.close()
        except BaseException:
            self._stats["fixed_context_session_close_calls"] += 1
            self._stats["fixed_context_session_close_failures"] += 1
            if self._resource_cache:
                self._stats["resource_cache_session_close_calls"] += 1
                self._stats["resource_cache_session_close_failures"] += 1
            raise
        self._stats["fixed_context_session_close_calls"] += 1
        if self._resource_cache:
            self._stats["resource_cache_session_close_calls"] += 1
        self._closed = True

    def _session_stats(self) -> dict[str, object]:
        try:
            session_stats = dict(self._identity_session.stats())
        except BaseException as exc:
            raise VideoDecodeError(
                f"amf-interop fixed-context session telemetry failed: {exc}"
            ) from exc
        if (
            not self._resource_cache
            and self._integer(session_stats, "cache_entries", default=0) != 0
        ):
            raise VideoDecodeError(
                "amf-interop resource cache was unexpectedly populated: "
                f"{session_stats}"
            )
        return session_stats

    def snapshot(self) -> dict[str, object]:
        stats = dict(self._stats)
        session_stats = self._session_stats()
        if self._resource_cache:
            for name in (
                "vulkan_memory_exports",
                "hip_external_memory_imports",
                "hip_mapped_buffer_acquires",
                "hip_mapped_buffer_releases",
                "hip_external_memory_destroys",
                "cache_hits",
                "cache_misses",
                "cache_entries",
                "cache_active_external_imports",
                "cache_active_mappings",
                "cache_raw_handle_identity_changes",
                "cache_stable_identity_raw_handle_changes",
                "cache_fd_export_calls",
                "cache_fd_export_failures",
                "cache_fd_stat_calls",
                "cache_fd_stat_failures",
                "cache_fd_close_calls",
                "cache_fd_close_failures",
                "cache_last_fd_close_errno",
                "cache_fd_ownership_transfers",
            ):
                if name in session_stats:
                    target = {
                        "cache_hits": "resource_cache_hits",
                        "cache_misses": "resource_cache_misses",
                    }.get(name, name)
                    stats[target] = self._integer(session_stats, name)
        if self._decode_copy_stream == "private-deferred":
            for name in self._DEFERRED_SESSION_COUNTERS:
                if name in session_stats:
                    stats[name] = self._integer(session_stats, name)
        stats.update(
            {
                "schema": "jasna.amf.vulkan-hip-transport.v1",
                "telemetry_source": "instrumented-per-reader",
                "non_hardcoded": True,
                "failed_bridge_copies": stats["copy_to_hip_failures"],
                "hip_non_d2d_copy_calls": 0,
                "host_frame_transfers": 0,
                "cpu_map_calls": 0,
                "staging_copy_calls": 0,
                "d2h_copy_calls": 0,
                "av_hwframe_transfer_data_calls": 0,
                "transport_reconfigures": 0,
                "transport_restarts": 0,
                "resource_strategy": (
                    "stable dma-buf identity cache retained for one reader epoch"
                    if self._resource_cache
                    else (
                        "per-frame Vulkan external-memory import/map retained until "
                        "consumer-event release"
                        if self._decode_copy_stream == "private-deferred"
                        else "per-frame Vulkan external-memory import/map with balanced release"
                    )
                ),
                "copy_synchronization": (
                    "null-stream-cache-retained"
                    if self._resource_cache
                    else (
                        "private-deferred-device-wait"
                        if self._decode_copy_stream == "private-deferred"
                        else "null-stream-source-release"
                    )
                ),
                "resource_cache_enabled": self._resource_cache,
                "fixed_context_identity": self._identity,
                "fixed_context_session_closed": bool(session_stats.get("closed", False)),
            }
        )
        return stats

    def validate_closed(self) -> dict[str, object]:
        stats = self.snapshot()
        calls = int(stats["copy_to_hip_calls"])
        if self._resource_cache:
            misses = int(stats["resource_cache_misses"])
            resource_valid = (
                int(stats["vulkan_memory_exports"]) == calls
                and int(stats["resource_cache_hits"]) + misses == calls
                and int(stats.get("cache_entries", -1)) == misses
                and int(stats["hip_external_memory_imports"]) == misses
                and int(stats["hip_mapped_buffer_acquires"]) == misses
                and int(stats["hip_mapped_buffer_releases"]) == misses
                and int(stats["hip_external_memory_destroys"]) == misses
                and int(stats.get("cache_active_external_imports", -1)) == 0
                and int(stats.get("cache_active_mappings", -1)) == 0
                and int(stats.get("cache_raw_handle_identity_changes", -1)) == 0
                and int(
                    stats.get("cache_stable_identity_raw_handle_changes", -1)
                )
                == 0
                and int(stats.get("cache_fd_export_calls", -1)) == calls
                and int(stats.get("cache_fd_export_failures", -1)) == 0
                and int(stats.get("cache_fd_stat_calls", -1)) == calls
                and int(stats.get("cache_fd_stat_failures", -1)) == 0
                and int(stats.get("cache_fd_close_calls", -1))
                == calls
                and int(stats.get("cache_fd_close_failures", -1)) == 0
                and int(stats.get("cache_last_fd_close_errno", -1)) == 0
                and int(stats.get("cache_fd_ownership_transfers", -1)) == 0
                and stats["resource_cache_session_create_calls"] == 1
                and stats["resource_cache_session_close_calls"] == 1
                and stats["resource_cache_session_close_failures"] == 0
            )
        else:
            resource_counts = {
                calls,
                int(stats["vulkan_memory_exports"]),
                int(stats["vulkan_export_fd_close_calls"]),
                int(stats["hip_external_memory_imports"]),
                int(stats["hip_mapped_buffer_acquires"]),
                int(stats["hip_mapped_buffer_releases"]),
                int(stats["hip_external_memory_destroys"]),
            }
            resource_valid = (
                len(resource_counts) == 1
                and stats["resource_cache_session_create_calls"] == 0
                and stats["resource_cache_session_close_calls"] == 0
                and stats["resource_cache_hits"] == 0
                and stats["resource_cache_misses"] == 0
            )
        common = (
            stats["copy_to_hip_successes"] == calls
            and stats["copy_to_hip_failures"] == 0
            and stats["vulkan_export_fd_close_failures"] == 0
            and stats["last_vulkan_export_fd_close_errno"] == 0
            and stats["hip_d2d_plane_copies"] == calls * 2
            and resource_valid
            and stats["fixed_context_session_create_calls"] == 1
            and stats["fixed_context_session_close_calls"] == 1
            and stats["fixed_context_session_close_failures"] == 0
            and stats["fixed_context_session_closed"]
        )
        if self._decode_copy_stream == "null":
            source_release_valid = (
                stats["decode_source_release_hip_stream_synchronize_calls"] == calls
            )
        else:
            pool_events = 6 if calls else 0
            source_release_valid = (
                stats["decode_source_release_hip_stream_synchronize_calls"] == 0
                and stats[
                    "decode_private_deferred_source_release_hip_async_copy_calls"
                ]
                == calls * 2
                and stats[
                    "decode_private_deferred_source_release_hip_stream_synchronize_calls"
                ]
                == 0
                and stats[
                    "decode_private_deferred_source_release_error_stream_synchronize_calls"
                ]
                == 0
                and stats[
                    "decode_private_deferred_source_release_hip_stream_create_calls"
                ]
                == (1 if calls else 0)
                and stats[
                    "decode_private_deferred_source_release_hip_stream_destroy_calls"
                ]
                == (1 if calls else 0)
                and stats[
                    "decode_private_deferred_source_release_hip_event_create_calls"
                ]
                == pool_events
                and stats[
                    "decode_private_deferred_source_release_hip_event_destroy_calls"
                ]
                == pool_events
                and stats[
                    "decode_private_deferred_source_release_hip_event_record_calls"
                ]
                == calls * 2
                and stats[
                    "decode_private_deferred_source_release_device_wait_calls"
                ]
                == calls
                and stats[
                    "decode_private_deferred_source_release_source_acquires"
                ]
                == calls
                and stats[
                    "decode_private_deferred_source_release_source_releases"
                ]
                == calls
                and stats[
                    "decode_private_deferred_source_release_hip_event_synchronize_calls"
                ]
                == stats["decode_private_deferred_source_release_forced_drains"]
                + stats["decode_private_deferred_source_release_close_drains"]
                and stats[
                    "decode_private_deferred_source_release_in_flight"
                ]
                == 0
                and (
                    calls == 0
                    or 0
                    < stats["decode_private_deferred_source_release_max_in_flight"]
                    <= 3
                )
                and all(
                    stats[name] == 0
                    for name in (
                        "decode_private_deferred_source_release_hip_stream_create_failures",
                        "decode_private_deferred_source_release_hip_stream_destroy_failures",
                        "decode_private_deferred_source_release_hip_event_create_failures",
                        "decode_private_deferred_source_release_hip_event_record_failures",
                        "decode_private_deferred_source_release_hip_event_synchronize_failures",
                        "decode_private_deferred_source_release_hip_event_destroy_failures",
                        "decode_private_deferred_source_release_device_wait_failures",
                        "decode_private_deferred_source_release_failures",
                    )
                )
            )
        if not common or not source_release_valid:
            raise VideoDecodeError(
                "amf-interop violated its GPU-only resource/lifetime contract: "
                f"{stats}"
            )
        return stats


class _BatchedAmdYuvConverter:
    """Pixel-exact AMD eager YUV math amortized across up to four frames."""

    def __init__(
        self,
        reference: YuvToRgbConverter,
        capacity: int,
        device: torch.device,
    ) -> None:
        self.height = reference.height
        self.width = reference.width
        self.is_10bit = reference.is_10bit
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("batched AMD YUV conversion capacity must be positive")
        self._luma_scale = reference._luma_scale
        self._chroma_matrix = reference._chroma_matrix
        self._offset = reference._offset
        self._dither2 = getattr(reference, "_dither2", None)
        self._rgb = torch.empty(
            (self.capacity, 3, self.height, self.width),
            dtype=torch.float32,
            device=device,
        )
        self._chroma = torch.empty(
            (self.capacity, 3, self.height // 2, self.width // 2),
            dtype=torch.float32,
            device=device,
        )
        self._codes = (
            torch.empty(
                (self.capacity, 3, self.height, self.width),
                dtype=torch.int32,
                device=device,
            )
            if self.is_10bit
            else None
        )

    def convert_into(self, packed: torch.Tensor, out: torch.Tensor) -> None:
        count = int(packed.shape[0])
        if out.shape != (count, 3, self.height, self.width):
            raise ValueError(
                f"Unexpected batched RGB destination: {tuple(out.shape)}"
            )
        for start in range(0, count, self.capacity):
            stop = min(start + self.capacity, count)
            self._convert_chunk(packed[start:stop], out[start:stop])

    def _convert_chunk(self, packed: torch.Tensor, out: torch.Tensor) -> None:
        count = int(packed.shape[0])
        H, W = self.height, self.width
        y = packed[:, :H]
        uv = packed[:, H:].view(count, H // 2, W // 2, 2)
        u, v = uv[..., 0], uv[..., 1]
        chroma = self._chroma[:count]
        for plane, (cu, cv) in enumerate(self._chroma_matrix):
            torch.mul(u, cu, out=chroma[:, plane])
            chroma[:, plane].add_(v, alpha=cv)
        rgb = self._rgb[:count]
        rgb.view(count, 3, H // 2, 2, W // 2, 2).copy_(
            chroma.unsqueeze(3).unsqueeze(5)
        )
        for plane, offset in enumerate(self._offset):
            rgb[:, plane].add_(offset)
        rgb.add_(y.unsqueeze(1), alpha=self._luma_scale)
        if self.is_10bit:
            codes = self._codes[:count]
            codes.copy_(rgb.round_().clamp_(0, 1023))
            codes.add_(self._dither2.unsqueeze(0)).bitwise_right_shift_(2).clamp_(
                0,
                255,
            )
            out.copy_(codes)
        else:
            out.copy_(rgb.round_().clamp_(0, 255))


class AmfInteropUploader:
    """Convert native AMF Vulkan NV12/P010 surfaces without host staging."""

    def __init__(
        self,
        *,
        file: str,
        batch_size: int,
        device: torch.device,
        metadata: VideoMetadata,
        height: int,
        width: int,
        full_range: bool,
        audit: _AmfInteropTransportAudit,
        consumer_stream: object | None = None,
    ) -> None:
        self.file = file
        self.batch_size = int(batch_size)
        self.device = device
        self.metadata = metadata
        self.height = int(height)
        self.width = int(width)
        self.full_range = bool(full_range)
        self.audit = audit
        self.consumer_stream = consumer_stream
        self.is_10bit = bool(metadata.is_10bit)
        self.software_format = "p010le" if self.is_10bit else "nv12"
        self.bytes_per_sample = 2 if self.is_10bit else 1

    def frames(self, decoded, group: list) -> Iterator[tuple[torch.Tensor, list[int]]]:
        if getattr(self.audit, "resource_cache", False):
            yield from self._frames_resource_cache(decoded, group)
            return
        H, W = self.height, self.width
        converter = YuvToRgbConverter(
            H,
            W,
            self.metadata.color_space,
            self.full_range,
            self.is_10bit,
            self.device,
        )
        deferred = getattr(self.audit, "decode_copy_stream", "null") == "private-deferred"
        if deferred:
            if self.consumer_stream is None:
                raise VideoDecodeError(
                    "private-deferred AMF upload requires the verified non-default "
                    "Torch consumer stream"
                )
            consumer_stream = self.consumer_stream
            consumer_stream_handle = _amf_interop_stream_handle(consumer_stream)
        else:
            consumer_stream = None
            consumer_stream_handle = None
        while group:
            with stream_context(consumer_stream):
                packed = torch.empty(
                    (len(group), H + H // 2, W),
                    dtype=torch.uint16 if self.is_10bit else torch.uint8,
                    device=self.device,
                )
                batch = torch.empty(
                    (len(group), 3, H, W),
                    dtype=torch.uint8,
                    device=self.device,
                )
                pts: list[int] = []
                for index, frame in enumerate(group):
                    frame_format = getattr(getattr(frame, "format", None), "name", None)
                    software_format = getattr(
                        getattr(frame, "sw_format", None), "name", None
                    )
                    if (
                        frame_format != "amf"
                        or software_format != self.software_format
                        or int(frame.width) != W
                        or int(frame.height) != H
                    ):
                        raise VideoDecodeError(
                            "amf-interop requires a native AMF Vulkan "
                            f"{self.software_format.upper()} frame; got "
                            f"format={frame_format}, sw_format={software_format}, "
                            f"size={getattr(frame, 'width', None)}x"
                            f"{getattr(frame, 'height', None)} for {self.file}. "
                            "Host fallback is forbidden."
                        )
                    if deferred:
                        copied = self.audit.copy_to_hip(
                            frame,
                            packed[index].data_ptr(),
                            packed[index].numel() * packed[index].element_size(),
                            consumer_stream_handle=consumer_stream_handle,
                        )
                    else:
                        copied = self.audit.copy_to_hip(
                            frame,
                            packed[index].data_ptr(),
                            packed[index].numel() * packed[index].element_size(),
                        )
                    if (
                        int(copied.get("width", -1)) != W
                        or int(copied.get("height", -1)) != H
                        or int(copied.get("bytes_per_sample", -1))
                        != self.bytes_per_sample
                    ):
                        raise VideoDecodeError(
                            "amf-interop bridge returned an invalid native copy result: "
                            f"{copied}"
                        )
                    converter.convert_into(
                        packed[index, :H],
                        packed[index, H:].view(H // 2, W // 2, 2),
                        batch[index],
                    )
                    pts.append(frame.pts)
            # The null route released every source before return.  The deferred
            # route instead queued each conversion behind a producer event on
            # this same Torch stream, and only the bridge later retires its
            # retained source after the consumer acknowledgement.  This group
            # synchronization remains the existing B8 handoff boundary; it is
            # not a source-release synchronization inside the per-frame route.
            (consumer_stream or current_stream(self.device)).synchronize()
            # A Python ``for`` target retains its final value after the loop.
            # Drop it and clear the previous AMF surface list before asking the
            # decoder for another group, otherwise native surfaces span two
            # decode batches despite the bridge source-release synchronization.
            del frame
            group.clear()
            yield batch, pts
            # Do not prefetch native AMF surfaces across the consumer boundary.
            # In particular, closing the generator after this yield must leave
            # no unconsumed decode group alive while the AMF decoder is torn down.
            group = self._read_group(decoded)

    def _frames_resource_cache(
        self,
        decoded,
        group: list,
    ) -> Iterator[tuple[torch.Tensor, list[int]]]:
        """Run the accepted B4 cache + batched-conversion overlap route.

        The cache copy synchronizes and releases every AMF source before the
        next decode request.  Two packed slots let the independent ROCm
        conversion stream process one group while decode/copy fills the other;
        RGB output tensors are never reused after they are yielded.
        """

        H, W = self.height, self.width
        dtype = torch.uint16 if self.is_10bit else torch.uint8
        packed_slots = [
            torch.empty(
                (self.batch_size, H + H // 2, W),
                dtype=dtype,
                device=self.device,
            )
            for _ in range(2)
        ]
        conversion_stream = new_stream(self.device)
        reference = YuvToRgbConverter(
            H,
            W,
            self.metadata.color_space,
            self.full_range,
            self.is_10bit,
            self.device,
        )
        converter = _BatchedAmdYuvConverter(
            reference,
            min(self.batch_size, 4),
            self.device,
        )

        def fill_slot(slot: int, frames: list) -> list[int]:
            pts: list[int] = []
            for index, frame in enumerate(frames):
                frame_format = getattr(getattr(frame, "format", None), "name", None)
                software_format = getattr(
                    getattr(frame, "sw_format", None), "name", None
                )
                if (
                    frame_format != "amf"
                    or software_format != self.software_format
                    or int(frame.width) != W
                    or int(frame.height) != H
                ):
                    raise VideoDecodeError(
                        "amf-interop cache requires a native AMF Vulkan "
                        f"{self.software_format.upper()} frame; got "
                        f"format={frame_format}, sw_format={software_format}, "
                        f"size={getattr(frame, 'width', None)}x"
                        f"{getattr(frame, 'height', None)} for {self.file}. "
                        "Host fallback is forbidden."
                    )
                copied = self.audit.copy_to_hip(
                    frame,
                    packed_slots[slot][index].data_ptr(),
                    packed_slots[slot][index].numel()
                    * packed_slots[slot][index].element_size(),
                )
                if (
                    int(copied.get("width", -1)) != W
                    or int(copied.get("height", -1)) != H
                    or int(copied.get("bytes_per_sample", -1))
                    != self.bytes_per_sample
                ):
                    raise VideoDecodeError(
                        "amf-interop cache returned an invalid native copy result: "
                        f"{copied}"
                    )
                pts.append(frame.pts)
            if frames:
                del frame
            frames.clear()
            return pts

        current_slot = 0
        current_pts = fill_slot(current_slot, group)
        while current_pts:
            count = len(current_pts)
            batch = torch.empty(
                (count, 3, H, W),
                dtype=torch.uint8,
                device=self.device,
            )
            with stream_context(conversion_stream):
                converter.convert_into(
                    packed_slots[current_slot][:count],
                    batch,
                )
            next_slot = 1 - current_slot
            next_group = self._read_group(decoded)
            next_pts = fill_slot(next_slot, next_group)
            conversion_stream.synchronize()
            yield batch, current_pts
            current_slot = next_slot
            current_pts = next_pts

    def _read_group(self, decoded) -> list:
        group = []
        while len(group) < self.batch_size:
            frame = next(decoded, None)
            if frame is None:
                break
            group.append(frame)
        return group


class WindowsD3D11HipResidentUploader:
    """Copy native AMF DX11 NV12 frames to HIP without host staging."""

    def __init__(
        self,
        *,
        file: str,
        batch_size: int,
        device: torch.device,
        metadata: VideoMetadata,
        height: int,
        width: int,
        full_range: bool,
        coordinator: object,
        role: str,
        decoder_context: object,
    ) -> None:
        self.file = file
        self.batch_size = int(batch_size)
        self.device = device
        self.metadata = metadata
        self.height = int(height)
        self.width = int(width)
        self.full_range = bool(full_range)
        self.coordinator = coordinator
        self.role = str(role)
        self.decoder_context = decoder_context
        self.consumer_stream = new_stream(device)
        self.consumer_stream_handle = _amf_interop_stream_handle(
            self.consumer_stream
        )
        self._bound = False

    def _validate_frame(self, frame: object) -> None:
        frame_format = getattr(getattr(frame, "format", None), "name", None)
        software_format = getattr(
            getattr(frame, "sw_format", None), "name", None
        )
        if (
            frame_format != "amf"
            or software_format != "nv12"
            or int(getattr(frame, "width", -1)) != self.width
            or int(getattr(frame, "height", -1)) != self.height
        ):
            raise VideoDecodeError(
                "The explicit Windows D3D11/HIP resident route requires a "
                "native AMF DX11 NV12 frame; got "
                f"format={frame_format}, sw_format={software_format}, "
                f"size={getattr(frame, 'width', None)}x"
                f"{getattr(frame, 'height', None)} for {self.file}. "
                "Host fallback is forbidden."
            )

    def frames(self, decoded, group: list) -> Iterator[tuple[torch.Tensor, list[int]]]:
        H, W = self.height, self.width
        converter = YuvToRgbConverter(
            H,
            W,
            self.metadata.color_space,
            self.full_range,
            False,
            self.device,
        )
        while group:
            with stream_context(self.consumer_stream):
                packed = torch.empty(
                    (len(group), H + H // 2, W),
                    dtype=torch.uint8,
                    device=self.device,
                )
                batch = torch.empty(
                    (len(group), 3, H, W),
                    dtype=torch.uint8,
                    device=self.device,
                )
                pts: list[int] = []
                for index, frame in enumerate(group):
                    self._validate_frame(frame)
                    if not self._bound:
                        self.coordinator.bind_decoder_frame(
                            self.role,
                            self.decoder_context,
                            frame,
                        )
                        self._bound = True
                    copied = self.coordinator.copy_decoded_to_hip(
                        self.role,
                        frame,
                        packed[index].data_ptr(),
                        packed[index].numel() * packed[index].element_size(),
                        self.consumer_stream_handle,
                    )
                    if (
                        int(copied.get("width", -1)) != W
                        or int(copied.get("height", -1)) != H
                        or int(copied.get("bytes_per_sample", -1)) != 1
                        or int(copied.get("slot_count", -1)) != 4
                    ):
                        raise VideoDecodeError(
                            "Windows D3D11/HIP resident bridge returned an "
                            f"invalid decode-copy result: {copied}"
                        )
                    converter.convert_into(
                        packed[index, :H],
                        packed[index, H:].view(H // 2, W // 2, 2),
                        batch[index],
                    )
                    pts.append(frame.pts)
            self.consumer_stream.synchronize()
            del frame
            group.clear()
            yield batch, pts
            group = self._read_group(decoded)

    def _read_group(self, decoded) -> list:
        group = []
        while len(group) < self.batch_size:
            frame = next(decoded, None)
            if frame is None:
                break
            group.append(frame)
        return group


class VideoReader:
    def __init__(
        self,
        file: str,
        batch_size: int,
        device: torch.device,
        metadata: VideoMetadata,
        *,
        frame_stride: int = 1,
        decode_backend: str | None = None,
        resident_coordinator: object | None = None,
        resident_role: str | None = None,
    ):
        frame_stride = int(frame_stride)
        if frame_stride <= 0:
            raise ValueError("frame_stride must be > 0")
        self.device = device
        self.file = file
        self.batch_size = batch_size
        self.metadata = metadata
        self.frame_stride = frame_stride
        self.decode_backend = decode_backend
        self.resident_coordinator = resident_coordinator
        self.resident_role = resident_role
        self.vendor = vendor_for_device(device)
        self._decoder_ctx = None
        self._vali_source: _ValiFrameSource | None = None
        self._software_only = False
        self._amf_interop_enabled = False
        self._amf_interop_bridge = None
        self._amf_interop_audit: _AmfInteropTransportAudit | None = None
        self._amf_interop_resource_cache = False
        self._amf_interop_decode_copy_stream = "null"
        self._amf_interop_consumer_stream = None
        self._amf_interop_backend_closed = False
        self._windows_resident_enabled = False
        self._windows_resident_consumer_stream = None
        self.amf_interop_stats: dict[str, object] | None = None
        self._decode_backend = DECODE_BACKEND
        self._raw_stream: int | None = None
        self.container = None
        self.video_stream = None

    def __enter__(self):
        self._decoder_ctx = None
        self._amd_hardware_decode = False
        self._vali_source = None
        self._amf_interop_enabled = False
        self._amf_interop_bridge = None
        self._amf_interop_audit = None
        self._amf_interop_resource_cache = False
        self._amf_interop_decode_copy_stream = "null"
        self._amf_interop_consumer_stream = None
        self._amf_interop_backend_closed = False
        self._windows_resident_enabled = False
        self._windows_resident_consumer_stream = None
        self.amf_interop_stats = None
        self.container = None
        self.video_stream = None
        current_stream(self.device)
        backend = self.decode_backend if self.decode_backend is not None else _decode_backend()
        if backend not in _DECODE_BACKENDS:
            raise ValueError(
                f"Unknown decode backend {backend!r}, expected {_DECODE_BACKENDS}"
            )
        self._decode_backend = backend
        if backend == WINDOWS_D3D11_HIP_RESIDENT_BACKEND:
            self._open_windows_d3d11_hip_resident_backend()
            return self
        if backend == "amf-interop":
            self._open_amf_interop_backend()
            return self
        if backend in ("auto", "vali"):
            if self.vendor is AcceleratorVendor.NVIDIA:
                try:
                    self._vali_source = _ValiFrameSource(
                        self.file,
                        self.batch_size,
                        self.device,
                        self.metadata,
                        self.frame_stride,
                    )
                except (ImportError, RuntimeError, ValueError, VideoDecodeError) as exc:
                    if backend == "vali":
                        raise VideoDecodeError(f"VALI cannot decode {self.file}: {exc}") from exc
                    log.warning(
                        "VALI cannot decode %s (codec %s): %s; falling back to PyAV",
                        self.file,
                        self.metadata.codec_name,
                        exc,
                    )
                else:
                    self.width = self._vali_source.width
                    self.height = self._vali_source.height
                    log.info("Using VALI NVDEC decoder for %s", self.file)
                    return self
            elif backend == "vali":
                raise VideoDecodeError("The VALI decode backend requires an NVIDIA device")
        if backend == "auto" and auto_amf_interop_eligible(self.metadata, self.vendor):
            # This is a narrow Linux AMD backend substitution, not a change to
            # the shared auto ordering. Native failures are terminal so a
            # bridge/lifetime regression cannot silently become CPU transport.
            if _amf_interop_cache_eligible(self.metadata):
                self._open_amf_interop_backend(
                    decode_copy_stream="null",
                    resource_cache=True,
                )
            else:
                self._open_amf_interop_backend(decode_copy_stream="private-deferred")
            return self
        windows_amd_software_decode = (
            backend == "auto"
            and _requires_windows_amd_software_decode(
                self.metadata,
                self.vendor,
            )
        )
        software_only = backend == "pyav-sw" or windows_amd_software_decode
        if windows_amd_software_decode:
            codec_name = str(self.metadata.codec_name).casefold()
            if codec_name == "av1":
                log.warning(
                    "Windows AMD AV1 PyAV AMF decoding is unreliable across frame "
                    "transfer and shutdown; using FFmpeg software decoding and "
                    "uploading frames to ROCm for %s",
                    self.file,
                )
            else:
                log.warning(
                    "Windows AMD HEVC Main10/P010 cannot transfer PyAV AMF hardware "
                    "frames; using FFmpeg software decoding and uploading frames to "
                    "ROCm for %s",
                    self.file,
                )
        self._open_pyav(software_only=software_only)
        return self

    def _open_pyav(self, *, software_only: bool, amf_interop: bool = False) -> None:
        """Open the established PyAV route without changing its policy."""

        self._software_only = software_only
        try:
            if not software_only and self.vendor is AcceleratorVendor.NVIDIA:
                self.container = av.open(self.file, hwaccel=_cuda_hwaccel(self.device))
            else:
                self.container = av.open(self.file)
            self.video_stream = self.container.streams.video[0]
        except av.FFmpegError as error:
            raise VideoDecodeError(f"Failed to open {self.file}: {error}") from error

        ctx = self.video_stream.codec_context
        if software_only:
            if _requires_single_slice_pyav_threads(self.metadata, self.vendor):
                ctx.thread_count = 1
                ctx.thread_type = "SLICE"
            else:
                ctx.thread_type = "AUTO"
        elif self.vendor is AcceleratorVendor.AMD:
            if amf_interop:
                self._setup_amf_decoder(ctx, fail_closed=True)
            else:
                self._setup_amf_decoder(ctx)
        elif not ctx.is_hwaccel:
            if self.vendor is AcceleratorVendor.NVIDIA:
                self._setup_nvdec_decoder(ctx)
            else:
                # Definite software decode: let FFmpeg pick frame/slice threading.
                # CUDA contexts must keep their default threading configuration.
                ctx.thread_type = "AUTO"
        self.width = ctx.width
        self.height = ctx.height
        self._full_range = (
            ctx.color_range == int(AvColorRange.JPEG)
            or self.metadata.color_range == AvColorRange.JPEG
        )

    def _open_amf_interop_backend(
        self,
        *,
        decode_copy_stream: str | None = None,
        resource_cache: bool | None = None,
    ) -> None:
        """Open the explicit, fail-closed AMF Vulkan -> HIP reader only."""

        self._validate_amf_interop_scope()
        cache_eligible = _amf_interop_cache_eligible(self.metadata)
        if resource_cache is None:
            resource_cache = _amf_interop_resource_cache_enabled(
                default=cache_eligible,
            )
        self._amf_interop_resource_cache = bool(resource_cache)
        if self._amf_interop_resource_cache and not cache_eligible:
            raise VideoDecodeError(
                f"{AMF_INTEROP_RESOURCE_CACHE_ENV}=1 is supported only for the "
                "validated Linux AMD AV1 Main NV12/P010 route"
            )
        if decode_copy_stream is None:
            decode_copy_stream = _amf_interop_decode_copy_stream(
                default="null"
            )
        elif decode_copy_stream not in _AMF_INTEROP_DECODE_COPY_STREAMS:
            raise VideoDecodeError(
                f"invalid AMF interop decode-copy mode: {decode_copy_stream!r}"
            )
        if self._amf_interop_resource_cache and decode_copy_stream != "null":
            raise VideoDecodeError(
                "The stable dma-buf identity cache requires the proven null-stream "
                "source-release synchronization contract"
            )
        bridge = _load_amf_interop_bridge()
        try:
            if self._amf_interop_resource_cache:
                identity_session = bridge.AmfVulkanHipInteropSession(
                    "decode",
                    resource_cache=True,
                )
            else:
                identity_session = bridge.AmfVulkanHipInteropSession("decode")
        except BaseException as exc:
            raise VideoDecodeError(
                "The explicit amf-interop backend could not create its fixed-context "
                f"bridge session: {exc}"
            ) from exc
        if self._amf_interop_resource_cache:
            session_copy_name = "copy_amf_surface_to_hip_resource_cache"
        elif decode_copy_stream == "private-deferred":
            session_copy_name = "copy_amf_surface_to_hip_private_deferred_stream"
        else:
            session_copy_name = "copy_amf_surface_to_hip"
        session_copy = getattr(identity_session, session_copy_name, None)
        session_close = getattr(identity_session, "close", None)
        session_stats = getattr(identity_session, "stats", None)
        missing_methods = [
            name
            for name, method in (
                (f"{session_copy_name}()", session_copy),
                ("close()", session_close),
                ("stats()", session_stats),
            )
            if not callable(method)
        ]
        if missing_methods:
            if callable(session_close):
                try:
                    session_close()
                except BaseException as close_error:
                    raise VideoDecodeError(
                        "The explicit amf-interop bridge session is incomplete and its "
                        f"cleanup also failed: {close_error}"
                    ) from close_error
            raise VideoDecodeError(
                "The explicit amf-interop bridge session is missing required methods: "
                + ", ".join(missing_methods)
            )
        if decode_copy_stream == "private-deferred":
            try:
                consumer_stream = _verify_amf_interop_private_deferred_stream_dependency(
                    bridge, self.device
                )
            except BaseException:
                try:
                    session_close()
                except BaseException as close_error:
                    log.warning(
                        "AMF private-deferred dependency probe cleanup failed for %s: %s",
                        self.file,
                        close_error,
                    )
                raise
        else:
            consumer_stream = None
        audit = _AmfInteropTransportAudit(
            inspect_amf_surface=bridge.inspect_amf_surface,
            copy_amf_surface_to_hip=session_copy,
            get_transport_stats=bridge.get_transport_stats,
            identity_session=identity_session,
            device=self.device,
            decode_copy_stream=decode_copy_stream,
            resource_cache=self._amf_interop_resource_cache,
        )
        self._amf_interop_bridge = bridge
        self._amf_interop_audit = audit
        self._amf_interop_decode_copy_stream = decode_copy_stream
        self._amf_interop_consumer_stream = consumer_stream
        self._amf_interop_backend_closed = False
        self._amf_interop_enabled = True
        try:
            self._open_pyav(software_only=False, amf_interop=True)
        except BaseException as error:
            cleanup_errors = []
            try:
                self._close_pyav()
            except BaseException as close_error:
                cleanup_errors.append(f"PyAV close: {close_error}")
            try:
                audit.close()
            except BaseException as close_error:
                cleanup_errors.append(f"interop session close: {close_error}")
            self._amf_interop_enabled = False
            self._amf_interop_audit = None
            self._amf_interop_bridge = None
            self._amf_interop_decode_copy_stream = "null"
            self._amf_interop_consumer_stream = None
            if cleanup_errors:
                log.error(
                    "AMF interop open failed for %s; isolated cleanup also reported: %s",
                    self.file,
                    "; ".join(cleanup_errors),
                )
            raise
        log.info(
            "Using explicit AMF Vulkan -> HIP D2D decoder for %s%s",
            self.file,
            " with stable dma-buf identity cache"
            if self._amf_interop_resource_cache
            else "",
        )

    def _open_windows_d3d11_hip_resident_backend(self) -> None:
        """Open the explicit Windows AMD AMF-DX11 resident reader."""

        failures: list[str] = []
        if sys.platform != "win32":
            failures.append("Windows is required")
        if self.vendor is not AcceleratorVendor.AMD:
            failures.append("an AMD HIP device is required")
        if str(self.metadata.codec_name).casefold() != "hevc":
            failures.append("HEVC input is required")
        if bool(self.metadata.is_10bit):
            failures.append("Main10/P010 has not passed its separate gate")
        if int(self.batch_size) not in {1, 2, 4}:
            failures.append("batch_size must be 1, 2, or 4")
        if self.resident_coordinator is None:
            failures.append("a pipeline-owned resident coordinator is required")
        if self.resident_role not in {"decode-detect", "blend-encode"}:
            failures.append("a known resident reader role is required")
        if failures:
            raise VideoDecodeError(
                "The explicit Windows D3D11/HIP resident backend cannot "
                "satisfy this reader: " + "; ".join(failures)
            )
        self._windows_resident_enabled = True
        try:
            self._open_pyav(software_only=False, amf_interop=True)
        except BaseException:
            self._windows_resident_enabled = False
            self._close_pyav()
            raise
        log.info(
            "Using explicit AMF DX11 -> HIP resident decoder for %s (%s)",
            self.file,
            self.resident_role,
        )

    def _validate_amf_interop_scope(self) -> None:
        failures = []
        if sys.platform != "linux":
            failures.append("Linux is required")
        if self.vendor is not AcceleratorVendor.AMD:
            failures.append("an AMD device is required")
        if not _amf_interop_format_supported(self.metadata):
            failures.append(
                "only fixed-format H.264 Main/High 8-bit NV12, HEVC Main 8-bit "
                "NV12, HEVC Main10 10-bit P010, AV1 Main 8-bit NV12, or AV1 "
                "Main 10-bit P010 is accepted"
            )
        if int(self.batch_size) not in _AMF_INTEROP_READER_BATCH_SIZES:
            failures.append(
                "batch_size must be one of "
                f"{sorted(_AMF_INTEROP_READER_BATCH_SIZES)}"
            )
        if failures:
            raise VideoDecodeError(
                "The explicit amf-interop backend cannot satisfy this reader: "
                + "; ".join(failures)
            )

    def _close_amf_interop_backend(self, *, validate: bool = True) -> None:
        audit = self._amf_interop_audit
        if audit is None or getattr(self, "_amf_interop_backend_closed", False):
            return
        # Do not clear the audit before both close and validation complete: a
        # native teardown error must remain observable rather than becoming a
        # silent best-effort cleanup.
        close_error: BaseException | None = None
        validation_error: BaseException | None = None
        try:
            audit.close()
        except BaseException as exc:
            close_error = exc
        # Preserve a concrete native copy/decode exception from the with-body.
        # Normal close still applies full lifecycle validation; exceptional
        # close records the audit without replacing the original native cause.
        try:
            self.amf_interop_stats = (
                audit.validate_closed() if validate else audit.snapshot()
            )
        except BaseException as exc:
            validation_error = exc
            try:
                self.amf_interop_stats = audit.snapshot()
            except BaseException:
                self.amf_interop_stats = None
        # Keep the complete machine-readable audit available without flooding
        # the GUI: adaptive scans may close many short readers on older plans.
        # The compact INFO line is sufficient for normal product runs, while
        # DEBUG retains every counter for a diagnostic transaction.
        log.debug(
            "%s%s",
            AMF_INTEROP_STATS_PREFIX,
            json.dumps(self.amf_interop_stats, sort_keys=True),
        )
        stats = self.amf_interop_stats if isinstance(self.amf_interop_stats, dict) else {}
        copies = int(stats.get("copy_to_hip_calls", 0))
        if copies:
            log.info(
                "AMF D2D audit: copies=%d/%d, cache hit/miss=%d/%d, "
                "FD close/fail=%d/%d, forbidden host/map/staging/D2H/bridge=%d/%d/%d/%d/%d",
                int(stats.get("copy_to_hip_successes", 0)),
                copies,
                int(stats.get("resource_cache_hits", 0)),
                int(stats.get("resource_cache_misses", 0)),
                int(stats.get("vulkan_export_fd_close_calls", 0)),
                int(stats.get("vulkan_export_fd_close_failures", 0)),
                int(stats.get("host_frame_transfers", 0)),
                int(stats.get("cpu_map_calls", 0)),
                int(stats.get("staging_copy_calls", 0)),
                int(stats.get("d2h_copy_calls", 0)),
                int(stats.get("failed_bridge_copies", 0)),
            )
        # Once close has been attempted this reader is no longer a valid lease,
        # even when validation reports a native imbalance.  Detaching these
        # owners prevents a failed AMF session from being reused by a later
        # reader in the same process.
        self._amf_interop_enabled = False
        # Keep the now-closed audit object available for diagnostics and for
        # callers that need to inspect the exact teardown counters.  The
        # native identity session has already been closed, and the explicit
        # closed flag prevents a second close attempt on repeated __exit__
        # calls.  Bridge/stream owners are still detached immediately below.
        self._amf_interop_backend_closed = True
        self._amf_interop_bridge = None
        self._amf_interop_decode_copy_stream = "null"
        self._amf_interop_consumer_stream = None
        if close_error is not None:
            raise close_error
        if validation_error is not None:
            raise validation_error

    def _close_pyav(self) -> None:
        # ``VideoStream`` keeps the separately-created AMF decoder and its
        # Vulkan surface pool reachable even after ``InputContainer.close()``.
        # Smart Render keeps the reader object alive until the worker thread
        # has joined, so leaving this reference on ``self`` retains several
        # GiB between spans. Detach every PyAV owner as part of ``__exit__``;
        # relying on destruction of the reader object is too late for the next
        # span in the same process.
        decoder_ctx, self._decoder_ctx = self._decoder_ctx, None
        video_stream, self.video_stream = self.video_stream, None
        container, self.container = self.container, None
        stream_sync_error: BaseException | None = None
        if getattr(self, "_amf_interop_enabled", False):
            # A caller may close the frame generator while the last AMF→HIP
            # conversion is still queued.  Synchronize the exact consumer
            # stream before destroying the decoder's AVHWFrames context so a
            # mapped Vulkan surface cannot be recycled underneath HIP work.
            try:
                self._synchronize_amf_consumer_stream()
            except BaseException as exc:
                stream_sync_error = exc
        decoder_close_error: BaseException | None = None
        if decoder_ctx is not None:
            close = getattr(decoder_ctx, "close", None)
            if callable(close):
                try:
                    # CodecContext.close() releases AMF/Vulkan surfaces and
                    # the AVHWFrames context.  Merely dropping the Python
                    # reference leaves those native pools reachable until a
                    # later GC pass, which is too late when a new reader is
                    # opened in the same isolated worker.
                    close()
                except BaseException as exc:
                    decoder_close_error = exc
        if container is not None:
            container.close()
        del decoder_ctx, video_stream, container
        if stream_sync_error is not None:
            raise stream_sync_error
        if decoder_close_error is not None:
            raise decoder_close_error

    def _synchronize_amf_consumer_stream(self) -> None:
        """Wait for HIP work that still references AMF-owned surfaces.

        The resource-cache teardown path closes the interop audit before
        closing PyAV.  Keep this operation separate from ``_close_pyav`` so
        that path can perform the same ordering while the consumer stream
        reference is still attached to the reader.
        """

        consumer_stream = (
            getattr(self, "_amf_interop_consumer_stream", None)
            or current_stream(self.device)
        )
        synchronize = getattr(consumer_stream, "synchronize", None)
        if callable(synchronize):
            synchronize()

    @property
    def start_pts(self) -> int:
        if self._vali_source is not None:
            return self.metadata.start_pts
        return resolve_video_start_pts(
            self.video_stream.start_time,
            self.metadata.start_pts,
        )

    def _setup_amf_decoder(self, source_ctx, *, fail_closed: bool = False) -> None:
        source_name = str(source_ctx.name).lower()
        # The unified runtime can expose the selected AMF decoder as the
        # stream context name already.  Only the explicit fail-closed path
        # normalizes that implementation detail; existing auto behavior keeps
        # its established source-context handling unchanged.
        if fail_closed and source_name.endswith("_amf"):
            source_name = source_name[: -len("_amf")]
        decoder_name = {
            "h264": "h264_amf",
            "hevc": "hevc_amf",
            "av1": "av1_amf",
        }.get(source_name)
        if decoder_name is None:
            if fail_closed:
                raise VideoDecodeError(
                    "amf-interop cannot create a native AMF decoder for "
                    f"codec {source_ctx.name!r}"
                )
            source_ctx.thread_type = "AUTO"
            return
        try:
            with _AMF_DECODER_OPEN_LOCK:
                hwaccel = HWAccel(
                    "amf",
                    device=str(self.device.index or 0),
                    allow_software_fallback=False,
                    # Native AMF surfaces must be owned by this explicit decoder;
                    # otherwise PyAV can materialize NV12 host frames even though
                    # AMF itself opened successfully.
                    is_hw_owned=bool(fail_closed),
                )
                decoder = av.CodecContext.create(
                    decoder_name,
                    "r",
                    hwaccel=hwaccel,
                )
                decoder.extradata = source_ctx.extradata
                decoder.width = source_ctx.width
                decoder.height = source_ctx.height
                codec_tag = getattr(source_ctx, "codec_tag", None)
                if codec_tag is not None:
                    decoder.codec_tag = codec_tag
                surface_pool_size = (
                    _amf_interop_surface_pool_size(
                        self.metadata,
                        batch_size=self.batch_size,
                        vendor=self.vendor,
                    )
                    if fail_closed
                    else None
                )
                reset_on_keyframe = (
                    fail_closed
                    and _auto_amf_interop_surface_pool_size(
                        self.metadata,
                        batch_size=self.batch_size,
                        vendor=self.vendor,
                    )
                    is not None
                )
                if surface_pool_size is not None:
                    decoder.options["surface_pool_size"] = str(surface_pool_size)
                    decoder.options["copy_output"] = "1"
                    log.info(
                        "%s AMF decoder surface pool: %d for %s",
                        (
                            "Override"
                            if os.environ.get(AMF_INTEROP_SURFACE_POOL_SIZE_ENV, "").strip()
                            else "Automatic 8K/B4"
                        ),
                        surface_pool_size,
                        self.file,
                    )
                if reset_on_keyframe:
                    # On Linux/RADV, a long-lived 8K Main10 AMF component can
                    # corrupt the final reordered pictures of later GOPs. A
                    # drain plus Terminate/Init at keyframes clears that native
                    # state. Independent output surfaces are required because
                    # downstream Vulkan/HIP copies may still own an earlier
                    # AVFrame when AMF reaches the boundary.
                    decoder.options["copy_output"] = "1"
                    decoder.options["reset_on_keyframe"] = "1"
                    log.info(
                        "AMF HEVC keyframe state reset enabled for %s",
                        self.file,
                    )
                # PyAV 18 rejects assigning time_base on a decoder ("Cannot access
                # 'time_base' as a decoder"); decoders take timing from packets.
                # Source contexts in the accepted runtime can legitimately omit
                # container framerate/SAR. Packets carry timing; assigning None
                # makes PyAV reject an otherwise valid decoder before it opens.
                if source_ctx.framerate is not None:
                    decoder.framerate = source_ctx.framerate
                if source_ctx.sample_aspect_ratio is not None:
                    decoder.sample_aspect_ratio = source_ctx.sample_aspect_ratio
                else:
                    decoder.sample_aspect_ratio = Fraction(1, 1)
                run_amf_decoder_open_with_watchdog(
                    lambda: decoder.open(strict=False),
                    description=f"{decoder_name} for {self.file}",
                )
            self._decoder_ctx = decoder
            log.info("Using AMF hardware decoder %s for %s", decoder_name, self.file)
        except (ValueError, av.FFmpegError, RuntimeError) as exc:
            if fail_closed:
                raise VideoDecodeError(
                    "amf-interop cannot configure a native AMF decoder for "
                    f"{self.file} (codec {self.metadata.codec_name}): {exc}"
                ) from exc
            source_ctx.thread_type = "AUTO"
            log.warning(
                "AMF cannot decode %s (codec %s): %s; using FFmpeg software "
                "decoding and uploading frames to ROCm",
                self.file,
                self.metadata.codec_name,
                exc,
            )

    def _setup_nvdec_decoder(self, source_ctx) -> None:
        decoder_name = _NVDEC_DECODER_OVERRIDES.get(str(self.metadata.codec_name).lower())
        if decoder_name is None:
            source_ctx.thread_type = "AUTO"
            return
        min_width, min_height = _NVDEC_MIN_CODED_SIZE[decoder_name]
        if source_ctx.width < min_width or source_ctx.height < min_height:
            source_ctx.thread_type = "AUTO"
            log.info(
                "Skipping NVDEC decoder %s for %s: %dx%d is below its %dx%d minimum",
                decoder_name,
                self.file,
                source_ctx.width,
                source_ctx.height,
                min_width,
                min_height,
            )
            return
        try:
            decoder = av.CodecContext.create(
                decoder_name,
                "r",
                hwaccel=_cuda_hwaccel(self.device),
            )
            decoder.extradata = source_ctx.extradata
            decoder.width = source_ctx.width
            decoder.height = source_ctx.height
            if source_ctx.pix_fmt is not None:
                decoder.pix_fmt = source_ctx.pix_fmt
            decoder.profile = source_ctx.profile
            decoder.open(strict=False)
            self._decoder_ctx = decoder
            log.info("Using NVDEC decoder %s for %s", decoder_name, self.file)
        except (ValueError, av.FFmpegError, RuntimeError) as exc:
            source_ctx.thread_type = "AUTO"
            log.warning(
                "NVDEC decoder %s unavailable for %s (codec %s): %s; using FFmpeg "
                "software decoding and uploading frames to CUDA",
                decoder_name,
                self.file,
                self.metadata.codec_name,
                exc,
            )

    def __exit__(self, exc_type, exc_value, traceback):
        if self._vali_source is not None:
            source, self._vali_source = self._vali_source, None
            source.close()
            return
        cleanup_errors: list[BaseException] = []
        # A cache retains HIP imports of decoder-owned Vulkan allocations for
        # the reader epoch.  Release those imports before closing the decoder;
        # the non-cache routes retain their established teardown order.
        if (
            self._amf_interop_audit is not None
            and self._amf_interop_resource_cache
        ):
            try:
                # The cache close detaches ``_amf_interop_consumer_stream``.
                # Synchronize first, while it still names the HIP stream that
                # owns the last deferred AMF→HIP copies; otherwise closing the
                # decoder can recycle a Vulkan surface underneath that work.
                if self._amf_interop_enabled:
                    self._synchronize_amf_consumer_stream()
                self._close_amf_interop_backend(validate=exc_type is None)
            except BaseException as error:
                cleanup_errors.append(error)
        try:
            self._close_pyav()
        except BaseException as error:
            cleanup_errors.append(error)
        if self._amf_interop_audit is not None:
            try:
                self._close_amf_interop_backend(validate=exc_type is None)
            except BaseException as error:
                cleanup_errors.append(error)
        self._windows_resident_enabled = False
        self._windows_resident_consumer_stream = None
        # A teardown failure makes this lease unsafe to reuse. Drop the Python
        # owner so it cannot also remain attached to a failed reader object.
        if self._amf_interop_consumer_stream is not None:
            self._amf_interop_consumer_stream = None
        if cleanup_errors:
            if exc_type is not None:
                # The body exception carries the actionable decode/copy cause.
                # Keep teardown faults visible in logs without replacing it.
                for error in cleanup_errors:
                    log.warning(
                        "Cleanup after AMF/video decode failure for %s also failed: %s",
                        self.file,
                        error,
                    )
            else:
                raise cleanup_errors[0]
        if self._raw_stream is None:
            return
        try:
            destroy_stream(self._raw_stream)
        except RuntimeError:
            if exc_type is None:
                raise
            log.warning("Could not destroy the decode stream while handling another error", exc_info=True)

    def _decode_packet(self, packet, consecutive_errors: int) -> tuple[list, int]:
        try:
            frames = (
                self._decoder_ctx.decode(packet)
                if self._decoder_ctx is not None
                else packet.decode()
            )
        except av.error.InvalidDataError as e:
            consecutive_errors += 1
            if consecutive_errors > CORRUPT_PACKET_TOLERANCE:
                raise VideoDecodeError(
                    f"Failed to decode {self.file}: too many consecutive corrupt packets "
                    f"({consecutive_errors}): {e}"
                ) from e
            log.warning("Recovered video corruption in %s: %s", self.file, e)
            return [], consecutive_errors
        except av.FFmpegError as e:
            if (sys.platform == "win32" and self._decoder_ctx is not None
                    and is_windows_amf_host_transfer_failure(
                        e, platform=sys.platform,
                        amd=getattr(self, "vendor", None) is AcceleratorVendor.AMD,
                        decoder_name=str(getattr(self._decoder_ctx, "name", "")),
                    )):
                raise NativeGpuContextUnusableError(
                    f"Windows AMF hardware-to-host frame transfer failed for {self.file}: {e}. "
                    "The GPU context is quarantined; restart Jasna before continuing."
                ) from e
            raise VideoDecodeError(f"Failed to decode {self.file}: {e}") from e
        if frames:
            consecutive_errors = 0
        return frames, consecutive_errors

    def _decoded_frames(self, seek_ts: float | None):
        target_pts = None
        if seek_ts is not None:
            start = resolve_video_start_pts(
                self.video_stream.start_time,
                self.metadata.start_pts,
            )
            target_pts = start + round(seek_ts / self.video_stream.time_base)
            self.container.seek(target_pts, stream=self.video_stream, backward=True)
            if self._decoder_ctx is not None:
                self._decoder_ctx.flush_buffers()

        consecutive_errors = 0
        for packet in demux_video(self.container, self.video_stream):
            frames, consecutive_errors = self._decode_packet(packet, consecutive_errors)
            # Consume packet-owned references before suspending this generator.
            # The consumer owns each yielded frame for as long as it needs it;
            # retaining the original list also retains already-consumed frames.
            while frames:
                if target_pts is not None and frames[0].pts is not None and frames[0].pts < target_pts:
                    frames.pop(0)
                    continue
                target_pts = None
                yield frames.pop(0)

    def _read_group(self, decoded) -> list:
        group = []
        while len(group) < self.batch_size:
            frame = next(decoded, None)
            if frame is None:
                break
            group.append(frame)
        return group

    def _selected_frames(self, decoded):
        if self.frame_stride == 1:
            yield from decoded
            return
        for frame_index, frame in enumerate(decoded):
            if frame_index % self.frame_stride == 0:
                yield frame

    def frames(
        self,
        seek_ts: float | None = None,
    ) -> Iterator[tuple[torch.Tensor, list[int]]]:
        if self._vali_source is not None:
            yield from self._vali_source.frames(seek_ts)
            return
        yield from self._frames_pyav(seek_ts)

    def _frames_pyav(
        self,
        seek_ts: float | None,
    ) -> Iterator[tuple[torch.Tensor, list[int]]]:
        # With seek_ts, strided selection re-anchors at the first decoded frame
        # after the seek instead of the start of the file: sample phase is only
        # stable relative to the seek target.
        # The first decoded frame's format is the final backend decision: a codec
        # can advertise a CUDA config and still fall back to software when
        # hardware initialization rejects a profile or pixel format. Dispatch
        # once here so neither per-frame loop carries a backend branch.
        decoded_frames = self._decoded_frames(seek_ts)
        decoded = self._selected_frames(decoded_frames)
        group = self._read_group(decoded)
        if not group:
            return
        vendor = getattr(self, "vendor", AcceleratorVendor.NVIDIA)
        if getattr(self, "_amf_interop_enabled", False):
            backend = self._frames_amf_interop(decoded, group)
        elif getattr(self, "_windows_resident_enabled", False):
            backend = self._frames_windows_d3d11_hip_resident(decoded, group)
        elif (
            vendor is AcceleratorVendor.NVIDIA
            and group[0].format.name == "cuda"
        ):
            backend = self._frames_hardware(decoded, group)
        else:
            if vendor is AcceleratorVendor.NVIDIA and not self._software_only:
                log.warning(
                    "CUDA/NVDEC cannot decode %s (codec %s, %s); using FFmpeg "
                    "software decoding and uploading frames to CUDA",
                    self.file,
                    self.metadata.codec_name,
                    group[0].format.name,
                )
            backend = self._frames_software(decoded, group)

        # The backend generator now owns the first group. Drop this outer
        # reference before yielding: retaining four 4K P010 NVDEC surfaces here
        # for the reader's lifetime costs about 96 MiB of avoidable VRAM.
        del group
        yield from backend

    def _frames_amf_interop(
        self,
        decoded,
        group: list,
    ) -> Iterator[tuple[torch.Tensor, list[int]]]:
        audit = self._amf_interop_audit
        if audit is None:
            raise VideoDecodeError("amf-interop bridge audit is unavailable")
        uploader = AmfInteropUploader(
            file=self.file,
            batch_size=self.batch_size,
            device=self.device,
            metadata=self.metadata,
            height=self.height,
            width=self.width,
            full_range=self._full_range,
            audit=audit,
            consumer_stream=self._amf_interop_consumer_stream,
        )
        yield from uploader.frames(decoded, group)

    def _frames_windows_d3d11_hip_resident(
        self,
        decoded,
        group: list,
    ) -> Iterator[tuple[torch.Tensor, list[int]]]:
        coordinator = self.resident_coordinator
        role = self.resident_role
        decoder_context = self._decoder_ctx
        if coordinator is None or role is None or decoder_context is None:
            raise VideoDecodeError(
                "Windows D3D11/HIP resident reader lost its native binding"
            )
        uploader = WindowsD3D11HipResidentUploader(
            file=self.file,
            batch_size=self.batch_size,
            device=self.device,
            metadata=self.metadata,
            height=self.height,
            width=self.width,
            full_range=self._full_range,
            coordinator=coordinator,
            role=role,
            decoder_context=decoder_context,
        )
        self._windows_resident_consumer_stream = uploader.consumer_stream
        yield from uploader.frames(decoded, group)

    def _frames_hardware(self, decoded, group: list) -> Iterator[tuple[torch.Tensor, list[int]]]:
        # FFmpeg 8 maps NVDEC output on CUDA stream 0. Conversion runs in a
        # blocking stream in that same context, so legacy-default-stream ordering
        # makes the decoded writes visible before this kernel without a race.
        # Decode one group ahead while conversion runs; keep both groups' frame
        # references alive until conversion is synchronized so their mapped
        # surfaces cannot be recycled underneath queued work.
        converter = YuvToRgbConverter(
            self.height,
            self.width,
            self.metadata.color_space,
            self._full_range,
            self.metadata.is_10bit,
            self.device,
        )
        if self._raw_stream is None:
            self._raw_stream, self.stream = _create_blocking_cuda_stream(self.device)
        while group:
            batch = torch.empty(
                (len(group), 3, self.height, self.width), device=self.device, dtype=torch.uint8
            )
            pts = [frame.pts for frame in group]
            with torch.cuda.stream(self.stream):
                converter.convert_frames_into(group, batch, self.stream.cuda_stream)

            next_group = self._read_group(decoded)
            self.stream.synchronize()
            group = next_group
            yield batch, pts

    def _frames_software(self, decoded, group: list) -> Iterator[tuple[torch.Tensor, list[int]]]:
        # Normalize CPU frames to the two layouts the CUDA conversion kernel
        # accepts (NV12 for <=8-bit sources, P010 above), keeping the resolved
        # matrix/range identical on both reformat sides so swscale changes only
        # layout/subsampling/depth. The one authoritative YUV->RGB conversion
        # stays in the CUDA kernel.
        depth = max(
            (component.bits for component in group[0].format.components if component.bits),
            default=10 if self.metadata.is_10bit else 8,
        )
        ten_bit = depth > 8
        if depth > 10:
            log.warning(
                "Reducing %d-bit source %s to 10-bit P010 before CUDA upload", depth, self.file
            )
        target_format = "p010le" if ten_bit else "nv12"
        dtype = torch.uint16 if ten_bit else torch.uint8
        bytes_per_sample = 2 if ten_bit else 1

        converter = YuvToRgbConverter(
            self.height,
            self.width,
            self.metadata.color_space,
            self._full_range,
            ten_bit,
            self.device,
        )
        reformatter = VideoReformatter()
        color_range = AvColorRange.JPEG if self._full_range else AvColorRange.MPEG
        H, W = self.height, self.width

        # Pinned host batch is shared. Device staging is gated on
        # AcceleratorVendor.AMD (not "if not NVIDIA") so Intel/CPU keep the
        # historical single-staging private-stream fallback unless AMD-specific.
        # NVIDIA: one staging frame on a private stream (H2D+convert ordered so
        # the next overwrite starts only after the prior kernel consumed it).
        # AMD (issue #252): batch device YUV on current_stream. Isolated D1 was
        # clean; residual glitches are pipeline-contended (Phase 0). Per-frame
        # staging reuse under multi-thread load can race; batch staging removes
        # overwrite races. Phase 4 on gfx1201 cleared full-pipeline static.
        # Extra VRAM ≈ (batch_size - 1) × (1.5 × H × W × bytes) — a few MiB at
        # batch 4 @ 1080p8.
        pinned = torch.empty((self.batch_size, H + H // 2, W), dtype=dtype, pin_memory=True)
        if self.vendor is AcceleratorVendor.AMD:
            device_yuv = torch.empty(
                (self.batch_size, H + H // 2, W), dtype=dtype, device=self.device
            )
            staging = None
            stream = current_stream(self.device)
        else:
            device_yuv = None
            staging = torch.empty((H + H // 2, W), dtype=dtype, device=self.device)
            stream = new_stream(self.device)

        while group:
            batch = torch.empty((len(group), 3, H, W), device=self.device, dtype=torch.uint8)
            pts = [frame.pts for frame in group]
            for i, frame in enumerate(group):
                try:
                    normalized = reformatter.reformat(
                        frame,
                        width=W,
                        height=H,
                        format=target_format,
                        src_colorspace=self.metadata.color_space,
                        dst_colorspace=self.metadata.color_space,
                        src_color_range=color_range,
                        dst_color_range=color_range,
                    )
                except av.FFmpegError as e:
                    raise VideoDecodeError(f"Failed to decode {self.file}: {e}") from e
                y_plane, uv_plane = normalized.planes
                y = torch.frombuffer(y_plane, dtype=dtype).reshape(
                    H, y_plane.line_size // bytes_per_sample
                )[:, :W]
                uv = torch.frombuffer(uv_plane, dtype=dtype).reshape(
                    H // 2, uv_plane.line_size // bytes_per_sample
                )[:, :W]
                pinned[i, :H].copy_(y)
                pinned[i, H:].copy_(uv)

            with stream_context(stream):
                # Shared copy/convert loop: AMD uses a per-frame plane from the
                # batch device buffer; NVIDIA reuses the single staging frame
                # (H2D + convert ordered on the private stream so the next
                # overwrite starts only after the prior kernel consumed it).
                for i in range(len(group)):
                    plane = staging if staging is not None else device_yuv[i]
                    plane.copy_(pinned[i], non_blocking=True)
                    converter.convert_into(
                        plane[:H],
                        plane[H:].view(H // 2, W // 2, 2),
                        batch[i],
                    )

            # CPU-to-pinned copies above are synchronous. Queued GPU work reads
            # pinned/device staging, not these PyAV frames or frombuffer views.
            # Release only software sources before overlapping the next decode;
            # the hardware path must retain its mapped surfaces until sync.
            del frame, normalized, y_plane, uv_plane, y, uv
            group.clear()
            next_group = self._read_group(decoded)
            # Sync before yield so other pipeline threads never see in-flight planes.
            stream.synchronize()
            group = next_group
            try:
                yield batch, pts
            except GeneratorExit:
                # Release the prefetched owner set and pinned staging even if
                # the caller retains the closed generator object.
                group.clear()
                del pinned, device_yuv, staging, plane
                raise
