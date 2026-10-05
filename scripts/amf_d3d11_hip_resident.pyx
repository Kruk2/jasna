"""Explicit Windows AMD AMF/D3D11/HIP resident transport.

This module is intentionally a small ABI boundary around the C++ helper.  It
accepts only the validated HEVC Main/NV12 fixed-geometry experiment and has no
Torch, host-frame, Vulkan, or fallback dependency.  Callers pass raw HIP
device pointers and stream handles as integers.
"""

from collections.abc import Mapping
from operator import index as _integer_index
import threading

# Force the native header to be the first FFmpeg-header consumer in the
# generated C++ translation unit. It includes the C APIs inside extern "C";
# PyAV's public Cython declarations can then reuse the guarded declarations
# without changing their linkage to C++.
cdef extern from "native/amf_d3d11_hip_resident_native.hpp":
    pass

from av.codec.context cimport CodecContext
from av.video.frame cimport VideoFrame, alloc_video_frame
from libc.stdint cimport int64_t, uint64_t, uintptr_t


_API_VERSION = 1
_SLOT_COUNT = 4
_SCHEMA = "jasna.windows.d3d11-hip-resident.transport.v1"
_MAX_UINT64 = (1 << 64) - 1
_MAX_UINT32 = (1 << 32) - 1
_MAX_INT32 = (1 << 31) - 1
_MIN_INT64 = -(1 << 63)
_MAX_INT64 = (1 << 63) - 1

# A strong, process-local registry is deliberate.  A resident root cannot be
# released merely because one Python reference to it was dropped while native
# AMF observer callbacks can still occur.
_PROCESS_ROOTS = {}
_PROCESS_ROOTS_LOCK = threading.RLock()


cdef extern from "native/amf_d3d11_hip_resident_native.hpp":
    cdef cppclass JasnaAmfD3d11HipResidentSession:
        pass

    ctypedef struct JasnaAmfD3d11HipResidentBindInfo:
        int api_version
        int state
        int hip_device
        int visible_width
        int visible_height
        int allocation_width
        int allocation_height
        int surface_format
        int sw_format
        int adapter_luid_match
        int dx11_device_match
        uintptr_t hw_frames_identity
        uintptr_t hw_device_identity
        uintptr_t amf_context_identity

    ctypedef struct JasnaAmfD3d11HipResidentCopyInfo:
        int api_version
        int slot_count
        int slot_index
        int width
        int height
        int bytes_per_sample
        uint64_t packed_size
        uint64_t y_pitch
        uint64_t uv_pitch
        uint64_t d3d_to_hip_fence_value
        uint64_t hip_to_d3d_fence_value
        int hip_result
        int amf_result
        int d3d_result
        int in_flight
        int64_t pts

    ctypedef struct JasnaAmfD3d11HipResidentStats:
        int api_version
        int state
        int hip_device
        int slot_count
        int free_output_slots
        int amf_owned_output_slots
        int retained_decoder_leases
        int adapter_luid_match
        int encoder_bound
        int prewarm_complete
        uint64_t root_create_calls
        uint64_t root_final_close_calls
        uint64_t decoder_bind_calls
        uint64_t encoder_bind_calls
        uint64_t decode_copy_calls
        uint64_t decode_copy_successes
        uint64_t encode_acquire_calls
        uint64_t encode_acquire_successes
        uint64_t bridge_slots_created
        uint64_t bridge_slots_destroyed
        uint64_t output_slots_created
        uint64_t output_slots_destroyed
        uint64_t prewarm_slots_completed
        uint64_t external_memory_imports
        uint64_t external_memory_destroys
        uint64_t mapped_arrays_created
        uint64_t mapped_arrays_destroyed
        uint64_t hip_surface_objects_created
        uint64_t hip_surface_objects_destroyed
        uint64_t d3d12_fences_created
        uint64_t d3d12_fences_destroyed
        uint64_t d3d11_opened_fences_created
        uint64_t d3d11_opened_fences_destroyed
        uint64_t shared_fence_handles_created
        uint64_t shared_fence_handles_closed
        uint64_t hip_external_semaphores_imported
        uint64_t hip_external_semaphores_destroyed
        uint64_t d3d_to_hip_signals
        uint64_t d3d_to_hip_waits
        uint64_t hip_to_d3d_signals
        uint64_t hip_to_d3d_waits
        uint64_t hip_to_hip_waits
        uint64_t decoder_frame_leases_acquired
        uint64_t decoder_frame_leases_released
        uint64_t encoder_wrapper_creates
        uint64_t encoder_wrapper_buffer_releases
        uint64_t observer_callbacks
        uint64_t observer_unexpected_callbacks
        uint64_t observer_leases_acquired
        uint64_t observer_leases_released
        uint64_t peak_in_flight
        uint64_t raw_d3d_map_calls
        uint64_t raw_h2d_bytes
        uint64_t raw_d2h_bytes
        uint64_t av_hwframe_transfer_calls
        uint64_t fifth_slot_allocation_attempts
        uint64_t hip_device_reset_calls
        uint64_t terminal_failures
        uint64_t teardown_failures
        uint64_t pending_wrapper_owners

    int jasna_amf_d3d11_hip_resident_api_version()
    JasnaAmfD3d11HipResidentSession* jasna_amf_d3d11_hip_resident_create(
        int hip_device,
        unsigned int visible_width,
        unsigned int visible_height,
        unsigned int allocation_width,
        unsigned int allocation_height,
        const char **error,
    )
    int jasna_amf_d3d11_hip_resident_bind_decoder_frame(
        JasnaAmfD3d11HipResidentSession *session,
        void *frame,
        JasnaAmfD3d11HipResidentBindInfo *info,
        const char **error,
    )
    int jasna_amf_d3d11_hip_resident_bind_encoder_context(
        JasnaAmfD3d11HipResidentSession *session,
        void *encoder,
        void *decoder,
        JasnaAmfD3d11HipResidentBindInfo *info,
        const char **error,
    )
    int jasna_amf_d3d11_hip_resident_copy_decoded_to_hip(
        JasnaAmfD3d11HipResidentSession *session,
        void *frame,
        uintptr_t destination,
        uint64_t destination_size,
        uintptr_t consumer_stream,
        JasnaAmfD3d11HipResidentCopyInfo *info,
        const char **error,
    )
    int jasna_amf_d3d11_hip_resident_acquire_encoder_frame(
        JasnaAmfD3d11HipResidentSession *session,
        void *output,
        uintptr_t source,
        uint64_t source_size,
        int64_t pts,
        uintptr_t producer_stream,
        JasnaAmfD3d11HipResidentCopyInfo *info,
        const char **error,
    ) nogil
    int jasna_amf_d3d11_hip_resident_begin_drain(
        JasnaAmfD3d11HipResidentSession *session,
        const char **error,
    )
    int jasna_amf_d3d11_hip_resident_close(
        JasnaAmfD3d11HipResidentSession *session,
        int timeout_ms,
        const char **error,
    )
    void jasna_amf_d3d11_hip_resident_get_stats(
        JasnaAmfD3d11HipResidentSession *session,
        JasnaAmfD3d11HipResidentStats *stats,
    )
    void jasna_amf_d3d11_hip_resident_destroy(
        JasnaAmfD3d11HipResidentSession *session,
    )


def api_version():
    """Return the pinned native ABI version without creating a root."""

    return int(jasna_amf_d3d11_hip_resident_api_version())


cdef str _native_error_text(const char *error):
    if error == NULL:
        return "native resident bridge did not report an error"
    return (<bytes>error).decode("utf-8", "replace")


def _exact_integer(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        return _integer_index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _nonnegative_integer(value, name, maximum):
    value = _exact_integer(value, name)
    if value < 0 or value > maximum:
        raise ValueError(f"{name} must be between 0 and {maximum}")
    return value


def _required_key(mapping, name):
    if name not in mapping:
        raise ValueError(f"session_key is missing required {name!r}")
    return mapping[name]


def _required_text(mapping, name):
    value = _required_key(mapping, name)
    if not isinstance(value, str):
        raise TypeError(f"session_key[{name!r}] must be str")
    value = value.strip().casefold()
    if not value:
        raise ValueError(f"session_key[{name!r}] must not be empty")
    return value


def _freeze_registry_value(value):
    """Make an explicitly supplied session-key value hashable and stable."""

    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value)
    if isinstance(value, (tuple, list)):
        return ("sequence", tuple(_freeze_registry_value(item) for item in value))
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("nested session_key mappings must have string keys")
            items.append((key, _freeze_registry_value(item)))
        return ("mapping", tuple(sorted(items)))
    raise TypeError(
        "session_key values must be scalar values, bytes, sequences, or mappings"
    )


def _validate_session_key(session_key):
    """Validate and normalize the narrow Main8/NV12 native contract."""

    if not isinstance(session_key, Mapping):
        raise TypeError("session_key must be a mapping")
    normalized = dict(session_key)
    if any(not isinstance(name, str) for name in normalized):
        raise TypeError("session_key keys must be strings")

    codec = _required_text(normalized, "codec")
    profile = _required_text(normalized, "profile")
    surface_format = _required_text(normalized, "surface_format")
    bit_depth = _nonnegative_integer(
        _required_key(normalized, "bit_depth"), "session_key['bit_depth']", 255
    )
    visible_width = _nonnegative_integer(
        _required_key(normalized, "visible_width"),
        "session_key['visible_width']",
        _MAX_UINT32,
    )
    visible_height = _nonnegative_integer(
        _required_key(normalized, "visible_height"),
        "session_key['visible_height']",
        _MAX_UINT32,
    )
    allocation_width = _nonnegative_integer(
        _required_key(normalized, "allocation_width"),
        "session_key['allocation_width']",
        _MAX_UINT32,
    )
    allocation_height = _nonnegative_integer(
        _required_key(normalized, "allocation_height"),
        "session_key['allocation_height']",
        _MAX_UINT32,
    )
    slot_count = _nonnegative_integer(
        _required_key(normalized, "slot_count"), "session_key['slot_count']", 255
    )
    bytes_per_sample = _nonnegative_integer(
        normalized.get("bytes_per_sample", 1),
        "session_key['bytes_per_sample']",
        255,
    )

    if codec != "hevc":
        raise ValueError("resident bridge accepts HEVC only")
    if profile != "main":
        raise ValueError("resident bridge accepts the HEVC Main profile only")
    if surface_format in {"p010", "p010le", "main10"} or bit_depth == 10:
        raise ValueError("P010/Main10 is structurally unsupported by this Main8/NV12 bridge")
    if surface_format != "nv12" or bit_depth != 8 or bytes_per_sample != 1:
        raise ValueError("resident bridge accepts 8-bit NV12 with one byte per sample only")
    if slot_count != _SLOT_COUNT:
        raise ValueError("resident bridge has exactly four slots")
    if (
        visible_width == 0
        or visible_height == 0
        or visible_width & 1
        or visible_height & 1
        or allocation_width & 1
        or allocation_height & 1
        or allocation_width < visible_width
        or allocation_height < visible_height
    ):
        raise ValueError(
            "resident bridge requires nonzero even fixed visible/allocation geometry"
        )

    normalized.update(
        {
            "codec": codec,
            "profile": profile,
            "surface_format": surface_format,
            "bit_depth": bit_depth,
            "bytes_per_sample": bytes_per_sample,
            "visible_width": visible_width,
            "visible_height": visible_height,
            "allocation_width": allocation_width,
            "allocation_height": allocation_height,
            "slot_count": slot_count,
        }
    )
    frozen = tuple(
        sorted((name, _freeze_registry_value(value)) for name, value in normalized.items())
    )
    return normalized, frozen


cdef dict _bind_info_dict(JasnaAmfD3d11HipResidentBindInfo *info):
    return {
        "schema": _SCHEMA,
        "api_version": int(info.api_version),
        "state": int(info.state),
        "hip_device": int(info.hip_device),
        "slot_count": _SLOT_COUNT,
        "visible_width": int(info.visible_width),
        "visible_height": int(info.visible_height),
        "allocation_width": int(info.allocation_width),
        "allocation_height": int(info.allocation_height),
        "surface_format": "nv12",
        "surface_format_id": int(info.surface_format),
        "sw_format": "nv12",
        "sw_format_id": int(info.sw_format),
        "adapter_luid_match": bool(info.adapter_luid_match),
        "dx11_device_match": bool(info.dx11_device_match),
        "hw_frames_identity": int(info.hw_frames_identity),
        "hw_device_identity": int(info.hw_device_identity),
        "amf_context_identity": int(info.amf_context_identity),
    }


cdef dict _copy_info_dict(
    JasnaAmfD3d11HipResidentCopyInfo *info,
    str direction,
):
    return {
        "schema": _SCHEMA,
        "direction": direction,
        "api_version": int(info.api_version),
        "slot_count": int(info.slot_count),
        "slot_index": int(info.slot_index),
        "width": int(info.width),
        "height": int(info.height),
        "bytes_per_sample": int(info.bytes_per_sample),
        "packed_size": int(info.packed_size),
        "y_pitch": int(info.y_pitch),
        "uv_pitch": int(info.uv_pitch),
        "d3d_to_hip_fence_value": int(info.d3d_to_hip_fence_value),
        "hip_to_d3d_fence_value": int(info.hip_to_d3d_fence_value),
        "hip_result": int(info.hip_result),
        "amf_result": int(info.amf_result),
        "d3d_result": int(info.d3d_result),
        "in_flight": int(info.in_flight),
        "pts": int(info.pts),
        "raw_d3d_map_calls": 0,
        "raw_h2d_bytes": 0,
        "raw_d2h_bytes": 0,
        "av_hwframe_transfer_calls": 0,
        "hip_device_reset_calls": 0,
    }


cdef dict _stats_dict(
    JasnaAmfD3d11HipResidentStats *stats,
    uint64_t root_reuse_calls,
):
    return {
        "schema": _SCHEMA,
        "api_version": int(stats.api_version),
        "state": int(stats.state),
        "hip_device": int(stats.hip_device),
        "slot_count": int(stats.slot_count),
        "free_output_slots": int(stats.free_output_slots),
        "amf_owned_output_slots": int(stats.amf_owned_output_slots),
        "retained_decoder_leases": int(stats.retained_decoder_leases),
        "adapter_luid_match": bool(stats.adapter_luid_match),
        "encoder_bound": bool(stats.encoder_bound),
        "prewarm_complete": bool(stats.prewarm_complete),
        "root_create_calls": int(stats.root_create_calls),
        "root_reuse_calls": int(root_reuse_calls),
        "root_final_close_calls": int(stats.root_final_close_calls),
        "decoder_bind_calls": int(stats.decoder_bind_calls),
        "encoder_bind_calls": int(stats.encoder_bind_calls),
        "decode_copy_calls": int(stats.decode_copy_calls),
        "decode_copy_successes": int(stats.decode_copy_successes),
        "encode_acquire_calls": int(stats.encode_acquire_calls),
        "encode_acquire_successes": int(stats.encode_acquire_successes),
        "bridge_slots_created": int(stats.bridge_slots_created),
        "bridge_slots_destroyed": int(stats.bridge_slots_destroyed),
        "output_slots_created": int(stats.output_slots_created),
        "output_slots_destroyed": int(stats.output_slots_destroyed),
        "prewarm_slots_completed": int(stats.prewarm_slots_completed),
        "external_memory_imports": int(stats.external_memory_imports),
        "external_memory_destroys": int(stats.external_memory_destroys),
        "mapped_arrays_created": int(stats.mapped_arrays_created),
        "mapped_arrays_destroyed": int(stats.mapped_arrays_destroyed),
        "hip_surface_objects_created": int(stats.hip_surface_objects_created),
        "hip_surface_objects_destroyed": int(stats.hip_surface_objects_destroyed),
        "d3d12_fences_created": int(stats.d3d12_fences_created),
        "d3d12_fences_destroyed": int(stats.d3d12_fences_destroyed),
        "d3d11_opened_fences_created": int(stats.d3d11_opened_fences_created),
        "d3d11_opened_fences_destroyed": int(stats.d3d11_opened_fences_destroyed),
        "shared_fence_handles_created": int(stats.shared_fence_handles_created),
        "shared_fence_handles_closed": int(stats.shared_fence_handles_closed),
        "hip_external_semaphores_imported": int(stats.hip_external_semaphores_imported),
        "hip_external_semaphores_destroyed": int(stats.hip_external_semaphores_destroyed),
        "d3d_to_hip_signals": int(stats.d3d_to_hip_signals),
        "d3d_to_hip_waits": int(stats.d3d_to_hip_waits),
        "hip_to_d3d_signals": int(stats.hip_to_d3d_signals),
        "hip_to_d3d_waits": int(stats.hip_to_d3d_waits),
        "hip_to_hip_waits": int(stats.hip_to_hip_waits),
        "decoder_frame_leases_acquired": int(stats.decoder_frame_leases_acquired),
        "decoder_frame_leases_released": int(stats.decoder_frame_leases_released),
        "encoder_wrapper_creates": int(stats.encoder_wrapper_creates),
        "encoder_wrapper_buffer_releases": int(stats.encoder_wrapper_buffer_releases),
        "observer_callbacks": int(stats.observer_callbacks),
        "observer_unexpected_callbacks": int(stats.observer_unexpected_callbacks),
        "observer_leases_acquired": int(stats.observer_leases_acquired),
        "observer_leases_released": int(stats.observer_leases_released),
        "peak_in_flight": int(stats.peak_in_flight),
        "raw_d3d_map_calls": int(stats.raw_d3d_map_calls),
        "raw_h2d_bytes": int(stats.raw_h2d_bytes),
        "raw_d2h_bytes": int(stats.raw_d2h_bytes),
        "av_hwframe_transfer_calls": int(stats.av_hwframe_transfer_calls),
        "fifth_slot_allocation_attempts": int(stats.fifth_slot_allocation_attempts),
        "hip_device_reset_calls": int(stats.hip_device_reset_calls),
        "terminal_failures": int(stats.terminal_failures),
        "teardown_failures": int(stats.teardown_failures),
        "pending_wrapper_owners": int(stats.pending_wrapper_owners),
        "closed": bool(stats.state == 5),
    }


cdef class ResidentSession:
    """One serialized, fixed-four-slot native resident root."""

    cdef JasnaAmfD3d11HipResidentSession *_session
    cdef object _registry_key
    cdef bint _closed
    cdef bint _failed
    cdef bint _close_failed
    cdef uint64_t _root_reuse_calls
    cdef object _final_stats

    def __cinit__(self):
        self._session = NULL
        self._registry_key = None
        self._closed = False
        self._failed = False
        self._close_failed = False
        self._root_reuse_calls = 0
        self._final_stats = None

    def __dealloc__(self):
        # A non-closed native session may still be referenced by an AMF
        # observer.  Intentionally retain it rather than convert lifecycle
        # misuse into a callback use-after-free.  close_session() is the only
        # operation allowed to destroy it after native Close() succeeds.
        if self._session != NULL and self._closed:
            jasna_amf_d3d11_hip_resident_destroy(self._session)
            self._session = NULL

    cdef void _ensure_submission_allowed(self, str operation):
        if self._session == NULL or self._closed:
            raise RuntimeError(f"resident session is closed during {operation}")
        if self._failed or self._close_failed:
            raise RuntimeError(
                f"resident session is failed and refuses new work during {operation}"
            )

    cdef void _raise_submission_failure(
        self,
        str operation,
        int status,
        const char *error,
    ):
        self._failed = True
        raise RuntimeError(
            f"resident {operation} failed ({status}): {_native_error_text(error)}"
        )

    def bind_decoder_frame(self, VideoFrame frame):
        """Validate and retain the decoder's exact AMF/D3D11 identity."""

        cdef JasnaAmfD3d11HipResidentBindInfo info
        cdef const char *error = NULL
        cdef int status
        self._ensure_submission_allowed("bind_decoder_frame")
        status = jasna_amf_d3d11_hip_resident_bind_decoder_frame(
            self._session,
            <void *>frame.ptr,
            &info,
            &error,
        )
        if status != 0:
            self._raise_submission_failure("bind_decoder_frame", status, error)
        return _bind_info_dict(&info)

    def bind_encoder_context(
        self,
        CodecContext encoder,
        CodecContext decoder,
    ):
        """Attach the decoder's AMF references before the encoder is opened."""

        cdef JasnaAmfD3d11HipResidentBindInfo info
        cdef const char *error = NULL
        cdef int status
        self._ensure_submission_allowed("bind_encoder_context")
        if encoder.is_open:
            self._failed = True
            raise RuntimeError(
                "resident encoder context must bind before CodecContext.open()"
            )
        if getattr(encoder, "hwaccel", None) is not None:
            self._failed = True
            raise RuntimeError(
                "resident encoder context must not be created with HWAccel(amf)"
            )
        status = jasna_amf_d3d11_hip_resident_bind_encoder_context(
            self._session,
            <void *>encoder.ptr,
            <void *>decoder.ptr,
            &info,
            &error,
        )
        if status != 0:
            self._raise_submission_failure("bind_encoder_context", status, error)
        return _bind_info_dict(&info)

    def copy_decoded_to_hip(
        self,
        VideoFrame frame,
        destination_ptr,
        destination_size,
        consumer_stream_handle,
    ):
        """Queue the two NV12 D2D copies onto the supplied consumer stream."""

        cdef uintptr_t destination = <uintptr_t>_nonnegative_integer(
            destination_ptr, "destination_ptr", _MAX_UINT64
        )
        cdef uint64_t size = <uint64_t>_nonnegative_integer(
            destination_size, "destination_size", _MAX_UINT64
        )
        cdef uintptr_t consumer_stream = <uintptr_t>_nonnegative_integer(
            consumer_stream_handle, "consumer_stream_handle", _MAX_UINT64
        )
        cdef JasnaAmfD3d11HipResidentCopyInfo info
        cdef const char *error = NULL
        cdef int status
        cdef JasnaAmfD3d11HipResidentSession *native_session
        cdef void *output_frame
        if destination == 0 or consumer_stream == 0:
            raise ValueError("destination_ptr and consumer_stream_handle must be nonzero")
        self._ensure_submission_allowed("copy_decoded_to_hip")
        status = jasna_amf_d3d11_hip_resident_copy_decoded_to_hip(
            self._session,
            <void *>frame.ptr,
            destination,
            size,
            consumer_stream,
            &info,
            &error,
        )
        if status != 0:
            self._raise_submission_failure("copy_decoded_to_hip", status, error)
        return _copy_info_dict(&info, "decode")

    def acquire_encoder_frame(
        self,
        source_ptr,
        source_size,
        pts,
        producer_stream_handle,
    ):
        """Return an AMF-surface VideoFrame backed by a resident output slot."""

        cdef uintptr_t source = <uintptr_t>_nonnegative_integer(
            source_ptr, "source_ptr", _MAX_UINT64
        )
        cdef uint64_t size = <uint64_t>_nonnegative_integer(
            source_size, "source_size", _MAX_UINT64
        )
        cdef int64_t native_pts
        cdef uintptr_t producer_stream = <uintptr_t>_nonnegative_integer(
            producer_stream_handle, "producer_stream_handle", _MAX_UINT64
        )
        cdef VideoFrame frame
        cdef JasnaAmfD3d11HipResidentCopyInfo info
        cdef const char *error = NULL
        cdef int status
        native_pts_value = _exact_integer(pts, "pts")
        if native_pts_value < _MIN_INT64 or native_pts_value > _MAX_INT64:
            raise ValueError("pts is outside the signed 64-bit range")
        native_pts = <int64_t>native_pts_value
        if source == 0:
            raise ValueError("source_ptr must be nonzero")
        # HIP represents the process-wide default stream as a null hipStream_t.
        # PyTorch ROCm exposes that valid stream as cuda_stream == 0.
        self._ensure_submission_allowed("acquire_encoder_frame")
        frame = alloc_video_frame()
        native_session = self._session
        output_frame = <void *>frame.ptr
        with nogil:
            status = jasna_amf_d3d11_hip_resident_acquire_encoder_frame(
                native_session,
                output_frame,
                source,
                size,
                native_pts,
                producer_stream,
                &info,
                &error,
            )
        if status != 0:
            self._raise_submission_failure("acquire_encoder_frame", status, error)
        frame._init_user_attributes()
        return frame, _copy_info_dict(&info, "encode")

    def begin_drain(self):
        """Stop new work while preserving close_session() as the cleanup path."""

        cdef const char *error = NULL
        cdef int status
        if self._closed:
            return None
        if self._session == NULL:
            raise RuntimeError("resident session is unavailable")
        # A previous native failure already makes the route terminal.  Do not
        # prevent the caller from reaching close_session(), whose native close
        # path can still retire resources from FAILED.
        if self._failed or self._close_failed:
            return None
        status = jasna_amf_d3d11_hip_resident_begin_drain(self._session, &error)
        if status != 0:
            self._failed = True
            raise RuntimeError(
                f"resident begin_drain failed ({status}): {_native_error_text(error)}"
            )
        return None

    def close_session(self, timeout_ms):
        """Close and destroy only after the native close ledger is balanced."""

        cdef int timeout = <int>_nonnegative_integer(
            timeout_ms, "timeout_ms", _MAX_INT32
        )
        cdef const char *error = NULL
        cdef JasnaAmfD3d11HipResidentStats native_stats
        cdef int status
        cdef dict result
        if self._closed:
            return dict(self._final_stats)
        if self._session == NULL:
            raise RuntimeError("resident session is unavailable")
        status = jasna_amf_d3d11_hip_resident_close(self._session, timeout, &error)
        if status != 0:
            # Preserve the exact native pointer and registry entry.  This is a
            # fail-closed state, but a subsequent explicit close may still
            # retire callbacks that made the first bounded close time out.
            self._failed = True
            self._close_failed = True
            raise RuntimeError(
                f"resident close_session failed ({status}): {_native_error_text(error)}"
            )
        jasna_amf_d3d11_hip_resident_get_stats(self._session, &native_stats)
        result = _stats_dict(&native_stats, self._root_reuse_calls)
        if int(result["slot_count"]) != _SLOT_COUNT:
            self._failed = True
            raise RuntimeError("native resident close returned a non-four-slot ledger")
        self._final_stats = result
        jasna_amf_d3d11_hip_resident_destroy(self._session)
        self._session = NULL
        self._closed = True
        self._close_failed = False
        with _PROCESS_ROOTS_LOCK:
            if _PROCESS_ROOTS.get(self._registry_key) is self:
                del _PROCESS_ROOTS[self._registry_key]
        return dict(result)

    def stats(self):
        """Return a copy of the current auditable native ownership ledger."""

        cdef JasnaAmfD3d11HipResidentStats native_stats
        if self._closed:
            return dict(self._final_stats)
        if self._session == NULL:
            raise RuntimeError("resident session is unavailable")
        jasna_amf_d3d11_hip_resident_get_stats(self._session, &native_stats)
        return _stats_dict(&native_stats, self._root_reuse_calls)


def create_or_get_process_root(hip_device, session_key):
    """Return the matching process-resident Main8/NV12 root.

    ``session_key`` must describe one fixed HEVC Main, NV12, 8-bit media
    class, including visible and allocated geometry.  P010/Main10 is rejected
    before the C++ constructor runs.
    """

    cdef int device = <int>_nonnegative_integer(hip_device, "hip_device", _MAX_INT32)
    cdef JasnaAmfD3d11HipResidentSession *native_session = NULL
    cdef const char *error = NULL
    cdef ResidentSession session = None
    normalized, frozen_key = _validate_session_key(session_key)
    registry_key = (device, frozen_key)

    with _PROCESS_ROOTS_LOCK:
        existing = _PROCESS_ROOTS.get(registry_key)
        if existing is not None:
            if existing._closed:
                del _PROCESS_ROOTS[registry_key]
            elif existing._failed or existing._close_failed:
                raise RuntimeError(
                    "matching resident process root is terminally failed; "
                    "close it successfully before requesting another root"
                )
            else:
                existing._root_reuse_calls += 1
                return existing

        native_session = jasna_amf_d3d11_hip_resident_create(
            device,
            <unsigned int>normalized["visible_width"],
            <unsigned int>normalized["visible_height"],
            <unsigned int>normalized["allocation_width"],
            <unsigned int>normalized["allocation_height"],
            &error,
        )
        if native_session == NULL:
            raise RuntimeError(
                "creating resident process root failed: " + _native_error_text(error)
            )
        try:
            session = ResidentSession()
            session._session = native_session
            session._registry_key = registry_key
            _PROCESS_ROOTS[registry_key] = session
        except BaseException:
            # Native creation itself only constructs inert bookkeeping.  It is
            # therefore safe to tear it down if Python cannot publish the
            # registry owner; once publication succeeds, close_session() owns
            # all destruction instead.
            if session is not None:
                session._session = NULL
            jasna_amf_d3d11_hip_resident_destroy(native_session)
            raise
        return session
