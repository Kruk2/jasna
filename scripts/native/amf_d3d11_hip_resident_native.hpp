#pragma once

// Windows-only implementation for the explicit, research-gated AMF/D3D11/HIP
// resident route.  It deliberately has no product routing, no host transport,
// and no fallback path.  The build script supplies precompiled DXBC headers;
// shader compilation is never performed by the loaded extension.

#ifndef _WIN32
#error "_jasna_amf_d3d11_hip_resident is Windows-only"
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <windows.h>
#include <d3d11_4.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif
// Cython supplies a generic function-like __has_attribute fallback before
// including this header.  ROCm's Windows headers interpret the mere presence
// of that macro as a Clang-capable frontend and expose GNU attribute syntax
// which MSVC cannot parse.  Direct MSVC HIP-runtime consumers intentionally
// leave it undefined; preserve that supported path, then restore Cython's
// fallback for the remainder of the generated translation unit.
#if defined(_MSC_VER) && defined(__has_attribute)
#define JASNA_RESTORE_CYTHON_HAS_ATTRIBUTE 1
#undef __has_attribute
#endif
#include <hip/hip_runtime_api.h>
#if defined(JASNA_RESTORE_CYTHON_HAS_ATTRIBUTE)
#define __has_attribute(name) 0
#undef JASNA_RESTORE_CYTHON_HAS_ATTRIBUTE
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/buffer.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>
}

#include <AMF/core/Factory.h>
#include <AMF/core/Context.h>
#include <AMF/core/Surface.h>

// FFmpeg builds hwcontext_amf.c as C, where AMF interfaces are global C
// structs. In this C++ extension those interfaces live in namespace amf,
// while the public FFmpeg header still uses their unqualified names.
using namespace amf;
#include <libavutil/hwcontext_amf.h>

// These two generated headers are written by
// scripts/build_amf_d3d11_hip_resident.py into its external build directory.
#include "amf_d3d11_hip_resident_vs.hpp"
#include "amf_d3d11_hip_resident_ps.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <thread>

using Microsoft::WRL::ComPtr;
using namespace amf;

constexpr int JASNA_AMF_D3D11_HIP_RESIDENT_API_VERSION = 1;
constexpr unsigned JASNA_AMF_D3D11_HIP_RESIDENT_SLOTS = 4;

enum JasnaAmfD3d11HipResidentState : int {
    JASNA_RESIDENT_UNBOUND = 0,
    JASNA_RESIDENT_DECODER_BOUND = 1,
    JASNA_RESIDENT_ENCODER_BOUND = 2,
    JASNA_RESIDENT_ACTIVE = 3,
    JASNA_RESIDENT_DRAINING = 4,
    JASNA_RESIDENT_CLOSED = 5,
    JASNA_RESIDENT_FAILED = 6,
};

struct JasnaAmfD3d11HipResidentBindInfo {
    int api_version;
    int state;
    int hip_device;
    int visible_width;
    int visible_height;
    int allocation_width;
    int allocation_height;
    int surface_format;
    int sw_format;
    int adapter_luid_match;
    int dx11_device_match;
    uintptr_t hw_frames_identity;
    uintptr_t hw_device_identity;
    uintptr_t amf_context_identity;
};

struct JasnaAmfD3d11HipResidentCopyInfo {
    int api_version;
    int slot_count;
    int slot_index;
    int width;
    int height;
    int bytes_per_sample;
    uint64_t packed_size;
    uint64_t y_pitch;
    uint64_t uv_pitch;
    uint64_t d3d_to_hip_fence_value;
    uint64_t hip_to_d3d_fence_value;
    int hip_result;
    int amf_result;
    int d3d_result;
    int in_flight;
    int64_t pts;
};

struct JasnaAmfD3d11HipResidentStats {
    int api_version;
    int state;
    int hip_device;
    int slot_count;
    int free_output_slots;
    int amf_owned_output_slots;
    int retained_decoder_leases;
    int adapter_luid_match;
    int encoder_bound;
    int prewarm_complete;
    uint64_t root_create_calls;
    uint64_t root_final_close_calls;
    uint64_t decoder_bind_calls;
    uint64_t encoder_bind_calls;
    uint64_t decode_copy_calls;
    uint64_t decode_copy_successes;
    uint64_t encode_acquire_calls;
    uint64_t encode_acquire_successes;
    uint64_t bridge_slots_created;
    uint64_t bridge_slots_destroyed;
    uint64_t output_slots_created;
    uint64_t output_slots_destroyed;
    uint64_t prewarm_slots_completed;
    uint64_t external_memory_imports;
    uint64_t external_memory_destroys;
    uint64_t mapped_arrays_created;
    uint64_t mapped_arrays_destroyed;
    uint64_t hip_surface_objects_created;
    uint64_t hip_surface_objects_destroyed;
    uint64_t d3d12_fences_created;
    uint64_t d3d12_fences_destroyed;
    uint64_t d3d11_opened_fences_created;
    uint64_t d3d11_opened_fences_destroyed;
    uint64_t shared_fence_handles_created;
    uint64_t shared_fence_handles_closed;
    uint64_t hip_external_semaphores_imported;
    uint64_t hip_external_semaphores_destroyed;
    uint64_t d3d_to_hip_signals;
    uint64_t d3d_to_hip_waits;
    uint64_t hip_to_d3d_signals;
    uint64_t hip_to_d3d_waits;
    uint64_t hip_to_hip_waits;
    uint64_t decoder_frame_leases_acquired;
    uint64_t decoder_frame_leases_released;
    uint64_t encoder_wrapper_creates;
    uint64_t encoder_wrapper_buffer_releases;
    uint64_t observer_callbacks;
    uint64_t observer_unexpected_callbacks;
    uint64_t observer_leases_acquired;
    uint64_t observer_leases_released;
    uint64_t peak_in_flight;
    uint64_t raw_d3d_map_calls;
    uint64_t raw_h2d_bytes;
    uint64_t raw_d2h_bytes;
    uint64_t av_hwframe_transfer_calls;
    uint64_t fifth_slot_allocation_attempts;
    uint64_t hip_device_reset_calls;
    uint64_t terminal_failures;
    uint64_t teardown_failures;
    uint64_t pending_wrapper_owners;
};

class JasnaAmfD3d11HipResidentSession;

// av_buffer_create stores this callback before the session class definition is
// complete.  Keep the declaration here so C++ does not rely on a permissive
// compiler accepting a later declaration.
inline void JasnaAmfD3d11HipResidentReleaseWrapper(void* opaque, uint8_t* data);

struct JasnaAmfD3d11HipResidentObserver final : AMFSurfaceObserver {
    JasnaAmfD3d11HipResidentSession* session = nullptr;
    int slot_index = -1;

    void AMF_STD_CALL OnSurfaceDataRelease(AMFSurface* surface) override;
};

struct JasnaAmfD3d11HipResidentWrapperOwner {
    JasnaAmfD3d11HipResidentSession* session = nullptr;
    int slot_index = -1;
    uint64_t generation = 0;
};

struct JasnaAmfD3d11HipResidentPlane {
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11RenderTargetView> rtv;
    ComPtr<ID3D11ShaderResourceView> srv;
    hipExternalMemory_t external_memory = nullptr;
    hipMipmappedArray_t mipmapped_array = nullptr;
    hipArray_t array = nullptr;
    hipSurfaceObject_t surface = 0;
    unsigned width = 0;
    unsigned height = 0;
    unsigned bytes_per_pixel = 0;
    uint64_t logical_bytes = 0;
};

struct JasnaAmfD3d11HipResidentBridgeSlot {
    JasnaAmfD3d11HipResidentPlane y;
    JasnaAmfD3d11HipResidentPlane uv;
    AVFrame* decoder_lease = nullptr;
    uint64_t decoder_d3d_fence_value = 0;
    // Latest D3D11 work which touched this bridge slot and must complete
    // before HIP may reuse it.
    uint64_t d3d_to_hip_fence_value = 0;
    // Latest HIP completion (or its reserved terminal signal) which D3D
    // must wait for before reusing this bridge slot.
    uint64_t hip_to_d3d_fence_value = 0;
    bool d3d_wait_queued = false;
};

struct JasnaAmfD3d11HipResidentOutputSlot {
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<ID3D11RenderTargetView1> y_rtv;
    ComPtr<ID3D11RenderTargetView1> uv_rtv;
    JasnaAmfD3d11HipResidentObserver observer;
    bool amf_owned = false;
    bool observer_returned = false;
    uint64_t generation = 0;
};

struct JasnaAmfD3d11HipResidentFenceBridge {
    ComPtr<ID3D12Device> d3d12_device;
    ComPtr<ID3D12Fence> d3d12_fence;
    ComPtr<ID3D11Fence> d3d11_fence;
    hipExternalSemaphore_t semaphore = nullptr;
};

struct JasnaAmfD3d11HipResidentShaderCopy {
    ComPtr<ID3D11VertexShader> vertex_shader;
    ComPtr<ID3D11PixelShader> pixel_shader;
    ComPtr<ID3D11RasterizerState> rasterizer;
};

struct JasnaAmfD3d11HipResidentFrameView {
    AVHWFramesContext* frames = nullptr;
    AVHWDeviceContext* device = nullptr;
    AVAMFDeviceContext* amf_device = nullptr;
    AMFSurface* surface = nullptr;
    AMFPlane* y_plane = nullptr;
    AMFPlane* uv_plane = nullptr;
    ComPtr<ID3D11Texture2D> texture;
    int y_hpitch = 0;
    int y_vpitch = 0;
    int uv_hpitch = 0;
    int uv_vpitch = 0;
};

class JasnaAmfD3d11HipResidentSession {
public:
    JasnaAmfD3d11HipResidentSession(
        int hip_device,
        unsigned visible_width,
        unsigned visible_height,
        unsigned allocation_width,
        unsigned allocation_height
    )
        : hip_device_(hip_device),
          visible_width_(visible_width),
          visible_height_(visible_height),
          allocation_width_(allocation_width),
          allocation_height_(allocation_height) {
        stats_.api_version = JASNA_AMF_D3D11_HIP_RESIDENT_API_VERSION;
        stats_.state = JASNA_RESIDENT_UNBOUND;
        stats_.hip_device = hip_device_;
        stats_.slot_count = static_cast<int>(JASNA_AMF_D3D11_HIP_RESIDENT_SLOTS);
    }

    JasnaAmfD3d11HipResidentSession(const JasnaAmfD3d11HipResidentSession&) = delete;
    JasnaAmfD3d11HipResidentSession& operator=(const JasnaAmfD3d11HipResidentSession&) = delete;

    ~JasnaAmfD3d11HipResidentSession() {
        // Destruction is only permitted after Close() proves that no observer or
        // AVBuffer callback can retain this object.  Deliberately leak rather
        // than turn an unbalanced caller lifecycle into a callback UAF.
        std::lock_guard<std::mutex> guard(mutex_);
        if (state_ == JASNA_RESIDENT_CLOSED) {
            DestroyRootLocked();
        }
    }

    int BindDecoderFrame(AVFrame* frame, JasnaAmfD3d11HipResidentBindInfo* info) {
        std::lock_guard<std::mutex> guard(mutex_);
        ResetBindInfo(info);
        ClearErrorLocked();
        if (!EnsureUsableLocked("bind_decoder_frame")) {
            return -1;
        }

        JasnaAmfD3d11HipResidentFrameView view;
        if (!InspectFrameLocked(frame, &view)) {
            return FailWithCurrentErrorLocked();
        }

        if (state_ == JASNA_RESIDENT_UNBOUND) {
            if (!InitializeRootFromFrameLocked(view)) {
                DestroyRootLocked();
                return FailLocked("initializing the decoder-owned DX11/HIP root failed");
            }
            decoder_frames_ref_ = av_buffer_ref(frame->hw_frames_ctx);
            decoder_device_ref_ = av_buffer_ref(view.frames->device_ref);
            if (!decoder_frames_ref_ || !decoder_device_ref_) {
                av_buffer_unref(&decoder_frames_ref_);
                av_buffer_unref(&decoder_device_ref_);
                DestroyRootLocked();
                return FailLocked("referencing the decoder AMF hardware contexts failed");
            }
            canonical_frames_identity_ = reinterpret_cast<uintptr_t>(decoder_frames_ref_->data);
            canonical_device_identity_ = reinterpret_cast<uintptr_t>(decoder_device_ref_->data);
            state_ = JASNA_RESIDENT_DECODER_BOUND;
            ++stats_.root_create_calls;
        } else if (reinterpret_cast<uintptr_t>(frame->hw_frames_ctx->data) !=
                       canonical_frames_identity_ ||
                   reinterpret_cast<uintptr_t>(view.frames->device_ref->data) !=
                       canonical_device_identity_ ||
                   view.amf_device->context != amf_context_) {
            return FailLocked("decoder AMF hardware context identity changed within one resident session");
        }

        ++stats_.decoder_bind_calls;
        FillBindInfoLocked(info, &view);
        return 0;
    }

    int BindEncoderContext(
        AVCodecContext* encoder,
        AVCodecContext* decoder,
        JasnaAmfD3d11HipResidentBindInfo* info
    ) {
        std::lock_guard<std::mutex> guard(mutex_);
        ResetBindInfo(info);
        ClearErrorLocked();
        if (!EnsureUsableLocked("bind_encoder_context")) {
            return -1;
        }
        if (state_ == JASNA_RESIDENT_UNBOUND || !decoder_frames_ref_ || !decoder_device_ref_) {
            return FailLocked("a validated decoder AMF frame must bind before the encoder");
        }
        if (!encoder || !decoder || decoder->codec_id != AV_CODEC_ID_HEVC ||
            encoder->codec_id != AV_CODEC_ID_HEVC) {
            return FailLocked("the initial resident route accepts HEVC decoder and encoder contexts only");
        }
        if ((decoder->profile != AV_PROFILE_UNKNOWN &&
             decoder->profile != AV_PROFILE_HEVC_MAIN) ||
            (encoder->profile != AV_PROFILE_UNKNOWN &&
             encoder->profile != AV_PROFILE_HEVC_MAIN)) {
            return FailLocked("the initial resident route accepts the HEVC Main profile only");
        }
        if (encoder->width != static_cast<int>(visible_width_) ||
            encoder->height != static_cast<int>(visible_height_)) {
            return FailLocked("encoder geometry does not match the fixed resident decoder geometry");
        }
        if (!decoder->hw_frames_ctx || !decoder->hw_device_ctx ||
            reinterpret_cast<uintptr_t>(decoder->hw_frames_ctx->data) !=
                canonical_frames_identity_ ||
            reinterpret_cast<uintptr_t>(decoder->hw_device_ctx->data) !=
                canonical_device_identity_) {
            return FailLocked("decoder CodecContext does not retain the canonical AMF contexts");
        }
        if (encoder->hw_frames_ctx || encoder->hw_device_ctx) {
            return FailLocked("encoder already owns a hardware context; resident binding must happen before open without HWAccel(amf)");
        }
        if (encoder_frames_ref_) {
            return FailLocked("encoder hardware context was already bound for this resident session");
        }

        AVBufferRef* encoder_device = av_buffer_ref(decoder->hw_device_ctx);
        AVBufferRef* encoder_frames = av_buffer_ref(decoder->hw_frames_ctx);
        if (!encoder_device || !encoder_frames) {
            av_buffer_unref(&encoder_device);
            av_buffer_unref(&encoder_frames);
            return FailLocked("referencing decoder AMF contexts for encoder binding failed");
        }
        encoder->hw_device_ctx = encoder_device;
        encoder->hw_frames_ctx = encoder_frames;
        encoder->pix_fmt = AV_PIX_FMT_AMF_SURFACE;
        encoder->sw_pix_fmt = AV_PIX_FMT_NV12;
        encoder_frames_ref_ = av_buffer_ref(encoder->hw_frames_ctx);
        if (!encoder_frames_ref_) {
            av_buffer_unref(&encoder->hw_frames_ctx);
            av_buffer_unref(&encoder->hw_device_ctx);
            return FailLocked("retaining encoder AMF frames context failed");
        }

        state_ = JASNA_RESIDENT_ENCODER_BOUND;
        stats_.encoder_bound = 1;
        ++stats_.encoder_bind_calls;
        FillBindInfoLocked(info, nullptr);
        return 0;
    }

    int CopyDecodedToHip(
        AVFrame* frame,
        uintptr_t destination,
        uint64_t destination_size,
        uintptr_t consumer_stream,
        JasnaAmfD3d11HipResidentCopyInfo* info
    ) {
        std::unique_lock<std::mutex> lock(mutex_);
        ResetCopyInfo(info);
        ClearErrorLocked();
        ++stats_.decode_copy_calls;
        if (!EnsureUsableLocked("copy_decoded_to_hip")) {
            return -1;
        }
        if (state_ == JASNA_RESIDENT_UNBOUND) {
            return FailLocked("a validated decoder AMF frame must bind before copying to HIP");
        }
        if (destination == 0 || consumer_stream == 0) {
            return FailLocked("decode destination and consumer HIP stream are required");
        }
        if (destination_size < PackedBytes()) {
            return FailLocked("decode destination is smaller than the fixed packed NV12 frame");
        }

        JasnaAmfD3d11HipResidentFrameView view;
        if (!InspectFrameLocked(frame, &view) ||
            reinterpret_cast<uintptr_t>(frame->hw_frames_ctx->data) != canonical_frames_identity_ ||
            view.amf_device->context != amf_context_) {
            return FailLocked("decoded AMF frame does not match the bound resident context");
        }

        int index = FindBridgeSlotForUseLocked();
        if (index < 0) {
            return FailLocked("all four resident bridge slots retain an unfinished decoder lease");
        }
        JasnaAmfD3d11HipResidentBridgeSlot& slot = bridge_slots_[static_cast<size_t>(index)];
        AVFrame* lease = av_frame_clone(frame);
        if (!lease) {
            return FailLocked("cloning decoder AVFrame ownership for the bridge slot failed");
        }

        const uint64_t d3d_value = NextFenceValueLocked();
        const uint64_t hip_value = NextFenceValueLocked();
        int d3d_result = 0;
        if (!ProjectDecoderToBridgeLocked(view, slot, d3d_value, &d3d_result)) {
            av_frame_free(&lease);
            return FailLocked("projecting decoder DX11 planes into a bridge slot failed");
        }
        slot.decoder_lease = lease;
        slot.decoder_d3d_fence_value = d3d_value;
        ++stats_.decoder_frame_leases_acquired;

        const hipError_t hip_result = CopyBridgeToHipLocked(
            slot, destination, consumer_stream, hip_value
        );
        if (hip_result != hipSuccess) {
            return FailLocked("copying a decoded bridge slot into HIP failed");
        }

        state_ = state_ == JASNA_RESIDENT_ENCODER_BOUND ? JASNA_RESIDENT_ACTIVE : state_;
        ++stats_.decode_copy_successes;
        FillCopyInfoLocked(info, index, d3d_value, hip_value, hip_result, AMF_OK, d3d_result, frame->pts);
        return 0;
    }

    int AcquireEncoderFrame(
        AVFrame* output,
        uintptr_t source,
        uint64_t source_size,
        int64_t pts,
        uintptr_t producer_stream,
        JasnaAmfD3d11HipResidentCopyInfo* info
    ) {
        std::unique_lock<std::mutex> lock(mutex_);
        ResetCopyInfo(info);
        ClearErrorLocked();
        ++stats_.encode_acquire_calls;
        if (!EnsureUsableLocked("acquire_encoder_frame")) {
            return -1;
        }
        if (state_ != JASNA_RESIDENT_ENCODER_BOUND && state_ != JASNA_RESIDENT_ACTIVE) {
            return FailLocked("the shared AMF encoder context must bind before acquiring output frames");
        }
        if (!output || !encoder_frames_ref_ || source == 0) {
            // A null hipStream_t is HIP's valid process-wide default stream;
            // PyTorch ROCm reports it as cuda_stream == 0.
            return FailLocked("encoder frame, shared frames context, and source pointer are required");
        }
        if (source_size < PackedBytes()) {
            return FailLocked("encode source is smaller than the fixed packed NV12 frame");
        }

        // Decode and encode share the session's four bridge textures.  A
        // batch decoder can have all four decoder leases in flight for a few
        // milliseconds even when an output surface is already reusable.  Do
        // not turn that bounded pressure into a permanent session failure:
        // wait for both resources while releasing the session mutex so AMF
        // observer callbacks and the peer decoder thread can make progress.
        const auto acquire_deadline = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(5000);
        int bridge_index = -1;
        int output_index = -1;
        while (bridge_index < 0 || output_index < 0) {
            bridge_index = FindBridgeSlotForUseLocked();
            output_index = bridge_index >= 0 ? FindOutputSlotLocked() : -1;
            if (bridge_index >= 0 && output_index >= 0) {
                break;
            }
            if (std::chrono::steady_clock::now() >= acquire_deadline) {
                return FailLocked(
                    bridge_index < 0
                        ? "timed out waiting for one of the four shared resident bridge slots"
                        : "timed out waiting for an observer-released resident output surface"
                );
            }
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            lock.lock();
            if (!EnsureUsableLocked("acquire_encoder_frame")) {
                return -1;
            }
        }

        JasnaAmfD3d11HipResidentBridgeSlot& bridge = bridge_slots_[static_cast<size_t>(bridge_index)];
        JasnaAmfD3d11HipResidentOutputSlot& output_slot = output_slots_[static_cast<size_t>(output_index)];
        const uint64_t hip_value = NextFenceValueLocked();
        const uint64_t d3d_complete_value = NextFenceValueLocked();
        const hipError_t hip_result = CopyHipToBridgeLocked(
            bridge, source, producer_stream, hip_value
        );
        if (hip_result != hipSuccess) {
            return FailLocked("copying packed HIP NV12 into a resident bridge slot failed");
        }

        int d3d_result = 0;
        if (!ProjectBridgeToOutputLocked(
                bridge, output_slot, hip_value, d3d_complete_value, &d3d_result
            )) {
            return FailLocked("projecting resident bridge planes into the AMF output texture failed");
        }

        AMFSurface* wrapper = nullptr;
        output_slot.generation += 1;
        output_slot.amf_owned = true;
        output_slot.observer_returned = false;
        AMF_RESULT amf_result = amf_context_->CreateSurfaceFromDX11Native(
            output_slot.texture.Get(), &wrapper, &output_slot.observer
        );
        if (amf_result == AMF_OK && wrapper &&
            wrapper->GetMemoryType() == AMF_MEMORY_DX11 &&
            wrapper->GetFormat() == AMF_SURFACE_NV12 && wrapper->GetPlanesCount() == 2) {
            amf_result = wrapper->SetCrop(
                0, 0, static_cast<amf_int32>(visible_width_),
                static_cast<amf_int32>(visible_height_)
            );
        }
        if (amf_result != AMF_OK || !wrapper ||
            wrapper->GetMemoryType() != AMF_MEMORY_DX11 ||
            wrapper->GetFormat() != AMF_SURFACE_NV12 || wrapper->GetPlanesCount() != 2) {
            // CreateSurfaceFromDX11Native may invoke the observer during Release;
            // never release a wrapper while this mutex is held.
            output_slot.amf_owned = false;
            output_slot.observer_returned = true;
            lock.unlock();
            if (wrapper) {
                wrapper->Release();
            }
            return FailWithoutLock("creating an AMF DX11 NV12 output wrapper failed");
        }
        ++stats_.observer_leases_acquired;

        auto* owner = new (std::nothrow) JasnaAmfD3d11HipResidentWrapperOwner{
            this, output_index, output_slot.generation
        };
        if (!owner) {
            lock.unlock();
            wrapper->Release();
            return FailWithoutLock("allocating the AMF AVBuffer owner failed");
        }
        ++stats_.pending_wrapper_owners;
        AVBufferRef* owner_buffer = av_buffer_create(
            reinterpret_cast<uint8_t*>(wrapper), sizeof(wrapper),
            &JasnaAmfD3d11HipResidentReleaseWrapper, owner,
            AV_BUFFER_FLAG_READONLY
        );
        if (!owner_buffer) {
            lock.unlock();
            wrapper->Release();
            OnWrapperBufferReleased(output_index, owner->generation);
            delete owner;
            return FailWithoutLock("creating AVBuffer ownership for the AMF output surface failed");
        }
        AVBufferRef* frame_context = av_buffer_ref(encoder_frames_ref_);
        if (!frame_context) {
            lock.unlock();
            av_buffer_unref(&owner_buffer);
            return FailWithoutLock("referencing the shared encoder hw_frames_ctx failed");
        }

        av_frame_unref(output);
        output->buf[0] = owner_buffer;
        output->data[0] = reinterpret_cast<uint8_t*>(wrapper);
        output->format = AV_PIX_FMT_AMF_SURFACE;
        output->width = static_cast<int>(visible_width_);
        output->height = static_cast<int>(visible_height_);
        output->pts = pts;
        output->hw_frames_ctx = frame_context;
        ++stats_.encoder_wrapper_creates;
        ++stats_.encode_acquire_successes;
        state_ = JASNA_RESIDENT_ACTIVE;
        UpdatePeakInFlightLocked();
        FillCopyInfoLocked(
            info, output_index, d3d_complete_value, hip_value,
            hip_result, amf_result, d3d_result, pts
        );
        return 0;
    }

    int BeginDrain() {
        std::lock_guard<std::mutex> guard(mutex_);
        ClearErrorLocked();
        if (state_ == JASNA_RESIDENT_CLOSED) {
            return SetErrorLocked("resident session is already closed");
        }
        if (state_ == JASNA_RESIDENT_FAILED) {
            return SetErrorLocked("resident session is permanently failed");
        }
        state_ = JASNA_RESIDENT_DRAINING;
        stats_.state = state_;
        return 0;
    }

    int Close(int timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        ClearErrorLocked();
        if (timeout_ms < 0) {
            return FailLocked("close timeout must not be negative");
        }
        if (state_ == JASNA_RESIDENT_CLOSED) {
            return 0;
        }
        if (state_ != JASNA_RESIDENT_FAILED) {
            state_ = JASNA_RESIDENT_DRAINING;
            stats_.state = state_;
        }
        const auto deadline = std::chrono::steady_clock::now() +
            std::chrono::milliseconds(timeout_ms);
        while (!AllObserversAndWrapperOwnersReturnedLocked()) {
            RetireCompletedDecoderLeasesLocked();
            if (std::chrono::steady_clock::now() >= deadline) {
                ++stats_.teardown_failures;
                return FailLocked("timed out waiting for AMFSurface observer and AVBuffer releases");
            }
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            lock.lock();
        }
        if (!RetireAllD3DLocked(deadline)) {
            ++stats_.teardown_failures;
            return FailLocked("retiring D3D11 work during resident close failed");
        }
        ReleaseAllDecoderLeasesLocked();
        DestroyRootLocked();
        if (!TeardownLedgerBalancedLocked()) {
            return FailLocked("resident close detected an unbalanced native ownership ledger");
        }
        state_ = JASNA_RESIDENT_CLOSED;
        stats_.state = state_;
        ++stats_.root_final_close_calls;
        return 0;
    }

    void GetStats(JasnaAmfD3d11HipResidentStats* stats) {
        if (!stats) {
            return;
        }
        std::lock_guard<std::mutex> guard(mutex_);
        RetireCompletedDecoderLeasesLocked();
        *stats = stats_;
        stats->state = state_;
        stats->free_output_slots = FreeOutputSlotsLocked();
        stats->amf_owned_output_slots = AmfOwnedOutputSlotsLocked();
        stats->retained_decoder_leases = RetainedDecoderLeasesLocked();
    }

    const char* LastError() const {
        return error_.empty() ? nullptr : error_.c_str();
    }

    void OnSurfaceReleased(int slot_index) {
        std::lock_guard<std::mutex> guard(mutex_);
        if (slot_index < 0 || slot_index >= static_cast<int>(JASNA_AMF_D3D11_HIP_RESIDENT_SLOTS)) {
            ++stats_.observer_unexpected_callbacks;
            return;
        }
        JasnaAmfD3d11HipResidentOutputSlot& slot = output_slots_[static_cast<size_t>(slot_index)];
        if (!slot.amf_owned) {
            ++stats_.observer_unexpected_callbacks;
            return;
        }
        slot.observer_returned = true;
        slot.amf_owned = false;
        ++stats_.observer_callbacks;
        ++stats_.observer_leases_released;
    }

    void OnWrapperBufferReleased(int slot_index, uint64_t generation) {
        std::lock_guard<std::mutex> guard(mutex_);
        (void)slot_index;
        (void)generation;
        if (stats_.pending_wrapper_owners == 0) {
            ++stats_.observer_unexpected_callbacks;
            return;
        }
        --stats_.pending_wrapper_owners;
        ++stats_.encoder_wrapper_buffer_releases;
    }

private:
    static constexpr unsigned kBytesPerSample = 1;

    static bool ComIdentityEqual(IUnknown* first, IUnknown* second) {
        if (!first || !second) {
            return false;
        }
        ComPtr<IUnknown> first_identity;
        ComPtr<IUnknown> second_identity;
        return SUCCEEDED(first->QueryInterface(IID_PPV_ARGS(&first_identity))) &&
            SUCCEEDED(second->QueryInterface(IID_PPV_ARGS(&second_identity))) &&
            first_identity.Get() == second_identity.Get();
    }

    static void ResetBindInfo(JasnaAmfD3d11HipResidentBindInfo* info) {
        if (info) {
            std::memset(info, 0, sizeof(*info));
            info->api_version = JASNA_AMF_D3D11_HIP_RESIDENT_API_VERSION;
        }
    }

    static void ResetCopyInfo(JasnaAmfD3d11HipResidentCopyInfo* info) {
        if (info) {
            std::memset(info, 0, sizeof(*info));
            info->api_version = JASNA_AMF_D3D11_HIP_RESIDENT_API_VERSION;
            info->slot_count = static_cast<int>(JASNA_AMF_D3D11_HIP_RESIDENT_SLOTS);
            info->slot_index = -1;
        }
    }

    uint64_t PackedBytes() const {
        return static_cast<uint64_t>(visible_width_) * visible_height_ * 3U / 2U;
    }

    uint64_t NextFenceValueLocked() {
        return ++fence_value_;
    }

    void ClearErrorLocked() {
        error_.clear();
    }

    int SetErrorLocked(const char* message) {
        error_ = message ? message : "native resident bridge failure";
        return -1;
    }

    int FailLocked(const char* message) {
        error_ = message ? message : "native resident bridge failure";
        state_ = JASNA_RESIDENT_FAILED;
        stats_.state = state_;
        ++stats_.terminal_failures;
        return -1;
    }

    int FailWithCurrentErrorLocked() {
        if (error_.empty()) {
            error_ = "native resident bridge failure";
        }
        state_ = JASNA_RESIDENT_FAILED;
        stats_.state = state_;
        ++stats_.terminal_failures;
        return -1;
    }

    int FailWithoutLock(const char* message) {
        std::lock_guard<std::mutex> guard(mutex_);
        return FailLocked(message);
    }

    bool EnsureUsableLocked(const char* operation) {
        if (state_ == JASNA_RESIDENT_CLOSED) {
            SetErrorLocked("resident session is closed");
            return false;
        }
        if (state_ == JASNA_RESIDENT_DRAINING) {
            SetErrorLocked("resident session is draining and accepts no new work");
            return false;
        }
        if (state_ == JASNA_RESIDENT_FAILED) {
            SetErrorLocked("resident session is permanently failed");
            return false;
        }
        if (!operation) {
            SetErrorLocked("resident operation name is missing");
            return false;
        }
        return true;
    }

    bool InspectFrameLocked(AVFrame* frame, JasnaAmfD3d11HipResidentFrameView* view) {
        if (!frame || !view) {
            SetErrorLocked("decoder AVFrame is missing");
            return false;
        }
        if (frame->format != AV_PIX_FMT_AMF_SURFACE || !frame->hw_frames_ctx ||
            frame->width != static_cast<int>(visible_width_) ||
            frame->height != static_cast<int>(visible_height_)) {
            SetErrorLocked("decoder frame is not a fixed visible-size AMF surface");
            return false;
        }
        view->frames = reinterpret_cast<AVHWFramesContext*>(frame->hw_frames_ctx->data);
        if (!view->frames || view->frames->format != AV_PIX_FMT_AMF_SURFACE ||
            view->frames->sw_format != AV_PIX_FMT_NV12 || !view->frames->device_ref ||
            view->frames->width != static_cast<int>(visible_width_) ||
            view->frames->height != static_cast<int>(visible_height_)) {
            SetErrorLocked("decoder hw_frames_ctx is not fixed NV12 AMF hardware context");
            return false;
        }
        view->device = reinterpret_cast<AVHWDeviceContext*>(view->frames->device_ref->data);
        if (!view->device || view->device->type != AV_HWDEVICE_TYPE_AMF || !view->device->hwctx) {
            SetErrorLocked("decoder hw_device_ctx is not AMF");
            return false;
        }
        view->amf_device = static_cast<AVAMFDeviceContext*>(view->device->hwctx);
        if (!view->amf_device->context) {
            SetErrorLocked("decoder AMF context is unavailable");
            return false;
        }
        view->surface = reinterpret_cast<AMFSurface*>(frame->data[0]);
        if (!view->surface || view->surface->GetMemoryType() != AMF_MEMORY_DX11 ||
            view->surface->GetFormat() != AMF_SURFACE_NV12 ||
            view->surface->GetPlanesCount() != 2) {
            SetErrorLocked("decoder AMF surface is not DX11 NV12");
            return false;
        }
        view->y_plane = view->surface->GetPlaneAt(0);
        view->uv_plane = view->surface->GetPlaneAt(1);
        if (!view->y_plane || !view->uv_plane || !view->y_plane->GetNative() ||
            !view->uv_plane->GetNative()) {
            SetErrorLocked("decoder AMF surface has no two native DX11 planes");
            return false;
        }
        view->y_hpitch = view->y_plane->GetHPitch();
        view->y_vpitch = view->y_plane->GetVPitch();
        view->uv_hpitch = view->uv_plane->GetHPitch();
        view->uv_vpitch = view->uv_plane->GetVPitch();
        if (view->y_plane->GetWidth() != static_cast<amf_int32>(visible_width_) ||
            view->y_plane->GetHeight() != static_cast<amf_int32>(visible_height_) ||
            view->y_hpitch < static_cast<int>(visible_width_) ||
            view->y_vpitch != static_cast<int>(allocation_height_) ||
            view->uv_plane->GetWidth() != static_cast<amf_int32>(visible_width_ / 2U) ||
            view->uv_plane->GetHeight() != static_cast<amf_int32>(visible_height_ / 2U) ||
            view->uv_hpitch < static_cast<int>(visible_width_) ||
            view->uv_vpitch != static_cast<int>(allocation_height_ / 2U)) {
            SetErrorLocked("decoder AMF plane pitch or allocation geometry changed");
            return false;
        }
        IUnknown* y_native = reinterpret_cast<IUnknown*>(view->y_plane->GetNative());
        IUnknown* uv_native = reinterpret_cast<IUnknown*>(view->uv_plane->GetNative());
        ComPtr<ID3D11Texture2D> uv_texture;
        if (FAILED(y_native->QueryInterface(IID_PPV_ARGS(&view->texture))) ||
            FAILED(uv_native->QueryInterface(IID_PPV_ARGS(&uv_texture))) ||
            !ComIdentityEqual(view->texture.Get(), uv_texture.Get())) {
            SetErrorLocked("decoder AMF planes do not expose one ID3D11Texture2D");
            return false;
        }
        D3D11_TEXTURE2D_DESC description{};
        view->texture->GetDesc(&description);
        if (description.Width != allocation_width_ ||
            description.Height != allocation_height_ ||
            description.Format != DXGI_FORMAT_NV12 || description.ArraySize != 1 ||
            description.MipLevels != 1 || description.SampleDesc.Count != 1 ||
            (description.BindFlags & D3D11_BIND_DECODER) == 0 ||
            (description.BindFlags & D3D11_BIND_SHADER_RESOURCE) == 0) {
            SetErrorLocked("decoder DX11 texture violates the fixed NV12 allocation contract");
            return false;
        }
        return true;
    }

    bool InitializeRootFromFrameLocked(const JasnaAmfD3d11HipResidentFrameView& view) {
        amf_context_ = view.amf_device->context;
        void* raw_device = amf_context_->GetDX11Device();
        if (!raw_device || FAILED(reinterpret_cast<IUnknown*>(raw_device)->QueryInterface(
                IID_PPV_ARGS(&d3d_device_)))) {
            SetErrorLocked("AMF context did not expose an ID3D11Device");
            return false;
        }
        ComPtr<ID3D11Device> texture_device;
        view.texture->GetDevice(&texture_device);
        if (!texture_device || !ComIdentityEqual(d3d_device_.Get(), texture_device.Get())) {
            SetErrorLocked("AMF DX11 device differs from decoder surface device");
            return false;
        }
        ID3D11DeviceContext* immediate = nullptr;
        d3d_device_->GetImmediateContext(&immediate);
        d3d_context_.Attach(immediate);
        if (!d3d_context_ || FAILED(d3d_device_.As(&device3_)) ||
            FAILED(d3d_device_.As(&device5_)) || FAILED(d3d_context_.As(&context4_))) {
            SetErrorLocked("required D3D11.3/11.4 fence and plane-view interfaces are unavailable");
            return false;
        }
        ComPtr<IDXGIDevice> dxgi_device;
        ComPtr<IDXGIAdapter> base_adapter;
        if (FAILED(d3d_device_.As(&dxgi_device)) ||
            FAILED(dxgi_device->GetAdapter(&base_adapter)) ||
            FAILED(base_adapter.As(&adapter_))) {
            SetErrorLocked("cannot obtain the DXGI adapter for the AMF DX11 device");
            return false;
        }
        DXGI_ADAPTER_DESC1 adapter_description{};
        hipDeviceProp_t hip_properties{};
        if (FAILED(adapter_->GetDesc1(&adapter_description)) ||
            hipGetDeviceProperties(&hip_properties, hip_device_) != hipSuccess ||
            std::memcmp(&adapter_description.AdapterLuid, hip_properties.luid,
                        sizeof(adapter_description.AdapterLuid)) != 0) {
            SetErrorLocked("HIP device LUID does not match the AMF DX11 adapter LUID");
            return false;
        }
        stats_.adapter_luid_match = 1;
        if (hipSetDevice(hip_device_) != hipSuccess) {
            SetErrorLocked("selecting the validated HIP device failed");
            return false;
        }
        D3D11_FEATURE_DATA_D3D11_OPTIONS4 options4{};
        D3D11_FEATURE_DATA_D3D11_OPTIONS5 options5{};
        if (FAILED(d3d_device_->CheckFeatureSupport(
                D3D11_FEATURE_D3D11_OPTIONS4, &options4, sizeof(options4))) ||
            FAILED(d3d_device_->CheckFeatureSupport(
                D3D11_FEATURE_D3D11_OPTIONS5, &options5, sizeof(options5))) ||
            !options4.ExtendedNV12SharedTextureSupported ||
            options5.SharedResourceTier < D3D11_SHARED_RESOURCE_TIER_1) {
            SetErrorLocked("D3D11 shared NV12 feature contract is not available");
            return false;
        }
        if (!CreatePlaneCopyShadersLocked() || !CreateBridgeSlotsLocked() ||
            !CreateOutputSlotsLocked() || !CreateFenceBridgeLocked() ||
            hipStreamCreateWithFlags(&prewarm_stream_, hipStreamNonBlocking) != hipSuccess) {
            SetErrorLocked("creating fixed resident DX11/HIP resources failed");
            return false;
        }
        if (!PrewarmAllSlotsLocked()) {
            SetErrorLocked("four-slot resident prewarm failed");
            return false;
        }
        return true;
    }

    bool CreatePlaneCopyShadersLocked() {
        if (FAILED(d3d_device_->CreateVertexShader(
                jasna_amf_d3d11_hip_resident_vs,
                sizeof(jasna_amf_d3d11_hip_resident_vs), nullptr,
                &shader_copy_.vertex_shader)) ||
            FAILED(d3d_device_->CreatePixelShader(
                jasna_amf_d3d11_hip_resident_ps,
                sizeof(jasna_amf_d3d11_hip_resident_ps), nullptr,
                &shader_copy_.pixel_shader))) {
            return false;
        }
        D3D11_RASTERIZER_DESC rasterizer{};
        rasterizer.FillMode = D3D11_FILL_SOLID;
        rasterizer.CullMode = D3D11_CULL_NONE;
        rasterizer.DepthClipEnable = TRUE;
        return SUCCEEDED(d3d_device_->CreateRasterizerState(
            &rasterizer, &shader_copy_.rasterizer
        ));
    }

    bool CreateImportedPlaneLocked(
        unsigned width,
        unsigned height,
        DXGI_FORMAT format,
        const hipChannelFormatDesc& channel,
        unsigned bytes_per_pixel,
        JasnaAmfD3d11HipResidentPlane* plane
    ) {
        if (!plane) {
            return false;
        }
        D3D11_TEXTURE2D_DESC description{};
        description.Width = width;
        description.Height = height;
        description.MipLevels = 1;
        description.ArraySize = 1;
        description.Format = format;
        description.SampleDesc.Count = 1;
        description.Usage = D3D11_USAGE_DEFAULT;
        description.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        description.MiscFlags = D3D11_RESOURCE_MISC_SHARED;
        if (FAILED(d3d_device_->CreateTexture2D(&description, nullptr, &plane->texture)) ||
            FAILED(d3d_device_->CreateRenderTargetView(
                plane->texture.Get(), nullptr, &plane->rtv)) ||
            FAILED(d3d_device_->CreateShaderResourceView(
                plane->texture.Get(), nullptr, &plane->srv))) {
            return false;
        }
        ComPtr<IDXGIResource> dxgi_resource;
        HANDLE kmt_handle = nullptr;
        if (FAILED(plane->texture.As(&dxgi_resource)) ||
            FAILED(dxgi_resource->GetSharedHandle(&kmt_handle)) || !kmt_handle) {
            return false;
        }
        // A D3D11 KMT resource handle is borrowed by HIP import.  It is not an
        // NT fence handle and must not be sent to CloseHandle.
        plane->width = width;
        plane->height = height;
        plane->bytes_per_pixel = bytes_per_pixel;
        plane->logical_bytes = static_cast<uint64_t>(width) * height * bytes_per_pixel;
        hipExternalMemoryHandleDesc memory_description{};
        memory_description.type = hipExternalMemoryHandleTypeD3D11ResourceKmt;
        memory_description.handle.win32.handle = kmt_handle;
        memory_description.size = static_cast<unsigned long long>(plane->logical_bytes);
        memory_description.flags = hipExternalMemoryDedicated;
        if (hipImportExternalMemory(&plane->external_memory, &memory_description) != hipSuccess) {
            return false;
        }
        ++stats_.external_memory_imports;
        hipExternalMemoryMipmappedArrayDesc array_description{};
        array_description.extent = make_hipExtent(width, height, 0);
        array_description.formatDesc = channel;
        array_description.numLevels = 1;
        if (hipExternalMemoryGetMappedMipmappedArray(
                &plane->mipmapped_array, plane->external_memory,
                &array_description
            ) != hipSuccess) {
            return false;
        }
        ++stats_.mapped_arrays_created;
        if (hipGetMipmappedArrayLevel(&plane->array, plane->mipmapped_array, 0) != hipSuccess) {
            return false;
        }
        hipResourceDesc resource_description{};
        resource_description.resType = hipResourceTypeArray;
        resource_description.res.array.array = plane->array;
        if (hipCreateSurfaceObject(&plane->surface, &resource_description) != hipSuccess) {
            return false;
        }
        ++stats_.hip_surface_objects_created;
        return true;
    }

    bool CreateBridgeSlotsLocked() {
        const hipChannelFormatDesc y_channel = hipCreateChannelDesc(
            8, 0, 0, 0, hipChannelFormatKindUnsigned
        );
        const hipChannelFormatDesc uv_channel = hipCreateChannelDesc(
            8, 8, 0, 0, hipChannelFormatKindUnsigned
        );
        for (unsigned index = 0; index < JASNA_AMF_D3D11_HIP_RESIDENT_SLOTS; ++index) {
            JasnaAmfD3d11HipResidentBridgeSlot& slot = bridge_slots_[index];
            if (!CreateImportedPlaneLocked(
                    visible_width_, visible_height_, DXGI_FORMAT_R8_UNORM,
                    y_channel, 1, &slot.y
                ) ||
                !CreateImportedPlaneLocked(
                    visible_width_ / 2U, visible_height_ / 2U,
                    DXGI_FORMAT_R8G8_UNORM, uv_channel, 2, &slot.uv
                )) {
                return false;
            }
            ++stats_.bridge_slots_created;
        }
        return stats_.external_memory_imports == JASNA_AMF_D3D11_HIP_RESIDENT_SLOTS * 2U;
    }

    bool CreateOutputSlotsLocked() {
        for (unsigned index = 0; index < JASNA_AMF_D3D11_HIP_RESIDENT_SLOTS; ++index) {
            JasnaAmfD3d11HipResidentOutputSlot& slot = output_slots_[index];
            D3D11_TEXTURE2D_DESC description{};
            description.Width = visible_width_;
            description.Height = visible_height_;
            description.MipLevels = 1;
            description.ArraySize = 1;
            description.Format = DXGI_FORMAT_NV12;
            description.SampleDesc.Count = 1;
            description.Usage = D3D11_USAGE_DEFAULT;
            description.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
            if (FAILED(d3d_device_->CreateTexture2D(&description, nullptr, &slot.texture))) {
                return false;
            }
            D3D11_RENDER_TARGET_VIEW_DESC1 y{};
            y.Format = DXGI_FORMAT_R8_UNORM;
            y.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
            y.Texture2D.MipSlice = 0;
            y.Texture2D.PlaneSlice = 0;
            D3D11_RENDER_TARGET_VIEW_DESC1 uv{};
            uv.Format = DXGI_FORMAT_R8G8_UNORM;
            uv.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
            uv.Texture2D.MipSlice = 0;
            uv.Texture2D.PlaneSlice = 1;
            if (FAILED(device3_->CreateRenderTargetView1(slot.texture.Get(), &y, &slot.y_rtv)) ||
                FAILED(device3_->CreateRenderTargetView1(slot.texture.Get(), &uv, &slot.uv_rtv))) {
                return false;
            }
            slot.observer.session = this;
            slot.observer.slot_index = static_cast<int>(index);
            ++stats_.output_slots_created;
        }
        return true;
    }

    bool CreateFenceBridgeLocked() {
        if (!adapter_ || FAILED(D3D12CreateDevice(
                adapter_.Get(), D3D_FEATURE_LEVEL_11_0,
                IID_PPV_ARGS(&fence_bridge_.d3d12_device)))) {
            return false;
        }
        if (FAILED(fence_bridge_.d3d12_device->CreateFence(
                0, D3D12_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence_bridge_.d3d12_fence)))) {
            return false;
        }
        ++stats_.d3d12_fences_created;
        HANDLE shared_handle = nullptr;
        if (FAILED(fence_bridge_.d3d12_device->CreateSharedHandle(
                fence_bridge_.d3d12_fence.Get(), nullptr, GENERIC_ALL,
                nullptr, &shared_handle)) || !shared_handle) {
            return false;
        }
        ++stats_.shared_fence_handles_created;
        const HRESULT opened = device5_->OpenSharedFence(
            shared_handle, IID_PPV_ARGS(&fence_bridge_.d3d11_fence)
        );
        if (SUCCEEDED(opened)) {
            ++stats_.d3d11_opened_fences_created;
        }
        hipExternalSemaphoreHandleDesc semaphore_description{};
        semaphore_description.type = hipExternalSemaphoreHandleTypeD3D12Fence;
        semaphore_description.handle.win32.handle = shared_handle;
        const hipError_t imported = SUCCEEDED(opened)
            ? hipImportExternalSemaphore(&fence_bridge_.semaphore, &semaphore_description)
            : hipErrorInvalidValue;
        // This is the sole CloseHandle call for the D3D12-created NT handle,
        // including all partial-import paths.
        const BOOL closed = CloseHandle(shared_handle);
        if (closed) {
            ++stats_.shared_fence_handles_closed;
        }
        if (imported != hipSuccess || !closed) {
            return false;
        }
        ++stats_.hip_external_semaphores_imported;
        return true;
    }

    bool PrewarmAllSlotsLocked() {
        for (unsigned index = 0; index < JASNA_AMF_D3D11_HIP_RESIDENT_SLOTS; ++index) {
            JasnaAmfD3d11HipResidentBridgeSlot& bridge = bridge_slots_[index];
            JasnaAmfD3d11HipResidentOutputSlot& output = output_slots_[index];
            const uint64_t d3d_value = NextFenceValueLocked();
            const uint64_t hip_value = NextFenceValueLocked();
            const uint64_t d3d_complete_value = NextFenceValueLocked();
            if (!LockDx11Locked()) {
                return false;
            }
            const float y_clear[4] = {static_cast<float>(16U + index) / 255.0F, 0.0F, 0.0F, 0.0F};
            const float uv_clear[4] = {0.5F, static_cast<float>(96U + index) / 255.0F, 0.0F, 0.0F};
            d3d_context_->ClearRenderTargetView(bridge.y.rtv.Get(), y_clear);
            d3d_context_->ClearRenderTargetView(bridge.uv.rtv.Get(), uv_clear);
            const HRESULT signaled = context4_->Signal(fence_bridge_.d3d11_fence.Get(), d3d_value);
            if (SUCCEEDED(signaled)) {
                ++stats_.d3d_to_hip_signals;
                d3d_context_->Flush();
            }
            const bool unlocked = UnlockDx11Locked();
            if (FAILED(signaled) || !unlocked) {
                return false;
            }
            hipExternalSemaphoreWaitParams wait{};
            wait.params.fence.value = d3d_value;
            if (hipWaitExternalSemaphoresAsync(
                    &fence_bridge_.semaphore, &wait, 1, prewarm_stream_
                ) != hipSuccess) {
                return false;
            }
            ++stats_.d3d_to_hip_waits;
            hipExternalSemaphoreSignalParams signal{};
            signal.params.fence.value = hip_value;
            if (hipSignalExternalSemaphoresAsync(
                    &fence_bridge_.semaphore, &signal, 1, prewarm_stream_
                ) != hipSuccess) {
                return false;
            }
            ++stats_.hip_to_d3d_signals;
            if (!LockDx11Locked()) {
                return false;
            }
            const HRESULT waited = context4_->Wait(fence_bridge_.d3d11_fence.Get(), hip_value);
            HRESULT completed_signal = E_FAIL;
            if (SUCCEEDED(waited)) {
                ++stats_.hip_to_d3d_waits;
                bridge.d3d_wait_queued = true;
                d3d_context_->ClearRenderTargetView(output.y_rtv.Get(), y_clear);
                d3d_context_->ClearRenderTargetView(output.uv_rtv.Get(), uv_clear);
                completed_signal = context4_->Signal(
                    fence_bridge_.d3d11_fence.Get(), d3d_complete_value
                );
                if (SUCCEEDED(completed_signal)) {
                    ++stats_.d3d_to_hip_signals;
                    bridge.d3d_to_hip_fence_value = d3d_complete_value;
                    bridge.hip_to_d3d_fence_value = 0;
                }
                d3d_context_->Flush();
            }
            const bool output_unlocked = UnlockDx11Locked();
            if (FAILED(waited) || FAILED(completed_signal) || !output_unlocked) {
                return false;
            }
            AMFSurface* wrapper = nullptr;
            const AMF_RESULT created = amf_context_->CreateSurfaceFromDX11Native(
                output.texture.Get(), &wrapper, nullptr
            );
            if (created != AMF_OK || !wrapper || wrapper->GetMemoryType() != AMF_MEMORY_DX11 ||
                wrapper->GetFormat() != AMF_SURFACE_NV12 || wrapper->GetPlanesCount() != 2) {
                if (wrapper) {
                    wrapper->Release();
                }
                return false;
            }
            wrapper->Release();
            ++stats_.prewarm_slots_completed;
        }
        stats_.prewarm_complete = stats_.prewarm_slots_completed ==
            JASNA_AMF_D3D11_HIP_RESIDENT_SLOTS;
        return stats_.prewarm_complete != 0;
    }

    bool CreateDecoderPlaneViewsLocked(
        ID3D11Texture2D* texture,
        ComPtr<ID3D11ShaderResourceView1>* y,
        ComPtr<ID3D11ShaderResourceView1>* uv
    ) {
        if (!texture || !y || !uv) {
            return false;
        }
        D3D11_SHADER_RESOURCE_VIEW_DESC1 y_description{};
        y_description.Format = DXGI_FORMAT_R8_UNORM;
        y_description.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        y_description.Texture2D.MostDetailedMip = 0;
        y_description.Texture2D.MipLevels = 1;
        y_description.Texture2D.PlaneSlice = 0;
        D3D11_SHADER_RESOURCE_VIEW_DESC1 uv_description{};
        uv_description.Format = DXGI_FORMAT_R8G8_UNORM;
        uv_description.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        uv_description.Texture2D.MostDetailedMip = 0;
        uv_description.Texture2D.MipLevels = 1;
        uv_description.Texture2D.PlaneSlice = 1;
        return SUCCEEDED(device3_->CreateShaderResourceView1(texture, &y_description, y->GetAddressOf())) &&
            SUCCEEDED(device3_->CreateShaderResourceView1(texture, &uv_description, uv->GetAddressOf()));
    }

    void DrawPlaneLocked(
        ID3D11ShaderResourceView* source,
        ID3D11RenderTargetView* target,
        unsigned width,
        unsigned height
    ) {
        D3D11_VIEWPORT viewport{};
        viewport.Width = static_cast<float>(width);
        viewport.Height = static_cast<float>(height);
        viewport.MinDepth = 0.0F;
        viewport.MaxDepth = 1.0F;
        d3d_context_->OMSetRenderTargets(1, &target, nullptr);
        d3d_context_->RSSetViewports(1, &viewport);
        d3d_context_->RSSetState(shader_copy_.rasterizer.Get());
        d3d_context_->IASetInputLayout(nullptr);
        d3d_context_->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
        d3d_context_->VSSetShader(shader_copy_.vertex_shader.Get(), nullptr, 0);
        d3d_context_->PSSetShader(shader_copy_.pixel_shader.Get(), nullptr, 0);
        d3d_context_->PSSetShaderResources(0, 1, &source);
        d3d_context_->Draw(3, 0);
        ID3D11ShaderResourceView* null_source = nullptr;
        ID3D11RenderTargetView* null_target = nullptr;
        d3d_context_->PSSetShaderResources(0, 1, &null_source);
        d3d_context_->OMSetRenderTargets(1, &null_target, nullptr);
    }

    bool LockDx11Locked() {
        return amf_context_ && amf_context_->LockDX11() == AMF_OK;
    }

    bool UnlockDx11Locked() {
        return amf_context_ && amf_context_->UnlockDX11() == AMF_OK;
    }

    HRESULT QueueLatestHipSignalWaitOnD3DLocked() {
        if (latest_hip_signal_value_ == 0) {
            return S_OK;
        }
        const HRESULT waited = context4_->Wait(
            fence_bridge_.d3d11_fence.Get(), latest_hip_signal_value_
        );
        if (SUCCEEDED(waited)) {
            ++stats_.hip_to_d3d_waits;
        }
        return waited;
    }

    bool PrepareBridgeForD3DLocked(JasnaAmfD3d11HipResidentBridgeSlot& slot) {
        if (slot.hip_to_d3d_fence_value != 0 && !slot.d3d_wait_queued) {
            if (!LockDx11Locked()) {
                return false;
            }
            const HRESULT waited = context4_->Wait(
                fence_bridge_.d3d11_fence.Get(), slot.hip_to_d3d_fence_value
            );
            if (SUCCEEDED(waited)) {
                ++stats_.hip_to_d3d_waits;
                slot.d3d_wait_queued = true;
                d3d_context_->Flush();
            }
            const bool unlocked = UnlockDx11Locked();
            if (FAILED(waited) || !unlocked) {
                return false;
            }
        }
        if (slot.decoder_lease) {
            if (fence_bridge_.d3d11_fence->GetCompletedValue() < slot.decoder_d3d_fence_value) {
                return false;
            }
            av_frame_free(&slot.decoder_lease);
            slot.decoder_d3d_fence_value = 0;
            ++stats_.decoder_frame_leases_released;
        }
        return true;
    }

    int FindBridgeSlotForUseLocked() {
        for (unsigned attempt = 0; attempt < JASNA_AMF_D3D11_HIP_RESIDENT_SLOTS; ++attempt) {
            const unsigned index = (bridge_cursor_ + attempt) % JASNA_AMF_D3D11_HIP_RESIDENT_SLOTS;
            if (PrepareBridgeForD3DLocked(bridge_slots_[index])) {
                bridge_cursor_ = (index + 1U) % JASNA_AMF_D3D11_HIP_RESIDENT_SLOTS;
                return static_cast<int>(index);
            }
        }
        return -1;
    }

    int FindOutputSlotLocked() {
        for (unsigned attempt = 0; attempt < JASNA_AMF_D3D11_HIP_RESIDENT_SLOTS; ++attempt) {
            const unsigned index = (output_cursor_ + attempt) % JASNA_AMF_D3D11_HIP_RESIDENT_SLOTS;
            JasnaAmfD3d11HipResidentOutputSlot& slot = output_slots_[index];
            if (!slot.amf_owned && slot.observer_returned) {
                slot.observer_returned = false;
                output_cursor_ = (index + 1U) % JASNA_AMF_D3D11_HIP_RESIDENT_SLOTS;
                return static_cast<int>(index);
            }
            if (!slot.amf_owned && slot.generation == 0) {
                output_cursor_ = (index + 1U) % JASNA_AMF_D3D11_HIP_RESIDENT_SLOTS;
                return static_cast<int>(index);
            }
        }
        return -1;
    }

    bool ProjectDecoderToBridgeLocked(
        const JasnaAmfD3d11HipResidentFrameView& view,
        JasnaAmfD3d11HipResidentBridgeSlot& bridge,
        uint64_t d3d_value,
        int* d3d_result
    ) {
        ComPtr<ID3D11ShaderResourceView1> y_source;
        ComPtr<ID3D11ShaderResourceView1> uv_source;
        if (!CreateDecoderPlaneViewsLocked(view.texture.Get(), &y_source, &uv_source) ||
            !LockDx11Locked()) {
            if (d3d_result) {
                *d3d_result = E_FAIL;
            }
            return false;
        }
        // This session intentionally imports one shared D3D12 timeline fence.
        // Every new D3D signal must therefore be ordered after the most recent
        // HIP signal, even when a different one of the four bridge textures is
        // selected.  Batch size one hid this requirement because Python
        // synchronized after every frame; a four-frame batch could otherwise
        // let D3D signal N+1 overtake the queued HIP signal N and permanently
        // block the consumer stream.
        const HRESULT ordered = QueueLatestHipSignalWaitOnD3DLocked();
        if (FAILED(ordered)) {
            UnlockDx11Locked();
            if (d3d_result) {
                *d3d_result = static_cast<int>(ordered);
            }
            return false;
        }
        DrawPlaneLocked(y_source.Get(), bridge.y.rtv.Get(), visible_width_, visible_height_);
        DrawPlaneLocked(uv_source.Get(), bridge.uv.rtv.Get(), visible_width_ / 2U, visible_height_ / 2U);
        const HRESULT signaled = context4_->Signal(fence_bridge_.d3d11_fence.Get(), d3d_value);
        if (SUCCEEDED(signaled)) {
            ++stats_.d3d_to_hip_signals;
            bridge.d3d_to_hip_fence_value = d3d_value;
            bridge.hip_to_d3d_fence_value = 0;
            d3d_context_->Flush();
        }
        const bool unlocked = UnlockDx11Locked();
        if (d3d_result) {
            *d3d_result = static_cast<int>(signaled);
        }
        return SUCCEEDED(signaled) && unlocked;
    }

    hipError_t WaitForLatestHipSignalOnHipLocked(hipStream_t stream) {
        if (latest_hip_signal_value_ == 0) {
            return hipSuccess;
        }
        hipExternalSemaphoreWaitParams wait{};
        wait.params.fence.value = latest_hip_signal_value_;
        const hipError_t result = hipWaitExternalSemaphoresAsync(
            &fence_bridge_.semaphore, &wait, 1, stream
        );
        if (result == hipSuccess) {
            ++stats_.hip_to_hip_waits;
        }
        return result;
    }

    hipError_t CopyBridgeToHipLocked(
        JasnaAmfD3d11HipResidentBridgeSlot& bridge,
        uintptr_t destination,
        uintptr_t consumer_stream,
        uint64_t hip_value
    ) {
        const hipStream_t stream = reinterpret_cast<hipStream_t>(consumer_stream);
        if (hipSetDevice(hip_device_) != hipSuccess) {
            return hipErrorInvalidDevice;
        }
        const uint64_t d3d_value = bridge.d3d_to_hip_fence_value;
        if (d3d_value == 0) {
            return hipErrorInvalidValue;
        }
        hipError_t result = WaitForLatestHipSignalOnHipLocked(stream);
        if (result != hipSuccess) {
            return result;
        }
        hipExternalSemaphoreWaitParams wait{};
        wait.params.fence.value = d3d_value;
        result = hipWaitExternalSemaphoresAsync(
            &fence_bridge_.semaphore, &wait, 1, stream
        );
        if (result != hipSuccess) {
            return result;
        }
        ++stats_.d3d_to_hip_waits;
        // From this point, a failed later HIP call must leave an unsignaled
        // fence dependency behind so Close() fails closed instead of releasing
        // a bridge whose producer stream may still be touching it.
        bridge.hip_to_d3d_fence_value = hip_value;
        bridge.d3d_to_hip_fence_value = 0;
        bridge.d3d_wait_queued = false;
        auto* y_destination = reinterpret_cast<uint8_t*>(destination);
        auto* uv_destination = y_destination + static_cast<size_t>(visible_width_) * visible_height_;
        result = hipMemcpy2DFromArrayAsync(
            y_destination, visible_width_, bridge.y.array, 0, 0,
            visible_width_, visible_height_, hipMemcpyDeviceToDevice, stream
        );
        if (result != hipSuccess) {
            return result;
        }
        result = hipMemcpy2DFromArrayAsync(
            uv_destination, visible_width_, bridge.uv.array, 0, 0,
            visible_width_, visible_height_ / 2U, hipMemcpyDeviceToDevice, stream
        );
        if (result != hipSuccess) {
            return result;
        }
        hipExternalSemaphoreSignalParams signal{};
        signal.params.fence.value = hip_value;
        result = hipSignalExternalSemaphoresAsync(
            &fence_bridge_.semaphore, &signal, 1, stream
        );
        if (result == hipSuccess) {
            ++stats_.hip_to_d3d_signals;
            latest_hip_signal_value_ = hip_value;
            bridge.hip_to_d3d_fence_value = hip_value;
            bridge.d3d_to_hip_fence_value = 0;
            bridge.d3d_wait_queued = false;
        }
        return result;
    }

    hipError_t WaitForBridgeReuseOnHipLocked(
        const JasnaAmfD3d11HipResidentBridgeSlot& bridge,
        hipStream_t stream
    ) {
        // Successful paths retain only the newest completion.  A reserved HIP
        // terminal signal can remain after a failed submission; selecting the
        // larger generated value keeps that failure path conservative too.
        const bool wait_for_d3d = bridge.d3d_to_hip_fence_value >=
            bridge.hip_to_d3d_fence_value;
        const uint64_t completion_value = wait_for_d3d
            ? bridge.d3d_to_hip_fence_value
            : bridge.hip_to_d3d_fence_value;
        if (completion_value == 0) {
            return hipSuccess;
        }
        hipExternalSemaphoreWaitParams wait{};
        wait.params.fence.value = completion_value;
        const hipError_t result = hipWaitExternalSemaphoresAsync(
            &fence_bridge_.semaphore, &wait, 1, stream
        );
        if (result == hipSuccess) {
            if (wait_for_d3d) {
                ++stats_.d3d_to_hip_waits;
            } else {
                ++stats_.hip_to_hip_waits;
            }
        }
        return result;
    }

    hipError_t CopyHipToBridgeLocked(
        JasnaAmfD3d11HipResidentBridgeSlot& bridge,
        uintptr_t source,
        uintptr_t producer_stream,
        uint64_t hip_value
    ) {
        const hipStream_t stream = reinterpret_cast<hipStream_t>(producer_stream);
        if (hipSetDevice(hip_device_) != hipSuccess) {
            return hipErrorInvalidDevice;
        }
        hipError_t result = WaitForLatestHipSignalOnHipLocked(stream);
        if (result != hipSuccess) {
            return result;
        }
        result = WaitForBridgeReuseOnHipLocked(bridge, stream);
        if (result != hipSuccess) {
            return result;
        }
        // The external wait was accepted by this stream, so all following
        // writes are ordered after the prior bridge owner.  Reserve the HIP
        // signal now to keep a later copy/signal failure fail-closed at close.
        bridge.hip_to_d3d_fence_value = hip_value;
        bridge.d3d_to_hip_fence_value = 0;
        bridge.d3d_wait_queued = false;
        const auto* y_source = reinterpret_cast<const uint8_t*>(source);
        const auto* uv_source = y_source + static_cast<size_t>(visible_width_) * visible_height_;
        result = hipMemcpy2DToArrayAsync(
            bridge.y.array, 0, 0, y_source, visible_width_, visible_width_,
            visible_height_, hipMemcpyDeviceToDevice, stream
        );
        if (result != hipSuccess) {
            return result;
        }
        result = hipMemcpy2DToArrayAsync(
            bridge.uv.array, 0, 0, uv_source, visible_width_, visible_width_,
            visible_height_ / 2U, hipMemcpyDeviceToDevice, stream
        );
        if (result != hipSuccess) {
            return result;
        }
        hipExternalSemaphoreSignalParams signal{};
        signal.params.fence.value = hip_value;
        result = hipSignalExternalSemaphoresAsync(
            &fence_bridge_.semaphore, &signal, 1, stream
        );
        if (result == hipSuccess) {
            ++stats_.hip_to_d3d_signals;
            latest_hip_signal_value_ = hip_value;
            bridge.hip_to_d3d_fence_value = hip_value;
            bridge.d3d_to_hip_fence_value = 0;
            bridge.d3d_wait_queued = false;
        }
        return result;
    }

    bool ProjectBridgeToOutputLocked(
        JasnaAmfD3d11HipResidentBridgeSlot& bridge,
        JasnaAmfD3d11HipResidentOutputSlot& output,
        uint64_t hip_value,
        uint64_t d3d_complete_value,
        int* d3d_result
    ) {
        if (!LockDx11Locked()) {
            if (d3d_result) {
                *d3d_result = E_FAIL;
            }
            return false;
        }
        const HRESULT waited = context4_->Wait(fence_bridge_.d3d11_fence.Get(), hip_value);
        HRESULT signaled = E_FAIL;
        if (SUCCEEDED(waited)) {
            ++stats_.hip_to_d3d_waits;
            bridge.d3d_wait_queued = true;
            DrawPlaneLocked(bridge.y.srv.Get(), output.y_rtv.Get(), visible_width_, visible_height_);
            DrawPlaneLocked(bridge.uv.srv.Get(), output.uv_rtv.Get(), visible_width_ / 2U, visible_height_ / 2U);
            signaled = context4_->Signal(
                fence_bridge_.d3d11_fence.Get(), d3d_complete_value
            );
            if (SUCCEEDED(signaled)) {
                ++stats_.d3d_to_hip_signals;
                bridge.d3d_to_hip_fence_value = d3d_complete_value;
                bridge.hip_to_d3d_fence_value = 0;
            }
            d3d_context_->Flush();
        }
        const bool unlocked = UnlockDx11Locked();
        if (d3d_result) {
            *d3d_result = static_cast<int>(SUCCEEDED(waited) ? signaled : waited);
        }
        return SUCCEEDED(waited) && SUCCEEDED(signaled) && unlocked;
    }

    void RetireCompletedDecoderLeasesLocked() {
        if (!fence_bridge_.d3d11_fence) {
            return;
        }
        const uint64_t completed = fence_bridge_.d3d11_fence->GetCompletedValue();
        for (auto& slot : bridge_slots_) {
            if (slot.decoder_lease && completed >= slot.decoder_d3d_fence_value) {
                av_frame_free(&slot.decoder_lease);
                slot.decoder_d3d_fence_value = 0;
                ++stats_.decoder_frame_leases_released;
            }
        }
    }

    bool RetireAllD3DLocked(const std::chrono::steady_clock::time_point& deadline) {
        // A validation or root-initialization failure can leave this session
        // FAILED before any D3D/HIP work exists.  That must still be
        // explicitly closable so the wrapper can destroy the inert native
        // object instead of preserving an unnecessary permanent leak.
        if (!d3d_device_ && !d3d_context_ && !context4_ &&
            !fence_bridge_.d3d11_fence && !fence_bridge_.semaphore &&
            !prewarm_stream_) {
            return true;
        }
        if (!d3d_device_ || !d3d_context_ || !context4_ || !fence_bridge_.d3d11_fence ||
            !LockDx11Locked()) {
            return false;
        }
        bool success = true;
        for (auto& bridge : bridge_slots_) {
            if (bridge.hip_to_d3d_fence_value != 0 && !bridge.d3d_wait_queued) {
                const HRESULT waited = context4_->Wait(
                    fence_bridge_.d3d11_fence.Get(), bridge.hip_to_d3d_fence_value
                );
                if (SUCCEEDED(waited)) {
                    ++stats_.hip_to_d3d_waits;
                    bridge.d3d_wait_queued = true;
                } else {
                    success = false;
                }
            }
        }
        D3D11_QUERY_DESC query_description{D3D11_QUERY_EVENT, 0};
        ComPtr<ID3D11Query> query;
        if (success && FAILED(d3d_device_->CreateQuery(&query_description, &query))) {
            success = false;
        }
        if (success) {
            d3d_context_->End(query.Get());
            d3d_context_->Flush();
        }
        if (!UnlockDx11Locked()) {
            success = false;
        }
        while (success) {
            const HRESULT result = d3d_context_->GetData(query.Get(), nullptr, 0, 0);
            if (result == S_OK) {
                break;
            }
            if (result != S_FALSE || std::chrono::steady_clock::now() >= deadline) {
                success = false;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (!success) {
            return false;
        }
        if (!LockDx11Locked()) {
            return false;
        }
        d3d_context_->ClearState();
        d3d_context_->Flush();
        if (!UnlockDx11Locked()) {
            return false;
        }
        for (auto& bridge : bridge_slots_) {
            bridge.d3d_to_hip_fence_value = 0;
            bridge.hip_to_d3d_fence_value = 0;
            bridge.d3d_wait_queued = false;
        }
        RetireCompletedDecoderLeasesLocked();
        return RetainedDecoderLeasesLocked() == 0;
    }

    bool AllObserversAndWrapperOwnersReturnedLocked() const {
        return AmfOwnedOutputSlotsLocked() == 0 && stats_.pending_wrapper_owners == 0;
    }

    bool TeardownLedgerBalancedLocked() const {
        return stats_.teardown_failures == 0 &&
            stats_.external_memory_imports == stats_.external_memory_destroys &&
            stats_.mapped_arrays_created == stats_.mapped_arrays_destroyed &&
            stats_.hip_surface_objects_created == stats_.hip_surface_objects_destroyed &&
            stats_.d3d12_fences_created == stats_.d3d12_fences_destroyed &&
            stats_.d3d11_opened_fences_created == stats_.d3d11_opened_fences_destroyed &&
            stats_.shared_fence_handles_created == stats_.shared_fence_handles_closed &&
            stats_.hip_external_semaphores_imported ==
                stats_.hip_external_semaphores_destroyed &&
            stats_.decoder_frame_leases_acquired == stats_.decoder_frame_leases_released &&
            stats_.observer_leases_acquired == stats_.observer_leases_released &&
            stats_.pending_wrapper_owners == 0;
    }

    int FreeOutputSlotsLocked() const {
        int count = 0;
        for (const auto& slot : output_slots_) {
            if (!slot.amf_owned &&
                (slot.generation == 0 || slot.observer_returned)) {
                ++count;
            }
        }
        return count;
    }

    int AmfOwnedOutputSlotsLocked() const {
        int count = 0;
        for (const auto& slot : output_slots_) {
            if (slot.amf_owned) {
                ++count;
            }
        }
        return count;
    }

    int RetainedDecoderLeasesLocked() const {
        int count = 0;
        for (const auto& slot : bridge_slots_) {
            if (slot.decoder_lease) {
                ++count;
            }
        }
        return count;
    }

    void ReleaseAllDecoderLeasesLocked() {
        for (auto& slot : bridge_slots_) {
            if (slot.decoder_lease) {
                av_frame_free(&slot.decoder_lease);
                ++stats_.decoder_frame_leases_released;
            }
            slot.decoder_d3d_fence_value = 0;
        }
    }

    void DestroyPlaneLocked(JasnaAmfD3d11HipResidentPlane* plane) {
        if (!plane) {
            return;
        }
        if (plane->surface != 0) {
            if (hipDestroySurfaceObject(plane->surface) == hipSuccess) {
                ++stats_.hip_surface_objects_destroyed;
            } else {
                ++stats_.teardown_failures;
            }
            plane->surface = 0;
        }
        if (plane->mipmapped_array) {
            if (hipFreeMipmappedArray(plane->mipmapped_array) == hipSuccess) {
                ++stats_.mapped_arrays_destroyed;
            } else {
                ++stats_.teardown_failures;
            }
            plane->mipmapped_array = nullptr;
            plane->array = nullptr;
        }
        if (plane->external_memory) {
            if (hipDestroyExternalMemory(plane->external_memory) == hipSuccess) {
                ++stats_.external_memory_destroys;
            } else {
                ++stats_.teardown_failures;
            }
            plane->external_memory = nullptr;
        }
        plane->srv.Reset();
        plane->rtv.Reset();
        plane->texture.Reset();
    }

    void DestroyRootLocked() {
        if (prewarm_stream_) {
            if (hipStreamDestroy(prewarm_stream_) != hipSuccess) {
                ++stats_.teardown_failures;
            }
            prewarm_stream_ = nullptr;
        }
        for (auto& bridge : bridge_slots_) {
            const bool had_resources = bridge.y.texture || bridge.uv.texture;
            DestroyPlaneLocked(&bridge.uv);
            DestroyPlaneLocked(&bridge.y);
            if (had_resources) {
                ++stats_.bridge_slots_destroyed;
            }
        }
        if (fence_bridge_.semaphore) {
            if (hipDestroyExternalSemaphore(fence_bridge_.semaphore) == hipSuccess) {
                ++stats_.hip_external_semaphores_destroyed;
            } else {
                ++stats_.teardown_failures;
            }
            fence_bridge_.semaphore = nullptr;
        }
        if (fence_bridge_.d3d11_fence) {
            ++stats_.d3d11_opened_fences_destroyed;
        }
        fence_bridge_.d3d11_fence.Reset();
        if (fence_bridge_.d3d12_fence) {
            ++stats_.d3d12_fences_destroyed;
        }
        fence_bridge_.d3d12_fence.Reset();
        fence_bridge_.d3d12_device.Reset();
        for (auto& output : output_slots_) {
            if (output.texture) {
                ++stats_.output_slots_destroyed;
            }
            output.uv_rtv.Reset();
            output.y_rtv.Reset();
            output.texture.Reset();
            output.observer.session = nullptr;
            output.observer.slot_index = -1;
        }
        shader_copy_.rasterizer.Reset();
        shader_copy_.pixel_shader.Reset();
        shader_copy_.vertex_shader.Reset();
        av_buffer_unref(&encoder_frames_ref_);
        av_buffer_unref(&decoder_device_ref_);
        av_buffer_unref(&decoder_frames_ref_);
        canonical_frames_identity_ = 0;
        canonical_device_identity_ = 0;
        amf_context_ = nullptr;  // Borrowed from decoder; never Terminate/Release here.
        context4_.Reset();
        device5_.Reset();
        device3_.Reset();
        d3d_context_.Reset();
        d3d_device_.Reset();
        adapter_.Reset();
    }

    void FillBindInfoLocked(
        JasnaAmfD3d11HipResidentBindInfo* info,
        const JasnaAmfD3d11HipResidentFrameView* view
    ) const {
        if (!info) {
            return;
        }
        info->state = state_;
        info->hip_device = hip_device_;
        info->visible_width = static_cast<int>(visible_width_);
        info->visible_height = static_cast<int>(visible_height_);
        info->allocation_width = static_cast<int>(allocation_width_);
        info->allocation_height = static_cast<int>(allocation_height_);
        info->surface_format = AMF_SURFACE_NV12;
        info->sw_format = AV_PIX_FMT_NV12;
        info->adapter_luid_match = stats_.adapter_luid_match;
        info->dx11_device_match = d3d_device_ ? 1 : 0;
        info->hw_frames_identity = canonical_frames_identity_;
        info->hw_device_identity = canonical_device_identity_;
        info->amf_context_identity = reinterpret_cast<uintptr_t>(amf_context_);
        if (view && view->amf_device && view->amf_device->context != amf_context_) {
            info->dx11_device_match = 0;
        }
    }

    void FillCopyInfoLocked(
        JasnaAmfD3d11HipResidentCopyInfo* info,
        int slot_index,
        uint64_t d3d_value,
        uint64_t hip_value,
        hipError_t hip_result,
        AMF_RESULT amf_result,
        int d3d_result,
        int64_t pts
    ) const {
        if (!info) {
            return;
        }
        info->slot_index = slot_index;
        info->slot_count = static_cast<int>(JASNA_AMF_D3D11_HIP_RESIDENT_SLOTS);
        info->width = static_cast<int>(visible_width_);
        info->height = static_cast<int>(visible_height_);
        info->bytes_per_sample = kBytesPerSample;
        info->packed_size = PackedBytes();
        info->y_pitch = visible_width_;
        info->uv_pitch = visible_width_;
        info->d3d_to_hip_fence_value = d3d_value;
        info->hip_to_d3d_fence_value = hip_value;
        info->hip_result = static_cast<int>(hip_result);
        info->amf_result = static_cast<int>(amf_result);
        info->d3d_result = d3d_result;
        info->in_flight = AmfOwnedOutputSlotsLocked();
        info->pts = pts;
    }

    void UpdatePeakInFlightLocked() {
        const uint64_t in_flight = static_cast<uint64_t>(AmfOwnedOutputSlotsLocked());
        if (in_flight > stats_.peak_in_flight) {
            stats_.peak_in_flight = in_flight;
        }
    }

    int hip_device_ = -1;
    unsigned visible_width_ = 0;
    unsigned visible_height_ = 0;
    unsigned allocation_width_ = 0;
    unsigned allocation_height_ = 0;
    mutable std::mutex mutex_;
    JasnaAmfD3d11HipResidentState state_ = JASNA_RESIDENT_UNBOUND;
    std::string error_;
    uint64_t canonical_frames_identity_ = 0;
    uint64_t canonical_device_identity_ = 0;
    uint64_t fence_value_ = 0;
    uint64_t latest_hip_signal_value_ = 0;
    unsigned bridge_cursor_ = 0;
    unsigned output_cursor_ = 0;
    AVBufferRef* decoder_frames_ref_ = nullptr;
    AVBufferRef* decoder_device_ref_ = nullptr;
    AVBufferRef* encoder_frames_ref_ = nullptr;
    AMFContext* amf_context_ = nullptr;
    ComPtr<ID3D11Device> d3d_device_;
    ComPtr<ID3D11DeviceContext> d3d_context_;
    ComPtr<ID3D11Device3> device3_;
    ComPtr<ID3D11Device5> device5_;
    ComPtr<ID3D11DeviceContext4> context4_;
    ComPtr<IDXGIAdapter1> adapter_;
    std::array<JasnaAmfD3d11HipResidentBridgeSlot,
               JASNA_AMF_D3D11_HIP_RESIDENT_SLOTS> bridge_slots_{};
    std::array<JasnaAmfD3d11HipResidentOutputSlot,
               JASNA_AMF_D3D11_HIP_RESIDENT_SLOTS> output_slots_{};
    JasnaAmfD3d11HipResidentFenceBridge fence_bridge_{};
    JasnaAmfD3d11HipResidentShaderCopy shader_copy_{};
    hipStream_t prewarm_stream_ = nullptr;
    JasnaAmfD3d11HipResidentStats stats_{};
};

inline void AMF_STD_CALL JasnaAmfD3d11HipResidentObserver::OnSurfaceDataRelease(
    AMFSurface* surface
) {
    (void)surface;
    if (session) {
        session->OnSurfaceReleased(slot_index);
    }
}

inline void JasnaAmfD3d11HipResidentReleaseWrapper(void* opaque, uint8_t* data) {
    auto* owner = static_cast<JasnaAmfD3d11HipResidentWrapperOwner*>(opaque);
    auto* surface = reinterpret_cast<AMFSurface*>(data);
    // Keep the owner count nonzero until Release has returned: Release can run
    // the AMF observer synchronously, and Close must not free observer storage
    // during that callback.
    if (surface) {
        surface->Release();
    }
    if (owner && owner->session) {
        owner->session->OnWrapperBufferReleased(owner->slot_index, owner->generation);
    }
    delete owner;
}

static thread_local std::string jasna_amf_d3d11_hip_resident_global_error;

inline void JasnaAmfD3d11HipResidentSetError(
    JasnaAmfD3d11HipResidentSession* session,
    const char** error
) {
    if (!error) {
        return;
    }
    if (session && session->LastError()) {
        *error = session->LastError();
        return;
    }
    *error = jasna_amf_d3d11_hip_resident_global_error.empty()
        ? nullptr : jasna_amf_d3d11_hip_resident_global_error.c_str();
}

extern "C" {

inline int jasna_amf_d3d11_hip_resident_api_version() {
    return JASNA_AMF_D3D11_HIP_RESIDENT_API_VERSION;
}

inline JasnaAmfD3d11HipResidentSession* jasna_amf_d3d11_hip_resident_create(
    int hip_device,
    unsigned visible_width,
    unsigned visible_height,
    unsigned allocation_width,
    unsigned allocation_height,
    const char** error
) {
    jasna_amf_d3d11_hip_resident_global_error.clear();
    if (error) {
        *error = nullptr;
    }
    if (hip_device < 0 || visible_width == 0 || visible_height == 0 ||
        allocation_width < visible_width || allocation_height < visible_height ||
        (visible_width & 1U) != 0 || (visible_height & 1U) != 0 ||
        (allocation_width & 1U) != 0 || (allocation_height & 1U) != 0) {
        jasna_amf_d3d11_hip_resident_global_error =
            "resident NV12 session requires nonzero even fixed visible/allocation geometry";
        JasnaAmfD3d11HipResidentSetError(nullptr, error);
        return nullptr;
    }
    try {
        return new JasnaAmfD3d11HipResidentSession(
            hip_device, visible_width, visible_height, allocation_width, allocation_height
        );
    } catch (const std::bad_alloc&) {
        jasna_amf_d3d11_hip_resident_global_error = "allocating resident session failed";
    } catch (...) {
        jasna_amf_d3d11_hip_resident_global_error = "constructing resident session threw an unexpected exception";
    }
    JasnaAmfD3d11HipResidentSetError(nullptr, error);
    return nullptr;
}

inline int jasna_amf_d3d11_hip_resident_bind_decoder_frame(
    JasnaAmfD3d11HipResidentSession* session,
    void* frame,
    JasnaAmfD3d11HipResidentBindInfo* info,
    const char** error
) {
    jasna_amf_d3d11_hip_resident_global_error.clear();
    if (!session) {
        jasna_amf_d3d11_hip_resident_global_error = "resident session is null";
        JasnaAmfD3d11HipResidentSetError(nullptr, error);
        return -1;
    }
    try {
        const int status = session->BindDecoderFrame(static_cast<AVFrame*>(frame), info);
        JasnaAmfD3d11HipResidentSetError(session, error);
        return status;
    } catch (...) {
        jasna_amf_d3d11_hip_resident_global_error = "binding decoder frame raised an unexpected native exception";
        JasnaAmfD3d11HipResidentSetError(nullptr, error);
        return -1;
    }
}

inline int jasna_amf_d3d11_hip_resident_bind_encoder_context(
    JasnaAmfD3d11HipResidentSession* session,
    void* encoder,
    void* decoder,
    JasnaAmfD3d11HipResidentBindInfo* info,
    const char** error
) {
    jasna_amf_d3d11_hip_resident_global_error.clear();
    if (!session) {
        jasna_amf_d3d11_hip_resident_global_error = "resident session is null";
        JasnaAmfD3d11HipResidentSetError(nullptr, error);
        return -1;
    }
    try {
        const int status = session->BindEncoderContext(
            static_cast<AVCodecContext*>(encoder), static_cast<AVCodecContext*>(decoder), info
        );
        JasnaAmfD3d11HipResidentSetError(session, error);
        return status;
    } catch (...) {
        jasna_amf_d3d11_hip_resident_global_error = "binding encoder context raised an unexpected native exception";
        JasnaAmfD3d11HipResidentSetError(nullptr, error);
        return -1;
    }
}

inline int jasna_amf_d3d11_hip_resident_copy_decoded_to_hip(
    JasnaAmfD3d11HipResidentSession* session,
    void* frame,
    uintptr_t destination,
    uint64_t destination_size,
    uintptr_t consumer_stream,
    JasnaAmfD3d11HipResidentCopyInfo* info,
    const char** error
) {
    jasna_amf_d3d11_hip_resident_global_error.clear();
    if (!session) {
        jasna_amf_d3d11_hip_resident_global_error = "resident session is null";
        JasnaAmfD3d11HipResidentSetError(nullptr, error);
        return -1;
    }
    try {
        const int status = session->CopyDecodedToHip(
            static_cast<AVFrame*>(frame), destination, destination_size, consumer_stream, info
        );
        JasnaAmfD3d11HipResidentSetError(session, error);
        return status;
    } catch (...) {
        jasna_amf_d3d11_hip_resident_global_error = "copying decoded frame raised an unexpected native exception";
        JasnaAmfD3d11HipResidentSetError(nullptr, error);
        return -1;
    }
}

inline int jasna_amf_d3d11_hip_resident_acquire_encoder_frame(
    JasnaAmfD3d11HipResidentSession* session,
    void* output,
    uintptr_t source,
    uint64_t source_size,
    int64_t pts,
    uintptr_t producer_stream,
    JasnaAmfD3d11HipResidentCopyInfo* info,
    const char** error
) {
    jasna_amf_d3d11_hip_resident_global_error.clear();
    if (!session) {
        jasna_amf_d3d11_hip_resident_global_error = "resident session is null";
        JasnaAmfD3d11HipResidentSetError(nullptr, error);
        return -1;
    }
    try {
        const int status = session->AcquireEncoderFrame(
            static_cast<AVFrame*>(output), source, source_size, pts, producer_stream, info
        );
        JasnaAmfD3d11HipResidentSetError(session, error);
        return status;
    } catch (...) {
        jasna_amf_d3d11_hip_resident_global_error = "acquiring encoder frame raised an unexpected native exception";
        JasnaAmfD3d11HipResidentSetError(nullptr, error);
        return -1;
    }
}

inline int jasna_amf_d3d11_hip_resident_begin_drain(
    JasnaAmfD3d11HipResidentSession* session,
    const char** error
) {
    jasna_amf_d3d11_hip_resident_global_error.clear();
    if (!session) {
        jasna_amf_d3d11_hip_resident_global_error = "resident session is null";
        JasnaAmfD3d11HipResidentSetError(nullptr, error);
        return -1;
    }
    try {
        const int status = session->BeginDrain();
        JasnaAmfD3d11HipResidentSetError(session, error);
        return status;
    } catch (...) {
        jasna_amf_d3d11_hip_resident_global_error =
            "beginning resident drain raised an unexpected native exception";
        JasnaAmfD3d11HipResidentSetError(nullptr, error);
        return -1;
    }
}

inline int jasna_amf_d3d11_hip_resident_close(
    JasnaAmfD3d11HipResidentSession* session,
    int timeout_ms,
    const char** error
) {
    jasna_amf_d3d11_hip_resident_global_error.clear();
    if (!session) {
        jasna_amf_d3d11_hip_resident_global_error = "resident session is null";
        JasnaAmfD3d11HipResidentSetError(nullptr, error);
        return -1;
    }
    try {
        const int status = session->Close(timeout_ms);
        JasnaAmfD3d11HipResidentSetError(session, error);
        return status;
    } catch (...) {
        jasna_amf_d3d11_hip_resident_global_error =
            "closing resident session raised an unexpected native exception";
        JasnaAmfD3d11HipResidentSetError(nullptr, error);
        return -1;
    }
}

inline void jasna_amf_d3d11_hip_resident_get_stats(
    JasnaAmfD3d11HipResidentSession* session,
    JasnaAmfD3d11HipResidentStats* stats
) {
    if (session && stats) {
        session->GetStats(stats);
    }
}

inline const char* jasna_amf_d3d11_hip_resident_last_error(
    JasnaAmfD3d11HipResidentSession* session
) {
    return session ? session->LastError() : nullptr;
}

inline void jasna_amf_d3d11_hip_resident_destroy(
    JasnaAmfD3d11HipResidentSession* session
) {
    delete session;
}

}  // extern "C"
