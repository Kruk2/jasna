"""R1 zero-copy encode bridge for AMD (Windows): HIP tensor -> D3D11 texture -> AMF.

PyAV only implements the hardware-frame bridge for CUDA (``from_dlpack`` hard-codes
kCuda/kCPU), so on Windows/AMD there is no way to hand a device-resident YUV frame
to ``hevc_amf``/``h264_amf`` without a host roundtrip.  This module re-implements
PyAV's CUDA branch of ``from_dlpack`` (video/frame.py:1737-1741) for d3d11 via
ctypes against the libavutil DLLs PyAV already ships:

  torch (HIP) packed NV12/P010 tensor
    -> hipMemcpy2DToArray into shared D3D11 "carrier" texture(s) (<=2048 rows each,
       the driver-side map cap measured on RDNA3; extra rows go into a second
       carrier bound as a second SRV)
    -> compute shader unpacks the carrier into a real NV12/P010 texture
       (plane-slice UAV views are the only primitive that can reformat)
    -> av_hwdevice_ctx_create(D3D11VA) + frames ctx (sw_format=nv12/p010le)
    -> av_hwframe_get_buffer, then AVFrame.data[0] is pointed at our texture
    -> the resulting av.VideoFrame carries hw_frames_ctx and is adopted by the
       AMF encoder exactly like an NVENC CUDA frame is on NVIDIA.

Measured on RX 7900 XT (report: E:/work2/Jasna-R1零拷贝编码桥实测-20261009.md):
encode leg 1.23x wall, 4.5x less CPU; decoded output PSNR-identical to the
host-frame path (43.8/53.0 dB).
"""
from __future__ import annotations

import ctypes
import glob
import os
from ctypes import (
    POINTER, Structure, Union, WINFUNCTYPE, WinDLL, byref, c_char_p, c_int,
    c_long, c_size_t, c_ubyte, c_uint, c_ulong, c_ulonglong, c_ushort, c_void_p, c_wchar_p,
)

import av  # noqa: F401  ensures the bundled av.libs DLLs are resident

__all__ = ["D3D11Bridge", "BridgeUnavailable"]


class BridgeUnavailable(RuntimeError):
    """Raised when the D3D11/HIP bridge cannot be set up on this machine."""


# ---------------------------------------------------------------------------
# COM / D3D11 plumbing (vtable indices for the Windows SDK d3d11.h layout)
# ---------------------------------------------------------------------------

HRESULT = c_long
S_OK = 0


class GUID(Structure):
    _fields_ = [("Data1", c_uint), ("Data2", c_ushort), ("Data3", c_ushort), ("Data4", c_ubyte * 8)]

    def __init__(self, s: str):
        super().__init__()
        p = s.strip("{}").split("-")
        self.Data1 = int(p[0], 16)
        self.Data2 = int(p[1], 16)
        self.Data3 = int(p[2], 16)
        b = bytes.fromhex(p[3] + p[4])
        for i in range(8):
            self.Data4[i] = b[i]


IID_IDXGIResource1 = GUID("30961379-4609-4a41-998e-54fe567ee0c1")

D3D_DRIVER_TYPE_HARDWARE = 1
D3D11_SDK_VERSION = 7
D3D11_CREATE_DEVICE_BGRA_SUPPORT = 0x20
D3D11_USAGE_DEFAULT = 0
D3D11_BIND_SHADER_RESOURCE = 0x8
D3D11_BIND_RENDER_TARGET = 0x20
D3D11_BIND_UNORDERED_ACCESS = 0x80
D3D11_RESOURCE_MISC_SHARED = 0x2
D3D11_RESOURCE_MISC_SHARED_NTHANDLE = 0x800
DXGI_SHARED_RESOURCE_READ = 0x80000000
DXGI_SHARED_RESOURCE_WRITE = 0x1

DXGI_FORMAT_R8_UNORM = 61
DXGI_FORMAT_R16_UINT = 57
DXGI_FORMAT_R32_UINT = 42
DXGI_FORMAT_R8G8_UNORM = 49
DXGI_FORMAT_NV12 = 103
DXGI_FORMAT_P010 = 104

# ID3D11Device (IUnknown -> +2)
DEV_CREATE_TEXTURE2D = 5
DEV_CREATE_UAV = 8
DEV_CREATE_COMPUTE_SHADER = 18
# ID3D11DeviceContext (ID3D11DeviceChild -> +6)
CTX_DISPATCH = 41
CTX_CS_SET_SRV = 67
CTX_CS_SET_UAV = 68
CTX_CS_SET_SHADER = 69

D3D11_SRV_DIMENSION_TEXTURE2D = 4
D3D11_UAV_DIMENSION_TEXTURE2DARRAY = 5

D2D = 3  # hipMemcpyDeviceToDevice
H2D = 1  # hipMemcpyHostToDevice

# Measured driver-side cap on hipExternalMemoryGetMappedMipmappedArray map
# extents (rows), independent of format.  Carriers are sliced to respect it.
_HIP_MAX_MAP_ROWS = 2048


def _vtbl(ptr):
    return ctypes.cast(ptr, POINTER(POINTER(c_void_p)))[0]


def method(ptr, idx, restype, *argtypes):
    fn = _vtbl(ptr)[idx]
    return WINFUNCTYPE(restype, c_void_p, *argtypes)(fn)


def release(ptr) -> None:
    if ptr:
        method(ptr, 2, c_ulong)(ptr)


class D3D11_SAMPLE_DESC(Structure):
    _fields_ = [("Count", c_uint), ("Quality", c_uint)]


class D3D11_TEXTURE2D_DESC(Structure):
    _fields_ = [
        ("Width", c_uint), ("Height", c_uint), ("MipLevels", c_uint), ("ArraySize", c_uint),
        ("Format", c_uint), ("SampleDesc", D3D11_SAMPLE_DESC),
        ("Usage", c_uint), ("BindFlags", c_uint), ("CPUAccessFlags", c_uint), ("MiscFlags", c_uint),
    ]


class D3D11_TEX2D_SRV(Structure):
    _fields_ = [("MostDetailedMip", c_uint), ("MipLevels", c_uint)]


class _SRVUnion(Union):
    _fields_ = [("Texture2D", D3D11_TEX2D_SRV), ("_pad", c_uint * 8)]


class D3D11_SHADER_RESOURCE_VIEW_DESC(Structure):
    _fields_ = [("Format", c_uint), ("ViewDimension", c_uint), ("u", _SRVUnion)]


class D3D11_TEX2D_ARRAY_UAV(Structure):
    _fields_ = [("MipSlice", c_uint), ("FirstArraySlice", c_uint), ("ArraySize", c_uint), ("PlaneSlice", c_uint)]


class _UAVUnion(Union):
    _fields_ = [("Texture2DArray", D3D11_TEX2D_ARRAY_UAV), ("_pad", c_uint * 8)]


class D3D11_UNORDERED_ACCESS_VIEW_DESC(Structure):
    _fields_ = [("Format", c_uint), ("ViewDimension", c_uint), ("u", _UAVUnion)]


# ---------------------------------------------------------------------------
# HIP runtime (the amdhip64 DLL shipped inside the torch-ROCm wheel)
# ---------------------------------------------------------------------------

_HIP_BIN_ENV = "JASNA_HIP_BIN"


class hipChannelFormatDesc(Structure):
    _fields_ = [("x", c_int), ("y", c_int), ("z", c_int), ("w", c_int), ("f", c_int)]


class hipExtent(Structure):
    _fields_ = [("width", c_size_t), ("height", c_size_t), ("depth", c_size_t)]


class _Win32Handle(Structure):
    _fields_ = [("handle", c_void_p), ("name", c_void_p)]


class _HandleUnion(Union):
    _fields_ = [("fd", c_int), ("win32", _Win32Handle), ("nvSciBufObject", c_void_p)]


class hipExternalMemoryHandleDesc(Structure):
    _fields_ = [("type", c_int), ("handle", _HandleUnion), ("size", c_ulonglong),
                ("flags", c_uint), ("reserved", c_uint * 16)]


class hipExtMipDesc(Structure):
    _fields_ = [("offset", c_ulonglong), ("formatDesc", hipChannelFormatDesc),
                ("extent", hipExtent), ("flags", c_uint), ("numLevels", c_uint)]


def _hip_bin() -> str:
    from_env = os.environ.get(_HIP_BIN_ENV)
    if from_env and os.path.isdir(from_env):
        return from_env
    import torch

    base = os.path.dirname(os.path.dirname(os.path.abspath(torch.__file__)))
    cand = os.path.join(base, "_rocm_sdk_core", "bin")
    if os.path.isdir(cand):
        return cand
    raise BridgeUnavailable(f"amdhip64_7.dll directory not found (set {_HIP_BIN_ENV})")


def _load_hip():
    hip_bin = _hip_bin()
    os.add_dll_directory(hip_bin)
    hip = WinDLL(os.path.join(hip_bin, "amdhip64_7.dll"))
    hip.hipInit.restype = c_int
    hip.hipInit.argtypes = [c_uint]
    hip.hipSetDevice.restype = c_int
    hip.hipSetDevice.argtypes = [c_int]
    hip.hipDeviceSynchronize.restype = c_int
    hip.hipDeviceSynchronize.argtypes = []
    hip.hipImportExternalMemory.restype = c_int
    hip.hipImportExternalMemory.argtypes = [POINTER(c_void_p), POINTER(hipExternalMemoryHandleDesc)]
    hip.hipDestroyExternalMemory.restype = c_int
    hip.hipDestroyExternalMemory.argtypes = [c_void_p]
    hip.hipExternalMemoryGetMappedMipmappedArray.restype = c_int
    hip.hipExternalMemoryGetMappedMipmappedArray.argtypes = [POINTER(c_void_p), c_void_p, POINTER(hipExtMipDesc)]
    hip.hipGetMipmappedArrayLevel.restype = c_int
    hip.hipGetMipmappedArrayLevel.argtypes = [POINTER(c_void_p), c_void_p, c_uint]
    hip.hipMemcpy2DToArray.restype = c_int
    hip.hipMemcpy2DToArray.argtypes = [c_void_p, c_size_t, c_size_t, c_void_p,
                                       c_size_t, c_size_t, c_size_t, c_int]
    return hip


# ---------------------------------------------------------------------------
# ctypes bridge to libavutil (same DLLs PyAV ships)
# ---------------------------------------------------------------------------

def _find_avlib(prefix: str) -> str:
    libs = os.path.join(os.path.dirname(os.path.dirname(av.__file__)), "av.libs")
    if os.path.isdir(libs):
        os.add_dll_directory(libs)
        hits = glob.glob(os.path.join(libs, prefix + "-*.dll")) or \
               glob.glob(os.path.join(libs, prefix + ".dll"))
        if hits:
            return hits[0]
    raise BridgeUnavailable(f"{prefix}*.dll not found next to PyAV")


# AVBufferRef = { AVBuffer *buffer; uint8_t *data; size_t size; }
OFF_AVBUFFER_DATA = 8
# AVHWDeviceContext = { const AVClass *av_class; enum AVHWDeviceType type; void *hwctx; }
OFF_HWDEVICE_HWCTX = 16
OFF_D3D11VA_DEVICE = 0        # AVD3D11VADeviceContext.device
OFF_D3D11VA_DEVICE_CTX = 8    # AVD3D11VADeviceContext.device_context
# AVHWFramesContext measured offsets (ffmpeg 7.1, verified on this machine):
OFF_HWFRAMES_FORMAT = 60
OFF_HWFRAMES_SW_FORMAT = 64
OFF_HWFRAMES_WIDTH = 68
OFF_HWFRAMES_HEIGHT = 72
# AVFrame measured offsets:
OFF_AVFRAME_FORMAT = 116
# PyAV VideoFrame C object: ptr field at +24 (vtab at +16)
OFF_VIDEOFRAME_PTR = 24

HWDEVICE_D3D11VA = 7  # AV_HWDEVICE_TYPE_D3D11VA (ffmpeg 7.1)


def _rptr(addr: int) -> int:
    return ctypes.cast(c_void_p(addr), POINTER(c_ulonglong))[0]


def _ri32(addr: int) -> int:
    return ctypes.cast(c_void_p(addr), POINTER(c_int))[0]


def _wptr(addr: int, val: int) -> None:
    ctypes.cast(c_void_p(addr), POINTER(c_ulonglong))[0] = val


def _wi32(addr: int, val: int) -> None:
    ctypes.cast(c_void_p(addr), POINTER(c_int))[0] = val


# ---------------------------------------------------------------------------
# Compute shaders: unpack the single-channel carrier into real NV12/P010
# ---------------------------------------------------------------------------
# Carrier is a plain single-channel texture HIP can write: R8_UNORM for NV12
# (one byte per texel), R32_UINT for P010 (one u32 texel = two little-endian
# u16 samples).  Plane-slice UAV views are the only D3D11 primitive that can
# reformat planes.  The shader guards its bounds because Jasna's frame heights
# (e.g. 1080) are not multiples of the 8x8 thread block.

_NV12_HLSL = """
Texture2D<float>   srcA : register(t0);   // carrier slice 0 (rows 0..SPLIT-1)
Texture2D<float>   srcB : register(t1);   // carrier slice 1 (never selected when single)
RWTexture2D<float>  dstY  : register(u0);
RWTexture2D<float2> dstUV : register(u1);
[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint x = tid.x, y = tid.y;
    if (x >= W2CONST || y >= H2CONST) return;
    uint sx = x * 2u, sy = y * 2u;
    bool bA = (sy < SPLITCONST);
    uint ly = bA ? sy : sy - SPLITCONST;
    dstY[uint2(sx,     sy)]     = bA ? srcA.Load(int3(sx,     ly,     0)) : srcB.Load(int3(sx,     ly,     0));
    dstY[uint2(sx + 1u, sy)]     = bA ? srcA.Load(int3(sx + 1u, ly,     0)) : srcB.Load(int3(sx + 1u, ly,     0));
    dstY[uint2(sx,     sy + 1u)] = bA ? srcA.Load(int3(sx,     ly + 1u, 0)) : srcB.Load(int3(sx,     ly + 1u, 0));
    dstY[uint2(sx + 1u, sy + 1u)] = bA ? srcA.Load(int3(sx + 1u, ly + 1u, 0)) : srcB.Load(int3(sx + 1u, ly + 1u, 0));
    uint uy = HCONST + y;
    bool uA = (uy < SPLITCONST);
    uint luy = uA ? uy : uy - SPLITCONST;
    float2 uv;
    uv.x = uA ? srcA.Load(int3(sx,     luy, 0)) : srcB.Load(int3(sx,     luy, 0));
    uv.y = uA ? srcA.Load(int3(sx + 1u, luy, 0)) : srcB.Load(int3(sx + 1u, luy, 0));
    dstUV[uint2(x, y)] = uv;
}
"""

_P010_HLSL = """
Texture2D<uint>   srcA : register(t0);   // carrier slice 0 (rows 0..SPLIT-1)
Texture2D<uint>   srcB : register(t1);   // carrier slice 1 (never selected when single)
RWTexture2D<uint> dstY  : register(u0);
RWTexture2D<uint> dstUV : register(u1);
[numthreads(8,8,1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    uint x = tid.x, y = tid.y;
    if (x >= W2CONST || y >= H2CONST) return;
    uint sx = x * 2u, sy = y * 2u;
    bool bA = (sy < SPLITCONST);
    uint ly = bA ? sy : sy - SPLITCONST;
    uint a = bA ? srcA.Load(int3(x, ly, 0)) : srcB.Load(int3(x, ly, 0));
    uint b = bA ? srcA.Load(int3(x, ly + 1u, 0)) : srcB.Load(int3(x, ly + 1u, 0));
    uint uy = HCONST + y;
    bool uA = (uy < SPLITCONST);
    uint luy = uA ? uy : uy - SPLITCONST;
    uint uv = uA ? srcA.Load(int3(x, luy, 0)) : srcB.Load(int3(x, luy, 0));
    dstY[uint2(sx,     sy)]      = a & 0xFFFFu;
    dstY[uint2(sx + 1u, sy)]     = a >> 16u;
    dstY[uint2(sx,     sy + 1u)]      = b & 0xFFFFu;
    dstY[uint2(sx + 1u, sy + 1u)]     = b >> 16u;
    dstUV[uint2(x, y)] = uv;
}
"""


def _compile_shader(height: int, split_rows: int, ten_bit: bool, w2: int, h2: int):
    src = (_P010_HLSL if ten_bit else _NV12_HLSL) \
        .replace("HCONST", str(height)).replace("SPLITCONST", str(split_rows)) \
        .replace("W2CONST", str(w2)).replace("H2CONST", str(h2)).encode()
    d3dc = WinDLL("d3dcompiler_47.dll")
    d3dc.D3DCompile.restype = HRESULT
    d3dc.D3DCompile.argtypes = [c_char_p, c_size_t, c_char_p, c_void_p, c_void_p,
                                c_char_p, c_char_p, c_uint, c_uint,
                                POINTER(c_void_p), POINTER(c_void_p)]
    blob = c_void_p()
    err = c_void_p()
    hr = d3dc.D3DCompile(src, len(src), b"cs", None, None, b"main", b"cs_5_0",
                         0x8000, 0, byref(blob), byref(err))
    if hr != S_OK:
        msg = ""
        if err.value:
            size = method(err, 4, c_ulong)(err)
            buf = method(err, 3, c_void_p)(err)
            msg = ctypes.string_at(buf, size).decode(errors="replace")
        raise BridgeUnavailable(f"D3DCompile hr=0x{hr & 0xFFFFFFFF:08x} {msg}")
    size = method(blob, 4, c_ulong)(blob)
    ptr = method(blob, 3, c_void_p)(blob)
    return ptr, size, blob


def _make_tex(dev, w: int, h: int, fmt: int, bind: int, misc: int):
    t = D3D11_TEXTURE2D_DESC()
    t.Width, t.Height = w, h
    t.MipLevels, t.ArraySize = 1, 1
    t.Format = fmt
    t.SampleDesc.Count, t.SampleDesc.Quality = 1, 0
    t.Usage = D3D11_USAGE_DEFAULT
    t.BindFlags = bind
    t.CPUAccessFlags = 0
    t.MiscFlags = misc
    tex = c_void_p()
    f = method(dev, DEV_CREATE_TEXTURE2D, HRESULT,
               POINTER(D3D11_TEXTURE2D_DESC), c_void_p, POINTER(c_void_p))
    hr = f(dev, byref(t), None, byref(tex))
    return hr, tex


def _share_nt(tex):
    res1 = c_void_p()
    method(tex, 0, HRESULT, POINTER(GUID), POINTER(c_void_p))(
        tex, byref(IID_IDXGIResource1), byref(res1))
    if not res1.value:
        return 0x80004005, None
    h = c_void_p()
    hr = method(res1, 13, HRESULT, c_void_p, c_uint, c_wchar_p, POINTER(c_void_p))(
        res1, None, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, None, byref(h))
    release(res1)
    return hr, h


class _Carrier:
    """One shared single-channel texture HIP writes and the shader reads."""

    def __init__(self, hip, dev, width: int, rows: int, elem_bytes: int, fmt: int):
        self.hip = hip
        self.rows = rows
        self.row_bytes = width * elem_bytes
        hr, self.tex = _make_tex(dev, width, rows, fmt, D3D11_BIND_SHADER_RESOURCE,
                                 D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE)
        if hr != S_OK:
            raise BridgeUnavailable(f"carrier CreateTexture2D hr=0x{hr & 0xFFFFFFFF:08x}")
        hr, self.handle = _share_nt(self.tex)
        if hr != S_OK:
            self.close()
            raise BridgeUnavailable(f"carrier CreateSharedHandle hr=0x{hr & 0xFFFFFFFF:08x}")
        d = hipExternalMemoryHandleDesc()
        d.type = 6  # hipExternalMemoryHandleTypeD3D11Resource
        d.handle.win32.handle = ctypes.cast(self.handle, c_void_p)
        d.size = self.row_bytes * rows
        self.ext = c_void_p()
        rc = hip.hipImportExternalMemory(byref(self.ext), byref(d))
        if rc != 0:
            self.close()
            raise BridgeUnavailable(f"hipImportExternalMemory rc={rc}")
        md = hipExtMipDesc()
        md.offset = 0
        md.formatDesc.x, md.formatDesc.y, md.formatDesc.z, md.formatDesc.w, md.formatDesc.f = \
            elem_bytes * 8, 0, 0, 0, 1
        md.extent.width, md.extent.height, md.extent.depth = width, rows, 1
        md.flags, md.numLevels = 0, 1
        self.mip = c_void_p()
        rc = hip.hipExternalMemoryGetMappedMipmappedArray(byref(self.mip), self.ext, byref(md))
        if rc != 0:
            self.close()
            raise BridgeUnavailable(f"map {width}x{rows} rc={rc}")
        self.array = c_void_p()
        rc = hip.hipGetMipmappedArrayLevel(byref(self.array), self.mip, 0)
        if rc != 0:
            self.close()
            raise BridgeUnavailable(f"hipGetMipmappedArrayLevel rc={rc}")
        sd = D3D11_SHADER_RESOURCE_VIEW_DESC()
        sd.Format, sd.ViewDimension = fmt, D3D11_SRV_DIMENSION_TEXTURE2D
        sd.u.Texture2D.MostDetailedMip, sd.u.Texture2D.MipLevels = 0, 1
        self.srv = c_void_p()
        hr = method(dev, 7, HRESULT, c_void_p,
                    POINTER(D3D11_SHADER_RESOURCE_VIEW_DESC), POINTER(c_void_p))(
            dev, self.tex, byref(sd), byref(self.srv))
        if hr != S_OK:
            self.close()
            raise BridgeUnavailable(f"carrier SRV hr=0x{hr & 0xFFFFFFFF:08x}")

    def upload(self, dev_ptr: int, kind: int = D2D) -> None:
        rc = self.hip.hipMemcpy2DToArray(
            self.array, 0, 0, c_void_p(dev_ptr), self.row_bytes,
            self.row_bytes, self.rows, kind)
        if rc != 0:
            raise BridgeUnavailable(f"hipMemcpy2DToArray rc={rc}")

    def close(self) -> None:
        hip = getattr(self, "hip", None)
        if hip is not None:
            ext = getattr(self, "ext", None)
            if ext is not None and ext.value:
                hip.hipDestroyExternalMemory(ext)
            self.hip = None
        for attr in ("array", "mip", "handle", "tex", "ext"):
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            if attr == "handle" and obj.value:
                try:
                    WinDLL("kernel32").CloseHandle(obj)
                except Exception:
                    pass
            elif attr == "tex" and obj.value:
                release(obj)
            setattr(self, attr, None)


class D3D11Bridge:
    """Encode-side zero-copy bridge: HIP YUV tensor -> D3D11 texture -> AMF frame.

    One instance per encoder geometry/format.  ``upload_and_wrap`` returns an
    ``av.VideoFrame`` whose data[0] aliases the just-uploaded D3D11 texture.
    Frames come from a ring of independent slots (own texture + own hw frame
    each): the consumer may hold up to ``RING`` returned frames alive before
    the ring wraps and reuses the oldest slot's texture.
    """

    # Ring depth.  Jasna's encoder queue holds ENCODE_BUFFER_SIZE in-flight
    # frames; any returned frame still referenced when its slot comes around
    # again would get its texture overwritten.
    RING = 32  # >= video_encoder.ENCODE_BUFFER_SIZE (8) + encoder lookahead headroom

    def __init__(self, device_index: int, width: int, height: int, frame_format: str):
        if os.name != "nt":
            raise BridgeUnavailable("d3d11 bridge is Windows-only")
        if frame_format not in ("nv12", "p010le"):
            raise BridgeUnavailable(f"unsupported frame format {frame_format!r}")
        if width % 2 or height % 2:
            raise BridgeUnavailable(f"4:2:0 requires even dimensions, got {width}x{height}")
        self.width = width
        self.height = height
        self.ten_bit = frame_format == "p010le"
        self.frame_format = frame_format
        self._closed = False

        self.hip = _load_hip()
        rc = self.hip.hipInit(0)
        if rc != 0:
            raise BridgeUnavailable(f"hipInit rc={rc}")
        rc = self.hip.hipSetDevice(device_index)
        if rc != 0:
            raise BridgeUnavailable(f"hipSetDevice({device_index}) rc={rc}")

        # libavutil D3D11VA device: the textures MUST live on the same device
        # the frames context uses, so we take ffmpeg's device, not our own.
        self._avutil = WinDLL(_find_avlib("avutil"))
        self._avutil.av_get_pix_fmt.restype = c_int
        self._avutil.av_get_pix_fmt.argtypes = [c_char_p]
        self._avutil.av_hwdevice_ctx_create.restype = c_int
        self._avutil.av_hwdevice_ctx_create.argtypes = [POINTER(c_void_p), c_int, c_char_p, c_void_p, c_int]
        self._avutil.av_hwframe_ctx_alloc.restype = c_void_p
        self._avutil.av_hwframe_ctx_alloc.argtypes = [c_void_p]
        self._avutil.av_hwframe_ctx_init.restype = c_int
        self._avutil.av_hwframe_ctx_init.argtypes = [c_void_p]
        self._avutil.av_hwframe_get_buffer.restype = c_int
        self._avutil.av_hwframe_get_buffer.argtypes = [c_void_p, c_void_p, c_int]
        self._avutil.av_buffer_unref.restype = None
        self._avutil.av_buffer_unref.argtypes = [POINTER(c_void_p)]

        self.device_ref = c_void_p()
        rc = self._avutil.av_hwdevice_ctx_create(byref(self.device_ref), HWDEVICE_D3D11VA, None, None, 0)
        if rc < 0:
            raise BridgeUnavailable(f"av_hwdevice_ctx_create(D3D11VA) rc={rc}")
        dctx = _rptr(self.device_ref.value + OFF_AVBUFFER_DATA)
        hwctx = _rptr(dctx + OFF_HWDEVICE_HWCTX)
        self._d3d_dev = c_void_p(_rptr(hwctx + OFF_D3D11VA_DEVICE))
        self._d3d_ctx = c_void_p(_rptr(hwctx + OFF_D3D11VA_DEVICE_CTX))

        pix_d3d11 = self._avutil.av_get_pix_fmt(b"d3d11")
        pix_sw = self._avutil.av_get_pix_fmt(frame_format.encode())
        self.frames_ref = c_void_p(self._avutil.av_hwframe_ctx_alloc(self.device_ref))
        fctx = _rptr(self.frames_ref.value + OFF_AVBUFFER_DATA)
        _wi32(fctx + OFF_HWFRAMES_FORMAT, pix_d3d11)
        _wi32(fctx + OFF_HWFRAMES_SW_FORMAT, pix_sw)
        _wi32(fctx + OFF_HWFRAMES_WIDTH, width)
        _wi32(fctx + OFF_HWFRAMES_HEIGHT, height)
        rc = self._avutil.av_hwframe_ctx_init(self.frames_ref)
        if rc < 0:
            self.close()
            raise BridgeUnavailable(f"av_hwframe_ctx_init rc={rc}")

        # Ring of dst slots: the real NV12/P010 surfaces AMF consumes, each
        # wrapped by its own hw frame from av_hwframe_get_buffer (patched once
        # at init so every slot is an independent, long-lived VideoFrame).
        # No MISC_SHARED: ffmpeg/AMF take the COM pointer directly.
        dst_fmt = DXGI_FORMAT_P010 if self.ten_bit else DXGI_FORMAT_NV12
        plane_fmts = ((DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R32_UINT) if self.ten_bit
                      else (DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM))
        self._slots = []
        self._slot_fmts = []
        for _ in range(self.RING):
            hr, tex = _make_tex(
                self._d3d_dev, width, height, dst_fmt,
                D3D11_BIND_UNORDERED_ACCESS | D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
                0)
            if hr != S_OK:
                self.close()
                raise BridgeUnavailable(f"dst {dst_fmt} CreateTexture2D hr=0x{hr & 0xFFFFFFFF:08x}")
            uavs = []
            for plane, fmt in enumerate(plane_fmts):
                ud = D3D11_UNORDERED_ACCESS_VIEW_DESC()
                ud.Format, ud.ViewDimension = fmt, D3D11_UAV_DIMENSION_TEXTURE2DARRAY
                ud.u.Texture2DArray.MipSlice, ud.u.Texture2DArray.FirstArraySlice = 0, 0
                ud.u.Texture2DArray.ArraySize, ud.u.Texture2DArray.PlaneSlice = 1, plane
                u = c_void_p()
                hr = method(self._d3d_dev, 8, HRESULT, c_void_p,
                            POINTER(D3D11_UNORDERED_ACCESS_VIEW_DESC), POINTER(c_void_p))(
                    self._d3d_dev, tex, byref(ud), byref(u))
                if hr != S_OK or not u.value:
                    release(tex)
                    self.close()
                    raise BridgeUnavailable(f"dst plane{plane} UAV hr=0x{hr & 0xFFFFFFFF:08x}")
                uavs.append(u)
            vf = self._alloc_hw_video_frame(tex)
            self._slots.append({"tex": tex, "uavs": uavs, "vf": vf,
                                "fptr": _rptr(id(vf) + OFF_VIDEOFRAME_PTR)})
            self._slot_fmts.append(pix_d3d11)

        # carrier(s): single-channel, HIP-writable, sliced at the 2048-row cap
        if self.ten_bit:
            carrier_fmt, cw, elem = DXGI_FORMAT_R32_UINT, width // 2, 4
        else:
            carrier_fmt, cw, elem = DXGI_FORMAT_R8_UNORM, width, 1
        ch = height * 3 // 2
        self._split = min(ch, _HIP_MAX_MAP_ROWS)
        slices = [(0, self._split)] if ch <= self._split else \
                 [(0, self._split), (self._split, ch)]
        self._carriers = [
            _Carrier(self.hip, self._d3d_dev, cw, hi - lo, elem, carrier_fmt)
            for lo, hi in slices
        ]

        sptr, ssize, self._blob = _compile_shader(
            height, self._split, self.ten_bit, width // 2, height // 2)
        self._csh = c_void_p()
        hr = method(self._d3d_dev, DEV_CREATE_COMPUTE_SHADER, HRESULT,
                    c_void_p, c_size_t, c_void_p, POINTER(c_void_p))(
            self._d3d_dev, sptr, ssize, None, byref(self._csh))
        if hr != S_OK:
            self.close()
            raise BridgeUnavailable(f"CreateComputeShader hr=0x{hr & 0xFFFFFFFF:08x}")

        self._slot = 0
        self._active = None
        self._fmt_d3d11 = pix_d3d11

    def _alloc_hw_video_frame(self, dst_tex) -> av.VideoFrame:
        # Emulate PyAV's CUDA branch of from_dlpack (video/frame.py:1737-1741):
        # hand the VideoFrame's AVFrame to our d3d11 frames context, then point
        # data[0] at our texture.
        vf = av.VideoFrame(self.width, self.height, self.frame_format)
        fptr = _rptr(id(vf) + OFF_VIDEOFRAME_PTR)
        rc = self._avutil.av_hwframe_get_buffer(self.frames_ref, c_void_p(fptr), 0)
        if rc < 0:
            raise BridgeUnavailable(f"av_hwframe_get_buffer rc={rc}")
        _wptr(fptr, dst_tex.value)        # AVFrame.data[0] -> our texture
        _wptr(fptr + 8, 0)                # data[1] = NULL
        return vf

    def _dispatch(self, slot: dict) -> None:
        ctx = self._d3d_ctx
        srvs = [c.srv for c in self._carriers]
        uavs = slot["uavs"]
        method(ctx, CTX_CS_SET_SHADER, None, c_void_p, c_void_p, c_uint)(ctx, self._csh, None, 0)
        method(ctx, CTX_CS_SET_SRV, None, c_uint, c_uint, c_void_p)(
            ctx, 0, len(srvs), ctypes.cast((c_void_p * len(srvs))(*srvs), c_void_p))
        method(ctx, CTX_CS_SET_UAV, None, c_uint, c_uint, c_void_p, c_void_p)(
            ctx, 0, 2, ctypes.cast((c_void_p * 2)(uavs[0], uavs[1]), c_void_p), None)
        method(ctx, CTX_DISPATCH, None, c_uint, c_uint, c_uint)(
            ctx, (self.width // 2 + 7) // 8, (self.height // 2 + 7) // 8, 1)
        method(ctx, CTX_CS_SET_UAV, None, c_uint, c_uint, c_void_p, c_void_p)(
            ctx, 0, 2, ctypes.cast((c_void_p * 2)(None, None), c_void_p), None)
        method(ctx, CTX_CS_SET_SHADER, None, c_void_p, c_void_p, c_uint)(ctx, None, None, 0)

    def upload_and_wrap(self, packed) -> av.VideoFrame:
        """Upload a packed (H*3/2, W) HIP tensor and return an encode-ready frame.

        ``packed`` must be contiguous, uint8 for nv12 / uint16 for p010le, laid
        out as Y rows followed by interleaved UV rows (Jasna's host YUV layout).
        The returned frame aliases a ring slot's D3D11 texture: keep at most
        ``RING`` returned frames alive before the slot is overwritten.
        """
        import torch

        if self._closed:
            raise RuntimeError("bridge is closed")
        if not packed.is_contiguous():
            packed = packed.contiguous()
        rows = self.height * 3 // 2
        host = not packed.is_cuda
        if self.ten_bit:
            if packed.dtype is not torch.uint16:
                raise ValueError(f"p010 bridge needs uint16 tensor, got {packed.dtype}")
            # one carrier texel = one u32 = two little-endian u16 samples, so the
            # (rows, W) u16 stream reinterprets as (rows, W/2) u32 rows.
            u32 = packed.view(torch.uint32)  # (rows, W/2)
            dev0 = u32.data_ptr()
            row_bytes = (self.width // 2) * 4
        else:
            if packed.dtype is not torch.uint8:
                raise ValueError(f"nv12 bridge needs uint8 tensor, got {packed.dtype}")
            dev0 = packed.data_ptr()
            row_bytes = self.width

        kind = H2D if host else D2D
        slot = self._slots[self._slot]
        self._carriers[0].upload(dev0, kind)
        if len(self._carriers) > 1:
            self._carriers[1].upload(dev0 + self._split * row_bytes, kind)
        # Required, not redundant: the blocking copies order HIP against the host,
        # but the *next* frame rewrites the same carrier while the previous frame's
        # D3D11 dispatch may still be reading it.  Measured 2026-10-10 on the
        # 1080p mosaic clip: with this barrier removed, one run in two corrupted a
        # single frame (29 dB vs a 47 dB run-to-run floor, and it did not reproduce
        # in the repeat), at a wall-clock saving of ~4-7 s per 1800 frames.
        self.hip.hipDeviceSynchronize()
        self._dispatch(slot)
        self._slot = (self._slot + 1) % len(self._slots)
        self._active = slot
        return slot["vf"]

    def wrap_frame(self, pts=None, time_base=None) -> av.VideoFrame:
        """Stamp the frame returned by the most recent upload_and_wrap."""
        slot = self._active
        _wi32(slot["fptr"] + OFF_AVFRAME_FORMAT, self._fmt_d3d11)
        vf = slot["vf"]
        if pts is not None:
            vf.pts = pts
        if time_base is not None:
            vf.time_base = time_base
        return vf

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for slot in getattr(self, "_slots", []) or []:
            for u in slot["uavs"]:
                release(u)
            if getattr(slot["tex"], "value", 0):
                release(slot["tex"])
        self._slots = []
        self._active = None
        for c in getattr(self, "_carriers", []) or []:
            c.close()
        self._carriers = []
        for attr in ("_csh", "frames_ref", "device_ref"):
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            if attr == "_uavs":
                for u in obj:
                    release(u)
            elif attr in ("frames_ref", "device_ref") and obj.value:
                self._avutil.av_buffer_unref(byref(obj))
            elif getattr(obj, "value", 0):
                release(obj)
            setattr(self, attr, None)
        self._blob = None
