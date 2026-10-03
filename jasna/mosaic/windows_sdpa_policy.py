"""Narrow, recorded Windows RF-DETR compatibility policy; no import-time changes.

Math SDPA is still GPU attention. Precision is selected independently by the
runner. This is a process default, not an override of explicit sdpa_kernel
contexts (in particular, it does not certify the LTX restoration path).
"""
from __future__ import annotations

import json
import logging
import os
import sys
import threading

logger = logging.getLogger(__name__)
POLICY_ENV = "JASNA_WINDOWS_AMD_SDPA_POLICY"
_TORCH_VERSION = "2.12.0+rocm10.0.0"
_HIP_VERSION = "7.15.26333"
_HIP_API = 71526333
_HIP_SHA256 = "546fb3d6e2d2194a9526fb94ec2fd3aa5b92a48a7595f04efece80162047ef69"
_LOCK = threading.Lock()
_BACKENDS = ("flash", "mem_efficient", "cudnn", "math")


def _runtime_identity() -> dict:
    from jasna.media.hip_kernel import hip_runtime_identity

    return hip_runtime_identity()


def _flags(torch_module) -> dict[str, bool]:
    cuda = torch_module.backends.cuda
    return {name: bool(getattr(cuda, name + "_sdp_enabled")()) for name in _BACKENDS}


def configure_windows_amd_sdpa(device, *, fp16: bool, torch_module=None) -> dict:
    """Apply the verified profile before constructing/forwarding RF-DETR.

    auto: Math only for the exact verified Windows gfx1100 binary identity.
    math: require that same profile, or fail before the first GPU forward.
    default: observe existing defaults, without resetting an external bootstrap.
    Linux, CPU and NVIDIA are never probed or changed by this Windows policy.
    """
    if torch_module is None:
        import torch as torch_module

    record = {"requested": os.environ.get(POLICY_ENV, "auto"),
              "resolved": "unchanged", "fp16": bool(fp16)}
    if (sys.platform != "win32" or getattr(device, "type", None) != "cuda"
            or not getattr(torch_module.version, "hip", None)):
        record["reason"] = "outside Windows AMD GPU scope"
        return record

    requested = record["requested"].strip().lower()
    if requested not in {"auto", "math", "default"}:
        raise ValueError(f"{POLICY_ENV} must be auto, math or default")
    record.update(requested=requested, torch=str(torch_module.__version__),
                  torch_hip=str(torch_module.version.hip))

    def unchanged(reason):
        record.update(reason=reason, flags=_flags(torch_module))
        if requested == "math":
            raise RuntimeError(f"Windows AMD Math SDPA profile not verified: {reason}")
        logger.info("Windows AMD SDPA policy: %s", json.dumps(record, sort_keys=True))
        return record

    if requested == "default":
        return unchanged("explicit default; existing backend flags retained")
    if (record["torch"] != _TORCH_VERSION or record["torch_hip"] != _HIP_VERSION):
        return unchanged("Torch/HIP version outside verified profile")
    properties = torch_module.cuda.get_device_properties(device)
    arch = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    record["gpu_arch"] = arch
    if arch != "gfx1100":
        return unchanged("GPU architecture outside verified profile")
    # No hipDeviceGetLuid dependency: this is the actual loaded HIP binary,
    # not the wheel's ROCm label, a GPU name, or an assumed install path.
    identity = _runtime_identity()
    record["hip_identity"] = identity
    if (identity.get("runtime_version") != _HIP_API
            or identity.get("torch_hip") != _HIP_VERSION
            or identity.get("runtime_dll_sha256") != _HIP_SHA256):
        return unchanged("loaded HIP binary identity outside verified profile")
    with _LOCK:
        before = _flags(torch_module)
        for name in _BACKENDS:
            getattr(torch_module.backends.cuda, "enable_" + name + "_sdp")(name == "math")
        after = _flags(torch_module)
        if after != {name: name == "math" for name in _BACKENDS}:
            raise RuntimeError("Windows AMD Math SDPA backend readback failed; GPU forward refused")
    record.update(resolved="math", flags=after, previous_flags=before,
                  reason="verified Windows gfx1100 Torch/HIP profile")
    logger.info("Windows AMD SDPA policy: %s", json.dumps(record, sort_keys=True))
    return record
