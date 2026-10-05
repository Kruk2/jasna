"""Typed native-context quarantine, without importing torch/av in the GUI."""
from __future__ import annotations


class NativeGpuContextUnusableError(RuntimeError):
    """Do not reuse this process's GPU context; restart to rebuild resources.

    A precaution after a native transfer failure, not proof of TDR/OOM or
    attribution to a particular decoder, encoder, model or driver.
    """


def native_context_failure(exc: BaseException) -> NativeGpuContextUnusableError | None:
    pending = [exc]
    seen = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, NativeGpuContextUnusableError):
            return current
        for linked in (current.__cause__, current.__context__):
            if linked is not None:
                pending.append(linked)
    return None


def is_windows_amf_host_transfer_failure(exc: BaseException, *, platform: str,
                                        amd: bool, decoder_name: str) -> bool:
    """Only quarantine the observed AMF hardware->host UnknownError contract.

    AVERROR_UNKNOWN is generic: by itself it does not identify AMF or device
    loss. Require the active AMF route AND its concrete AVHWFramesContext log.
    """
    if platform != "win32" or not amd or not decoder_name.endswith("_amf"):
        return False
    errno = getattr(exc, "errno", None)
    if isinstance(errno, bool) or not isinstance(errno, int) or abs(errno) != 1313558101:
        return False
    message = str(exc).casefold()
    return ("avhwframescontext" in message
            and "convert(amf::amf_memory_host) failed" in message)
