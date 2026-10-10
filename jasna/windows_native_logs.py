"""Process-lifetime native FFmpeg diagnostics for an explicit Windows worker.

This is not a general PyAV logging replacement. File-restoration workers do
not consume Capture output or enumerate devices. Native logging avoids a
Python/GIL callback during threaded codec destruction. The disposable worker
keeps this policy until process exit, including late native finalizers.

Scope: the admitted file-worker imports only. Device enumeration and loudnorm
stats are unsupported; the latter changes callbacks. Ordinary filters may load. Binding
checks cannot detect a callback changed directly by C or a previously captured
Python alias. They are not a native callback identity query.
"""
from __future__ import annotations

import os
import re
import sys
import threading

ENV = "JASNA_WINDOWS_NATIVE_FFMPEG_LOGS"
_POLICY = None


def requested(environ, platform):
    raw = environ.get(ENV, "0").strip().casefold()
    if raw in {"", "0", "false", "off", "no"}:
        return False
    if raw not in {"1", "true", "on", "yes"}:
        raise RuntimeError(f"invalid {ENV} switch")
    if (platform != "win32" or environ.get("JASNA_ISOLATED_VIDEO_JOB") != "1"
            or environ.get("JASNA_WINDOWS_WORKER_GPU_IDENTITY") != "1"
            or re.fullmatch(r"[0-9a-f]{32}", environ.get("JASNA_WINDOWS_WORKER_ATTEMPT_TOKEN", "")) is None):
        raise RuntimeError("native FFmpeg logs require an explicit guarded Windows video worker")
    return True


class NativeWorkerLogPolicy:
    def __init__(self, av_module):
        if av_module.__version__ != "18.1.0":
            raise RuntimeError("native worker logging requires the admitted PyAV 18.1.0 API")
        log = av_module.logging
        if (log.ERROR, log.WARNING) != (16, 24):
            raise RuntimeError("unexpected FFmpeg logging level constants")
        for name in ("set_level", "get_level", "set_libav_level", "restore_default_callback"):
            if not callable(getattr(log, name, None)):
                raise RuntimeError("native worker logging API is incomplete")
        # PyAV's audio-resampler C imports load av.filter/loudnorm during av
        # import, before the caller can install this policy. Validate their
        # original aliases, then redirect them and reject the public stats
        # entry points, which otherwise swap the global callback in C.
        filters = sys.modules.get("av.filter")
        loudnorm = sys.modules.get("av.filter.loudnorm")
        if (filters is None or loudnorm is None or
                not callable(getattr(loudnorm, "stats", None)) or
                getattr(filters, "stats", None) is not loudnorm.stats or
                getattr(loudnorm, "set_level", None) is not log.set_level or
                getattr(loudnorm, "get_level", None) is not log.get_level):
            raise RuntimeError("unexpected PyAV loudnorm aliases before native logging installation")
        self._filters = filters
        self._loudnorm = loudnorm
        self._logging = log
        self._native_level = log.set_libav_level
        self._restore_native = log.restore_default_callback
        self._level = log.WARNING
        self._requests = 0
        self._lock = threading.Lock()
        # Store bound methods once so identity checks can detect replacement.
        self._set = self.set_level
        self._get = self.get_level
        self._reject_stats = self.reject_loudnorm_stats
        self._native_level(self._level)
        self._restore_native()
        log.set_level = self._set
        log.set_libav_level = self._set
        log.get_level = self._get
        loudnorm.set_level = self._set
        loudnorm.get_level = self._get
        loudnorm.stats = self._reject_stats
        filters.stats = self._reject_stats

    def reject_loudnorm_stats(self, *args, **kwargs):
        raise RuntimeError("loudnorm stats are unsupported in the native-log Windows file worker")

    def set_level(self, level):
        """Change native verbosity without ever installing a Python callback.

        Within this file-worker policy, None restores WARNING. Requests are
        clamped to ERROR..WARNING: errors remain visible and verbose library
        chatter cannot exhaust the bounded parent diagnostic display budget.
        This deliberately does not preserve Capture-based device APIs.
        """
        if level is not None and (type(level) is not int or not -8 <= level <= 64):
            raise ValueError("FFmpeg worker log level must be None or an integer from -8 to 64")
        effective = 24 if level is None else max(16, min(24, level))
        with self._lock:
            self.assert_active()
            self._native_level(effective)
            self._level = effective
            self._requests += 1

    def get_level(self):
        return self._level

    def assert_active(self):
        log = self._logging
        if (log.set_level is not self._set or log.set_libav_level is not self._set
                or log.get_level is not self._get or log.restore_default_callback is not self._restore_native):
            raise RuntimeError("native worker logging bindings were replaced")
        if (sys.modules.get("av.filter") is not self._filters or
                sys.modules.get("av.filter.loudnorm") is not self._loudnorm or
                self._filters.stats is not self._reject_stats or
                self._loudnorm.stats is not self._reject_stats or
                self._loudnorm.set_level is not self._set or self._loudnorm.get_level is not self._get):
            raise RuntimeError("native worker loudnorm bindings were replaced")

    def snapshot(self):
        self.assert_active()
        return dict(mode="native_process_lifetime", level=self._level,
                    redirected_level_requests=self._requests, python_callback_setter_used=False)


def install_worker_native_logs(*, environ=None, platform=None):
    """Opt in before Processor/media imports; never restore Python callbacks."""
    global _POLICY
    environ = os.environ if environ is None else environ
    platform = sys.platform if platform is None else platform
    if not requested(environ, platform):
        return None
    if _POLICY is not None:
        _POLICY.assert_active()
        return _POLICY
    import av
    _POLICY = NativeWorkerLogPolicy(av)
    return _POLICY
