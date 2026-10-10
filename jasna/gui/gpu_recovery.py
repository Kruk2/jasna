"""Shared bounded post-worker whole-card VRAM recovery policy (no GPU imports)."""
import time


def wait_for_vram_recovery(reader, *, stop_event, on_log, min_headroom_bytes,
                           timeout_seconds=60, poll_seconds=.25, stable_samples=2,
                           allow_unavailable=False):
    """Wait for consecutive whole-card samples, with cancellation between reads.

    Reader ownership belongs to the caller. Linux's historical unavailable
    telemetry policy is explicit; new guarded Windows callers fail closed.
    This reserve is a recovery admission gate, not the offloader watermarks.
    """
    if type(min_headroom_bytes) is not int or min_headroom_bytes < 0:
        raise ValueError("minimum headroom must be nonnegative integer bytes")
    if type(stable_samples) is not int or stable_samples < 1:
        raise ValueError("stable sample count must be a positive integer")
    if (type(timeout_seconds) not in (int, float) or type(poll_seconds) not in (int, float)
            or not 0 < timeout_seconds <= 60 or not 0 < poll_seconds <= 1):
        raise ValueError("recovery timing exceeds the bounded contract")
    deadline = time.monotonic() + timeout_seconds
    stable = 0
    last_headroom = 0
    while not stop_event.is_set():
        try:
            sample = reader()
            if sample is None:
                if allow_unavailable:
                    return True
                raise ValueError("whole-card VRAM telemetry is unavailable")
            if (not isinstance(sample, tuple) or len(sample) != 2
                    or any(type(value) is not int for value in sample)):
                raise ValueError("whole-card VRAM telemetry must be two integer byte counts")
            used, total = sample
            if total <= 0 or not 0 <= used <= total:
                raise ValueError("whole-card VRAM telemetry is out of range")
            last_headroom = total - used
        except Exception as error:
            on_log("ERROR", f"Cannot verify whole-card GPU recovery: {error}"[:2048])
            return False
        if last_headroom >= min_headroom_bytes:
            stable += 1
            if stable >= stable_samples:
                return not stop_event.is_set()
        else:
            stable = 0
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            on_log("ERROR", "GPU memory did not recover after the isolated worker exited: "
                   f"only {last_headroom / (1024 ** 2):.0f} MiB headroom is available")
            return False
        stop_event.wait(min(poll_seconds, remaining))
    return False
