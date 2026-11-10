import threading

import pytest

from jasna.peak_sampler import PeakSampler


def _reads_then_signals(values: list[int], drained: threading.Event):
    """A read that returns ``values`` in turn, then sets ``drained`` and keeps returning 0."""
    remaining = iter(values)

    def read() -> int:
        value = next(remaining, None)
        if value is None:
            drained.set()
            return 0
        return value

    return read


def test_keeps_the_largest_value_read_until_stopped():
    drained = threading.Event()
    sampler = PeakSampler(_reads_then_signals([3, 9, 4], drained), interval_seconds=0)
    sampler.start()
    assert drained.wait(timeout=30)
    assert sampler.stop() == 9


def test_reads_once_as_it_starts():
    first_read = threading.Event()
    sampler = PeakSampler(lambda: first_read.set() or 7, interval_seconds=3600)
    sampler.start()
    assert first_read.wait(timeout=30)
    assert sampler.stop() == 7


def test_stop_raises_what_the_read_raised():
    def read() -> int:
        raise RuntimeError("no device")

    sampler = PeakSampler(read, interval_seconds=0)
    sampler.start()
    with pytest.raises(RuntimeError, match="no device"):
        sampler.stop()
