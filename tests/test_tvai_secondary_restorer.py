"""TvaiSecondaryRestorer: argument handling, output bookkeeping, and whole runs through a stand-in
for Topaz's ffmpeg (a script that echoes rawvideo frames with TVAI's output delay)."""
from __future__ import annotations

import os
import sys
import threading

import numpy as np
import pytest
import torch

from jasna.restorer.tvai_secondary_restorer import (
    TVAI_MIN_STREAM_FRAMES_TO_EMIT,
    TVAI_PIPELINE_DELAY,
    TvaiSecondaryRestorer,
    WorkerLedger,
    _parse_tvai_args_kv,
    frames_to_tensor,
    least_busy_worker,
)

needs_script_ffmpeg = pytest.mark.skipif(
    sys.platform == "win32", reason="the stand-in ffmpeg is a script started through its shebang"
)

_FRAME_BYTES = 256 * 256 * 3

_ECHO_WITH_TVAI_DELAY = f"""
held = []
received = 0
while len(data := stdin.read({_FRAME_BYTES})) == {_FRAME_BYTES}:
    received += 1
    held.append(data)
    if len(held) > {TVAI_PIPELINE_DELAY}:
        stdout.write(held.pop(0))
        stdout.flush()
if received >= {TVAI_MIN_STREAM_FRAMES_TO_EMIT}:
    stdout.write(b"".join(held))
"""
_READ_ALL_RETURN_NOTHING = f"""
while len(stdin.read({_FRAME_BYTES})) == {_FRAME_BYTES}:
    pass
"""
_EXIT_WITHOUT_READING_WHEN_TOLD = """
open(sys.argv[0] + ".exit").read()
sys.exit(1)
"""


@pytest.fixture
def tvai_install(monkeypatch, tmp_path):
    monkeypatch.setenv("TVAI_MODEL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("TVAI_MODEL_DIR", str(tmp_path))

    def ffmpeg(body: str) -> str:
        script = tmp_path / "ffmpeg"
        script.write_text(
            f"#!{sys.executable}\nimport sys\nstdin, stdout = sys.stdin.buffer, sys.stdout.buffer\n{body}"
        )
        script.chmod(0o755)
        return str(script)

    return ffmpeg


def _restorer(ffmpeg_path: str, *, num_workers: int = 1) -> TvaiSecondaryRestorer:
    return TvaiSecondaryRestorer(
        ffmpeg_path=ffmpeg_path, tvai_args="model=iris-2", scale=1, num_workers=num_workers, tvai_denoise=False
    )


def _numbered_frames(first: int, count: int) -> torch.Tensor:
    """Float frames whose uint8 value is their number."""
    values = (torch.arange(first, first + count, dtype=torch.float32) + 0.5) / 255.0
    return values.view(-1, 1, 1, 1).expand(count, 3, 256, 256).contiguous()


def _numbers(frames) -> list[int]:
    return [int(frame[0, 0, 0]) for frame in frames]


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ("", {}),
        (None, {}),
        ("   ", {}),
        ("model=iris-2", {"model": "iris-2"}),
        ("model=iris-2:scale=2:noise=0", {"model": "iris-2", "scale": "2", "noise": "0"}),
        ("model=iris-2:", {"model": "iris-2"}),
        (":model=iris-2", {"model": "iris-2"}),
        ("model=iris-2::scale=2", {"model": "iris-2", "scale": "2"}),
    ],
)
def test_parse_tvai_args_kv(args, expected):
    assert _parse_tvai_args_kv(args) == expected


@pytest.mark.parametrize(("args", "error"), [("model", "expected key=value"), ("=value", "empty key")])
def test_parse_tvai_args_kv_rejects_malformed(args, error):
    with pytest.raises(ValueError, match=error):
        _parse_tvai_args_kv(args)


def test_invalid_scale_raises():
    with pytest.raises(ValueError, match="Invalid tvai scale"):
        TvaiSecondaryRestorer(ffmpeg_path="ffmpeg.exe", tvai_args="model=iris-2", scale=3, num_workers=1, tvai_denoise=False)


def test_scale_and_size_arguments_come_from_the_restorer():
    restorer = TvaiSecondaryRestorer(
        ffmpeg_path="ffmpeg.exe",
        tvai_args="model=iris-2:scale=4:w=256:h=256:noise=0",
        scale=2,
        num_workers=2,
        tvai_denoise=False,
    )
    assert restorer.tvai_filter_args == "model=iris-2:scale=2:noise=0"


@pytest.mark.parametrize(
    ("data_dir", "model_dir", "ffmpeg_exists", "error", "match"),
    [
        (None, "dir", True, RuntimeError, "TVAI_MODEL_DATA_DIR"),
        ("dir", None, True, RuntimeError, "TVAI_MODEL_DIR"),
        ("file", "dir", True, RuntimeError, "not a directory"),
        ("dir", "dir", False, FileNotFoundError, "not found"),
    ],
)
def test_push_clip_rejects_a_broken_tvai_install(monkeypatch, tmp_path, data_dir, model_dir, ffmpeg_exists, error, match):
    not_a_dir = tmp_path / "not_a_dir"
    not_a_dir.write_bytes(b"")
    paths = {"dir": str(tmp_path), "file": str(not_a_dir)}
    for name, value in (("TVAI_MODEL_DATA_DIR", data_dir), ("TVAI_MODEL_DIR", model_dir)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, paths[value])
    ffmpeg = tmp_path / "ffmpeg.exe"
    if ffmpeg_exists:
        ffmpeg.write_bytes(b"")
    restorer = _restorer(str(ffmpeg))
    with pytest.raises(error, match=match):
        restorer.push_clip(torch.rand((1, 3, 256, 256)), keep_start=0, keep_end=1)


def test_frames_become_truncated_uint8_hwc_with_nan_as_black():
    frames = np.array([10.7 / 255, np.nan, 2.0, -1.0], dtype=np.float32).reshape(4, 1, 1, 1).repeat(3, axis=1)
    assert TvaiSecondaryRestorer._to_numpy_hwc(frames)[:, 0, 0, :].tolist() == [[10] * 3, [0] * 3, [255] * 3, [0] * 3]


def test_frames_to_tensor_stacks_hwc_frames_as_tchw():
    frames = [np.full((2, 4, 3), value, dtype=np.uint8) for value in (1, 2)]
    batch = frames_to_tensor(frames)
    assert batch.shape == (2, 3, 2, 4)
    assert batch[:, 0, 0, 0].tolist() == [1, 2]
    assert frames_to_tensor([]).numel() == 0


def _frames(count: int) -> list[np.ndarray]:
    return [np.full((1, 1, 3), index, dtype=np.uint8) for index in range(count)]


def test_ledger_matches_output_frames_to_clips_and_skips_filler():
    ledger = WorkerLedger()
    ledger.add_clip(seq=0, frame_count=2)
    ledger.add_filler(3)
    ledger.add_clip(seq=1, frame_count=1)

    assert ledger.consume(_frames(1)) == []
    completed = ledger.consume(_frames(5))

    assert [seq for seq, _ in completed] == [0, 1]
    assert [frame[0, 0, 0] for frame in completed[1][1]] == [4]
    assert ledger.pending_frames == 0


def test_ledger_extends_a_trailing_filler():
    ledger = WorkerLedger()
    ledger.add_clip(seq=0, frame_count=2)
    ledger.add_filler(TVAI_PIPELINE_DELAY)
    ledger.add_filler(TVAI_PIPELINE_DELAY)

    assert ledger.pending_frames == 2 + 2 * TVAI_PIPELINE_DELAY
    ledger.consume(_frames(2 + 2 * TVAI_PIPELINE_DELAY))
    ledger.add_clip(seq=1, frame_count=1)
    assert [seq for seq, _ in ledger.consume(_frames(1))] == [1]


def test_ledger_reports_waiting_and_unfinished_clips():
    ledger = WorkerLedger()
    ledger.add_clip(seq=3, frame_count=2)
    ledger.add_clip(seq=4, frame_count=2)
    ledger.consume(_frames(3))

    assert ledger.holds_clip()
    assert ledger.holds_clip({4})
    assert not ledger.holds_clip({3})
    assert ledger.unfinished_clips() == [4]


def test_ledger_ignores_frames_nothing_waits_for():
    ledger = WorkerLedger()
    assert ledger.consume(_frames(2)) == []
    assert not ledger.holds_clip()


def test_the_least_busy_worker_gets_the_next_clip():
    busy, idle, also_idle = WorkerLedger(), WorkerLedger(), WorkerLedger()
    busy.add_clip(seq=0, frame_count=5)
    assert least_busy_worker([busy, idle, also_idle]) == 1


@needs_script_ffmpeg
@pytest.mark.parametrize(("frame_count", "keep_start", "keep_end"), [(30, 5, 25), (3, 0, 3)], ids=["clip", "too-short-to-emit"])
def test_restore_returns_the_kept_frames(tvai_install, frame_count, keep_start, keep_end):
    restorer = _restorer(tvai_install(_ECHO_WITH_TVAI_DELAY))
    try:
        restored = restorer.restore(_numbered_frames(0, frame_count), keep_start=keep_start, keep_end=keep_end)
    finally:
        restorer.close()

    assert _numbers(restored) == list(range(keep_start, keep_end))
    assert restored[0].shape == (3, 256, 256) and restored[0].dtype == torch.uint8


@needs_script_ffmpeg
def test_clips_on_several_workers_come_back_in_push_order(tvai_install):
    restorer = _restorer(tvai_install(_ECHO_WITH_TVAI_DELAY), num_workers=2)
    try:
        seqs = [restorer.push_clip(_numbered_frames(10 * clip, 10), keep_start=0, keep_end=10) for clip in range(3)]
        empty_seq = restorer.push_clip(_numbered_frames(0, 4), keep_start=4, keep_end=4)
        restorer.flush_all()
        completed = restorer.pop_completed()
        pending = restorer.has_pending
    finally:
        restorer.close()

    assert [seq for seq, _ in completed] == sorted([*seqs, empty_seq])
    batches = dict(completed)
    assert [_numbers(batches[seq]) for seq in seqs] == [list(range(10 * clip, 10 * clip + 10)) for clip in range(3)]
    assert batches[empty_seq].numel() == 0
    assert not pending


@needs_script_ffmpeg
def test_restore_fails_when_ffmpeg_returns_too_few_frames(tvai_install):
    restorer = _restorer(tvai_install(_READ_ALL_RETURN_NOTHING))
    try:
        with pytest.raises(RuntimeError, match="every frame"):
            restorer.restore(_numbered_frames(0, 10), keep_start=0, keep_end=10)
    finally:
        restorer.close()


@needs_script_ffmpeg
def test_push_fails_instead_of_waiting_forever_when_ffmpeg_dies(tvai_install):
    # ffmpeg reads nothing, so the worker's writer is stuck on the first clip and the second
    # fills the write queue: the third push is waiting for room when ffmpeg exits.
    ffmpeg = tvai_install(_EXIT_WITHOUT_READING_WHEN_TOLD)
    os.mkfifo(ffmpeg + ".exit")
    restorer = _restorer(ffmpeg)
    restorer.push_clip(_numbered_frames(0, 1), keep_start=0, keep_end=1)
    restorer.push_clip(torch.zeros(360, 3, 256, 256), keep_start=0, keep_end=360)
    errors: list[BaseException] = []

    def push_third_clip() -> None:
        try:
            restorer.push_clip(_numbered_frames(0, 1), keep_start=0, keep_end=1)
        except RuntimeError as error:
            errors.append(error)

    pusher = threading.Thread(target=push_third_clip, daemon=True)
    pusher.start()
    with open(ffmpeg + ".exit", "w"):  # closing the fifo lets ffmpeg exit
        pass
    pusher.join(timeout=30)
    finished = not pusher.is_alive()
    if finished:
        restorer.close()

    assert finished
    assert errors
