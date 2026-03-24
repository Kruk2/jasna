"""Probe script to measure the actual pipeline delay of a TVAI model.

Usage:
    python probe_tvai_delay.py --model ahq-12
    python probe_tvai_delay.py --model iris-2
    python probe_tvai_delay.py --model ahq-12 --scale 2 --frames 60

Requires TVAI_MODEL_DATA_DIR and TVAI_MODEL_DIR environment variables set,
and Topaz Video ffmpeg accessible.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

TVAI_FFMPEG_PATH = os.environ.get(
    "TVAI_FFMPEG_PATH",
    r"C:\Program Files\Topaz Labs LLC\Topaz Video\ffmpeg.exe",
)
INPUT_SIZE = 256


def build_cmd(model: str, scale: int, extra_args: str = "") -> list[str]:
    size = f"{INPUT_SIZE}x{INPUT_SIZE}"
    out_size = INPUT_SIZE * scale
    filter_arg = f"tvai_up=model={model}:scale={scale}"
    if extra_args:
        filter_arg += f":{extra_args}"
    return [
        TVAI_FFMPEG_PATH,
        "-hide_banner",
        "-loglevel", "warning",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", size,
        "-r", "25",
        "-i", "pipe:0",
        "-sws_flags", "spline+accurate_rnd+full_chroma_int",
        "-filter_complex", filter_arg,
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1",
    ]


def probe_delay(model: str, scale: int, total_frames: int, extra_args: str = "") -> None:
    import threading
    from queue import Queue

    cmd = build_cmd(model, scale, extra_args)
    out_size = INPUT_SIZE * scale
    out_frame_bytes = out_size * out_size * 3

    print(f"Model: {model}, scale: {scale}, input: {INPUT_SIZE}x{INPUT_SIZE}, output: {out_size}x{out_size}")
    print(f"Pushing {total_frames} frames one-by-one, measuring when output appears...")
    print(f"Command: {' '.join(cmd)}")
    print()

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    frame = np.random.randint(0, 255, (INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    frame_bytes = frame.tobytes()

    received_queue: Queue[int] = Queue()

    def _reader():
        stdout = proc.stdout
        while True:
            data = stdout.read(out_frame_bytes)
            if len(data) < out_frame_bytes:
                break
            received_queue.put(1)
        received_queue.put(None)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    frames_pushed = 0
    frames_received = 0
    first_output_at = None

    for i in range(total_frames):
        proc.stdin.write(frame_bytes)
        proc.stdin.flush()
        frames_pushed += 1

        time.sleep(0.005)

        while not received_queue.empty():
            item = received_queue.get_nowait()
            if item is None:
                break
            frames_received += 1
            if first_output_at is None:
                first_output_at = frames_pushed

        if (i + 1) % 10 == 0 or i == 0:
            print(f"  pushed={frames_pushed:3d}  received={frames_received:3d}  delay={frames_pushed - frames_received}")

    # Wait a moment for any in-flight output
    time.sleep(0.5)
    while not received_queue.empty():
        item = received_queue.get_nowait()
        if item is None:
            break
        frames_received += 1
        if first_output_at is None:
            first_output_at = frames_pushed

    print(f"\n--- After pushing all {total_frames} frames ---")
    print(f"  pushed={frames_pushed}  received={frames_received}  buffered={frames_pushed - frames_received}")
    if first_output_at is not None:
        print(f"  First output appeared after pushing frame #{first_output_at}")
    else:
        print(f"  No output received yet!")

    print(f"\nClosing stdin to flush remaining frames...")
    proc.stdin.close()

    reader_thread.join(timeout=30)

    n_flushed = 0
    while not received_queue.empty():
        item = received_queue.get_nowait()
        if item is None:
            break
        n_flushed += 1
    frames_received += n_flushed

    proc.wait(timeout=30)

    stderr_out = proc.stderr.read().decode("utf-8", errors="replace").strip()
    if stderr_out:
        print(f"\nStderr:\n{stderr_out}")

    measured_delay = frames_pushed - (frames_received - n_flushed)
    print(f"\n--- Final ---")
    print(f"  Total pushed:   {frames_pushed}")
    print(f"  Total received: {frames_received}")
    print(f"  After close:    {n_flushed} frames flushed")
    print(f"  Pipeline delay: ~{measured_delay} frames")
    print(f"  Lost frames:    {frames_pushed - frames_received}")


def main():
    parser = argparse.ArgumentParser(description="Probe TVAI model pipeline delay")
    parser.add_argument("--model", required=True, help="TVAI model name (e.g. iris-2, ahq-12)")
    parser.add_argument("--scale", type=int, default=1, help="Scale factor (1, 2, 4)")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames to push")
    parser.add_argument("--extra-args", type=str, default="", help="Extra tvai_up params (e.g. 'estimate=8:blend=0.2')")
    args = parser.parse_args()

    if not Path(TVAI_FFMPEG_PATH).is_file():
        print(f"ERROR: TVAI ffmpeg not found at {TVAI_FFMPEG_PATH}")
        sys.exit(1)
    for var in ("TVAI_MODEL_DATA_DIR", "TVAI_MODEL_DIR"):
        val = os.environ.get(var)
        if not val or not Path(val).is_dir():
            print(f"ERROR: {var} not set or not a directory")
            sys.exit(1)

    probe_delay(args.model, args.scale, args.frames, args.extra_args)


if __name__ == "__main__":
    main()
