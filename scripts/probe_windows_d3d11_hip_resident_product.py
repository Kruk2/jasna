"""Guarded product-class smoke for the Windows AMF-D3D11-HIP resident path."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

from jasna import runtime_contract


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
    )


def _validate_output(
    runtime_root: Path,
    output_path: Path,
    expected_frames: int,
    expected_width: int,
    expected_height: int,
) -> dict[str, object]:
    ffprobe = runtime_root / "bin/ffprobe.exe"
    ffmpeg = runtime_root / "bin/ffmpeg.exe"
    for executable in (ffprobe, ffmpeg):
        if not executable.is_file():
            raise RuntimeError(f"unified runtime is missing {executable.name}")
    stream_probe = _run_checked(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=codec_name,profile,pix_fmt,width,height,nb_read_frames",
            "-of",
            "json",
            str(output_path),
        ]
    )
    stream_payload = json.loads(stream_probe.stdout)
    streams = stream_payload.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"expected one output video stream, observed {streams}")
    stream = streams[0]
    expected_stream = {
        "codec_name": "hevc",
        "profile": "Main",
        "pix_fmt": "yuv420p",
        "width": expected_width,
        "height": expected_height,
        "nb_read_frames": str(expected_frames),
    }
    mismatch = {
        name: {"expected": value, "observed": stream.get(name)}
        for name, value in expected_stream.items()
        if stream.get(name) != value
    }
    if mismatch:
        raise RuntimeError(f"strict output stream validation failed: {mismatch}")

    frame_probe = _run_checked(
        [
            str(ffprobe),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "frame=pts",
            "-of",
            "json",
            str(output_path),
        ]
    )
    frame_payload = json.loads(frame_probe.stdout)
    output_pts = [int(frame["pts"]) for frame in frame_payload.get("frames", [])]
    if len(output_pts) != expected_frames or not all(
        first < second for first, second in zip(output_pts, output_pts[1:])
    ):
        raise RuntimeError(
            "strict output PTS validation failed: "
            f"expected {expected_frames} increasing frames, observed {output_pts}"
        )
    decode = _run_checked(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-xerror",
            "-i",
            str(output_path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ]
    )
    return {
        "stream": stream,
        "output_pts": output_pts,
        "output_pts_strict": True,
        "strict_software_decode": decode.returncode == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--frames", type=int, default=2)
    parser.add_argument("--batch-size", type=int, choices=(1, 4), default=1)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dual-reader", action="store_true")
    parser.add_argument("--long-gate", action="store_true")
    parser.add_argument(
        "--transport",
        choices=("baseline", "resident"),
        default="resident",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if sys.platform != "win32":
        raise RuntimeError("This probe is Windows-only")
    if args.frames < 1:
        raise ValueError("--frames must be positive")
    if args.frames > 8 and (
        not args.long_gate or args.frames not in {60, 300, 600}
    ):
        raise ValueError(
            "more than 8 frames requires --long-gate and an accepted "
            "60, 300, or 600 frame count"
        )
    if args.long_gate and not args.dual_reader:
        raise ValueError("--long-gate requires the product dual-reader topology")
    if os.environ.get("JASNA_UNIFIED_RUNTIME") != "1":
        raise RuntimeError("The probe requires the pinned unified runtime")

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    report_path = Path(args.report).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    runtime_root = runtime_contract.default_runtime_root("win32").resolve()
    runtime_status = runtime_contract.validate_loaded_runtime(
        runtime_root,
        Path(os.environ["JASNA_REPO_ROOT"]).resolve(),
        platform="win32",
    )
    # Python 3.8+ ignores PATH when resolving extension-module dependencies.
    # validate_loaded_runtime() registers the pinned runtime's DLL directory;
    # only import PyAV-dependent product modules after that handle is retained.
    from jasna.media import get_video_meta_data
    from jasna.media.video_decoder import NvidiaVideoReader
    from jasna.media.video_encoder import NvidiaVideoEncoder
    from jasna.media.windows_d3d11_hip_resident import (
        WINDOWS_D3D11_HIP_RESIDENT_ADMITTED_GEOMETRIES,
        WINDOWS_D3D11_HIP_RESIDENT_BACKEND,
        WindowsD3D11HipResidentCoordinator,
    )

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)
    metadata = get_video_meta_data(input_path)
    input_width = int(metadata.video_width)
    input_height = int(metadata.video_height)
    if (
        str(metadata.codec_name).casefold() != "hevc"
        or bool(metadata.is_10bit)
        or (input_width, input_height)
        not in WINDOWS_D3D11_HIP_RESIDENT_ADMITTED_GEOMETRIES
    ):
        raise RuntimeError(
            "The product gate accepts only 1920x1080 or 3840x2160 "
            "HEVC Main8/NV12"
        )

    coordinator = (
        WindowsD3D11HipResidentCoordinator(
            device=device,
            metadata=metadata,
            batch_size=args.batch_size,
            output_codec="hevc",
        )
        if args.transport == "resident"
        else None
    )
    decode_backend = (
        WINDOWS_D3D11_HIP_RESIDENT_BACKEND
        if coordinator is not None
        else "auto"
    )
    started = time.perf_counter()
    pipeline_seconds: float | None = None
    encoded = 0
    pts: list[int] = []
    detect_pts: list[int] = []
    close_stats: dict[str, object] = {}
    output_validation: dict[str, object] = {}
    status = "FAILED"
    failure = ""
    torch.cuda.reset_peak_memory_stats(device)
    try:
        detect_context = (
            NvidiaVideoReader(
                str(input_path),
                batch_size=args.batch_size,
                device=device,
                metadata=metadata,
                decode_backend=decode_backend,
                resident_coordinator=coordinator,
                resident_role="decode-detect",
            )
            if args.dual_reader
            else nullcontext(None)
        )
        with detect_context as detect_reader, NvidiaVideoReader(
            str(input_path),
            batch_size=args.batch_size,
            device=device,
            metadata=metadata,
            decode_backend=decode_backend,
            resident_coordinator=coordinator,
            resident_role="blend-encode",
        ) as reader:
            frames = reader.frames()
            detect_frames = detect_reader.frames() if detect_reader is not None else None
            try:
                first_batch, first_pts = next(frames)
                first_detect_pts = (
                    next(detect_frames)[1] if detect_frames is not None else None
                )
                with NvidiaVideoEncoder(
                    str(output_path),
                    device=device,
                    metadata=metadata,
                    codec="hevc",
                    encoder_settings={"g": 60, "bf": 0},
                    output_fps=metadata.video_fps_exact,
                    mux_audio=False,
                    match_input_bit_depth=True,
                    resident_coordinator=coordinator,
                ) as encoder:
                    batch, batch_pts = first_batch, first_pts
                    detect_batch_pts = first_detect_pts
                    while encoded < args.frames:
                        if detect_batch_pts is not None:
                            observed_detect_pts = [int(value) for value in detect_batch_pts]
                            observed_blend_pts = [int(value) for value in batch_pts]
                            if observed_detect_pts != observed_blend_pts:
                                raise RuntimeError(
                                    "dual resident readers returned mismatched PTS: "
                                    f"detect={observed_detect_pts}, "
                                    f"blend={observed_blend_pts}"
                                )
                            detect_pts.extend(observed_detect_pts)
                        for index, frame_pts in enumerate(batch_pts):
                            encoder.encode(batch[index], int(frame_pts))
                            encoded += 1
                            pts.append(int(frame_pts))
                            if encoded >= args.frames:
                                break
                        if encoded >= args.frames:
                            break
                        batch, batch_pts = next(frames)
                        detect_batch_pts = (
                            next(detect_frames)[1]
                            if detect_frames is not None
                            else None
                        )
            finally:
                frames.close()
                if detect_frames is not None:
                    detect_frames.close()
        if coordinator is not None:
            close_stats = coordinator.close()
        pipeline_seconds = time.perf_counter() - started
        output_validation = _validate_output(
            runtime_root,
            output_path,
            encoded,
            input_width,
            input_height,
        )
        status = "PASSED"
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        try:
            if coordinator is not None:
                close_stats = coordinator.close()
        except BaseException as close_exc:
            failure += f"; close={type(close_exc).__name__}: {close_exc}"
        raise
    finally:
        elapsed = time.perf_counter() - started
        report = {
            "schema": "jasna.windows.d3d11-hip-resident.product-smoke.v1",
            "status": status,
            "failure": failure,
            "scope": (
                f"{input_width}x{input_height} HEVC Main8, dual product "
                "readers and writer"
                if args.dual_reader
                else f"{input_width}x{input_height} HEVC Main8, product "
                "reader and writer classes"
            ),
            "input": str(input_path),
            "input_sha256": _sha256(input_path),
            "output": str(output_path),
            "output_sha256": (
                _sha256(output_path) if output_path.is_file() else None
            ),
            "requested_frames": int(args.frames),
            "batch_size": int(args.batch_size),
            "encoded_frames": encoded,
            "transport": args.transport,
            "pts": pts,
            "pts_strict": all(a < b for a, b in zip(pts, pts[1:])),
            "dual_reader": bool(args.dual_reader),
            "long_gate": bool(args.long_gate),
            "detect_pts": detect_pts,
            "detect_pts_match": not args.dual_reader or detect_pts == pts,
            "wall_seconds": elapsed,
            "pipeline_seconds": (
                pipeline_seconds if pipeline_seconds is not None else elapsed
            ),
            "validation_seconds": (
                elapsed - pipeline_seconds if pipeline_seconds is not None else None
            ),
            "peak_torch_bytes": int(torch.cuda.max_memory_allocated(device)),
            "resident_stats": close_stats,
            "output_validation": output_validation,
            "runtime": runtime_status,
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if not args.quiet:
            print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
