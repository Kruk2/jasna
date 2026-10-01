"""Compare the current RGB roundtrip with a packed-YUV AMF probe path.

This is an isolated Linux AMD experiment.  It deliberately does not patch the
product pipeline or change any default.  Both arms use the product AMF decoder,
encoder, rate-control, PTS buffering, and host-native submission contracts.  The
only experimental change is that the ``native-yuv`` arm hands the second AMF
reader's packed NV12/P010 tensor to the encoder without a full-frame
YUV->RGB->YUV roundtrip.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import logging
from pathlib import Path
from statistics import median
import subprocess
import sys
import time
from typing import Iterator

import av
import torch

from jasna.accelerator import current_stream
from jasna.media import VideoMetadata, get_video_meta_data
from jasna.media import dual_gop_encoder
from jasna.media.video_decoder import NvidiaVideoReader, VideoDecodeError
from jasna.media.video_encoder import NvidiaVideoEncoder


log = logging.getLogger(__name__)
ARM_BASELINE = "baseline-rgb"
ARM_NATIVE_YUV = "native-yuv"
ARMS = (ARM_BASELINE, ARM_NATIVE_YUV)


def _expected_format(metadata: VideoMetadata) -> tuple[str, torch.dtype]:
    if metadata.is_10bit:
        return "p010le", torch.uint16
    return "nv12", torch.uint8


def _validate_native_frame(
    frame,
    *,
    software_format: str,
    width: int,
    height: int,
) -> None:
    frame_format = getattr(getattr(frame, "format", None), "name", None)
    observed_software = getattr(getattr(frame, "sw_format", None), "name", None)
    if (
        frame_format != "amf"
        or observed_software != software_format
        or int(frame.width) != width
        or int(frame.height) != height
    ):
        raise VideoDecodeError(
            "native-YUV probe requires a fixed AMF Vulkan "
            f"{software_format.upper()} frame; got format={frame_format}, "
            f"sw_format={observed_software}, size={getattr(frame, 'width', None)}x"
            f"{getattr(frame, 'height', None)}"
        )


def _validate_packed_tensor(
    packed: torch.Tensor,
    *,
    width: int,
    height: int,
    dtype: torch.dtype,
) -> None:
    expected_shape = (height + height // 2, width)
    if tuple(packed.shape) != expected_shape:
        raise ValueError(
            f"packed YUV shape is {tuple(packed.shape)}, expected {expected_shape}"
        )
    if packed.dtype != dtype:
        raise TypeError(f"packed YUV dtype is {packed.dtype}, expected {dtype}")
    if packed.device.type != "cuda":
        raise ValueError(f"packed YUV must reside on HIP, got {packed.device}")
    if not packed.is_contiguous():
        raise ValueError("packed YUV must be contiguous")


def packed_amf_batches(
    reader: NvidiaVideoReader,
    *,
    seek_seconds: float,
) -> Iterator[tuple[torch.Tensor, list[int]]]:
    """Yield product-audited AMF frames as packed HIP NV12/P010 batches."""

    if not getattr(reader, "_amf_interop_enabled", False):
        raise VideoDecodeError("native-YUV probe did not open the AMF interop reader")
    audit = getattr(reader, "_amf_interop_audit", None)
    if audit is None:
        raise VideoDecodeError("native-YUV probe has no AMF transport audit")
    if getattr(audit, "decode_copy_stream", "") != "null":
        raise VideoDecodeError(
            "native-YUV phase-1 probe requires the proven null-stream source-release "
            "contract"
        )

    height = int(reader.height)
    width = int(reader.width)
    software_format, dtype = _expected_format(reader.metadata)
    bytes_per_sample = 2 if reader.metadata.is_10bit else 1
    decoded = reader._selected_frames(reader._decoded_frames(seek_seconds))
    group = reader._read_group(decoded)
    while group:
        packed = torch.empty(
            (len(group), height + height // 2, width),
            dtype=dtype,
            device=reader.device,
        )
        pts: list[int] = []
        for index, frame in enumerate(group):
            _validate_native_frame(
                frame,
                software_format=software_format,
                width=width,
                height=height,
            )
            result = audit.copy_to_hip(
                frame,
                packed[index].data_ptr(),
                packed[index].numel() * packed[index].element_size(),
            )
            if (
                int(result.get("width", -1)) != width
                or int(result.get("height", -1)) != height
                or int(result.get("bytes_per_sample", -1)) != bytes_per_sample
            ):
                raise VideoDecodeError(
                    f"native-YUV bridge returned an invalid copy result: {result}"
                )
            pts.append(int(frame.pts))

        current_stream(reader.device).synchronize()
        del frame
        group.clear()
        yield packed, pts
        group = reader._read_group(decoded)


class PackedYuvVideoEncoder(NvidiaVideoEncoder):
    """Probe-only encoder that bypasses the product RGB-to-YUV converter."""

    packed_d2h_seconds: float
    amf_submit_seconds: float
    packed_frames: int

    def __enter__(self):
        entered = super().__enter__()
        self.packed_d2h_seconds = 0.0
        self.amf_submit_seconds = 0.0
        self.packed_frames = 0
        return entered

    def _encode_frame(
        self,
        frame: torch.Tensor,
        pts: int,
        *,
        apply_lut: bool = True,
    ) -> None:
        if apply_lut:
            raise RuntimeError("native-YUV probe cannot apply an RGB LUT")
        if self._lut_applier is not None or self._cas is not None:
            raise RuntimeError("native-YUV probe excludes LUT and CAS sharpening")

        height = int(self.metadata.video_height)
        width = int(self.metadata.video_width)
        dtype = torch.uint16 if self.spec.ten_bit else torch.uint8
        _validate_packed_tensor(
            frame,
            width=width,
            height=height,
            dtype=dtype,
        )
        if self.spec.frame_format not in {"nv12", "p010le"}:
            raise RuntimeError(
                f"native-YUV probe cannot submit {self.spec.frame_format!r}"
            )

        d2h_started = time.perf_counter()
        self.stream.synchronize()
        host_yuv = self._host_yuv
        if self._amf_host_zero_copy:
            # Match the accepted product ownership rule: AMF can retain a host
            # pointer after encode() returns, so every in-flight frame owns its
            # own pinned storage.
            host_yuv = torch.empty(
                (height + height // 2, width),
                dtype=dtype,
                pin_memory=True,
            )
        if host_yuv is None:
            raise RuntimeError("native-YUV probe has no pinned AMF input storage")
        host_yuv.copy_(frame, non_blocking=False)
        self.packed_d2h_seconds += time.perf_counter() - d2h_started

        video_frame = av.VideoFrame.from_dlpack(
            [host_yuv[:height], host_yuv[height:]],
            format=self.spec.frame_format,
        )
        video_frame.pts = int(pts)
        video_frame.time_base = self.metadata.time_base

        submit_started = time.perf_counter()
        try:
            packets = self.out_stream.encode(video_frame)
        except av.FFmpegError as exc:
            if not self._video_started:
                raise self._encoder_open_error(exc) from exc
            raise
        for packet in packets:
            self._mux_video(packet)
        self.amf_submit_seconds += time.perf_counter() - submit_started
        self.packed_frames += 1


class PackedYuvDualGopFrameWriter(dual_gop_encoder.AmdDualGopFrameWriter):
    """Probe-only dual-session writer that accepts packed NV12/P010."""

    packed_d2h_seconds: float
    packed_frames: int

    def __init__(self, template: NvidiaVideoEncoder) -> None:
        if template._lut_applier is not None or template._cas is not None:
            raise RuntimeError("native-YUV dual-GOP probe excludes LUT and CAS")
        super().__init__(template)
        self.packed_d2h_seconds = 0.0
        self.packed_frames = 0

    def _prepare(
        self,
        frame: torch.Tensor,
        *,
        apply_lut: bool,
    ) -> tuple[av.VideoFrame, torch.Tensor]:
        if apply_lut:
            raise RuntimeError("native-YUV dual-GOP probe cannot apply an RGB LUT")

        template = self.template
        height = int(template.metadata.video_height)
        width = int(template.metadata.video_width)
        dtype = torch.uint16 if template.spec.ten_bit else torch.uint8
        _validate_packed_tensor(
            frame,
            width=width,
            height=height,
            dtype=dtype,
        )
        if template.spec.frame_format not in {"nv12", "p010le"}:
            raise RuntimeError(
                "native-YUV dual-GOP probe cannot submit "
                f"{template.spec.frame_format!r}"
            )

        started = time.perf_counter()
        template.stream.synchronize()
        host_yuv = self.host_pool.acquire(self.failed)
        try:
            host_yuv.copy_(frame, non_blocking=False)
            video_frame = av.VideoFrame.from_dlpack(
                [host_yuv[:height], host_yuv[height:]],
                format=template.spec.frame_format,
            )
        except BaseException:
            self.host_pool.release(host_yuv)
            raise
        self.packed_d2h_seconds += time.perf_counter() - started
        self.packed_frames += 1
        return video_frame, host_yuv


@dataclass(frozen=True)
class ArmRun:
    arm: str
    output: Path
    wall_seconds: float
    frames: int
    reader_stats: dict[str, object] | None
    writer: str = "single"
    packed_d2h_seconds: float | None = None
    packed_frames: int | None = None
    amf_submit_seconds: float | None = None
    pinned_pool: dict[str, int] | None = None

    @property
    def fps(self) -> float:
        return self.frames / self.wall_seconds


def _build_encoder(
    encoder_type,
    *,
    output: Path,
    metadata: VideoMetadata,
    device: torch.device,
    pts_origin: int,
    dual_gop: bool = False,
) -> NvidiaVideoEncoder:
    encoder_settings: dict[str, object] = {}
    if dual_gop:
        encoder_settings.update(
            {
                "g": dual_gop_encoder.AMD_DUAL_GOP_SIZE,
                "bf": 0,
            }
        )
    return encoder_type(
        str(output),
        device,
        metadata,
        codec="hevc",
        encoder_settings=encoder_settings,
        mux_audio=False,
        pts_origin=int(pts_origin),
        match_input_bit_depth=True,
        smart_fragment=True,
        auto_source_rate=bool(dual_gop),
        prefer_amf_host_native=bool(dual_gop),
        fmp4=False,
    )


def _target_pts(metadata: VideoMetadata, seek_seconds: float) -> int:
    return int(metadata.start_pts or 0) + round(
        float(seek_seconds) / float(metadata.time_base)
    )


def run_arm(
    arm: str,
    *,
    source: Path,
    output: Path,
    metadata: VideoMetadata,
    device: torch.device,
    seek_seconds: float,
    frame_limit: int,
    batch_size: int,
    writer_mode: str,
) -> ArmRun:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    if writer_mode not in {"single", "dual"}:
        raise ValueError(f"unknown writer mode: {writer_mode!r}")
    encoder_type = (
        PackedYuvVideoEncoder
        if writer_mode == "single" and arm == ARM_NATIVE_YUV
        else NvidiaVideoEncoder
    )
    encoder = _build_encoder(
        encoder_type,
        output=output,
        metadata=metadata,
        device=device,
        pts_origin=_target_pts(metadata, seek_seconds),
        dual_gop=writer_mode == "dual",
    )
    reader = NvidiaVideoReader(
        str(source),
        batch_size=batch_size,
        device=device,
        metadata=metadata,
        decode_backend="amf-interop",
    )

    submitted = 0
    started = time.perf_counter()
    dual_writer = None
    if writer_mode == "dual":
        writer_type = (
            PackedYuvDualGopFrameWriter
            if arm == ARM_NATIVE_YUV
            else dual_gop_encoder.AmdDualGopFrameWriter
        )
        dual_writer = writer_type(encoder)
    try:
        with reader:
            batches = (
                packed_amf_batches(reader, seek_seconds=seek_seconds)
                if arm == ARM_NATIVE_YUV
                else reader.frames(seek_ts=seek_seconds)
            )
            try:
                if dual_writer is None:
                    encoder.__enter__()
                for batch, pts_list in batches:
                    take = min(len(pts_list), frame_limit - submitted)
                    for index in range(take):
                        if dual_writer is None:
                            encoder.encode(
                                batch[index],
                                pts_list[index],
                                apply_lut=False,
                            )
                        else:
                            dual_writer.write(
                                batch[index],
                                pts_list[index],
                                apply_lut=False,
                            )
                        submitted += 1
                    if submitted >= frame_limit:
                        break
            finally:
                close_batches = getattr(batches, "close", None)
                if callable(close_batches):
                    close_batches()
        if dual_writer is None:
            encoder.__exit__(None, None, None)
        else:
            dual_writer.close()
    except BaseException:
        if dual_writer is None:
            if getattr(encoder, "dst", None) is not None:
                encoder.__exit__(*sys.exc_info())
        else:
            dual_writer.abort()
        raise

    wall_seconds = time.perf_counter() - started
    if submitted != frame_limit:
        raise RuntimeError(
            f"{arm} reached EOF after {submitted} frames, expected {frame_limit}"
        )
    return ArmRun(
        arm=arm,
        output=output,
        wall_seconds=wall_seconds,
        frames=submitted,
        reader_stats=reader.amf_interop_stats,
        writer=writer_mode,
        packed_d2h_seconds=(
            dual_writer.packed_d2h_seconds
            if isinstance(dual_writer, PackedYuvDualGopFrameWriter)
            else (
                encoder.packed_d2h_seconds
                if isinstance(encoder, PackedYuvVideoEncoder)
                else None
            )
        ),
        packed_frames=(
            dual_writer.packed_frames
            if isinstance(dual_writer, PackedYuvDualGopFrameWriter)
            else (
                encoder.packed_frames
                if isinstance(encoder, PackedYuvVideoEncoder)
                else None
            )
        ),
        amf_submit_seconds=(
            encoder.amf_submit_seconds
            if isinstance(encoder, PackedYuvVideoEncoder)
            else None
        ),
        pinned_pool=(
            {
                "allocated": dual_writer.host_pool.allocated,
                "capacity": dual_writer.host_pool.capacity,
                "peak_in_use": dual_writer.host_pool.peak_in_use,
                "in_use": dual_writer.host_pool.in_use,
            }
            if dual_writer is not None
            else None
        ),
    )


def _run_checked(command: list[str], *, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def validate_reader_stats(
    stats: dict[str, object] | None,
    *,
    expected_frames: int,
) -> dict[str, object]:
    if stats is None:
        raise RuntimeError("AMF reader did not return transport statistics")

    expected = {
        "copy_to_hip_calls": expected_frames,
        "copy_to_hip_successes": expected_frames,
        "hip_d2d_plane_copies": expected_frames * 2,
        "hip_external_memory_imports": expected_frames,
        "hip_external_memory_destroys": expected_frames,
        "hip_mapped_buffer_acquires": expected_frames,
        "hip_mapped_buffer_releases": expected_frames,
        "vulkan_export_fd_close_calls": expected_frames,
        "fixed_context_session_create_calls": 1,
        "fixed_context_session_close_calls": 1,
    }
    must_be_zero = (
        "copy_to_hip_failures",
        "failed_bridge_copies",
        "vulkan_export_fd_close_failures",
        "fixed_context_session_close_failures",
        "hip_non_d2d_copy_calls",
        "host_frame_transfers",
        "cpu_map_calls",
        "staging_copy_calls",
        "d2h_copy_calls",
        "av_hwframe_transfer_data_calls",
        "transport_reconfigures",
        "transport_restarts",
    )
    failures = [
        f"{name}={stats.get(name)!r}, expected {value}"
        for name, value in expected.items()
        if stats.get(name) != value
    ]
    failures.extend(
        f"{name}={stats.get(name)!r}, expected 0"
        for name in must_be_zero
        if stats.get(name) != 0
    )
    if stats.get("fixed_context_session_closed") is not True:
        failures.append(
            "fixed_context_session_closed="
            f"{stats.get('fixed_context_session_closed')!r}"
        )
    if failures:
        raise RuntimeError(f"AMF reader transport validation failed: {failures}")
    return {
        "expected_frames": expected_frames,
        "balanced_external_memory_lifecycle": True,
        "forbidden_host_paths": 0,
        "passed": True,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_output(
    output: Path,
    *,
    expected_frames: int,
    expected_width: int,
    expected_height: int,
    expected_ten_bit: bool,
    expected_fps: object,
    verify_ffprobe: str,
    verify_ffmpeg: str,
) -> dict[str, object]:
    probe = _run_checked(
        [
            verify_ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,profile,pix_fmt,width,height,avg_frame_rate,time_base,"
            "duration,nb_frames,nb_read_frames:format=duration,size",
            "-of",
            "json",
            str(output),
        ],
        timeout=600,
    )
    parsed = json.loads(probe.stdout)
    streams = parsed.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"ffprobe returned {len(streams)} video streams for {output}")
    stream = streams[0]
    observed_frames = int(stream.get("nb_read_frames") or stream.get("nb_frames") or 0)
    expected_pix_fmts = (
        {"p010le", "yuv420p10le"}
        if expected_ten_bit
        else {"nv12", "yuv420p"}
    )
    failures = []
    if stream.get("codec_name") != "hevc":
        failures.append(f"codec={stream.get('codec_name')}")
    if stream.get("pix_fmt") not in expected_pix_fmts:
        failures.append(f"pix_fmt={stream.get('pix_fmt')}")
    if (int(stream.get("width", 0)), int(stream.get("height", 0))) != (
        expected_width,
        expected_height,
    ):
        failures.append(f"size={stream.get('width')}x{stream.get('height')}")
    if observed_frames != expected_frames:
        failures.append(f"frames={observed_frames}")
    expected_duration = expected_frames / float(Fraction(expected_fps))
    observed_duration = float(parsed.get("format", {}).get("duration") or 0.0)
    duration_tolerance = max(0.002, 2.0 / float(Fraction(expected_fps)))
    if abs(observed_duration - expected_duration) > duration_tolerance:
        failures.append(
            f"duration={observed_duration:.6f}, expected "
            f"{expected_duration:.6f}+/-{duration_tolerance:.6f}"
        )

    packets_result = _run_checked(
        [
            verify_ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_packets",
            "-show_entries",
            "packet=pts,dts,duration,flags",
            "-of",
            "json",
            str(output),
        ],
        timeout=300,
    )
    packets = json.loads(packets_result.stdout).get("packets", [])
    packet_pts = [packet.get("pts") for packet in packets]
    packet_dts = [packet.get("dts") for packet in packets]
    if len(packets) != expected_frames:
        failures.append(f"packets={len(packets)}")
    if any(value is None for value in packet_pts):
        failures.append("packet_pts_missing")
    elif any(int(right) <= int(left) for left, right in zip(packet_pts, packet_pts[1:])):
        failures.append("packet_pts_not_strictly_increasing")
    if any(value is None for value in packet_dts):
        failures.append("packet_dts_missing")
    elif any(int(right) <= int(left) for left, right in zip(packet_dts, packet_dts[1:])):
        failures.append("packet_dts_not_strictly_increasing")

    strict = subprocess.run(
        [
            verify_ffmpeg,
            "-v",
            "error",
            "-xerror",
            "-err_detect",
            "explode",
            "-i",
            str(output),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if strict.returncode != 0 or strict.stderr.strip():
        failures.append(
            f"strict_decode={strict.returncode}:{strict.stderr.strip()[:500]}"
        )
    if failures:
        raise RuntimeError(f"output validation failed for {output}: {failures}")
    return {
        "ffprobe": parsed,
        "packet_count": len(packets),
        "first_packet_pts": packet_pts[0] if packet_pts else None,
        "last_packet_pts": packet_pts[-1] if packet_pts else None,
        "first_packet_dts": packet_dts[0] if packet_dts else None,
        "last_packet_dts": packet_dts[-1] if packet_dts else None,
        "strict_decode_return_code": strict.returncode,
        "strict_decode_stderr": strict.stderr,
        "expected_duration": expected_duration,
        "observed_duration": observed_duration,
        "duration_tolerance": duration_tolerance,
        "passed": True,
    }


def _parse_order(value: str) -> tuple[str, ...]:
    aliases = {
        "baseline": ARM_BASELINE,
        "rgb": ARM_BASELINE,
        ARM_BASELINE: ARM_BASELINE,
        "native": ARM_NATIVE_YUV,
        "yuv": ARM_NATIVE_YUV,
        ARM_NATIVE_YUV: ARM_NATIVE_YUV,
    }
    order = tuple(aliases.get(item.strip().casefold(), "") for item in value.split(","))
    if not order or any(not item for item in order):
        raise argparse.ArgumentTypeError(
            "--order accepts comma-separated baseline/native arms"
        )
    return order


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--result-json")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4, choices=(1, 2, 4, 8))
    parser.add_argument(
        "--writer",
        choices=("single", "dual"),
        default="single",
        help="single AMF session or the product dual-GOP session topology",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--verify-ffmpeg",
        default="/usr/bin/ffmpeg",
        help="independent software FFmpeg used for strict decode",
    )
    parser.add_argument(
        "--verify-ffprobe",
        default="/usr/bin/ffprobe",
        help="independent ffprobe used for packet/frame validation",
    )
    parser.add_argument(
        "--order",
        type=_parse_order,
        default=(ARM_BASELINE, ARM_NATIVE_YUV),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.start < 0 or args.frames <= 0:
        raise SystemExit("--start must be non-negative and --frames must be positive")
    source = Path(args.input).expanduser().resolve()
    if not source.is_file():
        raise SystemExit(f"input does not exist: {source}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    result_path = (
        Path(args.result_json).expanduser().resolve()
        if args.result_json
        else output_dir / "REPORT.json"
    )

    metadata = get_video_meta_data(str(source))
    if metadata.codec_name.casefold() != "hevc":
        raise SystemExit(f"phase-1 probe accepts HEVC input, got {metadata.codec_name}")
    if metadata.pixel_format.casefold() not in {
        "nv12",
        "yuv420p",
        "p010le",
        "yuv420p10le",
    }:
        raise SystemExit(
            f"phase-1 probe accepts only NV12/P010-compatible input, got {metadata.pixel_format}"
        )

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    records = []
    for run_index, arm in enumerate(args.order):
        output = output_dir / f"{run_index:02d}-{arm}.mp4"
        run = run_arm(
            arm,
            source=source,
            output=output,
            metadata=metadata,
            device=device,
            seek_seconds=float(args.start),
            frame_limit=int(args.frames),
            batch_size=int(args.batch_size),
            writer_mode=str(args.writer),
        )
        validation = validate_output(
            output,
            expected_frames=int(args.frames),
            expected_width=int(metadata.video_width),
            expected_height=int(metadata.video_height),
            expected_ten_bit=bool(metadata.is_10bit),
            expected_fps=metadata.video_fps_exact,
            verify_ffprobe=str(args.verify_ffprobe),
            verify_ffmpeg=str(args.verify_ffmpeg),
        )
        reader_validation = validate_reader_stats(
            run.reader_stats,
            expected_frames=int(args.frames),
        )
        records.append(
            {
                "arm": arm,
                "run_index": run_index,
                "output": str(output),
                "output_size_bytes": output.stat().st_size,
                "frames": run.frames,
                "wall_seconds": run.wall_seconds,
                "fps": run.fps,
                "writer": run.writer,
                "output_sha256": _sha256(output),
                "packed_d2h_seconds": run.packed_d2h_seconds,
                "packed_frames": run.packed_frames,
                "amf_submit_seconds": run.amf_submit_seconds,
                "pinned_pool": run.pinned_pool,
                "reader_stats": run.reader_stats,
                "reader_validation": reader_validation,
                "validation": validation,
            }
        )
        torch.cuda.empty_cache()

    by_arm = {
        arm: [record for record in records if record["arm"] == arm]
        for arm in ARMS
    }
    median_wall = {
        arm: median(record["wall_seconds"] for record in rows)
        for arm, rows in by_arm.items()
        if rows
    }
    deterministic_by_arm = {
        arm: (
            len(rows) < 2
            or len({record["output_sha256"] for record in rows}) == 1
        )
        for arm, rows in by_arm.items()
        if rows
    }
    report = {
        "schema": "jasna.amd-native-yuv-roundtrip-ab.v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "isolated-probe-only; product defaults unchanged",
        "source": str(source),
        "source_metadata": {
            "codec": metadata.codec_name,
            "profile": metadata.profile,
            "pixel_format": metadata.pixel_format,
            "width": metadata.video_width,
            "height": metadata.video_height,
            "fps": str(metadata.video_fps_exact),
            "time_base": str(metadata.time_base),
            "ten_bit": metadata.is_10bit,
        },
        "start_seconds": float(args.start),
        "frames": int(args.frames),
        "batch_size": int(args.batch_size),
        "writer": str(args.writer),
        "order": list(args.order),
        "runs": records,
        "median_wall_seconds": median_wall,
        "byte_deterministic_by_arm": deterministic_by_arm,
        "native_yuv_speedup_percent": (
            (median_wall[ARM_BASELINE] / median_wall[ARM_NATIVE_YUV] - 1.0) * 100.0
            if all(arm in median_wall for arm in ARMS)
            else None
        ),
        "passed": (
            all(record["validation"]["passed"] for record in records)
            and all(record["reader_validation"]["passed"] for record in records)
            and all(deterministic_by_arm.values())
        ),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
