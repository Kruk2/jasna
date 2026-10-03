#!/usr/bin/env python3
"""Run Jasna with transaction-only RF-DETR B1/B2 selection and track capture."""

from __future__ import annotations

import argparse
import atexit
import json
from pathlib import Path
import sys
import time


def normalize_jasna_args(values: list[str]) -> list[str]:
    result = list(values)
    if result[:1] == ["--"]:
        result.pop(0)
    if not result:
        raise ValueError("missing Jasna CLI arguments after --")
    return result


def value_after(arguments: list[str], option: str) -> str:
    matches = [index for index, value in enumerate(arguments) if value == option]
    if len(matches) != 1 or matches[0] + 1 >= len(arguments):
        raise ValueError(f"expected exactly one {option} VALUE pair")
    return arguments[matches[0] + 1]


def install_candidate_manifest(manifest: Path) -> None:
    from jasna.mosaic import rfdetr_migraphx_runner

    candidate = manifest.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    original = rfdetr_migraphx_runner.discover_product_rfdetr_migraphx_manifest

    def discover(**kwargs):
        if original(**kwargs) is None:
            raise RuntimeError("B2 probe reached a host/model scope outside product eligibility")
        return candidate

    rfdetr_migraphx_runner.discover_product_rfdetr_migraphx_manifest = discover


def install_track_capture(destination: Path) -> None:
    from jasna.tracking.clip_tracker import ClipTracker

    records: list[dict[str, int]] = []
    original_update = ClipTracker.update
    original_flush = ClipTracker.flush

    def capture(ended) -> None:
        for item in ended:
            records.append(
                {
                    "track_id": int(item.clip.track_id),
                    "start_frame": int(item.clip.start_frame),
                    "end_frame": int(item.clip.end_frame),
                    "frame_count": int(item.clip.frame_count),
                }
            )

    def update(self, *args, **kwargs):
        ended, active = original_update(self, *args, **kwargs)
        capture(ended)
        return ended, active

    def flush(self, *args, **kwargs):
        ended = original_flush(self, *args, **kwargs)
        capture(ended)
        return ended

    def save() -> None:
        destination.write_text(
            json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    ClipTracker.update = update
    ClipTracker.flush = flush
    atexit.register(save)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("b1", "b2"), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("jasna_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.backend == "b2" and args.manifest is None:
        parser.error("--manifest is required for B2")
    return args


def main() -> None:
    args = parse_args()
    jasna_args = normalize_jasna_args(args.jasna_args)
    output = Path(value_after(jasna_args, "--output")).resolve()
    if args.backend == "b2":
        install_candidate_manifest(args.manifest)
    install_track_capture(output.parent / "tracks.json")
    sys.argv = ["jasna", *jasna_args]
    from jasna.main import main as jasna_main

    started = time.monotonic()
    try:
        jasna_main()
    finally:
        print(
            f"RFDETR_B2_PRODUCT_PROBE backend={args.backend} "
            f"wall_seconds={time.monotonic() - started:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
