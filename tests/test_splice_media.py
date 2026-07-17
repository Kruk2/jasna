from __future__ import annotations

import subprocess
from pathlib import Path

import av
import numpy as np
import pytest

from jasna.media import get_video_meta_data
from jasna.media.splice import (
    SpliceSpan,
    concatenate_fragments,
    create_copy_fragment,
    mux_final_output,
    normalize_fragment,
    probe_keyframes,
)
from jasna.os_utils import resolve_executable, subprocess_no_window_kwargs


def _ffmpeg(*args: str) -> None:
    completed = subprocess.run(
        [resolve_executable("ffmpeg"), "-hide_banner", "-y", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
        check=False,
        **subprocess_no_window_kwargs(),
    )
    if completed.returncode != 0:
        pytest.fail(completed.stderr)


@pytest.mark.parametrize(
    ("codec", "encoder", "source_options", "render_options"),
    [
        ("h264", "libx264", ["-g", "12", "-keyint_min", "12", "-sc_threshold", "0"], ["-g", "12", "-bf", "3", "-flags", "+cgop"]),
        ("hevc", "libx265", ["-x265-params", "keyint=12:min-keyint=12:scenecut=0:open-gop=0"], ["-x265-params", "keyint=12:bframes=4:open-gop=0"]),
        ("av1", "libsvtav1", ["-preset", "10", "-g", "12", "-svtav1-params", "scd=0"], ["-preset", "10", "-g", "9999"]),
    ],
)
def test_mixed_encoder_splice_decodes_with_exact_duration_and_audio(
    tmp_path: Path,
    codec: str,
    encoder: str,
    source_options: list[str],
    render_options: list[str],
) -> None:
    source = tmp_path / f"source-{codec}.mp4"
    _ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=12:duration=3",
        "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=3",
        "-c:v", encoder,
        "-pix_fmt", "yuv420p",
        *source_options,
        "-c:a", "aac",
        str(source),
    )
    metadata = get_video_meta_data(str(source))
    index = probe_keyframes(source, metadata)
    assert len(index.pts) >= 3

    raw_parts = [tmp_path / f"raw-{i}.nut" for i in range(3)]
    create_copy_fragment(source, SpliceSpan("copy", index.start_pts, index.pts[1]), index, raw_parts[0])
    create_copy_fragment(source, SpliceSpan("copy", index.pts[2], index.end_pts), index, raw_parts[2])
    _ffmpeg(
        "-ss", "1",
        "-i", str(source),
        "-t", "1",
        "-map", "0:v:0",
        "-an",
        "-vf", "hue=s=0",
        "-c:v", encoder,
        *render_options,
        "-f", "nut",
        str(raw_parts[1]),
    )

    suffix = ".ts" if codec in {"h264", "hevc"} else ".mkv"
    fragments = []
    for part_index, raw in enumerate(raw_parts):
        normalized = tmp_path / f"part-{part_index}{suffix}"
        normalize_fragment(raw, normalized, codec=codec)
        fragments.append((normalized, 1.0))
    assembled = tmp_path / f"assembled{suffix}"
    concatenate_fragments(
        fragments,
        manifest=tmp_path / "parts.ffconcat",
        destination=assembled,
        codec=codec,
    )
    output = tmp_path / f"output-{codec}.mp4"
    mux_final_output(assembled, source, output, codec=codec)

    with av.open(str(output)) as container:
        assert len(container.streams.video) == 1
        assert len(container.streams.audio) == 1
        assert container.streams.video[0].codec_context.name in {codec, "libdav1d"}
        output_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
        assert len(output_frames) == 36
        assert float(container.duration / av.time_base) == pytest.approx(3.0, abs=0.01)
    with av.open(str(source)) as container:
        source_frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    for frame_index in [*range(12), *range(24, 36)]:
        assert np.array_equal(output_frames[frame_index], source_frames[frame_index])
