from __future__ import annotations

import shutil
import subprocess

from jasna.media import get_video_meta_data
from jasna.media.spatial_metadata import (
    _canonical_spatial_atoms,
    _read_spatial_atoms,
    inject_vr180_spatial_metadata,
)
from jasna.os_utils import resolve_executable, subprocess_no_window_kwargs


def _make_mp4(path) -> None:
    subprocess.run(
        [
            resolve_executable("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=size=128x64:rate=2:duration=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
        **subprocess_no_window_kwargs(),
    )


def test_canonical_atoms_are_left_right_half_equirectangular() -> None:
    st3d, sv3d = _canonical_spatial_atoms()

    assert st3d[4:8] == b"st3d"
    assert st3d[-1] == 2
    assert sv3d[4:8] == b"sv3d"
    equi = sv3d.index(b"equi")
    bounds = [
        int.from_bytes(sv3d[equi + 8 + offset : equi + 12 + offset], "big")
        for offset in (0, 4, 8, 12)
    ]
    assert bounds == [0, 0, 0x40000000, 0x40000000]


def test_injects_and_ffprobe_verifies_vr180_metadata(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    _make_mp4(source)
    shutil.copyfile(source, output)

    inject_vr180_spatial_metadata(source, output)

    metadata = get_video_meta_data(str(output))
    assert metadata.stereo_layout == "side by side"
    assert metadata.spherical_projection == "tiled equirectangular"
    st3d, sv3d = _read_spatial_atoms(output)
    assert st3d is not None
    assert sv3d is not None


def test_copies_compatible_source_atoms(tmp_path) -> None:
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mov"
    _make_mp4(source)
    inject_vr180_spatial_metadata(source, source)
    shutil.copyfile(source, output)

    source_atoms = _read_spatial_atoms(source)
    inject_vr180_spatial_metadata(source, output)

    assert _read_spatial_atoms(output) == source_atoms


def test_non_mp4_source_uses_canonical_atoms(tmp_path) -> None:
    source = tmp_path / "source.mkv"
    output = tmp_path / "output.mp4"
    source.write_bytes(b"not an MP4 file")
    _make_mp4(output)

    inject_vr180_spatial_metadata(source, output)

    metadata = get_video_meta_data(str(output))
    assert metadata.stereo_layout == "side by side"
    assert metadata.spherical_projection == "tiled equirectangular"
