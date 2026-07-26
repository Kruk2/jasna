import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from benchmark_lada_flatpak import (
    build_command,
    canonical_clips,
    flatpak_cgroup_pids,
    remove_output_artifacts,
)


def test_canonical_clips_excludes_8k_vr_asset(tmp_path: Path) -> None:
    included = tmp_path / "SONE-610_bench_1080p_h264_8bit.mp4"
    excluded = tmp_path / "vr1_8k_vr_hevc_8bit_60fps.mp4"
    included.touch()
    excluded.touch()

    assert canonical_clips(tmp_path) == [included]


def test_flatpak_cgroup_pids_finds_only_requested_app(tmp_path: Path) -> None:
    matching = tmp_path / "101"
    other = tmp_path / "102"
    non_pid = tmp_path / "self"
    matching.mkdir()
    other.mkdir()
    non_pid.mkdir()
    (matching / "cgroup").write_text(
        "0::/user.slice/app-flatpak-io.github.ladaapp.lada-123.scope\n"
    )
    (other / "cgroup").write_text(
        "0::/user.slice/app-flatpak-com.example.Other-456.scope\n"
    )

    assert flatpak_cgroup_pids("io.github.ladaapp.lada", tmp_path) == {101}


def test_build_command_uses_flatpak_and_canonical_settings(tmp_path: Path) -> None:
    clip = tmp_path / "clips" / "test.mp4"
    output = tmp_path / "work" / "out.mp4"
    workdir = output.parent

    command = build_command(
        "io.github.ladaapp.lada",
        clip,
        output,
        workdir,
        device="cuda:0",
        detection_model="v2",
        encoding_preset="hevc-nvidia-gpu-hq",
        max_clip_length=180,
        extra_args=("--no-detect-face-mosaics",),
    )

    assert command[:2] == ["flatpak", "run"]
    assert f"--filesystem={clip.parent}:ro" in command
    assert f"--filesystem={workdir}" in command
    assert command[command.index("--device") + 1] == "cuda:0"
    assert command[command.index("--max-clip-length") + 1] == "180"
    assert command[command.index("--mosaic-detection-model") + 1] == "v2"
    assert (
        command[command.index("--encoding-preset") + 1]
        == "hevc-nvidia-gpu-hq"
    )
    assert command[-1] == "--no-detect-face-mosaics"


def test_remove_output_artifacts_removes_lada_temp_file(tmp_path: Path) -> None:
    output = tmp_path / "bench_out.mp4"
    temporary = tmp_path / "bench_out.tmp.mp4"
    unrelated = tmp_path / "bench_out.log"
    for path in (output, temporary, unrelated):
        path.touch()

    remove_output_artifacts(output, tmp_path)

    assert not output.exists()
    assert not temporary.exists()
    assert unrelated.exists()
