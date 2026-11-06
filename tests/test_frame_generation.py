import importlib.util
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from factories import FakeFrameGeneration, session_config, video_metadata
from jasna.frame_generation import FrameGenerationWriter
from jasna.media.splice import SmartRenderCompatibilityError
from jasna.media.video_encoder import resolve_encoder_options, AcceleratorVendor
from jasna.session_factory import FRAME_GENERATION_NEEDS_RTX_40, build_restoration_session, splice_source


class _Writer:
    def __init__(self):
        self.written: list[tuple[int, int]] = []
        self.finished = self.closed = False

    def write(self, frame, pts):
        self.written.append((int(frame.flatten()[0]), pts))

    def finish(self):
        self.finished = True

    def close(self):
        self.closed = True


def _frame(value: int) -> torch.Tensor:
    return torch.full((3, 2, 2), value, dtype=torch.uint8)


def _generate_mean(previous, current):
    return ((previous.int() + current.int()) // 2).to(torch.uint8)


def _write(pts: list[int]) -> tuple[_Writer, FakeFrameGeneration]:
    writer, generate = _Writer(), FakeFrameGeneration(_generate_mean)
    frame_generation = FrameGenerationWriter(writer, generate)
    for value, frame_pts in zip(range(0, 100, 10), pts):
        frame_generation.write(_frame(value), frame_pts)
    frame_generation.finish()
    frame_generation.close()
    return writer, generate


def test_a_frame_is_generated_at_each_midpoint_and_the_last_frame_fills_its_interval():
    writer, generate = _write([0, 3, 6])

    assert writer.written == [(0, 0), (5, 3), (10, 6), (15, 9), (20, 12), (20, 15)]
    assert generate.calls == 2
    assert writer.finished and writer.closed and generate.closed


def test_variable_frame_intervals_keep_their_midpoints():
    writer, _ = _write([0, 1, 3, 7])

    assert [pts for _, pts in writer.written] == [0, 1, 2, 4, 6, 10, 14, 18]


def test_one_frame_is_written_once():
    writer, generate = _write([5])

    assert writer.written == [(0, 10)]
    assert generate.calls == 0


def test_a_stopped_job_closes_without_the_held_frame():
    writer, generate = _Writer(), FakeFrameGeneration(_generate_mean)
    frame_generation = FrameGenerationWriter(writer, generate)
    frame_generation.write(_frame(0), 0)
    frame_generation.write(_frame(10), 1)
    frame_generation.close()

    assert writer.written == [(0, 0), (5, 1)]
    assert not writer.finished and writer.closed and generate.closed


def test_the_bitrate_cap_doubles_with_frame_generation():
    metadata = video_metadata(video_bitrate=4_000_000)
    options = {
        enabled: resolve_encoder_options(
            AcceleratorVendor.NVIDIA, "hevc", metadata, {}, smart_fragment=False, frame_generation=enabled
        )[1]
        for enabled in (False, True)
    }
    assert int(options[True]["maxrate"]) == 2 * int(options[False]["maxrate"])
    assert int(options[True]["bufsize"]) == 2 * int(options[False]["bufsize"])


def test_smart_rendering_is_refused_with_frame_generation():
    with pytest.raises(SmartRenderCompatibilityError, match="frame generation"):
        splice_source(session_config(frame_generation="2x"), Path("in.mp4"), Path("out.mp4"))


def test_ltx_auto_split_restores_the_whole_video_with_frame_generation(caplog):
    from jasna.session_factory import _auto_split_job

    config = session_config(restoration_model_name="ltx", ltx_auto_split_percent=12.0, frame_generation="2x")
    scan = MagicMock()

    assert _auto_split_job(config, Path("in.mp4"), Path("out.mp4"), scan) is None
    scan.assert_not_called()
    assert "frame generation" in caplog.text and "whole video with LTX" in caplog.text


@pytest.mark.parametrize(("vendor", "capability", "supported"), [
    ("nvidia", (8, 9), True), ("nvidia", (12, 0), True), ("nvidia", (8, 6), False), ("amd", (12, 0), False),
])
def test_frame_generation_needs_an_nvidia_rtx_40_or_newer(vendor, capability, supported):
    from jasna import accelerator

    with (
        patch.object(accelerator, "is_nvidia_device", return_value=vendor == "nvidia"),
        patch.object(accelerator.torch.cuda, "get_device_capability", return_value=capability),
    ):
        assert accelerator.supports_frame_generation("cuda:0") is supported


def test_a_session_refuses_frame_generation_on_an_unsupported_gpu():
    with (
        patch("jasna.accelerator.supports_frame_generation", return_value=False),
        pytest.raises(ValueError, match=FRAME_GENERATION_NEEDS_RTX_40),
    ):
        build_restoration_session(session_config(device="cpu", frame_generation="2x"), log_callback=None)


GPU_SCRIPT = """
import torch
from jasna.frame_generation import RtxFrameGeneration

device = torch.device("cuda")
torch.manual_seed(0)
scenes = [torch.rand(3, 27, 48) * 255 for _ in range(2)]
def frame(scene, shift):
    big = torch.nn.functional.interpolate(scenes[scene][None], size=(270, 480), mode="bilinear")[0]
    return big.roll(shift, dims=2).round().to(torch.uint8).to(device)

generate = RtxFrameGeneration(device, 480, 270)
expected = generate(frame(0, 0), frame(0, 8))
with torch.cuda.stream(torch.cuda.Stream(device, priority=-1)):
    side = generate(frame(0, 0), frame(0, 8))
cut = generate(frame(0, 8), frame(1, 0))
torch.cuda.synchronize()
generate.close()
print(tuple(side.shape), side.dtype, torch.equal(side, expected), torch.equal(cut, frame(0, 8)))
"""


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("nvvfx") is None, reason="needs CUDA and nvvfx"
)
def test_rtx_frame_generation_runs_from_a_side_stream_and_does_not_blend_across_a_cut():
    from jasna.accelerator import supports_frame_generation

    if not supports_frame_generation("cuda:0"):
        pytest.skip("needs an RTX 40-series or newer GPU")
    # A subprocess, since test_rtx_superres_restorer installs a fake nvvfx module.
    result = subprocess.run([sys.executable, "-c", GPU_SCRIPT], capture_output=True, text=True, check=True)
    assert result.stdout.split("\n")[-2] == "(3, 270, 480) torch.uint8 True True"
