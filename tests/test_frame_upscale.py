import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from factories import make_pipeline, session_config, video_metadata
from jasna.engine_compiler import EngineCompilationRequest
from jasna.media.splice import SmartRenderCompatibilityError
from jasna.models.span import SPAN_BLOCKS, Span
from jasna.session_factory import UPSCALE_RTX_NEEDS_NVIDIA, build_restoration_session, splice_source
from jasna.settings_tag import job_settings
from jasna.upscale import FrameUpscaleWriter, finish_frame, upscaled_metadata, upscaled_size
from jasna.upscale.span import span_engine_path, span_input_sizes
from jasna.vr180 import VrModeResolution

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("source", "target", "expected"),
    [
        ((1920, 1080), 2160, (3840, 2160)),
        ((1280, 720), 1080, (1920, 1080)),
        ((854, 480), 1080, (1922, 1080)),
        ((3840, 1920), 2160, (4320, 2160)),
        ((1920, 1080), 1080, None),
        ((3840, 2160), 1440, None),
    ],
)
def test_the_upscaled_size_keeps_the_aspect_ratio_with_an_even_width(source, target, expected):
    assert upscaled_size(*source, target) == expected


def test_the_encoder_sees_the_upscaled_size_and_a_bitrate_grown_with_the_pixels():
    metadata = upscaled_metadata(video_metadata(video_width=1920, video_height=1080, video_bitrate=5_000_000), (3840, 2160))

    assert (metadata.video_width, metadata.video_height, metadata.video_bitrate) == (3840, 2160, 20_000_000)


@pytest.mark.parametrize(
    ("source_height", "output_height", "passes"),
    [(1080, 2160, 1), (720, 1080, 1), (720, 1440, 1), (720, 2160, 2), (480, 2160, 2)],
)
def test_span_runs_twice_only_beyond_2x(source_height, output_height, passes):
    sizes = span_input_sizes((source_height * 16 // 9, source_height), (output_height * 16 // 9, output_height))

    assert [height for _, height in sizes] == [source_height * 2**index for index in range(passes)]


def test_each_span_input_size_and_precision_has_its_own_engine():
    paths = {span_engine_path(size, fp16=fp16).name for size in ((1920, 1080), (1280, 720)) for fp16 in (True, False)}

    assert len(paths) == 4
    assert all("span-nomosuni-2x" in name and name.endswith(".engine") for name in paths)


def test_span_engine_sizes_survive_the_compiler_subprocess_request():
    request = EngineCompilationRequest(device="cuda:0", fp16=True, span_input_sizes=[(1920, 1080), (3840, 2160)])

    assert EngineCompilationRequest.from_json(request.to_json()) == request


def test_a_model_output_is_resized_to_exactly_the_output_size():
    upscaled = torch.rand(1, 3, 16, 32)

    frame = finish_frame(upscaled, torch.zeros(3, 8, 16, dtype=torch.uint8), (24, 12), 1.0)

    assert frame.shape == (3, 12, 24) and frame.dtype == torch.uint8


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0])
def test_strength_mixes_the_model_output_with_a_plain_resize(strength):
    source = torch.full((3, 4, 4), 100, dtype=torch.uint8)
    upscaled = torch.full((1, 3, 8, 8), 200 / 255)

    frame = finish_frame(upscaled, source, (8, 8), strength)

    assert torch.all(frame == round(100 + strength * 100))


class _Writer:
    def __init__(self, *, fail_on_close: bool = False):
        self.written, self.finished, self.closed, self._fail = [], False, False, fail_on_close

    def write(self, frame, pts):
        self.written.append((tuple(frame.shape), pts))

    def after_write(self, frames_written, stop_event):
        pass

    def finish(self):
        self.finished = True

    def close(self):
        self.closed = True
        if self._fail:
            raise RuntimeError("encoder failed")


class _NearestUpscaler:
    def __init__(self):
        self.closed = False

    def __call__(self, frame):
        return F.interpolate(frame[None].float(), scale_factor=2, mode="nearest")[0].to(torch.uint8)

    def close(self):
        self.closed = True


def test_the_upscale_writer_writes_upscaled_frames_and_owns_the_upscaler():
    writer, upscaler = _Writer(), _NearestUpscaler()
    upscale = FrameUpscaleWriter(writer, upscaler)

    upscale.write(torch.zeros(3, 4, 6, dtype=torch.uint8), 7)
    upscale.finish()
    upscale.close()

    assert writer.written == [((3, 8, 12), 7)]
    assert writer.finished and writer.closed and upscaler.closed


def test_the_upscaler_is_closed_when_the_writer_fails_to_close():
    upscaler = _NearestUpscaler()

    with pytest.raises(RuntimeError, match="encoder failed"):
        FrameUpscaleWriter(_Writer(fail_on_close=True), upscaler).close()
    assert upscaler.closed


@pytest.mark.usefixtures("nvidia_vendor")
def test_an_upscaling_job_encodes_at_the_upscaled_size_and_upscales_every_frame():
    pipeline = make_pipeline(device=torch.device("cuda", 0), upscale_model="span", upscale_height=1080)
    metadata = video_metadata(video_width=960, video_height=540, video_bitrate=1_000_000)
    inner = _Writer()

    encoder = pipeline._video_encoder(metadata, pipeline._resolve_frame_rate(metadata), {"output": {}})
    with patch("jasna.pipeline.build_frame_upscaler", return_value=_NearestUpscaler()):
        pipeline._frame_writer(inner, metadata).write(torch.zeros(3, 540, 960, dtype=torch.uint8), 0)

    assert (encoder.metadata.video_width, encoder.metadata.video_height) == (1920, 1080)
    assert inner.written == [((3, 1080, 1920), 0)]


@pytest.mark.usefixtures("nvidia_vendor")
def test_a_video_already_at_the_target_height_is_not_upscaled():
    pipeline = make_pipeline(device=torch.device("cuda", 0), upscale_model="span", upscale_height=1080)
    metadata = video_metadata(video_width=1920, video_height=1080)
    inner = _Writer()

    assert pipeline._frame_writer(inner, metadata) is inner


def test_smart_rendering_is_refused_with_upscaling():
    with pytest.raises(SmartRenderCompatibilityError, match="upscaling"):
        splice_source(session_config(upscale_model="span"), Path("in.mp4"), Path("out.mp4"))


def test_an_amd_session_refuses_rtx_upscaling():
    with (
        patch("jasna.accelerator.is_amd_device", return_value=True),
        patch("jasna.accelerator.log_amd_backend"),
        pytest.raises(ValueError, match=UPSCALE_RTX_NEEDS_NVIDIA),
    ):
        build_restoration_session(session_config(device="cpu", upscale_model="rtx-super-res"), log_callback=None)


@pytest.mark.parametrize(
    ("overrides", "recorded"),
    [
        ({}, None),
        ({"upscale_model": "span", "upscale_height": 1440, "upscale_strength": 0.8},
         {"model": "span", "height": 1440, "strength": 0.8}),
        ({"upscale_model": "rtx-super-res", "upscale_rtx_quality": "ultra", "upscale_rtx_denoise": "low"},
         {"model": "rtx-super-res", "height": 2160, "strength": 1.0,
          "quality": "ultra", "denoise": "low", "deblur": "none"}),
    ],
)
def test_the_settings_tag_records_the_upscaling(overrides, recorded):
    settings = job_settings(
        session_config(**overrides), models=frozenset({"basicvsrpp"}), segments=None,
        vr=VrModeResolution("auto", "off", "", 16 / 9, "raw"), ltx_weights=None, ltx_large_canvas=None,
    )

    assert settings["output"]["upscale"] == recorded


def _training_span_state(generator: torch.Generator) -> dict[str, torch.Tensor]:
    """A SPAN training checkpoint with random weights: each Conv3XC as its 1x1 -> 3x3 -> 1x1 chain plus a skip."""
    state = {}

    def conv(name, out_channels, in_channels, kernel):
        state[f"{name}.weight"] = torch.randn(out_channels, in_channels, kernel, kernel, generator=generator) * 0.1
        state[f"{name}.bias"] = torch.randn(out_channels, generator=generator) * 0.1

    def conv3xc(name, in_channels, out_channels):
        conv(f"{name}.sk", out_channels, in_channels, 1)
        conv(f"{name}.conv.0", in_channels * 2, in_channels, 1)
        conv(f"{name}.conv.1", out_channels * 2, in_channels * 2, 3)
        conv(f"{name}.conv.2", out_channels, out_channels * 2, 1)

    conv3xc("conv_1", 3, 8)
    for block in range(1, SPAN_BLOCKS + 1):
        for index in (1, 2, 3):
            conv3xc(f"block_{block}.c{index}_r", 8, 8)
    conv3xc("conv_2", 8, 8)
    conv("conv_cat", 8, 32, 1)
    conv("upsampler.0", 12, 8, 3)
    return state


def _training_conv3xc(state, name, image):
    """How SPAN trains a Conv3XC: the three convolutions on a zero-padded input, plus the 1x1 skip."""
    padded = F.pad(image, (1, 1, 1, 1))
    out = F.conv2d(padded, state[f"{name}.conv.0.weight"], state[f"{name}.conv.0.bias"])
    out = F.conv2d(out, state[f"{name}.conv.1.weight"], state[f"{name}.conv.1.bias"])
    out = F.conv2d(out, state[f"{name}.conv.2.weight"], state[f"{name}.conv.2.bias"])
    return out + F.conv2d(image, state[f"{name}.sk.weight"], state[f"{name}.sk.bias"])


def test_folded_span_weights_compute_what_the_training_convolutions_compute_away_from_the_border():
    spec = importlib.util.spec_from_file_location("fuse_span_weights", REPO / "scripts" / "fuse_span_weights.py")
    fuse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fuse)
    state = _training_span_state(torch.Generator().manual_seed(0))
    image = torch.rand(1, 8, 12, 12, generator=torch.Generator().manual_seed(1))

    folded = fuse.fused_weights(state)
    network = Span(folded, scale=2)

    expected = _training_conv3xc(state, "block_1.c1_r", image)
    actual = network.blocks[0][0](image)
    assert torch.allclose(actual[..., 2:-2, 2:-2], expected[..., 2:-2, 2:-2], atol=1e-5)
    assert network(torch.rand(1, 3, 6, 10)).shape == (1, 3, 12, 20)


SPAN_GPU_SCRIPT = """
import torch
from pathlib import Path
from jasna.models.span import load_span
from jasna.upscale.span import SPAN_WEIGHTS_PATH, SpanFrameUpscaler, compile_span_engine, span_engine_path

device = torch.device("cuda")
torch.manual_seed(0)
frame = (torch.rand(3, 36, 64) * 255).to(torch.uint8).to(device)
engine = span_engine_path((64, 36), fp16=True)
compile_span_engine((64, 36), device, fp16=True)
try:
    upscaler = SpanFrameUpscaler(device, source_size=(64, 36), output_size=(128, 72), fp16=True, strength=1.0)
    with torch.cuda.stream(torch.cuda.Stream(device, priority=-1)):
        tensorrt = upscaler(frame)
    torch.cuda.synchronize()
    upscaler.close()
finally:
    engine.unlink()
network = load_span(SPAN_WEIGHTS_PATH, scale=2).to(device)
reference = network(frame[None].float() / 255.0).clamp(0, 1).mul(255).round()[0]
print(tuple(tensorrt.shape), tensorrt.dtype, (tensorrt.float() - reference).abs().mean().item() < 1.0)
"""


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("tensorrt") is None, reason="needs CUDA and TensorRT"
)
def test_the_span_tensorrt_engine_matches_the_pytorch_network():
    from jasna.upscale.span import SPAN_WEIGHTS_PATH

    if not (REPO / SPAN_WEIGHTS_PATH).exists():
        pytest.skip("needs the SPAN weights in model_weights")
    result = subprocess.run(
        [sys.executable, "-c", SPAN_GPU_SCRIPT], capture_output=True, text=True, check=True, cwd=REPO
    )
    assert result.stdout.split("\n")[-2] == "(3, 72, 128) torch.uint8 True"


RTX_GPU_SCRIPT = """
import torch
from jasna.upscale.rtx import RtxFrameUpscaler

device = torch.device("cuda")
torch.manual_seed(0)
frame = (torch.rand(3, 270, 480) * 255).to(torch.uint8).to(device)
upscaler = RtxFrameUpscaler(
    device, source_size=(480, 270), output_size=(960, 540), quality="high", denoise="low", deblur="none", strength=1.0
)
expected = upscaler(frame)
with torch.cuda.stream(torch.cuda.Stream(device, priority=-1)):
    side = upscaler(frame)
torch.cuda.synchronize()
upscaler.close()
print(tuple(side.shape), side.dtype, torch.equal(side, expected))
"""


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("nvvfx") is None, reason="needs CUDA and nvvfx"
)
def test_rtx_upscaling_runs_from_a_side_stream():
    # A subprocess, since test_rtx_superres_restorer installs a fake nvvfx module.
    result = subprocess.run([sys.executable, "-c", RTX_GPU_SCRIPT], capture_output=True, text=True, check=True)
    assert result.stdout.split("\n")[-2] == "(3, 540, 960) torch.uint8 True"
