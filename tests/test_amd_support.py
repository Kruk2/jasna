from __future__ import annotations

import sys
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from jasna.accelerator import (
    AcceleratorVendor,
    capabilities_for_device,
    vendor_for_device,
)
from jasna.media import VideoMetadata, validate_encoder_settings


def _metadata() -> VideoMetadata:
    return VideoMetadata(
        video_file="input.mp4",
        video_height=16,
        video_width=16,
        video_fps=30.0,
        average_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        codec_name="h264",
        duration=1.0,
        time_base=Fraction(1, 30),
        start_pts=0,
        color_range=AvColorRange.MPEG,
        color_space=AvColorspace.ITU709,
        num_frames=30,
        is_10bit=False,
    )


def test_rocm_uses_cuda_device_api_but_reports_amd(monkeypatch) -> None:
    monkeypatch.setattr(torch.version, "hip", "7.2.1")
    assert vendor_for_device("cuda:0") is AcceleratorVendor.AMD
    capabilities = capabilities_for_device("cuda:0")
    assert capabilities.migraphx is True
    assert capabilities.amf is True
    assert capabilities.tensorrt is False
    assert capabilities.nvcodec is False


def test_amd_basicvsrpp_skips_tensorrt_compilation(monkeypatch) -> None:
    import jasna.accelerator as accelerator
    import jasna.engine_compiler as compiler

    monkeypatch.setattr(accelerator, "is_nvidia_device", lambda _device: False)
    monkeypatch.setattr(accelerator, "is_amd_device", lambda _device: True)
    monkeypatch.setattr(
        compiler,
        "_basicvsrpp_engines_exist",
        MagicMock(side_effect=AssertionError("TensorRT probe on AMD")),
    )
    result = compiler.ensure_engines_compiled(
        compiler.EngineCompilationRequest(
            device="cuda:0",
            fp16=True,
            basicvsrpp=True,
            basicvsrpp_model_path="model.pth",
        )
    )
    assert result.use_basicvsrpp_tensorrt is False


def test_amf_encoder_settings_are_vendor_specific() -> None:
    assert validate_encoder_settings(
        {"preanalysis": 1, "cq": 24},
        codec="h264",
        vendor=AcceleratorVendor.AMD,
    ) == {"preanalysis": 1, "cq": 24}
    with pytest.raises(ValueError, match="temporal-aq"):
        validate_encoder_settings(
            {"temporal-aq": 1},
            codec="h264",
            vendor=AcceleratorVendor.AMD,
        )


def test_video_encoder_selects_amf_and_normalizes_cq(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        _metadata(),
        codec="h264",
        encoder_settings={"cq": 21},
    )
    assert encoder.encoder_name == "h264_amf"
    assert encoder.spec.frame_format == "nv12"
    assert encoder.encoder_options["qvbr_quality_level"] == "21"
    assert "cq" not in encoder.encoder_options


def test_amf_p010_host_input_reinterprets_signed_storage() -> None:
    import jasna.media.video_encoder as module

    packed = torch.tensor([-32768, -1, 0, 32767], dtype=torch.int16)
    host_input = module._amf_host_input(packed, ten_bit=True)

    assert host_input.dtype is torch.uint16
    assert torch.equal(host_input, packed.view(torch.uint16))


def test_smart_render_is_rejected_on_amd(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    with pytest.raises(ValueError, match="only with NVENC"):
        module.NvidiaVideoEncoder(
            str(tmp_path / "out.mp4"),
            torch.device("cuda:0"),
            _metadata(),
            codec="h264",
            encoder_settings={},
            smart_fragment=True,
        )


def test_streaming_encoder_selects_amf(monkeypatch, tmp_path) -> None:
    import jasna.streaming_encoder as module

    monkeypatch.setattr(module, "find_executable", lambda _name: "/ffmpeg")
    popen = MagicMock()
    popen.stderr = []
    monkeypatch.setattr(module.subprocess, "Popen", MagicMock(return_value=popen))
    encoder = module.StreamingEncoder(
        tmp_path,
        4.0,
        _metadata(),
        "missing.mp4",
        torch.device("cuda:0"),
    )
    encoder._vendor = AcceleratorVendor.AMD
    encoder._launch_ffmpeg(0)
    cmd = module.subprocess.Popen.call_args.args[0]
    assert cmd[cmd.index("-c:v") + 1] == "h264_amf"
    assert "-qvbr_quality_level" in cmd
    assert "h264_nvenc" not in cmd


def test_amf_decoder_context_is_created(monkeypatch) -> None:
    import jasna.media.video_decoder as module

    decoder = MagicMock()
    monkeypatch.setattr(
        module.av,
        "CodecContext",
        SimpleNamespace(create=MagicMock(return_value=decoder)),
    )
    reader = module.NvidiaVideoReader(
        "input.mp4",
        4,
        torch.device("cuda:0"),
        _metadata(),
    )
    source = SimpleNamespace(
        name="h264",
        extradata=b"header",
        width=16,
        height=16,
        time_base=Fraction(1, 30),
        framerate=Fraction(30, 1),
        sample_aspect_ratio=Fraction(1, 1),
        thread_type=None,
    )
    reader._setup_amf_decoder(source)
    create = module.av.CodecContext.create
    assert create.call_args.args[:2] == ("h264_amf", "r")
    decoder.open.assert_called_once_with(strict=False)
    assert reader._decoder_ctx is decoder
    assert reader._amd_hardware_decode is True


def test_migraphx_runner_provider_and_tensor_bridge(monkeypatch, tmp_path) -> None:
    import jasna.mosaic.migraphx_runner as module

    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx")
    input_node = SimpleNamespace(
        name="images",
        shape=[1, 3, 4, 4],
        type="tensor(float)",
    )
    output_node = SimpleNamespace(
        name="scores",
        shape=[1, 2],
        type="tensor(float)",
    )

    class FakeSession:
        def __init__(self, *_args, providers, **_kwargs):
            self.providers_arg = providers

        def get_providers(self):
            return ["MIGraphXExecutionProvider", "CPUExecutionProvider"]

        def get_inputs(self):
            return [input_node]

        def get_outputs(self):
            return [output_node]

        def run(self, names, feeds):
            assert names == ["scores"]
            assert feeds["images"].shape == (1, 3, 4, 4)
            return [np.array([[0.25, 0.75]], dtype=np.float32)]

    fake_ort = SimpleNamespace(
        get_available_providers=lambda: [
            "MIGraphXExecutionProvider",
            "CPUExecutionProvider",
        ],
        SessionOptions=lambda: SimpleNamespace(graph_optimization_level=None),
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=99),
        InferenceSession=FakeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(module, "_gpu_arch", lambda _device: "gfx-test")
    monkeypatch.setattr(module, "device_name", lambda _device: "AMD test GPU")

    runner = module.MigraphxRunner(
        model,
        input_shapes=[(1, 3, 4, 4)],
        device=torch.device("cpu"),
        fp16=True,
    )
    provider, options = runner.session.providers_arg[0]
    assert provider == "MIGraphXExecutionProvider"
    assert options["migraphx_fp16_enable"] == "1"
    assert options["migraphx_model_cache_dir"] == str(runner.cache_dir)
    result = runner.infer({"images": torch.ones(1, 3, 4, 4)})
    assert torch.equal(result["scores"], torch.tensor([[0.25, 0.75]]))


def test_migraphx_model_digest_is_cached_for_unchanged_file(tmp_path) -> None:
    import jasna.mosaic.migraphx_runner as module

    model = tmp_path / "model.onnx"
    model.write_bytes(b"onnx")
    module._cached_model_digest.cache_clear()

    assert module._model_digest(model) == module._model_digest(model)
    assert module._cached_model_digest.cache_info().hits == 1
