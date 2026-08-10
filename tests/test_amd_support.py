from __future__ import annotations

import sys
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def test_amf_hevc_uses_compatible_defaults() -> None:
    from jasna.media.video_encoder import AMF_ENCODER_SPECS

    options = AMF_ENCODER_SPECS["hevc"].default_options
    assert options["rc"] == "cqp"
    assert options["preanalysis"] == "0"
    assert options["vbaq"] == "0"
    assert options["qp_i"] == "25"
    assert options["qp_p"] == "25"
    assert "qvbr_quality_level" not in options


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


def test_amf_hevc_maps_cq_to_constant_qp(monkeypatch, tmp_path) -> None:
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
        codec="hevc",
        encoder_settings={"cq": 21},
    )
    assert encoder.spec.frame_format == "p010le"
    assert encoder.encoder_options["rc"] == "cqp"
    assert encoder.encoder_options["qp_i"] == "21"
    assert encoder.encoder_options["qp_p"] == "21"
    assert "cq" not in encoder.encoder_options
    assert "qvbr_quality_level" not in encoder.encoder_options


def test_amf_hevc_cqp_skips_source_bitrate_cap(monkeypatch, tmp_path) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    encoder = module.NvidiaVideoEncoder(
        str(tmp_path / "out.mp4"),
        torch.device("cuda:0"),
        replace(_metadata(), video_bitrate=20_000_000),
        codec="hevc",
        encoder_settings={"cq": 21},
    )
    assert "maxrate" not in encoder.encoder_options
    assert "bufsize" not in encoder.encoder_options


@pytest.mark.parametrize("rc", ["qvbr", "hqvbr", 4, 5])
def test_amf_hevc_rejects_qvbr_for_main10(
    monkeypatch, tmp_path, rc: str | int
) -> None:
    import jasna.media.video_encoder as module

    monkeypatch.setattr(
        module,
        "vendor_for_device",
        lambda _device: AcceleratorVendor.AMD,
    )
    with pytest.raises(ValueError, match="AMD HEVC Main10.*QVBR"):
        module.NvidiaVideoEncoder(
            str(tmp_path / "out.mp4"),
            torch.device("cuda:0"),
            _metadata(),
            codec="hevc",
            encoder_settings={"cq": 21, "rc": rc},
        )


def test_amf_hevc_8bit_allows_qvbr(monkeypatch, tmp_path) -> None:
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
        codec="hevc",
        encoder_settings={"cq": 21, "rc": "qvbr"},
        match_input_bit_depth=True,
    )
    assert encoder.spec.frame_format == "nv12"
    assert encoder.encoder_options["rc"] == "qvbr"
    assert encoder.encoder_options["qvbr_quality_level"] == "21"
    assert "qp_i" not in encoder.encoder_options
    assert "qp_p" not in encoder.encoder_options


def test_amf_p010_host_input_reinterprets_signed_storage() -> None:
    import jasna.media.video_encoder as module

    packed = torch.tensor([-32768, -1, 0, 32767], dtype=torch.int16)
    host_input = module._amf_host_input(packed, ten_bit=True)

    assert host_input.dtype is torch.uint16
    assert torch.equal(host_input, packed.view(torch.uint16))


@pytest.mark.parametrize("codec", ["h264", "hevc", "av1"])
def test_smart_render_uses_amf_fragment_options(
    monkeypatch,
    tmp_path,
    codec: str,
) -> None:
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
        codec=codec,
        encoder_settings={},
        smart_fragment=True,
    )

    assert encoder.encoder_name == f"{codec}_amf"
    assert encoder.encoder_options["forced_idr"] == "1"
    assert "forced-idr" not in encoder.encoder_options


def test_amf_h264_smart_settings_are_accepted() -> None:
    settings = {"bf": 3, "bf_ref": 1, "pa_adaptive_mini_gop": 0}

    assert validate_encoder_settings(
        settings,
        codec="h264",
        vendor=AcceleratorVendor.AMD,
    ) == settings


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


def test_amf_decoder_survives_pyav18_time_base_regression(monkeypatch) -> None:
    import jasna.media.video_decoder as module

    class FakeDecoder:
        def __init__(self):
            object.__setattr__(self, "opened", False)

        def __setattr__(self, name, value):
            if name == "time_base":
                raise RuntimeError("Cannot access 'time_base' as a decoder")
            object.__setattr__(self, name, value)

        def open(self, strict=False):
            object.__setattr__(self, "opened", True)

    decoder = FakeDecoder()
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
        name="hevc",
        extradata=b"header",
        width=16,
        height=16,
        time_base=Fraction(1, 30),
        framerate=Fraction(30, 1),
        sample_aspect_ratio=Fraction(1, 1),
        thread_type=None,
    )
    reader._setup_amf_decoder(source)
    assert decoder.opened is True
    assert reader._decoder_ctx is decoder
    assert reader._amd_hardware_decode is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_yuv_eager_converter_runs_on_gpu_planes(monkeypatch) -> None:
    import jasna.media.yuv_to_rgb as module

    monkeypatch.setattr(module, "is_nvidia_device", lambda _device: False)
    H = W = 16
    generator = torch.Generator().manual_seed(0)
    y = torch.randint(16, 236, (H, W), dtype=torch.uint8, generator=generator)
    uv = torch.randint(16, 240, (H // 2, W // 2, 2), dtype=torch.uint8, generator=generator)

    cpu = module.YuvToRgbConverter(
        H, W, AvColorspace.ITU709, False, False, torch.device("cpu")
    )
    expected = torch.empty((3, H, W), dtype=torch.uint8)
    cpu.convert_into(y, uv, expected)

    gpu = module.YuvToRgbConverter(
        H, W, AvColorspace.ITU709, False, False, torch.device("cuda:0")
    )
    out = torch.empty((3, H, W), dtype=torch.uint8, device="cuda:0")
    gpu.convert_into(y.cuda(), uv.cuda(), out)

    assert (out.cpu().int() - expected.int()).abs().max() <= 1


def test_amf_8bit_downgrade_drops_bitdepth(monkeypatch, tmp_path) -> None:
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
        codec="hevc",
        encoder_settings={},
        match_input_bit_depth=True,
        smart_fragment=True,
    )
    assert encoder.spec.frame_format == "nv12"
    assert encoder.encoder_options["profile"] == "main"
    assert encoder.encoder_options["forced_idr"] == "1"
    assert "bitdepth" not in encoder.encoder_options

def test_rfdetr_torch_runner_maps_outputs(monkeypatch, tmp_path) -> None:
    import jasna.mosaic.rfdetr_torch_runner as module

    weights = tmp_path / "rfdetr-v6.pt"
    weights.write_bytes(b"pt")

    class FakeCore:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, x):
            batch = x.shape[0]
            return {
                "pred_boxes": torch.zeros(batch, 5, 4),
                "pred_logits": torch.zeros(batch, 5, 3),
                "pred_masks": torch.zeros(batch, 5, 8, 8),
            }

    class FakeModel:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.model = SimpleNamespace(model=FakeCore())

    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace(RFDETRSegMedium=FakeModel))
    monkeypatch.setattr(
        module.torch,
        "load",
        lambda *_a, **_k: {"model": {"class_embed.weight": torch.zeros(3, 256)}},
    )

    runner = module.RfDetrTorchRunner(
        weights,
        input_shapes=[(2, 3, 576, 576)],
        device=torch.device("cpu"),
        fp16=False,
        resolution=576,
        variant="medium",
    )

    assert runner.input_names == ["input"]
    assert runner.input_dtypes == {"input": torch.float32}
    assert runner.output_names == ["dets", "labels", "masks"]
    assert runner.outputs["dets"].ndim == 3
    assert runner.outputs["dets"].shape[-1] == 4
    assert runner.outputs["labels"].ndim == 3
    assert runner.outputs["masks"].ndim == 4
    # num_classes derived from class_embed rows - 1 (rfdetr adds a reserve slot).
    assert runner._wrapper.kwargs["num_classes"] == 2

    out = runner.infer({"input": torch.zeros(2, 3, 576, 576)})
    assert set(out) == {"dets", "labels", "masks"}
    assert out["dets"].shape == (2, 5, 4)
    assert out["labels"].shape == (2, 5, 3)
    assert out["masks"].shape == (2, 5, 8, 8)

    runner.close()
    with pytest.raises(RuntimeError, match="closed"):
        runner.infer({"input": torch.zeros(2, 3, 576, 576)})


def test_rfdetr_torch_runner_rejects_unknown_variant(monkeypatch, tmp_path) -> None:
    import jasna.mosaic.rfdetr_torch_runner as module

    weights = tmp_path / "rfdetr-v6.pt"
    weights.write_bytes(b"pt")
    monkeypatch.setitem(sys.modules, "rfdetr", SimpleNamespace())

    with pytest.raises(RuntimeError, match="unsupported variant"):
        module.RfDetrTorchRunner(
            weights,
            input_shapes=[(1, 3, 576, 576)],
            device=torch.device("cpu"),
            fp16=False,
            resolution=576,
            variant="mystery",
        )


def test_amd_rfdetr_needs_no_detection_engine(monkeypatch) -> None:
    import jasna.accelerator as accelerator
    import jasna.engine_compiler as compiler

    monkeypatch.setattr(accelerator, "is_amd_device", lambda _device=None: True)

    assert compiler._detection_engine_exists(
        "rfdetr-v6",
        "rfdetr-v6.pt",
        batch_size=4,
        fp16=True,
        device="cpu",
    )
