"""End-to-end tests using the real test clip assets/test_clip1_1080p.mp4."""
from __future__ import annotations

import json
import os
from importlib.util import find_spec
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest
import torch
from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange

from factories import session_config
from jasna.media.probe import get_video_meta_data

TEST_CLIP = Path("assets/test_clip1_1080p.mp4")
REQUIRES_TEST_CLIP = pytest.mark.skipif(not TEST_CLIP.exists(), reason="test clip not found")
REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

RESTORATION_MODEL_PTH = Path("model_weights/lada_mosaic_restoration_model_generic_v1.2.pth")
RFDETR_ONNX = Path("model_weights/rfdetr-v5.onnx")

REQUIRES_NVVFX = pytest.mark.skipif(find_spec("nvvfx") is None, reason="nvvfx (RTX Video Effects) not available")
REQUIRES_RFDETR = pytest.mark.skipif(not RFDETR_ONNX.exists(), reason="rfdetr-v5 ONNX not found")
REQUIRES_RESTORER_PTH = pytest.mark.skipif(
    not RESTORATION_MODEL_PTH.exists(),
    reason="restoration model PTH weights not found",
)

TVAI_FFMPEG_PATH = os.environ.get("TVAI_FFMPEG_PATH", r"C:\Program Files\Topaz Labs LLC\Topaz Video\ffmpeg.exe")

def _tvai_available() -> bool:
    return (
        bool(os.environ.get("TVAI_MODEL_DATA_DIR"))
        and bool(os.environ.get("TVAI_MODEL_DIR"))
        and Path(TVAI_FFMPEG_PATH).is_file()
    )

REQUIRES_TVAI = pytest.mark.skipif(not _tvai_available(), reason="TVAI environment not configured")
TVAI_ARGS = "preblur=0:noise=0:details=0:halo=0:blur=0:compression=0:estimate=8:blend=0.2:device=-2:vram=1:instances=1"

_TEMPORAL_OVERLAPS = [pytest.param(0, id="overlap0"), pytest.param(8, id="overlap8")]


@REQUIRES_TEST_CLIP
def test_metadata_of_the_test_clip():
    meta = get_video_meta_data(str(TEST_CLIP))

    assert (meta.video_width, meta.video_height, meta.codec_name, meta.num_frames, meta.is_10bit) == (
        1920, 1080, "hevc", 300, False
    )
    assert meta.video_fps == pytest.approx(30.0)
    assert isinstance(meta.video_fps_exact, Fraction) and isinstance(meta.time_base, Fraction)
    assert meta.duration > 0
    assert (meta.color_space, meta.color_range) == (AvColorspace.ITU709, AvColorRange.MPEG)


@REQUIRES_TEST_CLIP
@REQUIRES_CUDA
class TestVideoDecoderE2E:
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_decode_first_batch(self, batch_size):
        meta = get_video_meta_data(str(TEST_CLIP))
        from jasna.media.video_decoder import VideoReader

        device = torch.device("cuda:0")
        with VideoReader(str(TEST_CLIP), batch_size=batch_size, device=device, metadata=meta) as reader:
            frames, pts_list = next(iter(reader.frames()))

        assert frames.shape == (batch_size, 3, 1080, 1920)
        assert frames.dtype == torch.uint8
        assert frames.device.type == "cuda"
        assert len(pts_list) == batch_size

    def test_decode_all_frames_count(self):
        meta = get_video_meta_data(str(TEST_CLIP))
        from jasna.media.video_decoder import VideoReader

        device = torch.device("cuda:0")
        total = 0
        with VideoReader(str(TEST_CLIP), batch_size=8, device=device, metadata=meta) as reader:
            for frames, pts_list in reader.frames():
                total += len(pts_list)
        assert total == 300


@REQUIRES_TEST_CLIP
@REQUIRES_CUDA
class TestDetectionE2E:
    def test_rfdetr_detection_on_real_frames(self):
        from jasna.mosaic.detection_registry import (
            build_detection_model,
            detection_model_weights_path,
        )

        model_path = detection_model_weights_path("rfdetr-v5")
        if not model_path.exists():
            model_path = Path("model_weights") / "rfdetr-v5.onnx"
        if not model_path.exists():
            pytest.skip("rfdetr-v5 model weights not found")

        meta = get_video_meta_data(str(TEST_CLIP))
        from jasna.media.video_decoder import VideoReader

        device = torch.device("cuda:0")
        bs = 4
        model = build_detection_model(
            "rfdetr-v5",
            model_path,
            batch_size=bs,
            device=device,
            score_threshold=0.25,
            fp16=True,
        )

        with VideoReader(str(TEST_CLIP), batch_size=bs, device=device, metadata=meta) as reader:
            for frames, pts_list in reader.frames():
                det = model(frames, target_hw=(meta.video_height, meta.video_width))
                assert len(det.boxes_xyxy) == bs
                assert len(det.masks) == bs
                for boxes in det.boxes_xyxy:
                    assert boxes.ndim == 2
                    assert boxes.shape[1] == 4
                break


@REQUIRES_TEST_CLIP
@REQUIRES_CUDA
class TestEncoderE2E:
    def test_encode_decoded_frames(self, tmp_path):
        meta = get_video_meta_data(str(TEST_CLIP))
        from jasna.media.video_decoder import VideoReader
        from jasna.media.video_encoder import VideoEncoder

        device = torch.device("cuda:0")
        output_path = tmp_path / "out.mkv"

        with VideoReader(str(TEST_CLIP), batch_size=4, device=device, metadata=meta) as reader:
            with VideoEncoder(
                str(output_path),
                device=device,
                metadata=meta,
                codec="hevc",
                encoder_settings={},
            ) as encoder:
                count = 0
                for frames, pts_list in reader.frames():
                    for i, pts in enumerate(pts_list):
                        encoder.encode(frames[i], pts)
                        count += 1
                    if count >= 30:
                        break

        assert output_path.exists()
        out_meta = get_video_meta_data(str(output_path))
        assert out_meta.video_width == 1920
        assert out_meta.video_height == 1080
        assert out_meta.num_frames >= 20


def _changed_share_per_frame(source: Path, output: Path) -> np.ndarray:
    """Per frame, the share of pixels whose output differs from the source by more than
    encoding noise does."""
    with av.open(str(source)) as source_container, av.open(str(output)) as output_container:
        shares = [
            float((np.abs(a.to_ndarray(format="gray").astype(np.int16) - b.to_ndarray(format="gray")) > 48).mean())
            for a, b in zip(source_container.decode(video=0), output_container.decode(video=0), strict=True)
        ]
    return np.array(shares)


@REQUIRES_TEST_CLIP
@REQUIRES_CUDA
@REQUIRES_RFDETR
@REQUIRES_RESTORER_PTH
class TestFullPipelineE2E:
    """Restores the test clip through the CLI and GUI path: ``SessionConfig`` ->
    ``build_restoration_session`` -> ``build_pipeline``."""

    def _restore(self, tmp_path, **overrides) -> Path:
        from jasna.session_factory import build_pipeline, build_restoration_session

        output = tmp_path / "output.mkv"
        config = session_config(
            device="cuda:0",
            batch_size=4,
            detection_model_name="rfdetr-v5",
            detection_model_path=RFDETR_ONNX,
            restoration_model_path=RESTORATION_MODEL_PTH,
            compile_basicvsrpp=False,
            max_clip_size=60,
            max_detection_gap=0,
            min_detection_duration=0,
            codec="hevc",
            encoder_settings={},
            disable_progress=True,
            **overrides,
        )
        session = build_restoration_session(config, log_callback=None)
        try:
            build_pipeline(config, session, TEST_CLIP, output).run()
        finally:
            session.close()
        return output

    def _assert_restored_video(self, output: Path) -> None:
        meta = get_video_meta_data(str(output))
        assert (meta.num_frames, meta.video_width, meta.video_height, meta.codec_name) == (300, 1920, 1080, "hevc")
        assert meta.video_fps == pytest.approx(30.0)

        changed = _changed_share_per_frame(TEST_CLIP, output)
        assert changed.max() > 0.001, "no mosaic was restored"
        assert np.median(changed) < 0.05, "restoration changed more than the mosaic areas"

    def test_normal_restoration(self, tmp_path):
        output = self._restore(tmp_path)

        self._assert_restored_video(output)
        with av.open(str(output)) as container:
            (comment,) = [value for key, value in container.metadata.items() if key.lower() == "comment"]
        tag = json.loads(comment)
        assert tag["basicvsrpp"]["max_clip_size"] == 60
        assert tag["detection"]["threshold"] == 0.25
        assert "ltx" not in tag

    @REQUIRES_TVAI
    @pytest.mark.parametrize("overlap", _TEMPORAL_OVERLAPS)
    def test_tvai_secondary(self, tmp_path, overlap):
        output = self._restore(
            tmp_path,
            secondary_restoration="tvai",
            tvai_ffmpeg_path=TVAI_FFMPEG_PATH,
            tvai_model="iris-2",
            tvai_scale=1,
            tvai_args=TVAI_ARGS,
            tvai_workers=2,
            temporal_overlap=overlap,
            denoise_strength="low",
        )

        self._assert_restored_video(output)

    @REQUIRES_NVVFX
    @pytest.mark.parametrize("overlap", _TEMPORAL_OVERLAPS)
    def test_rtx_secondary(self, tmp_path, overlap):
        output = self._restore(
            tmp_path,
            secondary_restoration="rtx-super-res",
            rtx_scale=4,
            rtx_quality="high",
            rtx_denoise="medium",
            temporal_overlap=overlap,
            denoise_strength="low",
        )

        self._assert_restored_video(output)
