"""End-to-end tests using the real test clip assets/test_clip1_1080p.mp4."""
from __future__ import annotations

import os
import threading
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

from jasna.media import get_video_meta_data, VideoMetadata

TEST_CLIP = Path("assets/test_clip1_1080p.mp4")
REQUIRES_TEST_CLIP = pytest.mark.skipif(not TEST_CLIP.exists(), reason="test clip not found")
REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")

RESTORATION_MODEL_PTH = Path("model_weights/lada_mosaic_restoration_model_generic_v1.2.pth")
RESTORATION_ENGINE_CLIP10 = Path("model_weights/lada_mosaic_restoration_model_generic_v1.2_clip10.trt_fp16.win.engine")
RFDETR_ONNX = Path("model_weights/rfdetr-v5.onnx")

TVAI_FFMPEG_PATH = os.environ.get("TVAI_FFMPEG_PATH", r"C:\Program Files\Topaz Labs LLC\Topaz Video\ffmpeg.exe")

def _tvai_available() -> bool:
    return (
        bool(os.environ.get("TVAI_MODEL_DATA_DIR"))
        and bool(os.environ.get("TVAI_MODEL_DIR"))
        and Path(os.environ.get("TVAI_MODEL_DATA_DIR", "")).is_dir()
        and Path(os.environ.get("TVAI_MODEL_DIR", "")).is_dir()
        and Path(TVAI_FFMPEG_PATH).is_file()
    )

def _nvvfx_available() -> bool:
    try:
        import nvvfx  # noqa: F401
        return True
    except ImportError:
        return False

REQUIRES_TVAI = pytest.mark.skipif(not _tvai_available(), reason="TVAI environment not available")
REQUIRES_NVVFX = pytest.mark.skipif(not _nvvfx_available(), reason="nvvfx (RTX Video Effects) not available")
REQUIRES_RFDETR = pytest.mark.skipif(not RFDETR_ONNX.exists(), reason="rfdetr-v5 ONNX not found")
REQUIRES_RESTORER = pytest.mark.skipif(
    not (RESTORATION_ENGINE_CLIP10.exists() or RESTORATION_MODEL_PTH.exists()),
    reason="restoration model weights not found",
)
REQUIRES_RESTORER_PTH = pytest.mark.skipif(
    not RESTORATION_MODEL_PTH.exists(),
    reason="restoration model PTH weights not found",
)


@REQUIRES_TEST_CLIP
class TestVideoMetadataE2E:
    def test_metadata_basic(self):
        meta = get_video_meta_data(str(TEST_CLIP))
        assert meta.video_width == 1920
        assert meta.video_height == 1080
        assert meta.video_fps == pytest.approx(30.0)
        assert meta.codec_name == "hevc"
        assert meta.num_frames == 300
        assert meta.is_10bit is False

    def test_metadata_types(self):
        meta = get_video_meta_data(str(TEST_CLIP))
        assert isinstance(meta.video_fps_exact, Fraction)
        assert isinstance(meta.time_base, Fraction)
        assert isinstance(meta.duration, float)
        assert meta.duration > 0

    def test_metadata_color(self):
        from av.video.reformatter import Colorspace as AvColorspace, ColorRange as AvColorRange
        meta = get_video_meta_data(str(TEST_CLIP))
        assert meta.color_space == AvColorspace.ITU709
        assert meta.color_range == AvColorRange.MPEG


@REQUIRES_TEST_CLIP
@REQUIRES_CUDA
class TestVideoDecoderE2E:
    def test_decode_first_batch(self):
        meta = get_video_meta_data(str(TEST_CLIP))
        from jasna.media.video_decoder import NvidiaVideoReader

        device = torch.device("cuda:0")
        with NvidiaVideoReader(str(TEST_CLIP), batch_size=4, device=device, metadata=meta) as reader:
            for frames, pts_list in reader.frames():
                assert frames.shape == (4, 3, 1080, 1920)
                assert frames.dtype == torch.uint8
                assert frames.device.type == "cuda"
                assert len(pts_list) == 4
                break

    def test_decode_all_frames_count(self):
        meta = get_video_meta_data(str(TEST_CLIP))
        from jasna.media.video_decoder import NvidiaVideoReader

        device = torch.device("cuda:0")
        total = 0
        with NvidiaVideoReader(str(TEST_CLIP), batch_size=8, device=device, metadata=meta) as reader:
            for frames, pts_list in reader.frames():
                total += len(pts_list)
        assert total == 300

    def test_decode_batch_size_1(self):
        meta = get_video_meta_data(str(TEST_CLIP))
        from jasna.media.video_decoder import NvidiaVideoReader

        device = torch.device("cuda:0")
        with NvidiaVideoReader(str(TEST_CLIP), batch_size=1, device=device, metadata=meta) as reader:
            for frames, pts_list in reader.frames():
                assert frames.shape[0] == 1
                assert len(pts_list) == 1
                break


@REQUIRES_TEST_CLIP
@REQUIRES_CUDA
class TestDetectionE2E:
    def test_rfdetr_detection_on_real_frames(self):
        from jasna.mosaic.detection_registry import detection_model_weights_path

        model_path = detection_model_weights_path("rfdetr-v5")
        if not model_path.exists():
            model_path = Path("model_weights") / "rfdetr-v5.onnx"
        if not model_path.exists():
            pytest.skip("rfdetr-v5 model weights not found")

        from jasna.mosaic.rfdetr import RfDetrMosaicDetectionModel

        meta = get_video_meta_data(str(TEST_CLIP))
        from jasna.media.video_decoder import NvidiaVideoReader

        device = torch.device("cuda:0")
        bs = 4
        model = RfDetrMosaicDetectionModel(
            onnx_path=model_path,
            batch_size=bs,
            device=device,
            fp16=True,
        )

        with NvidiaVideoReader(str(TEST_CLIP), batch_size=bs, device=device, metadata=meta) as reader:
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
        from jasna.media.video_decoder import NvidiaVideoReader
        from jasna.media.video_encoder import NvidiaVideoEncoder

        device = torch.device("cuda:0")
        output_path = tmp_path / "out.mkv"

        with NvidiaVideoReader(str(TEST_CLIP), batch_size=4, device=device, metadata=meta) as reader:
            with NvidiaVideoEncoder(
                str(output_path),
                device=device,
                metadata=meta,
                codec="hevc",
                encoder_settings={},
                stream_mode=False,
                working_directory=tmp_path,
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


# ---------------------------------------------------------------------------
# Spy helpers — wrap real methods to count calls, no mocking
# ---------------------------------------------------------------------------


class _DetectionSpy:
    """Wraps a detection model callable to count batches and detected mosaics."""

    def __init__(self, model):
        self._model = model
        self.batch_count = 0
        self.total_detections = 0
        self.frames_with_detections = 0
        self._lock = threading.Lock()

    def __call__(self, frames, *, target_hw):
        result = self._model(frames, target_hw=target_hw)
        with self._lock:
            self.batch_count += 1
            for boxes in result.boxes_xyxy:
                if len(boxes) > 0:
                    self.frames_with_detections += 1
                    self.total_detections += len(boxes)
        return result


class _MethodSpy:
    """Counts calls to an instance method while executing the original."""

    def __init__(self, obj, method_name):
        self._original = getattr(obj, method_name)
        self.call_count = 0
        self._lock = threading.Lock()
        setattr(obj, method_name, self)

    def __call__(self, *args, **kwargs):
        result = self._original(*args, **kwargs)
        with self._lock:
            self.call_count += 1
        return result


# ---------------------------------------------------------------------------
# Full Pipeline.run() E2E tests — real video, real models, no mocking
# ---------------------------------------------------------------------------


@REQUIRES_TEST_CLIP
@REQUIRES_CUDA
@REQUIRES_RFDETR
@REQUIRES_RESTORER_PTH
class TestFullPipelineE2E:

    EXPECTED_FRAMES = 300
    BATCH_SIZE = 4

    def _build(self, tmp_path, *, secondary_restorer=None):
        from jasna.pipeline import Pipeline
        from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
        from jasna.restorer.restoration_pipeline import RestorationPipeline

        device = torch.device("cuda:0")
        output = tmp_path / "output.mkv"

        restorer = BasicvsrppMosaicRestorer(
            checkpoint_path=str(RESTORATION_MODEL_PTH),
            device=device,
            max_clip_size=60,
            use_tensorrt=False,
            fp16=True,
        )
        rp = RestorationPipeline(restorer=restorer, secondary_restorer=secondary_restorer)

        pipeline = Pipeline(
            input_video=TEST_CLIP,
            output_video=output,
            detection_model_name="rfdetr-v5",
            detection_model_path=RFDETR_ONNX,
            detection_score_threshold=0.25,
            restoration_pipeline=rp,
            codec="hevc",
            encoder_settings={},
            batch_size=self.BATCH_SIZE,
            device=device,
            max_clip_size=60,
            temporal_overlap=8,
            enable_crossfade=True,
            fp16=True,
            disable_progress=True,
            working_directory=tmp_path,
        )

        det_spy = _DetectionSpy(pipeline.detection_model)
        pipeline.detection_model = det_spy

        primary_spy = _MethodSpy(rp, "prepare_and_run_primary")
        secondary_spy = _MethodSpy(rp, "run_secondary_from_primary")

        return pipeline, output, det_spy, primary_spy, secondary_spy

    def _assert_output_video(self, output):
        assert output.exists()
        assert output.stat().st_size > 1024, "output video suspiciously small"

        meta = get_video_meta_data(str(output))
        assert meta.num_frames == self.EXPECTED_FRAMES
        assert meta.video_width == 1920
        assert meta.video_height == 1080
        assert meta.codec_name == "hevc"
        assert meta.video_fps == pytest.approx(30.0)

        from jasna.media.video_decoder import NvidiaVideoReader

        device = torch.device("cuda:0")
        with NvidiaVideoReader(str(output), batch_size=4, device=device, metadata=meta) as reader:
            for frames, pts_list in reader.frames():
                assert frames.dtype == torch.uint8
                assert frames.any(), "decoded output frames should not be all black"
                break

        return meta

    def test_normal_restoration(self, tmp_path):
        pipeline, output, det_spy, primary_spy, secondary_spy = self._build(tmp_path)
        pipeline.run()

        self._assert_output_video(output)

        expected_batches = self.EXPECTED_FRAMES // self.BATCH_SIZE
        assert det_spy.batch_count == expected_batches
        assert det_spy.total_detections > 0, "test clip should contain detectable mosaics"
        assert det_spy.frames_with_detections > 0

        assert primary_spy.call_count > 0, "at least one clip should be restored"
        assert secondary_spy.call_count == primary_spy.call_count

    @REQUIRES_TVAI
    def test_tvai_secondary(self, tmp_path):
        from jasna.restorer.tvai_secondary_restorer import TvaiSecondaryRestorer

        tvai = TvaiSecondaryRestorer(
            ffmpeg_path=TVAI_FFMPEG_PATH,
            tvai_args="model=iris-2:scale=1",
            scale=1,
            num_workers=1,
        )
        try:
            pipeline, output, det_spy, primary_spy, secondary_spy = self._build(
                tmp_path, secondary_restorer=tvai,
            )
            tvai_restore_spy = _MethodSpy(tvai, "restore")
            pipeline.run()
        finally:
            tvai.close()

        self._assert_output_video(output)

        assert det_spy.total_detections > 0
        assert primary_spy.call_count > 0
        assert secondary_spy.call_count == primary_spy.call_count
        assert tvai_restore_spy.call_count == primary_spy.call_count

    @REQUIRES_NVVFX
    def test_rtx_superres_secondary(self, tmp_path):
        from jasna.restorer.rtx_superres_secondary_restorer import RtxSuperresSecondaryRestorer

        rtx = RtxSuperresSecondaryRestorer(
            device=torch.device("cuda:0"),
            scale=4,
            quality="high",
            denoise="medium",
        )
        try:
            pipeline, output, det_spy, primary_spy, secondary_spy = self._build(
                tmp_path, secondary_restorer=rtx,
            )
            rtx_restore_spy = _MethodSpy(rtx, "restore")
            pipeline.run()
        finally:
            rtx.close()

        self._assert_output_video(output)

        assert det_spy.total_detections > 0
        assert primary_spy.call_count > 0
        assert secondary_spy.call_count == primary_spy.call_count
        assert rtx_restore_spy.call_count == primary_spy.call_count


# ---------------------------------------------------------------------------
# Helpers for restoration pipeline E2E tests
# ---------------------------------------------------------------------------

NUM_CLIP_FRAMES = 6

def _make_restorer(device: torch.device):
    from jasna.restorer.basicvsrpp_mosaic_restorer import BasicvsrppMosaicRestorer
    return BasicvsrppMosaicRestorer(
        checkpoint_path=str(RESTORATION_MODEL_PTH),
        device=device, max_clip_size=10, use_tensorrt=False, fp16=True,
    )


def _decode_frames(device: torch.device, n: int = NUM_CLIP_FRAMES) -> list[torch.Tensor]:
    from jasna.media.video_decoder import NvidiaVideoReader
    meta = get_video_meta_data(str(TEST_CLIP))
    frames: list[torch.Tensor] = []
    with NvidiaVideoReader(str(TEST_CLIP), batch_size=n, device=device, metadata=meta) as reader:
        for batch, pts_list in reader.frames():
            for i in range(len(pts_list)):
                frames.append(batch[i])
            break
    return frames[:n]


def _make_synthetic_clip(n: int, h: int, w: int, device: torch.device):
    from jasna.tracking.clip_tracker import TrackedClip
    cx, cy, half = w // 2, h // 2, 80
    bbox = np.array([cx - half, cy - half, cx + half, cy + half], dtype=np.float32)
    mask = torch.zeros(h // 4, w // 4, dtype=torch.bool, device=device)
    mask[cy // 4 - half // 4 : cy // 4 + half // 4, cx // 4 - half // 4 : cx // 4 + half // 4] = True
    return TrackedClip(
        track_id=0, start_frame=0,
        mask_resolution=(h // 4, w // 4),
        bboxes=[bbox] * n,
        masks=[mask] * n,
    )


# ---------------------------------------------------------------------------
# RestorationPipeline E2E with TVAI secondary
# ---------------------------------------------------------------------------

@REQUIRES_TEST_CLIP
@REQUIRES_CUDA
@REQUIRES_RESTORER
@REQUIRES_TVAI
class TestRestorationPipelineTvaiE2E:
    def test_restore_clip(self):
        from jasna.restorer.tvai_secondary_restorer import TvaiSecondaryRestorer
        from jasna.restorer.restoration_pipeline import RestorationPipeline

        device = torch.device("cuda:0")
        frames = _decode_frames(device)
        n = len(frames)
        h, w = frames[0].shape[1], frames[0].shape[2]
        clip = _make_synthetic_clip(n, h, w, device)

        restorer = _make_restorer(device)
        tvai = TvaiSecondaryRestorer(
            ffmpeg_path=TVAI_FFMPEG_PATH, tvai_args="model=iris-2:scale=1",
            scale=1, num_workers=1,
        )
        try:
            pipeline = RestorationPipeline(restorer=restorer, secondary_restorer=tvai)
            result = pipeline.restore_clip(clip, frames, keep_start=0, keep_end=n)

            assert len(result.restored_frames) == n
            for f in result.restored_frames:
                assert f.dtype == torch.uint8
                assert f.dim() == 3
            assert result.frame_shape == (h, w)
        finally:
            tvai.close()


# ---------------------------------------------------------------------------
# RestorationPipeline E2E with RTX Super Res secondary
# ---------------------------------------------------------------------------

@REQUIRES_TEST_CLIP
@REQUIRES_CUDA
@REQUIRES_RESTORER
@REQUIRES_NVVFX
class TestRestorationPipelineRtxE2E:
    def test_restore_clip(self):
        from jasna.restorer.rtx_superres_secondary_restorer import RtxSuperresSecondaryRestorer
        from jasna.restorer.restoration_pipeline import RestorationPipeline

        device = torch.device("cuda:0")
        frames = _decode_frames(device)
        n = len(frames)
        h, w = frames[0].shape[1], frames[0].shape[2]
        clip = _make_synthetic_clip(n, h, w, device)

        restorer = _make_restorer(device)
        rtx = RtxSuperresSecondaryRestorer(
            device=device, scale=4, quality="high", denoise="medium",
        )
        try:
            pipeline = RestorationPipeline(restorer=restorer, secondary_restorer=rtx)
            result = pipeline.restore_clip(clip, frames, keep_start=0, keep_end=n)

            assert len(result.restored_frames) == n
            for f in result.restored_frames:
                assert f.dtype == torch.uint8
                assert f.dim() == 3
            assert result.frame_shape == (h, w)
        finally:
            rtx.close()
