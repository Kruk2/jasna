from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

from jasna.crop_buffer import compute_enlarged_bbox
from jasna.media import VideoMetadata
from jasna.mosaic.detections import Detections
from jasna.vr180 import (
    FISHEYE_STUDIO_TOKENS,
    DIRECT_STUDIO_TOKENS,
    GnomonicProjector,
    SbsDetectionAdapter,
    resolve_vr_mode,
)


def _metadata(
    *,
    width: int = 3840,
    height: int = 1920,
    sample_aspect_ratio: Fraction = Fraction(1, 1),
    stereo_layout: str = "",
    spherical_projection: str = "",
) -> VideoMetadata:
    return VideoMetadata(
        video_file="movie.mp4",
        video_height=height,
        video_width=width,
        video_fps=30.0,
        average_fps=30.0,
        video_fps_exact=Fraction(30, 1),
        codec_name="hevc",
        duration=1.0,
        time_base=Fraction(1, 30),
        start_pts=0,
        color_range=None,
        color_space=None,
        num_frames=30,
        is_10bit=True,
        sample_aspect_ratio=sample_aspect_ratio,
        stereo_layout=stereo_layout,
        spherical_projection=spherical_projection,
    )


@pytest.mark.parametrize("token", sorted(FISHEYE_STUDIO_TOKENS))
def test_auto_resolves_known_fisheye_studios(token: str) -> None:
    result = resolve_vr_mode("auto", _metadata(), Path(f"{token}-001.mp4"))
    assert result.resolved == "sbs-fisheye"
    assert token in result.reason


@pytest.mark.parametrize("token", sorted(DIRECT_STUDIO_TOKENS))
def test_auto_resolves_known_direct_sbs_studios(token: str) -> None:
    result = resolve_vr_mode("auto", _metadata(), Path(f"{token}-001.mp4"))
    assert result.resolved == "sbs"
    assert token in result.reason


def test_auto_fisheye_studio_overrides_direct_token() -> None:
    result = resolve_vr_mode("auto", _metadata(), Path("VRKM-FSVSS-001.mp4"))
    assert result.resolved == "sbs-fisheye"


def test_auto_matches_studio_code_glued_to_number() -> None:
    # Real 8K releases glue the studio code to the number (savr00327-2);
    # detection is a substring match, not a separator-bounded token.
    assert resolve_vr_mode(
        "auto", _metadata(), Path("savr00327-2.mp4")
    ).resolved == "sbs-fisheye"
    assert resolve_vr_mode(
        "auto", _metadata(), Path("mdvr00271-2.mp4")
    ).resolved == "sbs"


def test_auto_uses_spatial_metadata_for_sbs() -> None:
    metadata = _metadata(
        stereo_layout="side by side",
        spherical_projection="equirectangular",
    )
    result = resolve_vr_mode("auto", metadata, Path("unknown.mp4"))
    assert result.resolved == "sbs"


@pytest.mark.parametrize("name", ["generic-vr.mp4", "3DSVR-001.mp4", "unknown.mp4"])
def test_auto_uses_high_resolution_2_to_1_geometry(name: str) -> None:
    result = resolve_vr_mode("auto", _metadata(), Path(name))
    assert result.resolved == "sbs"
    assert result.reason == "2:1 frame above 1080p"


def test_auto_uses_exact_frame_geometry_even_with_non_square_pixels() -> None:
    result = resolve_vr_mode(
        "auto",
        _metadata(width=4096, height=2048, sample_aspect_ratio=Fraction(3, 4)),
        Path("unknown.mp4"),
    )
    assert result.resolved == "sbs"


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (1920, 960),
        (2160, 1080),
        (3840, 2160),
        (3842, 1920),
    ],
)
def test_auto_does_not_infer_vr_below_threshold_or_without_exact_geometry(
    width: int,
    height: int,
) -> None:
    assert resolve_vr_mode(
        "auto",
        _metadata(width=width, height=height),
        Path("unknown.mp4"),
    ).resolved == "off"


def test_explicit_sbs_rejects_odd_width() -> None:
    with pytest.raises(ValueError, match="even frame width"):
        resolve_vr_mode("sbs", _metadata(width=3839), Path("movie.mp4"))


def test_explicit_mode_overrides_auto_detection() -> None:
    assert resolve_vr_mode(
        "sbs-fisheye", _metadata(), Path("unknown.mp4")
    ).resolved == "sbs-fisheye"
    assert resolve_vr_mode(
        "off", _metadata(), Path("FSVSS-001.mp4")
    ).resolved == "off"


class _FakeDetector:
    def __init__(self) -> None:
        self.detect_calls: list[tuple[torch.Tensor, tuple[int, int]]] = []
        self.scan_calls: list[tuple[torch.Tensor, tuple[int, int]]] = []

    def __call__(
        self,
        frames: torch.Tensor,
        *,
        target_hw: tuple[int, int],
    ) -> Detections:
        self.detect_calls.append((frames.clone(), target_hw))
        batch_size = int(frames.shape[0])
        value = int(frames[0, 0, 0, 0])
        boxes = [
            np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
            for _ in range(batch_size)
        ]
        masks = [
            torch.full(
                (1, 4, 5),
                value > 0,
                dtype=torch.bool,
                device=frames.device,
            )
            for _ in range(batch_size)
        ]
        return Detections(boxes_xyxy=boxes, masks=masks)

    def scan_scores_masks(
        self,
        frames: torch.Tensor,
        *,
        mask_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.scan_calls.append((frames.clone(), mask_hw))
        scores = frames[:, 0, 0, 0].float()
        masks = torch.ones(
            (frames.shape[0], *mask_hw),
            dtype=torch.bool,
            device=frames.device,
        )
        return scores, masks


def test_sbs_detection_adapter_merges_boxes_and_full_canvas_masks() -> None:
    detector = _FakeDetector()
    adapter = SbsDetectionAdapter(detector)
    frames = torch.zeros((2, 3, 6, 8), dtype=torch.uint8)
    frames[:, :, :, 4:] = 9

    result = adapter(frames, target_hw=(6, 8))

    assert len(detector.detect_calls) == 2
    assert detector.detect_calls[0][0].shape == (2, 3, 6, 4)
    assert detector.detect_calls[1][0].shape == (2, 3, 6, 4)
    assert detector.detect_calls[0][1] == (6, 4)
    np.testing.assert_array_equal(
        result.boxes_xyxy[0],
        np.array(
            [[1.0, 2.0, 3.0, 4.0], [5.0, 2.0, 7.0, 4.0]],
            dtype=np.float32,
        ),
    )
    assert result.masks[0].shape == (2, 4, 10)
    assert not result.masks[0][0, :, 5:].any()
    assert result.masks[0][1, :, 5:].all()
    assert not result.masks[0][1, :, :5].any()


def test_sbs_scan_adapter_uses_each_eye_and_merges_scores_masks() -> None:
    detector = _FakeDetector()
    adapter = SbsDetectionAdapter(detector)
    frames = torch.zeros((2, 3, 6, 8), dtype=torch.uint8)
    frames[0, :, :, :4] = 3
    frames[0, :, :, 4:] = 7
    frames[1, :, :, :4] = 8
    frames[1, :, :, 4:] = 2

    scores, masks = adapter.scan_scores_masks(frames, mask_hw=(5, 9))

    assert [call[1] for call in detector.scan_calls] == [(5, 4), (5, 5)]
    assert scores.tolist() == [7.0, 8.0]
    assert masks.shape == (2, 5, 9)
    assert masks.all()


def test_sbs_adapter_rejects_odd_source_width() -> None:
    with pytest.raises(ValueError, match="even frame width"):
        SbsDetectionAdapter(_FakeDetector())(
            torch.zeros((1, 3, 4, 7), dtype=torch.uint8),
            target_hw=(4, 7),
        )


def _gradient_sbs_frame(eye_w: int, height: int) -> torch.Tensor:
    xs = torch.arange(eye_w, dtype=torch.float32)
    ys = torch.arange(height, dtype=torch.float32)
    eye = ys[:, None] * 0.5 + xs[None, :] * 0.7
    frame = torch.zeros((3, height, eye_w * 2), dtype=torch.float32)
    frame[:, :, :eye_w] = eye
    frame[:, :, eye_w:] = eye + 1000.0
    return frame


def test_gnomonic_extract_region_crop_matches_enlarged_bbox() -> None:
    projector = GnomonicProjector(eye_width=64, height=64, device=torch.device("cpu"))
    frame = _gradient_sbs_frame(64, 64)
    bbox = np.array([10.0, 12.0, 34.0, 40.0], dtype=np.float32)

    raw = projector.extract_region_crop(frame, bbox, 64, 128, x_bounds=(0, 64))

    expected = compute_enlarged_bbox(bbox, 64, 128, (0, 64))
    x1, y1, x2, y2 = expected
    assert raw.enlarged_bbox == expected
    assert raw.crop_shape == (y2 - y1, x2 - x1)
    assert tuple(raw.crop.shape) == (3, y2 - y1, x2 - x1)


def test_gnomonic_projection_round_trips_a_region() -> None:
    # A wide eye keeps the ~256 px crop to a small angular span; a horizontal-only
    # gradient is invariant to the vertical warp, isolating the reprojection error.
    eye_w, height = 1600, 64
    projector = GnomonicProjector(eye_width=eye_w, height=height, device=torch.device("cpu"))
    ramp = (torch.arange(eye_w, dtype=torch.float32) * 0.15).expand(height, eye_w)
    frame = torch.zeros((3, height, eye_w * 2), dtype=torch.float32)
    frame[:, :, :eye_w] = ramp
    bbox = np.array([640.0, 12.0, 900.0, 52.0], dtype=np.float32)

    raw = projector.extract_region_crop(frame, bbox, height, eye_w * 2, x_bounds=(0, eye_w))
    region = projector.source_region_from_patch(raw.crop.float(), raw.enlarged_bbox)

    x1, y1, x2, y2 = raw.enlarged_bbox
    truth = frame[:, y1:y2, x1:x2]
    margin = 3
    interior = (region - truth)[:, margin:-margin, margin:-margin]
    dynamic_range = (truth.max() - truth.min()).item()
    assert interior.abs().mean().item() < 0.02 * dynamic_range


def test_gnomonic_extract_selects_the_correct_eye() -> None:
    projector = GnomonicProjector(eye_width=64, height=64, device=torch.device("cpu"))
    frame = torch.zeros((3, 64, 128), dtype=torch.float32)
    frame[:, :, :64] = 30.0
    frame[:, :, 64:] = 200.0

    left = projector.extract_region_crop(
        frame, np.array([8.0, 8.0, 28.0, 28.0], dtype=np.float32), 64, 128,
        x_bounds=(0, 64),
    )
    right = projector.extract_region_crop(
        frame, np.array([72.0, 8.0, 92.0, 28.0], dtype=np.float32), 64, 128,
        x_bounds=(64, 128),
    )

    assert left.crop.max().item() <= 31.0
    assert right.crop.min().item() >= 199.0
    assert right.enlarged_bbox[0] >= 64


def test_gnomonic_projector_rejects_degenerate_dimensions() -> None:
    with pytest.raises(ValueError, match="Invalid eye dimensions"):
        GnomonicProjector(eye_width=0, height=64, device=torch.device("cpu"))
