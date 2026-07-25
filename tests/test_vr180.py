from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import torch

from jasna.media import VideoMetadata
from jasna.mosaic.detections import Detections
from jasna.vr180 import (
    DIRECT_STUDIO_TOKENS,
    FISHEYE_STUDIO_TOKENS,
    PROJECTION_CHOICES,
    STUDIO_PROJECTION,
    SbsDetectionAdapter,
    resolve_projection,
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
    assert result.resolved == "sbs"
    assert result.projection == "fisheye"
    assert token in result.reason


@pytest.mark.parametrize("token", sorted(DIRECT_STUDIO_TOKENS))
def test_auto_resolves_known_direct_sbs_studios(token: str) -> None:
    result = resolve_vr_mode("auto", _metadata(), Path(f"{token}-001.mp4"))
    assert result.resolved == "sbs"
    assert token in result.reason


def test_auto_fisheye_studio_overrides_direct_token() -> None:
    result = resolve_vr_mode("auto", _metadata(), Path("VRKM-FSVSS-001.mp4"))
    assert result.resolved == "sbs"
    assert result.projection == "fisheye"


def test_auto_matches_studio_code_glued_to_number() -> None:
    # Real 8K releases glue the studio code to the number (savr00327-2);
    # detection is a substring match, not a separator-bounded token.
    savr = resolve_vr_mode("auto", _metadata(), Path("savr00327-2.mp4"))
    assert savr.resolved == "sbs"
    assert savr.projection == "fisheye"
    mdvr = resolve_vr_mode("auto", _metadata(), Path("mdvr00271-2.mp4"))
    assert mdvr.resolved == "sbs"
    assert mdvr.projection == "raw"


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
    explicit_fe = resolve_vr_mode("sbs-fisheye", _metadata(), Path("unknown.mp4"))
    assert explicit_fe.resolved == "sbs"
    assert explicit_fe.projection == "fisheye"
    assert resolve_vr_mode(
        "off", _metadata(), Path("FSVSS-001.mp4")
    ).resolved == "off"


@pytest.mark.parametrize("projection", PROJECTION_CHOICES[1:])
def test_projection_override_applies_to_detected_vr(projection: str) -> None:
    result = resolve_vr_mode(
        "auto",
        _metadata(),
        Path("pxvr-001.mp4"),
        projection=projection,
    )

    assert result.is_sbs
    assert result.projection == projection


def test_projection_override_does_not_enable_vr_layout() -> None:
    result = resolve_vr_mode(
        "auto",
        _metadata(width=1920, height=1080),
        Path("movie.mp4"),
        projection="gnomonic",
    )

    assert not result.is_sbs
    assert result.projection == "none"


@pytest.mark.parametrize(("code", "kind"), sorted(STUDIO_PROJECTION.items()))
def test_resolve_projection_routes_confident_studios(code: str, kind: str) -> None:
    assert resolve_projection(Path(f"{code}-001.mp4")) == kind
    assert resolve_projection(Path(f"{code.lower()}00123-4.mp4")) == kind


def test_resolve_projection_falls_back_to_raw_for_unknown_studio() -> None:
    assert resolve_projection(Path("unknownvr-001.mp4")) == "raw"
    # Direct-SBS token with no routing entry stays raw (its studio prior).
    assert resolve_projection(Path("mdvr00271-2.mp4")) == "raw"


def test_resolve_projection_fisheye_token_without_table_entry() -> None:
    # SAVR/URVRSP are fisheye-shot but absent from the confident table;
    # the fisheye-token prior still routes them to fisheye.
    assert resolve_projection(Path("savr00327-2.mp4")) == "fisheye"
    assert resolve_projection(Path("urvrsp00285-3.mp4")) == "fisheye"


def test_resolve_projection_explicit_override_wins() -> None:
    assert resolve_projection(Path("ipvr-001.mp4"), requested="gnomonic") == "gnomonic"
    assert resolve_projection(Path("pxvr-001.mp4"), requested="raw") == "raw"


def test_resolve_projection_rejects_unknown_override() -> None:
    with pytest.raises(ValueError, match="Unknown VR projection"):
        resolve_projection(Path("movie.mp4"), requested="cylindrical")


def test_resolve_projection_strips_release_site_tag() -> None:
    assert resolve_projection(Path("[98T.TV]VRPRD-004-A.mp4")) == "gnomonic"


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
