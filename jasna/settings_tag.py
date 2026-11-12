"""The restoration settings Jasna writes into each output's ``comment`` tag.

The tag is one line of JSON whose first key is ``"jasna"`` (the version), so readers can tell
it apart from a source comment. ``comment`` is the free-text field MP4, MOV and Matroska all
keep, also through a plain FFmpeg remux. Bump ``SETTINGS_TAG_SCHEMA`` when a key changes meaning.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from jasna import __version__
from jasna.ltx.model_files import LTX_MODEL_BRANCH, resolved_vaes
from jasna.segments import SegmentRange
from jasna.session_config import RestorationModelName, SessionConfig

if TYPE_CHECKING:
    from jasna.vr180 import VrModeResolution

SETTINGS_TAG_SCHEMA = 1


def _rtx_settings(config: SessionConfig) -> dict[str, object]:
    return {"quality": config.rtx_quality, "denoise": config.rtx_denoise, "deblur": config.rtx_deblur}


def _secondary_settings(config: SessionConfig) -> dict[str, object]:
    name = config.secondary_restoration
    if name == "tvai":
        return {
            "name": name,
            "model": config.tvai_model,
            "scale": config.tvai_scale,
            "args": config.tvai_args,
            "denoise": config.tvai_denoise,
        }
    if name == "rtx-super-res":
        return {"name": name, "scale": config.rtx_scale, **_rtx_settings(config)}
    return {"name": name}


def _upscale_settings(config: SessionConfig) -> dict[str, object] | None:
    if config.upscale_model == "none":
        return None
    settings: dict[str, object] = {
        "model": config.upscale_model, "height": config.upscale_height, "strength": config.upscale_strength,
    }
    if config.upscale_model == "rtx-super-res":
        settings.update(
            quality=config.upscale_rtx_quality, denoise=config.upscale_rtx_denoise, deblur=config.upscale_rtx_deblur
        )
    if config.upscale_model == "dlss5":
        settings.update(
            preset=config.upscale_dlss5_preset, style=config.upscale_dlss5_style,
            intensity=config.upscale_dlss5_intensity,
        )
    return settings


def _segments_summary(segments: tuple[SegmentRange, ...]) -> dict[str, object]:
    """Per model: how many ranges and seconds, plus every LTX seed. Fixed size, so a job with
    hundreds of ranges keeps a short tag (the smart-render mux passes it on a command line)."""
    summary: dict[str, object] = {"count": len(segments)}
    for model in sorted({segment.restoration.model for segment in segments}):
        ranges = [segment for segment in segments if segment.restoration.model == model]
        part: dict[str, object] = {
            "count": len(ranges),
            "seconds": round(sum(segment.duration for segment in ranges), 3),
        }
        if model == "ltx":
            part["seeds"] = sorted({segment.restoration.ltx_seed for segment in ranges})
        summary[model] = part
    return summary


# Windows caps a command line at 32 767 characters; the settings themselves stay under 2 000.
SOURCE_COMMENT_MAX_CHARS = 16_000


def job_settings(
    config: SessionConfig,
    *,
    models: frozenset[RestorationModelName],
    segments: tuple[SegmentRange, ...] | None,
    vr: VrModeResolution,
    ltx_weights: str | None,
    ltx_large_canvas: bool | None,
) -> dict[str, object]:
    """Everything that shapes the restored picture, except the encoder options, which
    ``settings_comment`` adds. ``segments`` carry their resolved restorations;
    ``ltx_large_canvas`` is the 768 px decision the run made, not the request."""
    settings: dict[str, object] = {
        "jasna": __version__,
        "schema": SETTINGS_TAG_SCHEMA,
        "detection": {
            "model": config.detection_model_name,
            "threshold": config.detection_score_threshold,
            "max_gap": config.max_detection_gap,
            "min_duration": config.min_detection_duration,
            "scene_cuts": config.scene_detection,
        },
    }
    if "basicvsrpp" in models:
        from jasna.accelerator import is_amd_device

        basicvsrpp: dict[str, object] = {
            "model": config.model_path("basicvsrpp").name,
            "fp16": config.fp16,
            "tensorrt": config.compile_basicvsrpp and not is_amd_device(config.device),
            "max_clip_size": config.max_clip_size,
            "temporal_overlap": config.temporal_overlap,
            "denoise": config.denoise_strength,
            "secondary": _secondary_settings(config),
            "vr": vr.resolved,
        }
        if vr.is_stereo:
            basicvsrpp["vr_projection"] = vr.projection
        settings["basicvsrpp"] = basicvsrpp
    if "ltx" in models:
        vae_encoder, vae_decoder = resolved_vaes(config.ltx_model, config.ltx_vae_encoder, config.ltx_vae_decoder)
        ltx: dict[str, object] = {
            "weights": ltx_weights,
            "weights_release": LTX_MODEL_BRANCH,
            "vae_encoder": vae_encoder,
            "vae_decoder": vae_decoder,
            "window_hop": config.ltx_window_hop,
            "large_canvas": ltx_large_canvas,
        }
        if not segments:
            ltx["seed"] = config.ltx_seed
        if config.secondary_restoration == "rtx-super-res":
            ltx["super_res"] = _rtx_settings(config)
        if config.ltx_trial:
            ltx["trial"] = True
        settings["ltx"] = ltx
    if config.restoration_model_name == "ltx" and config.ltx_auto_split_percent is not None:
        settings["auto_split_percent"] = config.ltx_auto_split_percent
    if segments:
        settings["segments"] = _segments_summary(segments)
    settings["output"] = {
        "codec": config.codec,
        "lut": Path(config.lut_path).name if config.lut_path else None,
        "sharpen": config.sharpen_strength,
        "retarget_high_fps": config.retarget_high_fps,
        "frame_generation": config.frame_generation,
        "upscale": _upscale_settings(config),
    }
    return settings


def settings_comment(settings: Mapping[str, object], *, encoder: str, options: Mapping[str, str]) -> str:
    output = {**settings["output"], "encoder": encoder, "encoder_options": dict(options)}
    return json.dumps({**settings, "output": output}, separators=(",", ":"))


def with_source_comment(comment: str, source_tags: Mapping[str, str]) -> str:
    """``comment`` replaces the source's own comment tag; keep that text as ``source_comment``, cut
    to ``SOURCE_COMMENT_MAX_CHARS`` so the smart-render mux command line stays within Windows' limit."""
    source_comment = next((value for key, value in source_tags.items() if key.lower() == "comment" and value), None)
    if source_comment is None:
        return comment
    tag = {**json.loads(comment), "source_comment": source_comment[:SOURCE_COMMENT_MAX_CHARS]}
    return json.dumps(tag, separators=(",", ":"))
