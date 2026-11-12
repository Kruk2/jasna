"""Data models for GUI state management."""

import itertools
import json
import logging
import threading
from dataclasses import dataclass, field, fields, asdict
from enum import Enum
from pathlib import Path

from jasna.user_settings import read_settings, update_settings
from jasna.segments import SegmentRange
from jasna.session_config import (
    DEFAULT_CODEC,
    DEFAULT_DETECTION_MODEL_NAME,
    DEFAULT_MAX_CLIP_SIZE,
    DEFAULT_MAX_DETECTION_GAP,
    DEFAULT_MIN_DETECTION_DURATION,
    DEFAULT_TEMPORAL_OVERLAP,
    DEFAULT_UPSCALE_HEIGHT,
    DEFAULT_UPSCALE_STRENGTH,
    LTX_AUTO_DEFAULT_PERCENT,
    LTX_DEFAULT_MODEL,
    LTX_DEFAULT_OVERLAP,
    LTX_DEFAULT_SEED,
    RTX_DEFAULT_DEBLUR,
    RTX_DEFAULT_DENOISE,
    RTX_DEFAULT_QUALITY,
    RTX_DEFAULT_SCALE,
    SD15_DEFAULT_SEED,
    SD15_DEFAULT_STEPS,
    SD15_DEFAULT_STRENGTH,
    SD15_DEFAULT_VARIANTS,
    TVAI_DEFAULT_ARGS,
    TVAI_DEFAULT_FFMPEG_PATH,
    UPSCALE_RTX_DEFAULT_DEBLUR,
    UPSCALE_RTX_DEFAULT_DENOISE,
    UPSCALE_RTX_DEFAULT_QUALITY,
    TVAI_DEFAULT_MODEL,
    TVAI_DEFAULT_SCALE,
    TVAI_DEFAULT_WORKERS,
)

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


DEFAULT_OUTPUT_PATTERN = "{original}_restored.mp4"

_job_id_counter = itertools.count(1)


@dataclass(frozen=True)
class JobProcessingSnapshot:
    segments: tuple[SegmentRange, ...]
    detection_model: str | None
    detection_score_threshold: float | None
    vr_projection: str | None


@dataclass
class JobItem:
    path: Path
    output_path: Path | None = None
    id: int = field(default_factory=lambda: next(_job_id_counter))
    status: JobStatus = JobStatus.PENDING
    duration_seconds: float | None = None
    progress: float = 0.0
    elapsed_seconds: float | None = None
    error_message: str = ""
    has_conflict: bool = False  # True if output file already exists
    unfinished_output: Path | None = None
    segments: tuple[SegmentRange, ...] = ()
    detection_model: str | None = None
    detection_score_threshold: float | None = None
    vr_projection: str | None = None
    _state_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
        compare=False,
    )
    
    @property
    def filename(self) -> str:
        return self.path.name
    
    @property
    def duration_str(self) -> str:
        if self.duration_seconds is None:
            return ""
        mins, secs = divmod(int(self.duration_seconds), 60)
        return f"{mins}m {secs}s"

    def snapshot_segments(self) -> tuple[SegmentRange, ...]:
        with self._state_lock:
            return self.segments

    def try_set_video_options(
        self,
        segments: tuple[SegmentRange, ...],
        *,
        detection_model: str,
        detection_score_threshold: float,
        vr_projection: str,
    ) -> bool:
        with self._state_lock:
            if self.status is not JobStatus.PENDING:
                return False
            self.segments = tuple(segments)
            self.detection_model = str(detection_model)
            self.detection_score_threshold = float(detection_score_threshold)
            self.vr_projection = str(vr_projection)
            return True

    def begin_processing(self) -> JobProcessingSnapshot | None:
        with self._state_lock:
            if self.status is not JobStatus.PENDING:
                return None
            self.status = JobStatus.PROCESSING
            return JobProcessingSnapshot(
                segments=self.segments,
                detection_model=self.detection_model,
                detection_score_threshold=self.detection_score_threshold,
                vr_projection=self.vr_projection,
            )


@dataclass
class AppSettings:
    # Restoration model
    restoration_model: str = "basicvsrpp"  # basicvsrpp, ltx
    ltx_model: str = LTX_DEFAULT_MODEL  # distilled1, distilled2, distilled4, distilled, undistilled
    ltx_seed: int = LTX_DEFAULT_SEED
    ltx_fast: bool = False
    ltx_large_canvas: bool = False
    ltx_overlap: int = LTX_DEFAULT_OVERLAP
    ltx_trial: bool = False
    ltx_auto_split: bool = False
    ltx_auto_split_percent: int = LTX_AUTO_DEFAULT_PERCENT
    ltx_vae_encoder: str = "auto"  # auto, full, half, max
    ltx_vae_decoder: str = "auto"  # auto, full, half, max

    # Basic processing
    batch_size: int = 4
    max_clip_size: int = DEFAULT_MAX_CLIP_SIZE
    temporal_overlap: int = DEFAULT_TEMPORAL_OVERLAP
    vr_mode: str = "auto"
    vr_projection: str = "auto"
    fp16_mode: bool = True
    decode_backend: str = "auto"  # auto, vali, pyav-hw, pyav-sw
    
    # Denoising
    denoise_strength: str = "none"  # none, low, medium, high
    
    # Secondary restoration
    secondary_restoration: str = "none"  # none, unet-4x, tvai, rtx-super-res
    tvai_ffmpeg_path: str = TVAI_DEFAULT_FFMPEG_PATH
    tvai_model: str = TVAI_DEFAULT_MODEL
    tvai_scale: int = TVAI_DEFAULT_SCALE
    tvai_workers: int = TVAI_DEFAULT_WORKERS
    tvai_args: str = TVAI_DEFAULT_ARGS
    tvai_denoise: bool = False
    rtx_scale: int = RTX_DEFAULT_SCALE  # 2, 4
    rtx_quality: str = RTX_DEFAULT_QUALITY  # low, medium, high, ultra
    rtx_denoise: str = RTX_DEFAULT_DENOISE  # none, low, medium, high, ultra
    rtx_deblur: str = RTX_DEFAULT_DEBLUR  # none, low, medium, high, ultra

    # Whole-video upscaling
    upscale_model: str = "none"  # none, span, rtx-super-res
    upscale_height: int = DEFAULT_UPSCALE_HEIGHT
    upscale_strength: float = DEFAULT_UPSCALE_STRENGTH
    upscale_rtx_quality: str = UPSCALE_RTX_DEFAULT_QUALITY
    upscale_rtx_denoise: str = UPSCALE_RTX_DEFAULT_DENOISE
    upscale_rtx_deblur: str = UPSCALE_RTX_DEFAULT_DEBLUR
    
    # Detection
    detection_model: str = DEFAULT_DETECTION_MODEL_NAME  # RF-DETR, Lada YOLO, or ZeLeFans VR YOLO registry name
    detection_score_threshold: float = 0.35
    max_detection_gap: int = DEFAULT_MAX_DETECTION_GAP
    min_detection_duration: int = DEFAULT_MIN_DETECTION_DURATION
    scene_detection: bool = True
    compile_basicvsrpp: bool = True
    
    # Image restoration (SD 1.5 inpaint; used only for still-image inputs)
    image_restore_steps: int = SD15_DEFAULT_STEPS
    image_restore_strength: float = SD15_DEFAULT_STRENGTH
    image_restore_freeu: bool = True
    image_restore_seed: int = SD15_DEFAULT_SEED
    image_restore_variants: int = SD15_DEFAULT_VARIANTS

    # Encoding
    codec: str = DEFAULT_CODEC
    encoder_cq: int | None = None
    encoder_custom_args: str = ""
    sharpen_strength: float = 0.0
    lut_path: str = ""
    retarget_high_fps: bool = False
    frame_generation: str = "none"  # none, 2x
    fmp4: bool = False

    # Post-export action
    post_export_action: str = "none"  # none, shutdown, command
    post_export_command: str = ""
    post_export_video_command: str = ""
    
    # Output
    output_folder: str = ""
    output_pattern: str = DEFAULT_OUTPUT_PATTERN
    file_conflict: str = "auto_rename"  # auto_rename, overwrite, skip
    saved_progress_conflict: str = "ask"  # ask, start_over, skip
    working_directory: str = ""  # empty = same directory as the output video


# Factory default preset - frozen, matches CLI defaults
DEFAULT_SETTINGS = AppSettings()


_LEGACY_CODEC_SPELLINGS = {
    "hevc": "hevc",
    "h265": "hevc",
    "h.265": "hevc",
    "h264": "h264",
    "h.264": "h264",
    "avc": "h264",
    "av1": "av1",
    "av01": "av1",
}


def _normalize_preset_codec(value: object) -> str:
    canonical = _LEGACY_CODEC_SPELLINGS.get(str(value).strip().lower())
    if canonical is None:
        logger.warning("Unknown codec %r in preset; falling back to %s", value, DEFAULT_CODEC)
        return DEFAULT_CODEC
    return canonical


# Presets saved with the old "less overlap" switch off used this overlap.
_OLD_FULL_LTX_OVERLAP = 33


def _migrate_preset_dict(preset_dict: dict) -> dict:
    known_fields = {f.name for f in fields(AppSettings)}
    migrated = {k: v for k, v in preset_dict.items() if k in known_fields}
    custom_args = migrated.get("encoder_custom_args")
    if custom_args:
        from jasna.media.encoder_settings import CQ_CONTROLS, parse_encoder_settings

        try:
            parsed_args = parse_encoder_settings(custom_args)
        except (ValueError, json.JSONDecodeError):
            logger.warning("Preset has unreadable custom encoder settings %r; keeping them as they are", custom_args)
        else:
            # A preset may come from either GPU, so every CQ control moves to the slider.
            cq_key = next((key for key in CQ_CONTROLS if key in parsed_args), None)
            if cq_key is not None:
                cq = parsed_args[cq_key]
                if isinstance(cq, int) and not isinstance(cq, bool):
                    migrated["encoder_cq"] = cq
                    del parsed_args[cq_key]
            migrated["encoder_custom_args"] = ",".join(
                f"{key}={value}" for key, value in parsed_args.items()
            )
    if "codec" in migrated:
        migrated["codec"] = _normalize_preset_codec(migrated["codec"])
    if preset_dict.get("ltx_less_overlap") is False and "ltx_overlap" not in migrated:
        migrated["ltx_overlap"] = _OLD_FULL_LTX_OVERLAP
    return migrated


class PresetManager:
    """Manages user presets with persistence to settings.json."""

    FACTORY_PRESETS = {"Default": DEFAULT_SETTINGS}
    
    def __init__(self):
        self._user_presets: dict[str, AppSettings] = {}
        self._last_selected: str = "Default"
        self._last_output_folder: str = ""
        self._last_output_pattern: str = DEFAULT_OUTPUT_PATTERN
        self._system_check_passed_version: str = ""
        self._settings_view: str = "simple"
        self._load()
        
    def _load(self):
        """Load user presets from settings.json."""
        data = read_settings()
        self._last_selected = data.get("last_selected", "Default")
        self._last_output_folder = data.get("last_output_folder", "")
        self._last_output_pattern = data.get("last_output_pattern", DEFAULT_OUTPUT_PATTERN)
        self._system_check_passed_version = data.get("system_check_passed_version", "")
        view = data.get("settings_view")
        self._settings_view = view if view in ("advanced", "expert") else "simple"

        for name, preset_dict in data.get("user_presets", {}).items():
            try:
                self._user_presets[name] = AppSettings(**_migrate_preset_dict(preset_dict))
            except (TypeError, ValueError) as e:
                logger.warning("Skipping invalid preset %r: %s", name, e)

    def _save(self, **values) -> None:
        update_settings(**values)

    def _save_user_presets(self) -> None:
        self._save(user_presets={name: asdict(preset) for name, preset in self._user_presets.items()})
            
    def get_all_preset_names(self) -> tuple[list[str], list[str]]:
        """Return (factory_names, user_names)."""
        return list(self.FACTORY_PRESETS.keys()), list(self._user_presets.keys())
    
    def get_preset(self, name: str) -> AppSettings | None:
        """Get preset by name, checking factory first, then user."""
        if name in self.FACTORY_PRESETS:
            return self.FACTORY_PRESETS[name]
        return self._user_presets.get(name)
    
    def is_factory_preset(self, name: str) -> bool:
        """Check if preset is a factory preset."""
        return name in self.FACTORY_PRESETS

    def resolve(self, name: str) -> tuple[str, AppSettings]:
        """Return (name, preset), falling back to Default for unknown names."""
        preset = self.get_preset(name)
        if preset is None:
            return "Default", self.FACTORY_PRESETS["Default"]
        return name, preset
    
    def create_preset(self, name: str, settings: AppSettings) -> bool:
        """Create a new user preset. Returns False if name is invalid."""
        name = name.strip()
        if not name or name in self.FACTORY_PRESETS:
            return False
        self._user_presets[name] = settings
        self._save_user_presets()
        return True
    
    def update_preset(self, name: str, settings: AppSettings) -> bool:
        """Update an existing user preset. Returns False if not found or factory."""
        if name in self.FACTORY_PRESETS or name not in self._user_presets:
            return False
        self._user_presets[name] = settings
        self._save_user_presets()
        return True
    
    def delete_preset(self, name: str) -> bool:
        """Delete a user preset. Returns False if not found or factory."""
        if name in self.FACTORY_PRESETS or name not in self._user_presets:
            return False
        del self._user_presets[name]
        self._save_user_presets()
        return True
    
    def get_last_selected(self) -> str:
        """Get last selected preset name."""
        # Verify the preset still exists
        if self._last_selected in self.FACTORY_PRESETS or self._last_selected in self._user_presets:
            return self._last_selected
        return "Default"
    
    def set_last_selected(self, name: str):
        """Set last selected preset name."""
        self._last_selected = name
        self._save(last_selected=name)

    def get_last_output_folder(self) -> str:
        return self._last_output_folder

    def set_last_output_folder(self, path: str):
        self._last_output_folder = path or ""
        self._save(last_output_folder=self._last_output_folder)

    def get_last_output_pattern(self) -> str:
        return self._last_output_pattern

    def set_last_output_pattern(self, pattern: str):
        self._last_output_pattern = pattern or DEFAULT_OUTPUT_PATTERN
        self._save(last_output_pattern=self._last_output_pattern)

    def get_settings_view(self) -> str:
        """``"simple"``, ``"advanced"`` or ``"expert"``; new users start with the simple view."""
        return self._settings_view

    def set_settings_view(self, view: str):
        self._settings_view = view
        self._save(settings_view=view)

    def get_system_check_passed_version(self) -> str:
        return self._system_check_passed_version

    def set_system_check_passed_version(self, version: str):
        self._system_check_passed_version = version or ""
        self._save(system_check_passed_version=self._system_check_passed_version)
