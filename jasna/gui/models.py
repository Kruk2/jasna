"""Data models for GUI state management."""

import itertools
import json
import logging
import re
import threading
from dataclasses import dataclass, field, fields, asdict
from enum import Enum
from pathlib import Path
from typing import Callable

from jasna.gui.paths import get_settings_path
from jasna.segments import SegmentRange

logger = logging.getLogger(__name__)

ENCODER_RATE_MODE_AUTO_SOURCE = "auto_source"
ENCODER_RATE_MODE_MANUAL_CQ = "manual_cq"
ENCODER_RATE_MODES = {
    ENCODER_RATE_MODE_AUTO_SOURCE,
    ENCODER_RATE_MODE_MANUAL_CQ,
}


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"
    SKIPPED = "skipped"


class SegmentSelectionMode(Enum):
    """Where a queued video's restoration ranges came from.

    ``DEFAULT`` leaves routing to the global pre-scan policy. ``MANUAL`` keeps
    non-empty ranges strict, while ``FULL`` records that the user deliberately
    saved an empty range list in the segment editor.
    """

    DEFAULT = "default"
    MANUAL = "manual"
    FULL = "full"


_job_id_counter = itertools.count(1)


@dataclass(frozen=True)
class JobProcessingSnapshot:
    segments: tuple[SegmentRange, ...]
    detection_model: str | None
    detection_score_threshold: float | None
    vr_projection: str | None
    segment_selection_mode: SegmentSelectionMode = SegmentSelectionMode.DEFAULT


@dataclass
class JobItem:
    path: Path
    output_path: Path | None = None
    input_root: Path | None = None
    id: int = field(default_factory=lambda: next(_job_id_counter))
    status: JobStatus = JobStatus.PENDING
    duration_seconds: float | None = None
    progress: float = 0.0
    error_message: str = ""
    has_conflict: bool = False  # True if output file already exists
    segments: tuple[SegmentRange, ...] = ()
    segment_selection_mode: SegmentSelectionMode = SegmentSelectionMode.DEFAULT
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

    def try_set_segments(self, segments: tuple[SegmentRange, ...]) -> bool:
        with self._state_lock:
            if self.status is not JobStatus.PENDING:
                return False
            self.segments = tuple(segments)
            self.segment_selection_mode = (
                SegmentSelectionMode.MANUAL
                if self.segments
                else SegmentSelectionMode.FULL
            )
            return True

    def try_reset_segments(self) -> bool:
        """Return a pending job to global automatic range selection."""

        with self._state_lock:
            if self.status is not JobStatus.PENDING:
                return False
            self.segments = ()
            self.segment_selection_mode = SegmentSelectionMode.DEFAULT
            return True

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
            self.segment_selection_mode = (
                SegmentSelectionMode.MANUAL
                if self.segments
                else SegmentSelectionMode.FULL
            )
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
                segment_selection_mode=self.segment_selection_mode,
                detection_model=self.detection_model,
                detection_score_threshold=self.detection_score_threshold,
                vr_projection=self.vr_projection,
            )


@dataclass
class ProcessingState:
    is_running: bool = False
    is_paused: bool = False
    current_job_index: int = -1
    current_filename: str = ""
    progress_percent: float = 0.0
    fps: float = 0.0
    eta_seconds: float = 0.0
    frames_processed: int = 0
    total_frames: int = 0


@dataclass
class AppSettings:
    # Basic processing
    batch_size: int = 4
    max_clip_size: int = 90
    temporal_overlap: int = 8
    enable_crossfade: bool = True
    vr_mode: str = "auto"
    vr_projection: str = "auto"
    fp16_mode: bool = True
    
    # Denoising
    denoise_strength: str = "none"  # none, low, medium, high
    denoise_step: str = "after_primary"  # after_primary, after_secondary
    
    # Secondary restoration
    secondary_restoration: str = "none"  # none, unet-4x, tvai, rtx-super-res
    tvai_ffmpeg_path: str = r"C:\Program Files\Topaz Labs LLC\Topaz Video\ffmpeg.exe"
    tvai_model: str = "iris-2"
    tvai_scale: int = 4
    tvai_workers: int = 2
    tvai_args: str = "preblur=0:noise=0:details=0:halo=0:blur=0:compression=0:estimate=8:blend=0.2:device=-2:vram=1:instances=1"
    tvai_denoise: bool = False
    rtx_scale: int = 4  # 2, 4
    rtx_quality: str = "high"  # low, medium, high, ultra
    rtx_denoise: str = "medium"  # none, low, medium, high, ultra
    rtx_deblur: str = "none"  # none, low, medium, high, ultra
    
    # Detection
    detection_model: str = "rfdetr-v6"  # RF-DETR, Lada YOLO, or ZeLeFans VR YOLO registry name
    detection_score_threshold: float = 0.35
    pre_scan_policy: str = "auto"  # auto, scan, off
    pre_scan_full_threshold: float = 0.85
    pre_scan_coarse_interval: float = 4.0
    pre_scan_fine_interval: float = 0.5
    pre_scan_pad_seconds: str = "auto"  # auto, 0.0, 0.5, 1.0, 2.0, 5.0
    max_detection_gap: int = 2
    min_detection_duration: int = 2
    scene_detection: bool = True
    compile_basicvsrpp: bool = True
    
    # Image restoration (SD 1.5 inpaint; used only for still-image inputs)
    image_restore_steps: int = 25
    image_restore_strength: float = 0.6
    image_restore_freeu: bool = True
    image_restore_seed: int = 0
    image_restore_variants: int = 1

    # Encoding
    codec: str = "hevc"
    encoder_rate_mode: str = ENCODER_RATE_MODE_AUTO_SOURCE
    encoder_cq: int | None = None
    encoder_custom_args: str = ""
    amd_dual_gop_encode: bool = False
    sharpen_strength: float = 0.0
    lut_path: str = ""
    retarget_high_fps: bool = False
    fmp4: bool = False

    # Post-export action
    post_export_action: str = "none"  # none, shutdown, command
    post_export_command: str = ""
    post_export_video_command: str = ""
    
    # Output
    output_same_as_input: bool = True
    output_folder: str = ""
    output_pattern: str = "{original}_restored.mp4"
    file_conflict: str = "auto_rename"  # auto_rename, overwrite, skip
    working_directory: str = ""  # empty = same directory as the output video

    # Diagnostics
    save_run_log: bool = False


# The GUI factory preset enables the validated Linux AMD HEVC experiment for
# the next full-video comparison.  AppSettings itself deliberately keeps the
# CLI/session default off so non-GUI callers must opt in explicitly.
DEFAULT_SETTINGS = AppSettings(amd_dual_gop_encode=True)


# Old presets carry PyNvVideoCodec-era encoder option names; the encoder now
# speaks ffmpeg hevc_nvenc. Renames plus the two one-to-many expansions below.
_OLD_ENCODER_ARG_RENAMES = {
    "nonrefp": "nonref_p",
    "gop": "g",
    "maxbitrate": "maxrate",
    "vbvbufsize": "bufsize",
    "temporalaq": "temporal-aq",
    "lookahead": "rc-lookahead",
    "tflevel": "tf_level",
}
_OLD_TUNING_INFO_VALUES = {
    "high_quality": "hq",
    "low_latency": "ll",
    "ultra_low_latency": "ull",
    "lossless": "lossless",
}


def _migrate_encoder_custom_args(value: str) -> str:
    from jasna.gui.hardware_policy import split_batch_size_custom_arg
    from jasna.media import parse_encoder_settings

    try:
        batch_size, encoder_args = split_batch_size_custom_arg(value)
        settings = parse_encoder_settings(encoder_args)
    except (ValueError, json.JSONDecodeError):
        return value

    migrated: dict[str, object] = {}
    for key, v in settings.items():
        if key == "aq":
            migrated["spatial_aq"] = 1
            migrated["aq-strength"] = v
        elif key == "initqp":
            migrated["init_qpI"] = v
            migrated["init_qpP"] = v
            migrated["init_qpB"] = v
        elif key == "tuning_info":
            migrated["tune"] = _OLD_TUNING_INFO_VALUES.get(str(v), str(v))
        elif key == "preset" and isinstance(v, str) and re.fullmatch(r"P[1-7]", v):
            migrated["preset"] = v.lower()
        elif key == "vbvinit":
            continue  # no hevc_nvenc equivalent
        elif key in _OLD_ENCODER_ARG_RENAMES:
            migrated[_OLD_ENCODER_ARG_RENAMES[key]] = v
        else:
            migrated[key] = v
    parts = []
    if batch_size is not None:
        parts.append(f"--batch-size {batch_size}")
    parts.extend(f"{k}={v}" for k, v in migrated.items())
    return ",".join(parts)


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
        logger.warning("Unknown codec %r in preset; falling back to hevc", value)
        return "hevc"
    return canonical


def _migrate_preset_dict(preset_dict: dict) -> dict:
    known_fields = {f.name for f in fields(AppSettings)}
    migrated = {k: v for k, v in preset_dict.items() if k in known_fields}
    custom_args = migrated.get("encoder_custom_args")
    if custom_args:
        from jasna.gui.hardware_policy import split_batch_size_custom_arg
        from jasna.media import parse_encoder_settings

        migrated_args = _migrate_encoder_custom_args(custom_args)
        try:
            batch_size, encoder_args = split_batch_size_custom_arg(migrated_args)
            parsed_args = parse_encoder_settings(encoder_args)
        except (ValueError, json.JSONDecodeError):
            migrated["encoder_custom_args"] = migrated_args
        else:
            cq_key = next(
                (
                    key
                    for key in ("cq", "qvbr_quality_level")
                    if key in parsed_args
                ),
                None,
            )
            if cq_key is not None:
                cq = parsed_args[cq_key]
                if isinstance(cq, int) and not isinstance(cq, bool):
                    migrated["encoder_cq"] = cq
                    del parsed_args[cq_key]
            parts = []
            if batch_size is not None:
                parts.append(f"--batch-size {batch_size}")
            parts.extend(f"{key}={value}" for key, value in parsed_args.items())
            migrated["encoder_custom_args"] = ",".join(parts)
    if "codec" in migrated:
        migrated["codec"] = _normalize_preset_codec(migrated["codec"])
    return migrated


class PresetManager:
    """Manages user presets with persistence to settings.json."""

    FACTORY_PRESETS = {"Default": DEFAULT_SETTINGS}
    
    def __init__(self):
        self._user_presets: dict[str, AppSettings] = {}
        self._last_selected: str = "Default"
        self._last_output_folder: str = ""
        self._last_output_pattern: str = "{original}_restored.mp4"
        self._last_preserve_input_structure: bool = False
        self._system_check_passed_version: str = ""
        self._load()
        
    def _load(self):
        """Load user presets from settings.json."""
        path = get_settings_path()
        if not path.exists():
            return
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            self._last_selected = data.get("last_selected", "Default")
            self._last_output_folder = data.get("last_output_folder", "")
            self._last_output_pattern = data.get("last_output_pattern", "{original}_restored.mp4")
            self._last_preserve_input_structure = bool(
                data.get("last_preserve_input_structure", False)
            )
            self._system_check_passed_version = data.get("system_check_passed_version", "")
            
            for name, preset_dict in data.get("user_presets", {}).items():
                try:
                    self._user_presets[name] = AppSettings(**_migrate_preset_dict(preset_dict))
                except (TypeError, ValueError):
                    pass  # Skip invalid presets
        except (json.JSONDecodeError, IOError):
            pass
            
    def _save(self):
        """Save user presets to settings.json."""
        path = get_settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                data = {}

        data["last_selected"] = self._last_selected
        data["user_presets"] = {name: asdict(preset) for name, preset in self._user_presets.items()}
        data["last_output_folder"] = self._last_output_folder
        data["last_output_pattern"] = self._last_output_pattern
        data["last_preserve_input_structure"] = self._last_preserve_input_structure
        data["system_check_passed_version"] = self._system_check_passed_version
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except IOError:
            pass
            
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
        self._save()
        return True
    
    def update_preset(self, name: str, settings: AppSettings) -> bool:
        """Update an existing user preset. Returns False if not found or factory."""
        if name in self.FACTORY_PRESETS or name not in self._user_presets:
            return False
        self._user_presets[name] = settings
        self._save()
        return True
    
    def delete_preset(self, name: str) -> bool:
        """Delete a user preset. Returns False if not found or factory."""
        if name in self.FACTORY_PRESETS or name not in self._user_presets:
            return False
        del self._user_presets[name]
        self._save()
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
        self._save()

    def get_last_output_folder(self) -> str:
        return self._last_output_folder

    def set_last_output_folder(self, path: str):
        self._last_output_folder = path or ""
        self._save()

    def get_last_output_pattern(self) -> str:
        return self._last_output_pattern

    def set_last_output_pattern(self, pattern: str):
        self._last_output_pattern = pattern or "{original}_restored.mp4"
        self._save()

    def get_last_preserve_input_structure(self) -> bool:
        return self._last_preserve_input_structure

    def set_last_preserve_input_structure(self, enabled: bool):
        self._last_preserve_input_structure = bool(enabled)
        self._save()

    def get_system_check_passed_version(self) -> str:
        return self._system_check_passed_version

    def set_system_check_passed_version(self, version: str):
        self._system_check_passed_version = version or ""
        self._save()
