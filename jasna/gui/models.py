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


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"
    SKIPPED = "skipped"


_job_id_counter = itertools.count(1)


@dataclass(frozen=True)
class JobProcessingSnapshot:
    segments: tuple[SegmentRange, ...]
    detection_model: str | None
    detection_score_threshold: float | None


@dataclass
class JobItem:
    path: Path
    id: int = field(default_factory=lambda: next(_job_id_counter))
    status: JobStatus = JobStatus.PENDING
    duration_seconds: float | None = None
    progress: float = 0.0
    error_message: str = ""
    has_conflict: bool = False  # True if output file already exists
    segments: tuple[SegmentRange, ...] = ()
    detection_model: str | None = None
    detection_score_threshold: float | None = None
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
            return True

    def try_set_video_options(
        self,
        segments: tuple[SegmentRange, ...],
        *,
        detection_model: str,
        detection_score_threshold: float,
    ) -> bool:
        with self._state_lock:
            if self.status is not JobStatus.PENDING:
                return False
            self.segments = tuple(segments)
            self.detection_model = str(detection_model)
            self.detection_score_threshold = float(detection_score_threshold)
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
    rtx_scale: int = 4  # 2, 4
    rtx_quality: str = "high"  # low, medium, high, ultra
    rtx_denoise: str = "medium"  # none, low, medium, high, ultra
    rtx_deblur: str = "none"  # none, low, medium, high, ultra
    
    # Detection
    detection_model: str = "rfdetr-v5"  # RF-DETR, Lada YOLO, or ZeLeFans VR YOLO registry name
    detection_score_threshold: float = 0.25
    max_detection_gap: int = 2
    min_detection_duration: int = 2
    compile_basicvsrpp: bool = True
    
    # Image restoration (SD 1.5 inpaint; used only for still-image inputs)
    image_restore_steps: int = 25
    image_restore_strength: float = 0.6
    image_restore_freeu: bool = True
    image_restore_seed: int = 0
    image_restore_variants: int = 1

    # Encoding
    codec: str = "hevc"
    encoder_cq: int = 22
    encoder_custom_args: str = ""
    lut_path: str = ""
    retarget_high_fps: bool = False

    # Post-export action
    post_export_action: str = "none"  # none, shutdown, command
    post_export_command: str = ""
    
    # Output
    output_same_as_input: bool = True
    output_folder: str = ""
    output_pattern: str = "{original}_restored.mp4"
    file_conflict: str = "auto_rename"  # auto_rename, overwrite, skip
    working_directory: str = ""  # empty = same directory as the output video


# Factory default preset - frozen, matches CLI defaults
DEFAULT_SETTINGS = AppSettings()


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
    from jasna.media import parse_encoder_settings

    try:
        settings = parse_encoder_settings(value)
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
    return ",".join(f"{k}={v}" for k, v in migrated.items())


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
        migrated["encoder_custom_args"] = _migrate_encoder_custom_args(custom_args)
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

    def get_system_check_passed_version(self) -> str:
        return self._system_check_passed_version

    def set_system_check_passed_version(self, version: str):
        self._system_check_passed_version = version or ""
        self._save()
