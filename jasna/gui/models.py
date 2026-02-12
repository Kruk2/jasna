"""Data models for GUI state management."""

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Callable

from jasna import os_utils


def get_settings_path() -> Path:
    return os_utils.get_user_config_dir("jasna") / "settings.json"


class JobStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"
    SKIPPED = "skipped"


@dataclass
class JobItem:
    path: Path
    status: JobStatus = JobStatus.PENDING
    duration_seconds: float | None = None
    progress: float = 0.0
    error_message: str = ""
    has_conflict: bool = False  # True if output file already exists
    
    @property
    def filename(self) -> str:
        return self.path.name
    
    @property
    def duration_str(self) -> str:
        if self.duration_seconds is None:
            return ""
        mins, secs = divmod(int(self.duration_seconds), 60)
        return f"{mins}m {secs}s"


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
    max_clip_size: int = 60
    temporal_overlap: int = 8
    enable_crossfade: bool = True
    fp16_mode: bool = True
    
    # Denoising
    denoise_strength: str = "none"  # none, low, medium, high
    denoise_step: str = "after_primary"  # after_primary, after_secondary
    
    # Secondary restoration
    secondary_restoration: str = "none"  # none, swin2sr, tvai
    swin2sr_batch_size: int = 8
    swin2sr_tensorrt: bool = True
    tvai_ffmpeg_path: str = r"C:\Program Files\Topaz Labs LLC\Topaz Video AI\ffmpeg.exe"
    tvai_model: str = "iris-2"
    tvai_scale: int = 4
    tvai_workers: int = 2
    tvai_args: str = "preblur=0:noise=0:details=0:halo=0:blur=0:compression=0:estimate=8:blend=0.2:device=-2:vram=1:instances=1"
    
    # Detection
    detection_model: str = "rfdetr-v4"  # rfdetr-v2, rfdetr-v3, rfdetr-v4, lada-yolo-v2, lada-yolo-v4
    detection_score_threshold: float = 0.2
    compile_basicvsrpp: bool = True
    
    # Encoding
    codec: str = "hevc"
    encoder_cq: int = 22
    encoder_custom_args: str = ""
    
    # Output
    output_same_as_input: bool = True
    output_folder: str = ""
    output_pattern: str = "{original}_restored.mp4"
    file_conflict: str = "auto_rename"  # auto_rename, overwrite, skip


# Factory default preset - frozen, matches CLI defaults
DEFAULT_SETTINGS = AppSettings()


class PresetManager:
    """Manages user presets with persistence to settings.json."""
    
    FACTORY_PRESETS = {"Default": DEFAULT_SETTINGS}
    
    def __init__(self):
        self._user_presets: dict[str, AppSettings] = {}
        self._last_selected: str = "Default"
        self._last_output_folder: str = ""
        self._last_output_pattern: str = "{original}_restored.mp4"
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
            
            for name, preset_dict in data.get("user_presets", {}).items():
                try:
                    self._user_presets[name] = AppSettings(**preset_dict)
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
