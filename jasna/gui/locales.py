"""Localization system for Jasna GUI."""

import json
import locale as _locale
from pathlib import Path
from typing import Callable

from jasna import os_utils


def get_settings_path() -> Path:
    return os_utils.get_user_config_dir("jasna") / "settings.json"


def _get_cli_descriptions() -> dict[str, str]:
    """Extract descriptions from CLI argument parser."""
    from jasna.main import build_parser
    parser = build_parser()
    
    descriptions = {}
    for action in parser._actions:
        if action.dest and action.help and action.help != "==SUPPRESS==":
            # Clean up help text - remove default placeholders
            help_text = action.help
            if "%(default)s" in help_text:
                help_text = help_text.replace(" (default: %(default)s)", "")
                help_text = help_text.replace("(default: %(default)s)", "")
            
            # Map CLI arg names to GUI keys
            key_map = {
                "fp16": "fp16_mode",
                "compile_basicvsrpp": "compile_basicvsrpp",
                "max_clip_size": "max_clip_size",
                "temporal_overlap": "temporal_overlap",
                "enable_crossfade": "enable_crossfade",
                "denoise": "denoise_strength",
                "denoise_step": "denoise_step",
                "secondary_restoration": "secondary_restoration",
                "swin2sr_batch_size": "swin2sr_batch_size",
                "swin2sr_compilation": "swin2sr_compilation",
                "tvai_ffmpeg_path": "tvai_ffmpeg_path",
                "tvai_model": "tvai_model",
                "tvai_scale": "tvai_scale",
                "tvai_workers": "tvai_workers",
                "detection_score_threshold": "detection_score_threshold",
                "codec": "codec",
                "encoder_settings": "encoder_custom_args",
            }
            
            dest = action.dest
            if dest in key_map:
                descriptions[key_map[dest]] = help_text
            elif dest.replace("-", "_") in key_map:
                descriptions[key_map[dest.replace("-", "_")]] = help_text
                
    return descriptions


# English translations (base language, synced from CLI where applicable)
_CLI_DESCRIPTIONS = None

def get_cli_descriptions() -> dict[str, str]:
    """Lazy load CLI descriptions."""
    global _CLI_DESCRIPTIONS
    if _CLI_DESCRIPTIONS is None:
        _CLI_DESCRIPTIONS = _get_cli_descriptions()
    return _CLI_DESCRIPTIONS


TRANSLATIONS = {
    "en": {
        # App
        "app_title": "JASNA GUI",
        "status_idle": "IDLE",
        "status_processing": "PROCESSING",
        "status_paused": "PAUSED",
        "status_completed": "COMPLETED",
        "status_error": "ERROR",
        
        # Header
        "btn_help": "Help",
        "btn_about": "About",
        "language": "Language",
        
        # Queue Panel
        "btn_add_files": "📁 Add Files",
        "queue_empty": "Drag and drop files here\nor use buttons above",
        "items_queued": "{count} item(s) queued",
        "btn_clear": "🗑 Clear",
        "btn_clear_completed": "✓ Clear Done",
        "output_location": "OUTPUT LOCATION",
        "output_pattern_placeholder": "{original}_restored.mp4",
        "same_as_input": "Same as input",
        "select_video_files": "Select Video Files",
        "select_folder": "Select Folder",
        "select_output_folder": "Select Output Folder",
        
        # Job Status
        "job_pending": "Pending",
        "job_processing": "Processing",
        "job_completed": "Completed",
        "job_error": "Error",
        "job_paused": "Paused",
        "job_skipped": "Skipped",
        
        # Settings Panel
        "preset": "Preset:",
        "btn_create": "+",
        "btn_save": "💾",
        "btn_delete": "🗑",
        "btn_reset": "↺",
        
        # Sections
        "section_basic": "Basic Processing",
        "section_advanced": "Advanced Processing",
        "section_secondary": "Secondary Restoration",
        "section_encoding": "Encoding",
        
        # Basic Processing
        "max_clip_size": "Max Clip Size",
        "detection_model": "Detection Model",
        "detection_threshold": "Detection Threshold",
        "fp16_mode": "FP16 Mode",
        "compile_basicvsrpp": "Compile BasicVSR++",
        "file_conflict": "File Conflict",
        "file_conflict_auto_rename": "Auto-Rename",
        "file_conflict_overwrite": "Overwrite",
        "file_conflict_skip": "Skip",
        "file_conflict_overwrite_warning": "Existing files will be replaced permanently",
        "tip_file_conflict": "What to do if output file already exists",
        "conflict_tooltip": "Output file already exists",
        "renamed_output": "Output file exists. Renamed to {filename}",
        
        # Advanced Processing
        "temporal_overlap": "Temporal Overlap",
        "enable_crossfade": "Enable Crossfade",
        "denoise_strength": "Denoise Strength",
        "denoise_step": "Denoise Apply After",
        "denoise_none": "None",
        "denoise_low": "Low",
        "denoise_medium": "Medium",
        "denoise_high": "High",
        "after_primary": "After Primary",
        "after_secondary": "After Secondary",
        
        # Secondary Restoration
        "secondary_none": "None",
        "secondary_swin2sr": "Swin2SR",
        "secondary_tvai": "Topaz TVAI",
        "batch_size": "Batch Size",
        "compile_model": "Compile Model",
        "ffmpeg_path": "FFmpeg Path",
        "model": "Model",
        "scale": "Scale",
        "workers": "Workers",
        
        # Encoding
        "codec": "Codec",
        "quality_cq": "Quality (CQ)",
        "custom_args": "Custom Args",
        
        # Control Bar
        "btn_start": "▶ Start",
        "btn_pause": "⏸ Pause",
        "btn_resume": "▶ Resume",
        "btn_stop": "⏹ Stop",
        "progress": "Progress",
        "time_remaining": "Remaining",
        "no_file_processing": "No file processing",
        "queue_label": "QUEUE",
        "logs_btn": ">_ LOGS",
        
        # Log Panel
        "logs": "Logs",
        "btn_export": "Export",
        "btn_toggle_logs": "Logs ▼",
        "filter_all": "All",
        "filter_debug": "Debug",
        "filter_info": "Info",
        "filter_warn": "Warn",
        "filter_error": "Error",
        "system_output": "SYSTEM OUTPUT",
        "filter_all_levels": "All Levels",
        "filter_errors_only": "Errors Only",
        "filter_warnings_plus": "Warnings+",
        "filter_info_plus": "Info+",
        
        # Wizard
        "wizard_title": "System Check",
        "wizard_subtitle": "Checking required dependencies...",
        "wizard_checking": "Checking...",
        "wizard_found": "Found: {path}",
        "wizard_not_found": "Not found in PATH",
        "wizard_all_passed": "✓ All checks passed! You're ready to go.",
        "wizard_some_failed": "⚠ Some dependencies are missing. Check the README for setup instructions.",
        "btn_get_started": "Get Started",
        "btn_continue_anyway": "Continue Anyway",
        "btn_ok": "OK",
        
        # Dialogs
        "dialog_create_preset": "Create Preset",
        "preset_name": "Preset Name",
        "preset_placeholder": "My Custom Preset",
        "error_name_empty": "Name cannot be empty",
        "error_name_exists": "Name already exists",
        "btn_create_preset": "Create",
        "btn_cancel": "Cancel",
        "dialog_delete_preset": "Delete Preset",
        "confirm_delete": "Delete preset '{name}'?",
        "btn_delete_confirm": "Delete",
        
        # Toasts
        "toast_preset_saved": "Preset '{name}' saved",
        "toast_preset_created": "Preset '{name}' created",
        "toast_preset_deleted": "Preset '{name}' deleted",
        "toast_settings_reset": "Settings reset",
        "toast_no_files": "No files in queue",
        "toast_started": "Processing started",
        "toast_paused": "Processing paused",
        "toast_resumed": "Processing resumed",
        "toast_stopped": "Processing stopped",
        # Buy Me a Coffee
        "bmc_support": "Support",
        
        # Tooltips (from CLI)
        "tip_max_clip_size": "Maximum clip size for tracking",
        "tip_temporal_overlap": "Discard margin for overlap+discard clip splitting. Each split uses 2*temporal_overlap input overlap and discards temporal_overlap frames at each split boundary",
        "tip_enable_crossfade": "Cross-fade between clip boundaries to reduce flickering at seams. Uses frames that are already processed but otherwise discarded, so no extra GPU cost",
        "tip_fp16_mode": "Use FP16 where supported (restoration + TensorRT). Reduces VRAM usage and might improve performance",
        "tip_compile_basicvsrpp": "Compile BasicVSR++ for big performance boost (at cost of VRAM usage). Not recommended to use big clip sizes",
        "tip_denoise_strength": "Spatial denoising strength applied to restored crops. Reduces noise artifacts",
        "tip_denoise_step": "When to apply denoising: after_primary (before secondary) or after_secondary (right before blend)",
        "tip_secondary_restoration": "Secondary restoration after primary model",
        "tip_swin2sr_batch_size": "Batch size for Swin2SR secondary restoration",
        "tip_swin2sr_compilation": "Enable Swin2SR TensorRT compilation/usage where supported",
        "tip_tvai_ffmpeg_path": "Path to Topaz Video AI ffmpeg.exe",
        "tip_tvai_model": "Topaz model name for tvai_up (e.g. iris-2, prob-4, iris-3)",
        "tip_tvai_scale": "Topaz tvai_up scale (1=no scale). Output size is 256*scale",
        "tip_tvai_workers": "Number of parallel TVAI ffmpeg workers",
        "tip_detection_model": "Detection model version",
        "tip_detection_score_threshold": "Detection score threshold",
        "tip_codec": "Output video codec (only HEVC supported for now)",
        "tip_encoder_cq": "Constant quality value for encoder (lower = better quality, larger file)",
        "tip_encoder_custom_args": "Encoder settings as comma-separated key=value pairs (e.g. cq=22,lookahead=32)",
        
        # Preset button tooltips
        "tip_preset_reset": "Reset to saved values",
        "tip_preset_delete": "Delete preset",
        "tip_preset_save": "Save preset",
        "tip_preset_create": "Create new preset",
    },
    
    "zh": {
        # App
        "app_title": "JASNA 图形界面",
        "status_idle": "空闲",
        "status_processing": "处理中",
        "status_paused": "已暂停",
        "status_completed": "已完成",
        "status_error": "错误",
        
        # Header
        "btn_help": "帮助",
        "btn_about": "关于",
        "language": "语言",
        
        # Queue Panel
        "btn_add_files": "📁 添加文件",
        "queue_empty": "拖放文件到这里\n或使用上方按钮",
        "items_queued": "队列中有 {count} 个项目",
        "btn_clear": "🗑 清空",
        "btn_clear_completed": "✓ 清除完成",
        "output_location": "输出位置",
        "output_pattern_placeholder": "{original}_restored.mp4",
        "same_as_input": "与输入相同",
        "select_video_files": "选择视频文件",
        "select_folder": "选择文件夹",
        "select_output_folder": "选择输出文件夹",
        
        # Job Status
        "job_pending": "等待中",
        "job_processing": "处理中",
        "job_completed": "已完成",
        "job_error": "错误",
        "job_paused": "已暂停",
        "job_skipped": "已跳过",
        
        # Settings Panel
        "preset": "预设:",
        "btn_create": "+",
        "btn_save": "💾",
        "btn_delete": "🗑",
        "btn_reset": "↺",
        
        # Sections
        "section_basic": "基本处理",
        "section_advanced": "高级处理",
        "section_secondary": "二次修复",
        "section_encoding": "编码设置",
        
        # Basic Processing
        "max_clip_size": "最大片段大小",
        "detection_model": "检测模型",
        "detection_threshold": "检测阈值",
        "fp16_mode": "FP16 模式",
        "compile_basicvsrpp": "编译 BasicVSR++",
        "file_conflict": "文件冲突",
        "file_conflict_auto_rename": "自动重命名",
        "file_conflict_overwrite": "覆盖",
        "file_conflict_skip": "跳过",
        "file_conflict_overwrite_warning": "现有文件将被永久替换",
        "tip_file_conflict": "输出文件已存在时的处理方式",
        "conflict_tooltip": "输出文件已存在",
        "renamed_output": "输出文件已存在。已重命名为 {filename}",
        
        # Advanced Processing
        "temporal_overlap": "时间重叠",
        "enable_crossfade": "启用交叉淡入淡出",
        "denoise_strength": "降噪强度",
        "denoise_step": "降噪应用时机",
        "denoise_none": "无",
        "denoise_low": "低",
        "denoise_medium": "中",
        "denoise_high": "高",
        "after_primary": "主修复后",
        "after_secondary": "二次修复后",
        
        # Secondary Restoration
        "secondary_none": "无",
        "secondary_swin2sr": "Swin2SR",
        "secondary_tvai": "Topaz TVAI",
        "batch_size": "批处理大小",
        "compile_model": "编译模型",
        "ffmpeg_path": "FFmpeg 路径",
        "model": "模型",
        "scale": "缩放",
        "workers": "工作线程数",
        
        # Encoding
        "codec": "编解码器",
        "quality_cq": "质量 (CQ)",
        "custom_args": "自定义参数",
        
        # Control Bar
        "btn_start": "▶ 开始",
        "btn_pause": "⏸ 暂停",
        "btn_resume": "▶ 继续",
        "btn_stop": "⏹ 停止",
        "progress": "进度",
        "time_remaining": "剩余时间",
        "no_file_processing": "无文件处理",
        "queue_label": "队列",
        "logs_btn": ">_ 日志",
        
        # Log Panel
        "logs": "日志",
        "btn_export": "导出",
        "btn_toggle_logs": "日志 ▼",
        "filter_all": "全部",
        "filter_debug": "调试",
        "filter_info": "信息",
        "filter_warn": "警告",
        "filter_error": "错误",
        "system_output": "系统输出",
        "filter_all_levels": "全部级别",
        "filter_errors_only": "仅错误",
        "filter_warnings_plus": "警告+",
        "filter_info_plus": "信息+",
        
        # Wizard
        "wizard_title": "系统检查",
        "wizard_subtitle": "正在检查依赖项...",
        "wizard_checking": "检查中...",
        "wizard_found": "已找到: {path}",
        "wizard_not_found": "未在 PATH 中找到",
        "wizard_all_passed": "✓ 所有检查已通过！可以开始使用了。",
        "wizard_some_failed": "⚠ 缺少部分依赖项。请查看 README 获取安装说明。",
        "btn_get_started": "开始使用",
        "btn_continue_anyway": "仍然继续",
        "btn_ok": "确定",
        
        # Dialogs
        "dialog_create_preset": "创建预设",
        "preset_name": "预设名称",
        "preset_placeholder": "我的自定义预设",
        "error_name_empty": "名称不能为空",
        "error_name_exists": "名称已存在",
        "btn_create_preset": "创建",
        "btn_cancel": "取消",
        "dialog_delete_preset": "删除预设",
        "confirm_delete": "删除预设 '{name}'?",
        "btn_delete_confirm": "删除",
        
        # Toasts
        "toast_preset_saved": "预设 '{name}' 已保存",
        "toast_preset_created": "预设 '{name}' 已创建",
        "toast_preset_deleted": "预设 '{name}' 已删除",
        "toast_settings_reset": "设置已重置",
        "toast_no_files": "队列中没有文件",
        "toast_started": "处理已开始",
        "toast_paused": "处理已暂停",
        "toast_resumed": "处理已继续",
        "toast_stopped": "处理已停止",
        # Buy Me a Coffee
        "bmc_support": "支持",
        
        # Tooltips
        "tip_max_clip_size": "跟踪的最大片段大小",
        "tip_temporal_overlap": "重叠+丢弃片段分割的丢弃边距。每次分割使用 2*temporal_overlap 输入重叠，并在每个分割边界丢弃 temporal_overlap 帧",
        "tip_enable_crossfade": "在片段边界之间进行交叉淡入淡出以减少接缝处的闪烁。使用已处理但原本会被丢弃的帧，因此没有额外的 GPU 开销",
        "tip_fp16_mode": "在支持的地方使用 FP16 (修复 + TensorRT)。减少显存使用并可能提高性能",
        "tip_compile_basicvsrpp": "编译 BasicVSR++ 以获得显著的性能提升（以显存使用为代价）。不建议使用大的片段大小",
        "tip_denoise_strength": "应用于修复区域的空间降噪强度。减少噪点伪影",
        "tip_denoise_step": "何时应用降噪：after_primary（二次修复前）或 after_secondary（混合前）",
        "tip_secondary_restoration": "主模型之后的二次修复",
        "tip_swin2sr_batch_size": "Swin2SR 二次修复的批处理大小",
        "tip_swin2sr_compilation": "在支持的情况下启用 Swin2SR TensorRT 编译/使用",
        "tip_tvai_ffmpeg_path": "Topaz Video AI ffmpeg.exe 的路径",
        "tip_tvai_model": "tvai_up 的 Topaz 模型名称（例如 iris-2、prob-4、iris-3）",
        "tip_tvai_scale": "Topaz tvai_up 缩放（1=不缩放）。输出大小为 256*scale",
        "tip_tvai_workers": "并行 TVAI ffmpeg 工作线程数",
        "tip_detection_model": "检测模型版本",
        "tip_detection_score_threshold": "检测分数阈值",
        "tip_codec": "输出视频编解码器（目前仅支持 HEVC）",
        "tip_encoder_cq": "编码器的恒定质量值（越低 = 质量越好，文件越大）",
        "tip_encoder_custom_args": "编码器设置，以逗号分隔的 key=value 对（例如 cq=22,lookahead=32）",
        
        # Preset button tooltips
        "tip_preset_reset": "重置为保存的值",
        "tip_preset_delete": "删除预设",
        "tip_preset_save": "保存预设",
        "tip_preset_create": "创建新预设",
    },
    
    "es": {
        "app_title": "JASNA GUI",
        "status_idle": "INACTIVO",
        "status_processing": "PROCESANDO",
        "btn_help": "Ayuda",
        "btn_about": "Acerca de",
        "language": "Idioma",
        "btn_add_files": "📁 Añadir Archivos",
        "btn_start": "▶ Iniciar",
        "btn_pause": "⏸ Pausar",
        "btn_stop": "⏹ Detener",
        "btn_ok": "OK",
        # Add more as needed...
    },
    
    "de": {
        "app_title": "JASNA GUI",
        "status_idle": "BEREIT",
        "status_processing": "VERARBEITUNG",
        "btn_help": "Hilfe",
        "btn_about": "Über",
        "language": "Sprache",
        "btn_add_files": "📁 Dateien hinzufügen",
        "btn_start": "▶ Starten",
        "btn_pause": "⏸ Pause",
        "btn_stop": "⏹ Stopp",
        "btn_ok": "OK",
        # Add more as needed...
    },
}


LANGUAGE_NAMES = {
    "en": "English",
    "zh": "简体中文",
    "es": "Español",
    "de": "Deutsch",
}


class LocaleManager:
    """Manages language selection and translation lookup."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._current_lang = "en"
        self._listeners: list[Callable[[], None]] = []
        self._load()
        
    def _load(self):
        """Load language preference from settings."""
        path = get_settings_path()
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._current_lang = data.get("language", "en")
            except (json.JSONDecodeError, IOError):
                pass
        else:
            # If settings.json is missing, try to autodetect system language
            try:
                lang, _ = _locale.getdefaultlocale()
                if lang and lang.startswith("zh"):
                    self._current_lang = "zh"
            except Exception:
                # Fall back to default 'en'
                pass
                
    def _save(self):
        """Save language preference to settings."""
        path = get_settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        data["language"] = self._current_lang
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except IOError:
            pass
            
    @property
    def current_language(self) -> str:
        return self._current_lang
    
    @property
    def available_languages(self) -> list[str]:
        return list(LANGUAGE_NAMES.keys())
    
    def get_language_name(self, code: str) -> str:
        return LANGUAGE_NAMES.get(code, code)
    
    def set_language(self, lang: str):
        """Set current language and notify listeners."""
        if lang not in TRANSLATIONS:
            lang = "en"
        self._current_lang = lang
        self._save()
        for listener in self._listeners:
            listener()
            
    def add_listener(self, callback: Callable[[], None]):
        """Add a callback to be called when language changes."""
        self._listeners.append(callback)
        
    def remove_listener(self, callback: Callable[[], None]):
        """Remove a language change listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)
    
    def get(self, key: str, **kwargs) -> str:
        """Get translation for key. Falls back to English if not found."""
        translations = TRANSLATIONS.get(self._current_lang, TRANSLATIONS["en"])
        text = translations.get(key)
        
        # Fallback to English
        if text is None:
            text = TRANSLATIONS["en"].get(key, key)
            
        # Format with kwargs
        if kwargs:
            try:
                text = text.format(**kwargs)
            except (KeyError, ValueError):
                pass
                
        return text
    
    def __call__(self, key: str, **kwargs) -> str:
        """Shorthand for get()."""
        return self.get(key, **kwargs)


# Global instance
_locale = None

def get_locale() -> LocaleManager:
    """Get the global LocaleManager instance."""
    global _locale
    if _locale is None:
        _locale = LocaleManager()
    return _locale


def t(key: str, **kwargs) -> str:
    """Translate a key. Shorthand for get_locale().get(key)."""
    return get_locale().get(key, **kwargs)
