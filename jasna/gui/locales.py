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

        # Engine compilation / first run warnings
        "engine_first_run_title": "First run may be slow",
        "engine_first_run_body": "Some TensorRT engines are missing and may be compiled for your GPU. This is normal on the first run. The application may appear unresponsive during compilation. Do not close it.",
        "engine_first_run_missing": "Missing engines:",
        "engine_name_rfdetr": "RF-DETR (detection)",
        "engine_name_yolo": "YOLO (detection)",
        "engine_name_basicvsrpp": "BasicVSR++ (restoration)",
        "engine_name_swin2sr": "Swin2SR (secondary)",
        "engine_basicvsrpp_risky_title": "BasicVSR++ compilation warning",
        "engine_basicvsrpp_risky_body": "BasicVSR++ TensorRT compilation may be risky with your GPU VRAM.\n\nGPU VRAM (approx): {vram_gb} GB\nRequested clip size: {requested_clip}\nApprox safe max: {safe_clip}\n\nContinue with compilation anyway? This can take a long time and may run out of VRAM.",
        # About dialog
        "dialog_about_title": "About Jasna",
        "dialog_about_version": "Version {version}",
        "dialog_about_description": "JAV mosaic restoration tool",
        "dialog_about_credit": "Inspired by Lada",
        "btn_close": "Close",

        # Language change dialog
        "dialog_language_changed": "Language Changed",
        "dialog_language_restart": "Please restart the application for full language change.",

        # App messages
        "toast_select_output": "Please select an output folder before starting",
        "error_cannot_start": "Cannot start processing:",
        "error_invalid_tvai": "Invalid TVAI configuration",

        # Settings panel
        "dialog_select_tvai_ffmpeg": "Select Topaz Video AI ffmpeg.exe",
        "placeholder_encoder_args": "e.g. lookahead=32",

        # Wizard check labels
        "wizard_window_title": "Jasna - System Check",
        "wizard_check_ffmpeg": "FFmpeg",
        "wizard_check_ffprobe": "FFprobe",
        "wizard_check_mkvmerge": "MKVmerge",
        "wizard_check_gpu": "NVIDIA GPU",
        "wizard_check_cuda": "CUDA Runtime",
        "wizard_check_hags": "Hardware Accelerated GPU Scheduling",
        "wizard_not_checked": "Not checked",
        "wizard_not_callable": "Not callable: {path}",
        "wizard_found_version": "Found: {path} ({version})",
        "wizard_found_no_major": "Found: {path} (could not detect major version)",
        "wizard_found_bad_major": "Found: {path} (major={major}, expected=8)",
        "wizard_found_major": "Found: {path} (major={major})",
        "wizard_no_cuda": "No CUDA device",
        "wizard_gpu_compute_too_low": "Compute capability 7.5+ required (GPU: {major}.{minor})",
        "wizard_cuda_version": "CUDA {version}",
        "wizard_cuda_version_compute": "CUDA {version}, compute {major}.{minor}",
        "wizard_not_available": "Not available",

        # Validation errors
        "error_tvai_data_dir_not_set": "TVAI_MODEL_DATA_DIR env var is not set",
        "error_tvai_model_dir_not_set": "TVAI_MODEL_DIR env var is not set",
        "error_tvai_data_dir_missing": "TVAI_MODEL_DATA_DIR does not point to an existing directory: {path}",
        "error_tvai_model_dir_missing": "TVAI_MODEL_DIR does not point to an existing directory: {path}",
        "error_tvai_ffmpeg_not_found": "TVAI ffmpeg not found: {path}",
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
        "btn_clear_completed": "✓ 清除已完成",
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
        "no_file_processing": "未在处理文件",
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

        # Engine compilation / first run warnings
        "engine_first_run_title": "首次运行可能较慢",
        "engine_first_run_body": "检测到部分 TensorRT 引擎缺失，Jasna 可能需要为你的 GPU 进行编译。首次运行出现这种情况是正常的。编译期间应用可能看起来无响应，请不要关闭。",
        "engine_first_run_missing": "缺失的引擎：",
        "engine_name_rfdetr": "RF-DETR（检测）",
        "engine_name_yolo": "YOLO（检测）",
        "engine_name_basicvsrpp": "BasicVSR++（修复）",
        "engine_name_swin2sr": "Swin2SR（二次）",
        "engine_basicvsrpp_risky_title": "BasicVSR++ 编译警告",
        "engine_basicvsrpp_risky_body": "BasicVSR++ TensorRT 编译可能会因显存不足而存在风险。\n\n显存（约）：{vram_gb} GB\n请求的片段大小：{requested_clip}\n建议安全上限：{safe_clip}\n\n仍要继续编译吗？这可能耗时很长并且可能会因显存不足而失败。",
        # About dialog
        "dialog_about_title": "关于 Jasna",
        "dialog_about_version": "版本 {version}",
        "dialog_about_description": "JAV 马赛克修复工具",
        "dialog_about_credit": "灵感来源于 Lada",
        "btn_close": "关闭",

        # Language change dialog
        "dialog_language_changed": "语言已更改",
        "dialog_language_restart": "请重启应用程序以完成语言切换。",

        # App messages
        "toast_select_output": "请在开始前选择输出文件夹",
        "error_cannot_start": "无法开始处理：",
        "error_invalid_tvai": "无效的 TVAI 配置",

        # Settings panel
        "dialog_select_tvai_ffmpeg": "选择 Topaz Video AI ffmpeg.exe",
        "placeholder_encoder_args": "例如 lookahead=32",

        # Wizard check labels
        "wizard_window_title": "Jasna - 系统检查",
        "wizard_check_ffmpeg": "FFmpeg",
        "wizard_check_ffprobe": "FFprobe",
        "wizard_check_mkvmerge": "MKVmerge",
        "wizard_check_gpu": "NVIDIA GPU",
        "wizard_check_cuda": "CUDA 运行时",
        "wizard_check_hags": "硬件加速 GPU 调度",
        "wizard_not_checked": "未检查",
        "wizard_not_callable": "无法调用：{path}",
        "wizard_found_version": "已找到：{path}（{version}）",
        "wizard_found_no_major": "已找到：{path}（无法检测主版本号）",
        "wizard_found_bad_major": "已找到：{path}（主版本={major}，期望=8）",
        "wizard_found_major": "已找到：{path}（主版本={major}）",
        "wizard_no_cuda": "无 CUDA 设备",
        "wizard_gpu_compute_too_low": "需要计算能力 7.5 或更高（当前 GPU：{major}.{minor}）",
        "wizard_cuda_version": "CUDA {version}",
        "wizard_cuda_version_compute": "CUDA {version}，计算能力 {major}.{minor}",
        "wizard_not_available": "不可用",

        # Validation errors
        "error_tvai_data_dir_not_set": "环境变量 TVAI_MODEL_DATA_DIR 未设置",
        "error_tvai_model_dir_not_set": "环境变量 TVAI_MODEL_DIR 未设置",
        "error_tvai_data_dir_missing": "TVAI_MODEL_DATA_DIR 指向的目录不存在：{path}",
        "error_tvai_model_dir_missing": "TVAI_MODEL_DIR 指向的目录不存在：{path}",
        "error_tvai_ffmpeg_not_found": "TVAI ffmpeg 未找到：{path}",
    },

    "ja": {
        # App
        "app_title": "JASNA GUI",
        "status_idle": "待機中",
        "status_processing": "処理中",
        "status_paused": "一時停止",
        "status_completed": "完了",
        "status_error": "エラー",

        # Header
        "btn_help": "ヘルプ",
        "btn_about": "このアプリについて",
        "language": "言語",

        # Queue Panel
        "btn_add_files": "📁 ファイル追加",
        "queue_empty": "ここにファイルをドラッグ＆ドロップ\nまたは上のボタンを使用",
        "items_queued": "{count} 件がキューに追加済み",
        "btn_clear": "🗑 クリア",
        "btn_clear_completed": "✓ 完了済みを削除",
        "output_location": "出力先",
        "output_pattern_placeholder": "{original}_restored.mp4",
        "same_as_input": "入力と同じ",
        "select_video_files": "動画ファイルを選択",
        "select_folder": "フォルダを選択",
        "select_output_folder": "出力フォルダを選択",

        # Job Status
        "job_pending": "待機中",
        "job_processing": "処理中",
        "job_completed": "完了",
        "job_error": "エラー",
        "job_paused": "一時停止",
        "job_skipped": "スキップ",

        # Settings Panel
        "preset": "プリセット:",
        "btn_create": "+",
        "btn_save": "💾",
        "btn_delete": "🗑",
        "btn_reset": "↺",

        # Sections
        "section_basic": "基本設定",
        "section_advanced": "詳細設定",
        "section_secondary": "二次修復",
        "section_encoding": "エンコード",

        # Basic Processing
        "max_clip_size": "最大クリップサイズ",
        "detection_model": "検出モデル",
        "detection_threshold": "検出しきい値",
        "fp16_mode": "FP16 モード",
        "compile_basicvsrpp": "BasicVSR++ コンパイル",
        "file_conflict": "ファイル競合",
        "file_conflict_auto_rename": "自動リネーム",
        "file_conflict_overwrite": "上書き",
        "file_conflict_skip": "スキップ",
        "file_conflict_overwrite_warning": "既存ファイルは完全に置き換えられます",
        "tip_file_conflict": "出力ファイルが既に存在する場合の動作",
        "conflict_tooltip": "出力ファイルが既に存在します",
        "renamed_output": "出力ファイルが既に存在します。{filename} にリネームしました",

        # Advanced Processing
        "temporal_overlap": "時間オーバーラップ",
        "enable_crossfade": "クロスフェード有効化",
        "denoise_strength": "ノイズ除去強度",
        "denoise_step": "ノイズ除去適用タイミング",
        "denoise_none": "なし",
        "denoise_low": "低",
        "denoise_medium": "中",
        "denoise_high": "高",
        "after_primary": "一次修復後",
        "after_secondary": "二次修復後",

        # Secondary Restoration
        "secondary_none": "なし",
        "secondary_swin2sr": "Swin2SR",
        "secondary_tvai": "Topaz TVAI",
        "batch_size": "バッチサイズ",
        "compile_model": "モデルコンパイル",
        "ffmpeg_path": "FFmpeg パス",
        "model": "モデル",
        "scale": "スケール",
        "workers": "ワーカー数",

        # Encoding
        "codec": "コーデック",
        "quality_cq": "品質 (CQ)",
        "custom_args": "カスタム引数",

        # Control Bar
        "btn_start": "▶ 開始",
        "btn_pause": "⏸ 一時停止",
        "btn_resume": "▶ 再開",
        "btn_stop": "⏹ 停止",
        "progress": "進捗",
        "time_remaining": "残り時間",
        "no_file_processing": "処理中のファイルなし",
        "queue_label": "キュー",
        "logs_btn": ">_ ログ",

        # Log Panel
        "logs": "ログ",
        "btn_export": "エクスポート",
        "btn_toggle_logs": "ログ ▼",
        "filter_all": "すべて",
        "filter_debug": "デバッグ",
        "filter_info": "情報",
        "filter_warn": "警告",
        "filter_error": "エラー",
        "system_output": "システム出力",
        "filter_all_levels": "全レベル",
        "filter_errors_only": "エラーのみ",
        "filter_warnings_plus": "警告以上",
        "filter_info_plus": "情報以上",

        # Wizard
        "wizard_title": "システムチェック",
        "wizard_subtitle": "必要な依存関係を確認中...",
        "wizard_checking": "確認中...",
        "wizard_found": "検出: {path}",
        "wizard_not_found": "PATH に見つかりません",
        "wizard_all_passed": "✓ すべてのチェックに合格しました！準備完了です。",
        "wizard_some_failed": "⚠ 一部の依存関係が不足しています。README のセットアップ手順を確認してください。",
        "btn_get_started": "開始する",
        "btn_continue_anyway": "続行する",
        "btn_ok": "OK",

        # Dialogs
        "dialog_create_preset": "プリセット作成",
        "preset_name": "プリセット名",
        "preset_placeholder": "カスタムプリセット",
        "error_name_empty": "名前を入力してください",
        "error_name_exists": "この名前は既に使用されています",
        "btn_create_preset": "作成",
        "btn_cancel": "キャンセル",
        "dialog_delete_preset": "プリセット削除",
        "confirm_delete": "プリセット '{name}' を削除しますか？",
        "btn_delete_confirm": "削除",

        # Toasts
        "toast_preset_saved": "プリセット '{name}' を保存しました",
        "toast_preset_created": "プリセット '{name}' を作成しました",
        "toast_preset_deleted": "プリセット '{name}' を削除しました",
        "toast_settings_reset": "設定をリセットしました",
        "toast_no_files": "キューにファイルがありません",
        "toast_started": "処理を開始しました",
        "toast_paused": "処理を一時停止しました",
        "toast_resumed": "処理を再開しました",
        "toast_stopped": "処理を停止しました",
        # Buy Me a Coffee
        "bmc_support": "応援する",

        # Tooltips (from CLI)
        "tip_max_clip_size": "トラッキングの最大クリップサイズ",
        "tip_temporal_overlap": "オーバーラップ＋破棄によるクリップ分割の破棄マージン。各分割は 2*temporal_overlap の入力オーバーラップを使用し、各境界で temporal_overlap フレームを破棄します",
        "tip_enable_crossfade": "クリップ境界間でクロスフェードを行い、つなぎ目のちらつきを軽減します。処理済みだが破棄されるフレームを使用するため、追加の GPU コストはかかりません",
        "tip_fp16_mode": "対応する処理で FP16 を使用（修復 + TensorRT）。VRAM 使用量を削減し、パフォーマンスが向上する場合があります",
        "tip_compile_basicvsrpp": "BasicVSR++ をコンパイルして大幅なパフォーマンス向上を実現（VRAM 使用量が増加します）。大きなクリップサイズの使用は推奨されません",
        "tip_denoise_strength": "修復されたクロップに適用する空間ノイズ除去の強度。ノイズアーティファクトを低減します",
        "tip_denoise_step": "ノイズ除去の適用タイミング: after_primary（二次修復前）または after_secondary（ブレンド直前）",
        "tip_secondary_restoration": "一次モデルの後に行う二次修復",
        "tip_swin2sr_batch_size": "Swin2SR 二次修復のバッチサイズ",
        "tip_swin2sr_compilation": "対応環境で Swin2SR TensorRT コンパイル/使用を有効化",
        "tip_tvai_ffmpeg_path": "Topaz Video AI の ffmpeg.exe のパス",
        "tip_tvai_model": "tvai_up の Topaz モデル名（例: iris-2, prob-4, iris-3）",
        "tip_tvai_scale": "Topaz tvai_up のスケール（1=スケールなし）。出力サイズは 256*scale",
        "tip_tvai_workers": "並列 TVAI ffmpeg ワーカー数",
        "tip_detection_model": "検出モデルのバージョン",
        "tip_detection_score_threshold": "検出スコアのしきい値",
        "tip_codec": "出力動画コーデック（現在は HEVC のみ対応）",
        "tip_encoder_cq": "エンコーダーの固定品質値（低い値 = 高品質・大きなファイル）",
        "tip_encoder_custom_args": "エンコーダー設定（カンマ区切りの key=value 形式。例: cq=22,lookahead=32）",

        # Preset button tooltips
        "tip_preset_reset": "保存済みの値にリセット",
        "tip_preset_delete": "プリセットを削除",
        "tip_preset_save": "プリセットを保存",
        "tip_preset_create": "新しいプリセットを作成",

        # Engine compilation / first run warnings
        "engine_first_run_title": "初回起動は時間がかかる場合があります",
        "engine_first_run_body": "一部の TensorRT エンジンが見つからず、お使いの GPU 向けにコンパイルされる場合があります。初回起動時にはこれは正常です。コンパイル中はアプリケーションが応答しないように見える場合がありますが、閉じないでください。",
        "engine_first_run_missing": "不足しているエンジン:",
        "engine_name_rfdetr": "RF-DETR（検出）",
        "engine_name_yolo": "YOLO（検出）",
        "engine_name_basicvsrpp": "BasicVSR++（修復）",
        "engine_name_swin2sr": "Swin2SR（二次）",
        "engine_basicvsrpp_risky_title": "BasicVSR++ コンパイル警告",
        "engine_basicvsrpp_risky_body": "BasicVSR++ TensorRT のコンパイルは、GPU の VRAM 不足によりリスクがあります。\n\nGPU VRAM（概算）: {vram_gb} GB\n要求クリップサイズ: {requested_clip}\n推定安全上限: {safe_clip}\n\nこのままコンパイルを続行しますか？長時間かかる可能性があり、VRAM 不足で失敗する場合があります。",
        # About dialog
        "dialog_about_title": "Jasna について",
        "dialog_about_version": "バージョン {version}",
        "dialog_about_description": "JAV モザイク修復ツール",
        "dialog_about_credit": "Lada にインスパイア",
        "btn_close": "閉じる",

        # Language change dialog
        "dialog_language_changed": "言語が変更されました",
        "dialog_language_restart": "言語変更を完全に反映するには、アプリケーションを再起動してください。",

        # App messages
        "toast_select_output": "開始前に出力フォルダを選択してください",
        "error_cannot_start": "処理を開始できません:",
        "error_invalid_tvai": "無効な TVAI 設定",

        # Settings panel
        "dialog_select_tvai_ffmpeg": "Topaz Video AI の ffmpeg.exe を選択",
        "placeholder_encoder_args": "例: lookahead=32",

        # Wizard check labels
        "wizard_window_title": "Jasna - システムチェック",
        "wizard_check_ffmpeg": "FFmpeg",
        "wizard_check_ffprobe": "FFprobe",
        "wizard_check_mkvmerge": "MKVmerge",
        "wizard_check_gpu": "NVIDIA GPU",
        "wizard_check_cuda": "CUDA ランタイム",
        "wizard_check_hags": "ハードウェアアクセラレータによる GPU スケジューリング",
        "wizard_not_checked": "未確認",
        "wizard_not_callable": "実行不可: {path}",
        "wizard_found_version": "検出: {path}（{version}）",
        "wizard_found_no_major": "検出: {path}（メジャーバージョンを検出できません）",
        "wizard_found_bad_major": "検出: {path}（major={major}, 期待値=8）",
        "wizard_found_major": "検出: {path}（major={major}）",
        "wizard_no_cuda": "CUDA デバイスなし",
        "wizard_gpu_compute_too_low": "Compute capability 7.5 以上が必要です（GPU: {major}.{minor}）",
        "wizard_cuda_version": "CUDA {version}",
        "wizard_cuda_version_compute": "CUDA {version}、compute {major}.{minor}",
        "wizard_not_available": "利用不可",

        # Validation errors
        "error_tvai_data_dir_not_set": "環境変数 TVAI_MODEL_DATA_DIR が設定されていません",
        "error_tvai_model_dir_not_set": "環境変数 TVAI_MODEL_DIR が設定されていません",
        "error_tvai_data_dir_missing": "TVAI_MODEL_DATA_DIR が既存のディレクトリを指していません: {path}",
        "error_tvai_model_dir_missing": "TVAI_MODEL_DIR が既存のディレクトリを指していません: {path}",
        "error_tvai_ffmpeg_not_found": "TVAI ffmpeg が見つかりません: {path}",
    },

}


LANGUAGE_NAMES = {
    "en": "English",
    "zh": "简体中文",
    "ja": "日本語",
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
                elif lang and lang.startswith("ja"):
                    self._current_lang = "ja"
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
