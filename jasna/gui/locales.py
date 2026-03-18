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
                "tvai_ffmpeg_path": "tvai_ffmpeg_path",
                "tvai_model": "tvai_model",
                "tvai_scale": "tvai_scale",
                "tvai_workers": "tvai_workers",
                "detection_score_threshold": "detection_score_threshold",
                "codec": "codec",
                "encoder_settings": "encoder_custom_args",
                "working_directory": "working_directory",
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
        "btn_system_check": "System Check",
        "language": "Language",
        
        # Queue Panel
        "btn_add_files": "📁 Add Files",
        "queue_empty": "Drag and drop files here\nor use buttons above",
        "items_queued": "{count} item(s) queued",
        "btn_clear": "🗑 Clear",
        "btn_clear_completed": "✓ Clear Done",
        "output_location": "OUTPUT LOCATION",
        "output_pattern": "OUTPUT PATTERN",
        "output_pattern_placeholder": "{original}_restored.mp4",
        "same_as_input": "Same as input",
        "select_video_files": "Select Video Files",
        "select_folder": "Select Folder",
        "select_output_folder": "Select Output Folder",
        
        # Job Status
        "job_pending": "Pending",
        "job_processing": "Processing",
        "job_completed": "Completed",
        "completed_in": "Completed in",
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
        "working_directory": "Working Directory",
        "working_directory_placeholder": "Optional, same as output folder",
        "dialog_select_working_directory": "Select Working Directory",
        
        # Secondary Restoration
        "secondary_none": "None",
        "secondary_tvai": "Topaz TVAI",
        "secondary_tvai_hint": "best quality, slowest",
        "secondary_rtx_super_res": "RTX Super Res",
        "secondary_rtx_hint": "fast, may flicker",
        "rtx_scale": "Scale",
        "rtx_quality": "Quality",
        "rtx_denoise": "Denoise",
        "rtx_deblur": "Deblur",
        "recommended": "Recommended",
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
        
        # Tooltips
        "tip_max_clip_size": "How many frames are processed at once. Larger values can improve quality but use more VRAM.\n\nRecommended: 60 or higher. Use 60 even if it means disabling model compilation.\nGuidance: 60 (safe), 90 (good balance), 180 (best quality, needs 24 GB+ VRAM).\nDefault: 60",
        "tip_temporal_overlap": "Overlap between processed clips to reduce flickering at boundaries.\nHigher = smoother transitions but slightly slower. Going above 20 has little benefit.\n\nRecommended values based on clip size:\n- Clip 60 → overlap 6-8\n- Clip 90 → overlap 8-12\n- Clip 180 → overlap 15-20\nDefault: 8",
        "tip_enable_crossfade": "Smoothly blends clip boundaries to reduce flickering. Reuses already-processed frames so there is zero extra GPU cost.\n\nRecommended: Always ON.\nDefault: ON",
        "tip_fp16_mode": "Uses half-precision math to reduce VRAM usage and often run faster. No visible quality loss on modern GPUs.\n\nRecommended: ON for RTX 20-series and newer.\nDefault: ON",
        "tip_compile_basicvsrpp": "Compiles the restoration model into TensorRT sub-engines for a big speed boost (~2-3x faster).\nFirst compilation takes 15-60 minutes. Close all other applications (including browsers) and avoid using the PC during compilation.\nEngines are cached and reused on subsequent runs.\n\nEngine VRAM: ~1.9 GB (clip 60), ~5.4 GB (clip 180).\nPeak VRAM during processing: ~7.6 GB (clip 60), ~14.7 GB (clip 180).\nWithout compilation: ~6 GB (clip 60), ~10.4 GB (clip 180).\n\nIf you run out of VRAM, disable this or lower clip size.\n\nRecommended: ON with clip size 60-90.\nDefault: ON",
        "tip_denoise_strength": "Reduces noise and grain in restored areas. Higher = smoother but may lose fine detail.\n\nNone: no denoising. Low/Medium: good starting point. High: heavy smoothing.\nDefault: None",
        "tip_denoise_step": "When to apply denoising in the pipeline:\n- After Primary: before upscaling (secondary restoration). Denoises at 256x256.\n- After Secondary: after upscaling, right before final output. Denoises at full resolution.\n\nDefault: After Primary",
        "tip_secondary_restoration": "Optional second pass that upscales restored areas from 256x256 to 512 or 1024 pixels. Improves sharpness, especially for close-ups and 4K video.\n\nRTX Super Res is recommended — fast, free, no extra setup.\nTopaz TVAI requires a separate purchase and install.",
        "tip_secondary_tvai": "Best quality but slowest. Requires Topaz Video AI installed separately (paid).\nRecommended model: iris-2.",
        "tip_secondary_rtx": "Very fast and free. In some videos may produce a flickering effect — test on a short clip first.",
        "tip_tvai_ffmpeg_path": "Full path to the ffmpeg.exe bundled with Topaz Video AI.\n\nDefault location:\nC:\\Program Files\\Topaz Labs LLC\\Topaz Video\\ffmpeg.exe",
        "tip_tvai_model": "Which Topaz AI model to use for upscaling.\n\niris-2: good default, balanced quality.\niris-3, prob-4, nyx-1: experiment to see which looks best for your videos.\nDefault: iris-2",
        "tip_tvai_scale": "How much to enlarge the restored area.\n1x = no enlargement (256px). 2x = 512px. 4x = 1024px.\nHigher scale = sharper result but larger file and slower.\n\nDefault: 4x",
        "tip_tvai_workers": "How many Topaz upscale tasks run in parallel. More = faster overall but uses more CPU/GPU.\n\nDefault: 2",
        "tip_rtx_scale": "How much to enlarge the restored area.\n2x = 512px. 4x = 1024px.\nHigher = sharper but slower.\n\nDefault: 4x",
        "tip_rtx_quality": "Upscaling quality. Higher = better looking but slower.\n\nDefault: High",
        "tip_rtx_denoise": "Removes noise using RTX hardware. Set to None to skip.\n\nDefault: Medium",
        "tip_rtx_deblur": "Sharpens blurry areas using RTX hardware. Set to None to skip.\n\nDefault: None",
        "tip_detection_model": "AI model used to find areas that need restoration.\nrfdetr-v5: latest, most accurate — recommended.\nLada YOLO models may work better for 2D animations.\n\nDefault: rfdetr-v5",
        "tip_detection_score_threshold": "How confident the AI must be before marking an area for restoration.\nLower = detects more (may include false positives).\nHigher = detects less (may miss some).\n\nDefault: 0.25 (works well for most videos)",
        "tip_codec": "Output video format. Only HEVC (H.265) is supported.\nHEVC provides excellent quality at smaller file sizes.",
        "tip_encoder_cq": "Video quality level (Constant Quality). Lower number = better quality but larger file.\n\n18-22: high quality (recommended).\n22-28: balanced.\n28+: smaller files, lower quality.\nDefault: 22",
        "tip_encoder_custom_args": "Advanced encoder parameters as comma-separated key=value pairs.\nLeave empty unless you know what you're doing.\n\nExample: lookahead=32",
        "tip_working_directory": "Folder for temporary files created during encoding.\nUsing a fast SSD can improve speed. Leave empty to use the output folder.\n\nDefault: same as output folder",
        "tip_output_location": "Folder where processed videos are saved.\nLeave empty to save next to the original file.",
        "tip_output_pattern": "Filename template for output files.\nUse {original} as a placeholder for the input filename (without extension).\n\nExample: {original}_restored.mp4 → my_video_restored.mp4",
        
        # Preset button tooltips
        "tip_preset_reset": "Reset to saved values",
        "tip_preset_delete": "Delete preset",
        "tip_preset_save": "Save preset",
        "tip_preset_create": "Create new preset",

        # Engine compilation / first run warnings
        "engine_first_run_title": "First run may be slow",
        "engine_first_run_body": "Some TensorRT engines need to be compiled for your GPU. This is normal on the first run and can take 15-60 minutes.\n\nClose all other applications (browsers, games, etc.) and do not use the PC during compilation. The application may appear unresponsive — do not close it.\n\nEngines are cached and reused on all future runs.",
        "engine_first_run_missing": "Missing engines:",
        "engine_name_rfdetr": "RF-DETR (detection)",
        "engine_name_yolo": "YOLO (detection)",
        "engine_name_basicvsrpp": "BasicVSR++ (restoration)",
        "engine_basicvsrpp_risky_title": "BasicVSR++ compilation warning",
        "engine_basicvsrpp_risky_body": "BasicVSR++ TensorRT compilation may be risky with your GPU VRAM.\n\nGPU VRAM (approx): {vram_gb} GB\nRequested clip size: {requested_clip}\nApprox safe max: {safe_clip}\n\nEngine VRAM: ~1.9 GB (clip 60), ~5.4 GB (clip 180).\nPeak VRAM during processing: ~7.6 GB (clip 60), ~14.7 GB (clip 180).\n\nCompilation takes 15-60 minutes and may run out of VRAM.\nClose all other applications (including browsers) and do not use the PC during compilation.\nContinue anyway?",
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
        "btn_system_check": "系统检查",
        "language": "语言",
        
        # Queue Panel
        "btn_add_files": "📁 添加文件",
        "queue_empty": "拖放文件到这里\n或使用上方按钮",
        "items_queued": "队列中有 {count} 个项目",
        "btn_clear": "🗑 清空",
        "btn_clear_completed": "✓ 清除已完成",
        "output_location": "输出位置",
        "output_pattern": "输出文件名模板",
        "output_pattern_placeholder": "{original}_restored.mp4",
        "same_as_input": "与输入相同",
        "select_video_files": "选择视频文件",
        "select_folder": "选择文件夹",
        "select_output_folder": "选择输出文件夹",
        
        # Job Status
        "job_pending": "等待中",
        "job_processing": "处理中",
        "job_completed": "已完成",
        "completed_in": "完成用时",
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
        "working_directory": "工作目录",
        "working_directory_placeholder": "可选，默认与输出文件夹相同",
        "dialog_select_working_directory": "选择工作目录",

        # Secondary Restoration
        "secondary_none": "无",
        "secondary_tvai": "Topaz TVAI",
        "secondary_tvai_hint": "最佳质量，最慢",
        "secondary_rtx_super_res": "RTX Super Res",
        "secondary_rtx_hint": "速度快，可能闪烁",
        "rtx_scale": "缩放",
        "rtx_quality": "质量",
        "rtx_denoise": "降噪",
        "rtx_deblur": "去模糊",
        "recommended": "推荐",
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
        "tip_max_clip_size": "一次处理多少帧画面。数值越大，效果可能越好，但占用更多显存。\n\n建议：60 或更高。即使需要关闭模型编译，也建议至少使用 60。\n参考：60（安全）、90（平衡）、180（最佳质量，需要 24GB+ 显存）。\n默认值：60",
        "tip_temporal_overlap": "处理片段之间的重叠帧数，用于减少拼接处的闪烁。\n数值越高过渡越平滑，但速度稍慢。超过 20 效果提升不明显。\n\n根据片段大小推荐：\n- 片段 60 → 重叠 6-8\n- 片段 90 → 重叠 8-12\n- 片段 180 → 重叠 15-20\n默认值：8",
        "tip_enable_crossfade": "在片段边界处进行平滑过渡，减少画面闪烁。使用已处理的帧，不会增加任何额外 GPU 开销。\n\n建议：始终开启。\n默认值：开启",
        "tip_fp16_mode": "使用半精度计算来减少显存占用，通常还能提升速度。在现代显卡上几乎无画质损失。\n\n建议：RTX 20 系列及以上显卡开启。\n默认值：开启",
        "tip_compile_basicvsrpp": "将修复模型编译为 TensorRT 子引擎，大幅提升速度（约 2-3 倍）。\n首次编译需要 15-60 分钟。请关闭所有其他应用程序（包括浏览器），编译期间请勿使用电脑。\n引擎会被缓存，后续运行自动复用。\n\n引擎显存：约 1.9 GB（片段 60）、约 5.4 GB（片段 180）。\n处理时峰值显存：约 7.6 GB（片段 60）、约 14.7 GB（片段 180）。\n不编译时：约 6 GB（片段 60）、约 10.4 GB（片段 180）。\n\n如果显存不足，请关闭此选项或降低片段大小。\n\n建议：配合片段大小 60-90 开启。\n默认值：开启",
        "tip_denoise_strength": "降低修复区域的噪点和颗粒感。强度越高画面越平滑，但可能丢失细节。\n\n无：不降噪。低/中：推荐起步值。高：强力平滑。\n默认值：无",
        "tip_denoise_step": "降噪在处理流程中的应用时机：\n- 主修复后：在放大（二次修复）之前降噪，在 256x256 分辨率下处理。\n- 二次修复后：在放大之后、最终输出之前降噪，在完整分辨率下处理。\n\n默认值：主修复后",
        "tip_secondary_restoration": "可选的第二步处理，将修复区域从 256x256 放大到 512 或 1024 像素。可提升清晰度，特别是近景和 4K 视频。\n\nRTX Super Res 推荐使用 — 速度快、免费、无需额外配置。\nTopaz TVAI 需要单独购买和安装。",
        "tip_secondary_tvai": "质量最好但速度最慢。需要单独安装 Topaz Video AI（付费）。\n推荐模型：iris-2。",
        "tip_secondary_rtx": "速度非常快且免费。某些视频可能出现闪烁 — 建议先用短片段测试。",
        "tip_tvai_ffmpeg_path": "Topaz Video AI 自带的 ffmpeg.exe 完整路径。\n\n默认位置：\nC:\\Program Files\\Topaz Labs LLC\\Topaz Video\\ffmpeg.exe",
        "tip_tvai_model": "用于放大的 Topaz AI 模型。\n\niris-2：推荐默认值，质量均衡。\niris-3、prob-4、nyx-1：可尝试不同模型看哪个效果最好。\n默认值：iris-2",
        "tip_tvai_scale": "修复区域的放大倍数。\n1x = 不放大（256px）。2x = 512px。4x = 1024px。\n倍数越高越清晰，但文件更大、速度更慢。\n\n默认值：4x",
        "tip_tvai_workers": "同时运行的 Topaz 放大任务数。越多越快，但占用更多 CPU/GPU 资源。\n\n默认值：2",
        "tip_rtx_scale": "修复区域的放大倍数。\n2x = 512px。4x = 1024px。\n倍数越高越清晰，但速度更慢。\n\n默认值：4x",
        "tip_rtx_quality": "放大质量。越高画面越好，但速度越慢。\n\n默认值：High",
        "tip_rtx_denoise": "使用 RTX 硬件去除噪点。设为 None 跳过降噪。\n\n默认值：Medium",
        "tip_rtx_deblur": "使用 RTX 硬件锐化模糊区域。设为 None 跳过锐化。\n\n默认值：None",
        "tip_detection_model": "用于寻找需要修复区域的 AI 模型。\nrfdetr-v5：最新、最准确 — 推荐使用。\nLada YOLO 模型可能更适合 2D 动画。\n\n默认值：rfdetr-v5",
        "tip_detection_score_threshold": "AI 标记修复区域所需的置信度。\n数值越低 = 检测更多区域（可能误检）。\n数值越高 = 检测更少区域（可能漏检）。\n\n默认值：0.25（适合大多数视频）",
        "tip_codec": "输出视频格式。目前仅支持 HEVC (H.265)。\nHEVC 在较小文件体积下提供优秀画质。",
        "tip_encoder_cq": "视频质量等级（恒定质量模式）。数值越低画质越好，但文件越大。\n\n18-22：高画质（推荐）。\n22-28：画质与体积平衡。\n28 以上：较小文件，画质较低。\n默认值：22",
        "tip_encoder_custom_args": "高级编码器参数，以逗号分隔的 key=value 格式。\n如果不清楚用途，请留空。\n\n示例：lookahead=32",
        "tip_working_directory": "编码过程中产生的临时文件存放目录。\n使用高速 SSD 可提升速度。留空则使用输出文件夹。\n\n默认值：与输出文件夹相同",
        "tip_output_location": "处理后视频的保存文件夹。\n留空则保存在原始文件旁边。",
        "tip_output_pattern": "输出文件的命名模板。\n使用 {original} 作为输入文件名（不含扩展名）的占位符。\n\n示例：{original}_restored.mp4 → my_video_restored.mp4",

        # Preset button tooltips
        "tip_preset_reset": "重置为保存的值",
        "tip_preset_delete": "删除预设",
        "tip_preset_save": "保存预设",
        "tip_preset_create": "创建新预设",

        # Engine compilation / first run warnings
        "engine_first_run_title": "首次运行可能较慢",
        "engine_first_run_body": "部分 TensorRT 引擎需要为你的 GPU 进行编译。这在首次运行时是正常的，可能需要 15-60 分钟。\n\n请关闭所有其他应用程序（浏览器、游戏等），编译期间请勿使用电脑。应用可能看起来无响应——请不要关闭。\n\n引擎会被缓存，后续运行自动复用。",
        "engine_first_run_missing": "缺失的引擎：",
        "engine_name_rfdetr": "RF-DETR（检测）",
        "engine_name_yolo": "YOLO（检测）",
        "engine_name_basicvsrpp": "BasicVSR++（修复）",
        "engine_basicvsrpp_risky_title": "BasicVSR++ 编译警告",
        "engine_basicvsrpp_risky_body": "BasicVSR++ TensorRT 编译可能会因显存不足而存在风险。\n\n显存（约）：{vram_gb} GB\n请求的片段大小：{requested_clip}\n建议安全上限：{safe_clip}\n\n引擎显存：约 1.9 GB（片段 60）、约 5.4 GB（片段 180）。\n处理时峰值显存：约 7.6 GB（片段 60）、约 14.7 GB（片段 180）。\n\n编译需要 15-60 分钟，可能因显存不足而失败。\n请关闭所有其他应用程序（包括浏览器），编译期间请勿使用电脑。\n仍要继续吗？",
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
        "btn_system_check": "システムチェック",
        "language": "言語",

        # Queue Panel
        "btn_add_files": "📁 ファイル追加",
        "queue_empty": "ここにファイルをドラッグ＆ドロップ\nまたは上のボタンを使用",
        "items_queued": "{count} 件がキューに追加済み",
        "btn_clear": "🗑 クリア",
        "btn_clear_completed": "✓ 完了済みを削除",
        "output_location": "出力先",
        "output_pattern": "出力ファイル名テンプレート",
        "output_pattern_placeholder": "{original}_restored.mp4",
        "same_as_input": "入力と同じ",
        "select_video_files": "動画ファイルを選択",
        "select_folder": "フォルダを選択",
        "select_output_folder": "出力フォルダを選択",

        # Job Status
        "job_pending": "待機中",
        "job_processing": "処理中",
        "job_completed": "完了",
        "completed_in": "完了時間",
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
        "working_directory": "作業ディレクトリ",
        "working_directory_placeholder": "オプション。未指定の場合は出力フォルダと同じ",
        "dialog_select_working_directory": "作業ディレクトリを選択",

        # Secondary Restoration
        "secondary_none": "なし",
        "secondary_tvai": "Topaz TVAI",
        "secondary_tvai_hint": "最高品質、最も遅い",
        "secondary_rtx_super_res": "RTX Super Res",
        "secondary_rtx_hint": "高速、ちらつきの可能性あり",
        "rtx_scale": "スケール",
        "rtx_quality": "品質",
        "rtx_denoise": "ノイズ除去",
        "rtx_deblur": "ブレ除去",
        "recommended": "推奨",
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

        # Tooltips
        "tip_max_clip_size": "一度に処理するフレーム数です。大きいほど品質が向上する可能性がありますが、VRAM を多く使います。\n\n推奨：60 以上。モデルコンパイルを無効にしてでも 60 は維持しましょう。\n目安：60（安全）、90（バランス良）、180（最高品質、24GB 以上の VRAM が必要）。\nデフォルト：60",
        "tip_temporal_overlap": "処理クリップ間の重なりフレーム数で、つなぎ目のちらつきを軽減します。\n大きいほど滑らかですがやや遅くなります。20 を超えても効果はほとんど変わりません。\n\nクリップサイズ別の推奨値：\n- クリップ 60 → オーバーラップ 6-8\n- クリップ 90 → オーバーラップ 8-12\n- クリップ 180 → オーバーラップ 15-20\nデフォルト：8",
        "tip_enable_crossfade": "クリップの境目を滑らかにつなぎ、ちらつきを軽減します。処理済みフレームを再利用するため、追加の GPU 負荷はゼロです。\n\n推奨：常に ON。\nデフォルト：ON",
        "tip_fp16_mode": "半精度計算で VRAM 使用量を削減し、速度も向上することが多いです。最新の GPU では画質の劣化はほぼありません。\n\n推奨：RTX 20 シリーズ以降で ON。\nデフォルト：ON",
        "tip_compile_basicvsrpp": "修復モデルを TensorRT サブエンジンにコンパイルし、大幅に高速化します（約 2-3 倍）。\n初回コンパイルには 15～60 分かかります。他のアプリ（ブラウザ含む）をすべて閉じ、コンパイル中は PC を使用しないでください。\nエンジンはキャッシュされ、次回以降自動的に再利用されます。\n\nエンジン VRAM：約 1.9 GB（クリップ 60）、約 5.4 GB（クリップ 180）。\n処理時ピーク VRAM：約 7.6 GB（クリップ 60）、約 14.7 GB（クリップ 180）。\nコンパイルなし：約 6 GB（クリップ 60）、約 10.4 GB（クリップ 180）。\n\nVRAM 不足の場合は、この設定を無効にするかクリップサイズを下げてください。\n\n推奨：クリップサイズ 60-90 で ON。\nデフォルト：ON",
        "tip_denoise_strength": "修復された部分のノイズや粒子を低減します。高いほど滑らかになりますが、細部が失われることがあります。\n\nなし：ノイズ除去なし。低/中：まずこちらをお試しください。高：強力な平滑化。\nデフォルト：なし",
        "tip_denoise_step": "ノイズ除去を適用するタイミング：\n- 一次修復後：アップスケール（二次修復）前に適用。256x256 の解像度で処理。\n- 二次修復後：アップスケール後、最終出力の直前に適用。フル解像度で処理。\n\nデフォルト：一次修復後",
        "tip_secondary_restoration": "修復された領域を 256x256 から 512 または 1024 ピクセルに拡大するオプションの追加処理です。アップ時や 4K 動画で鮮明さが向上します。\n\nRTX Super Res がおすすめ — 高速、無料、追加設定不要。\nTopaz TVAI は別途購入とインストールが必要です。",
        "tip_secondary_tvai": "最高品質ですが最も遅いです。Topaz Video AI の別途インストールが必要です（有料）。\n推奨モデル：iris-2。",
        "tip_secondary_rtx": "非常に高速で無料です。一部の動画でちらつきが発生する場合があります — まず短いクリップでテストしてください。",
        "tip_tvai_ffmpeg_path": "Topaz Video AI に付属の ffmpeg.exe のフルパス。\n\nデフォルトの場所：\nC:\\Program Files\\Topaz Labs LLC\\Topaz Video\\ffmpeg.exe",
        "tip_tvai_model": "アップスケールに使用する Topaz AI モデル。\n\niris-2：おすすめのデフォルト、バランスの良い品質。\niris-3、prob-4、nyx-1：動画に合う最適なモデルを試してみてください。\nデフォルト：iris-2",
        "tip_tvai_scale": "修復領域の拡大倍率。\n1x = 拡大なし（256px）。2x = 512px。4x = 1024px。\n高倍率ほど鮮明ですが、ファイルが大きく処理が遅くなります。\n\nデフォルト：4x",
        "tip_tvai_workers": "同時に実行する Topaz アップスケールタスクの数。多いほど速くなりますが、CPU/GPU の負荷が増します。\n\nデフォルト：2",
        "tip_rtx_scale": "修復領域の拡大倍率。\n2x = 512px。4x = 1024px。\n高倍率ほど鮮明ですが、速度は低下します。\n\nデフォルト：4x",
        "tip_rtx_quality": "アップスケールの品質。高いほど綺麗ですが、速度が低下します。\n\nデフォルト：High",
        "tip_rtx_denoise": "RTX ハードウェアでノイズを除去します。None でスキップ。\n\nデフォルト：Medium",
        "tip_rtx_deblur": "RTX ハードウェアでぼやけた部分をシャープにします。None でスキップ。\n\nデフォルト：None",
        "tip_detection_model": "修復が必要な領域を検出する AI モデル。\nrfdetr-v5：最新かつ最も正確 — おすすめ。\nLada YOLO モデルは 2D アニメーションに適している場合があります。\n\nデフォルト：rfdetr-v5",
        "tip_detection_score_threshold": "AI が修復対象としてマークするために必要な確信度。\n低い値 = より多くの領域を検出（誤検出の可能性あり）。\n高い値 = より少ない領域を検出（見逃す可能性あり）。\n\nデフォルト：0.25（ほとんどの動画に適しています）",
        "tip_codec": "出力動画のフォーマット。現在は HEVC (H.265) のみ対応。\nHEVC は小さなファイルサイズで優れた画質を提供します。",
        "tip_encoder_cq": "動画の品質レベル（固定品質モード）。数値が低いほど高品質ですが、ファイルが大きくなります。\n\n18-22：高画質（おすすめ）。\n22-28：バランス型。\n28 以上：小さいファイル、画質は控えめ。\nデフォルト：22",
        "tip_encoder_custom_args": "上級者向けのエンコーダーパラメータ（カンマ区切りの key=value 形式）。\nよくわからない場合は空欄のままにしてください。\n\n例：lookahead=32",
        "tip_working_directory": "エンコード中の一時ファイルを保存するフォルダ。\n高速な SSD を使用すると処理が速くなります。空欄の場合は出力フォルダを使用します。\n\nデフォルト：出力フォルダと同じ",
        "tip_output_location": "処理済み動画の保存先フォルダ。\n空欄の場合、元のファイルと同じ場所に保存されます。",
        "tip_output_pattern": "出力ファイルの名前テンプレート。\n{original} は入力ファイル名（拡張子なし）のプレースホルダーです。\n\n例：{original}_restored.mp4 → my_video_restored.mp4",

        # Preset button tooltips
        "tip_preset_reset": "保存済みの値にリセット",
        "tip_preset_delete": "プリセットを削除",
        "tip_preset_save": "プリセットを保存",
        "tip_preset_create": "新しいプリセットを作成",

        # Engine compilation / first run warnings
        "engine_first_run_title": "初回起動は時間がかかる場合があります",
        "engine_first_run_body": "一部の TensorRT エンジンを GPU 向けにコンパイルする必要があります。初回起動時にはこれは正常で、15～60 分かかる場合があります。\n\n他のアプリ（ブラウザ、ゲームなど）をすべて閉じ、コンパイル中は PC を使用しないでください。アプリが応答しないように見える場合がありますが、閉じないでください。\n\nエンジンはキャッシュされ、次回以降自動的に再利用されます。",
        "engine_first_run_missing": "不足しているエンジン:",
        "engine_name_rfdetr": "RF-DETR（検出）",
        "engine_name_yolo": "YOLO（検出）",
        "engine_name_basicvsrpp": "BasicVSR++（修復）",
        "engine_basicvsrpp_risky_title": "BasicVSR++ コンパイル警告",
        "engine_basicvsrpp_risky_body": "BasicVSR++ TensorRT のコンパイルは、GPU の VRAM 不足によりリスクがあります。\n\nGPU VRAM（概算）: {vram_gb} GB\n要求クリップサイズ: {requested_clip}\n推定安全上限: {safe_clip}\n\nエンジン VRAM：約 1.9 GB（クリップ 60）、約 5.4 GB（クリップ 180）。\n処理時ピーク VRAM：約 7.6 GB（クリップ 60）、約 14.7 GB（クリップ 180）。\n\nコンパイルには 15～60 分かかり、VRAM 不足で失敗する場合があります。\n他のアプリ（ブラウザ含む）をすべて閉じ、コンパイル中は PC を使用しないでください。\n続行しますか？",
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

    "ko": {
        # App
        "app_title": "JASNA GUI",
        "status_idle": "대기 중",
        "status_processing": "처리 중",
        "status_paused": "일시 정지",
        "status_completed": "완료",
        "status_error": "오류",

        # Header
        "btn_help": "도움말",
        "btn_about": "정보",
        "language": "언어",

        # Queue Panel
        "btn_add_files": "📁 파일 추가",
        "queue_empty": "여기에 파일을 드래그 앤 드롭하거나\n위의 버튼을 사용하세요",
        "items_queued": "{count}개 항목 대기 중",
        "btn_clear": "🗑 비우기",
        "btn_clear_completed": "✓ 완료 항목 제거",
        "output_location": "출력 위치",
        "output_pattern": "출력 파일명 템플릿",
        "output_pattern_placeholder": "{original}_restored.mp4",
        "same_as_input": "입력과 동일",
        "select_video_files": "동영상 파일 선택",
        "select_folder": "폴더 선택",
        "select_output_folder": "출력 폴더 선택",

        # Job Status
        "job_pending": "대기 중",
        "job_processing": "처리 중",
        "job_completed": "완료",
        "completed_in": "소요 시간",
        "job_error": "오류",
        "job_paused": "일시 정지",
        "job_skipped": "건너뜀",

        # Settings Panel
        "preset": "프리셋:",
        "btn_create": "+",
        "btn_save": "💾",
        "btn_delete": "🗑",
        "btn_reset": "↺",

        # Sections
        "section_basic": "기본 처리",
        "section_advanced": "고급 처리",
        "section_secondary": "2차 복원",
        "section_encoding": "인코딩",

        # Basic Processing
        "max_clip_size": "최대 클립 크기",
        "detection_model": "감지 모델",
        "detection_threshold": "감지 임계값",
        "fp16_mode": "FP16 모드",
        "compile_basicvsrpp": "BasicVSR++ 컴파일",
        "file_conflict": "파일 충돌",
        "file_conflict_auto_rename": "자동 이름 변경",
        "file_conflict_overwrite": "덮어쓰기",
        "file_conflict_skip": "건너뛰기",
        "file_conflict_overwrite_warning": "기존 파일이 영구적으로 대체됩니다",
        "tip_file_conflict": "출력 파일이 이미 존재할 때 수행할 작업",
        "conflict_tooltip": "출력 파일이 이미 존재합니다",
        "renamed_output": "출력 파일이 이미 존재합니다. {filename}(으)로 이름이 변경되었습니다",

        # Advanced Processing
        "temporal_overlap": "시간 오버랩",
        "enable_crossfade": "크로스페이드 활성화",
        "denoise_strength": "노이즈 제거 강도",
        "denoise_step": "노이즈 제거 적용 시점",
        "denoise_none": "없음",
        "denoise_low": "낮음",
        "denoise_medium": "중간",
        "denoise_high": "높음",
        "after_primary": "1차 복원 후",
        "after_secondary": "2차 복원 후",
        "working_directory": "작업 디렉토리",
        "working_directory_placeholder": "선택 사항, 출력 폴더와 동일",
        "dialog_select_working_directory": "작업 디렉토리 선택",

        # Secondary Restoration
        "secondary_none": "없음",
        "secondary_tvai": "Topaz TVAI",
        "secondary_tvai_hint": "최고 품질, 가장 느림",
        "secondary_rtx_super_res": "RTX Super Res",
        "secondary_rtx_hint": "빠름, 깜빡임 가능성",
        "rtx_quality": "품질",
        "rtx_denoise": "노이즈 제거",
        "rtx_deblur": "블러 제거",
        "recommended": "권장",
        "batch_size": "배치 크기",
        "compile_model": "모델 컴파일",
        "ffmpeg_path": "FFmpeg 경로",
        "model": "모델",
        "scale": "스케일",
        "workers": "워커 수",

        # Encoding
        "codec": "코덱",
        "quality_cq": "품질 (CQ)",
        "custom_args": "사용자 정의 인수",

        # Control Bar
        "btn_start": "▶ 시작",
        "btn_pause": "⏸ 일시 정지",
        "btn_resume": "▶ 재개",
        "btn_stop": "⏹ 중지",
        "progress": "진행률",
        "time_remaining": "남은 시간",
        "no_file_processing": "처리 중인 파일 없음",
        "queue_label": "대기열",
        "logs_btn": ">_ 로그",

        # Log Panel
        "logs": "로그",
        "btn_export": "내보내기",
        "btn_toggle_logs": "로그 ▼",
        "filter_all": "전체",
        "filter_debug": "디버그",
        "filter_info": "정보",
        "filter_warn": "경고",
        "filter_error": "오류",
        "system_output": "시스템 출력",
        "filter_all_levels": "전체 수준",
        "filter_errors_only": "오류만",
        "filter_warnings_plus": "경고 이상",
        "filter_info_plus": "정보 이상",

        # Wizard
        "wizard_title": "시스템 점검",
        "wizard_subtitle": "필수 종속성을 확인하는 중...",
        "wizard_checking": "확인 중...",
        "wizard_found": "발견: {path}",
        "wizard_not_found": "PATH에서 찾을 수 없음",
        "wizard_all_passed": "✓ 모든 점검을 통과했습니다! 사용할 준비가 되었습니다.",
        "wizard_some_failed": "⚠ 일부 종속성이 누락되었습니다. README의 설치 안내를 확인하세요.",
        "btn_get_started": "시작하기",
        "btn_continue_anyway": "계속 진행",
        "btn_ok": "확인",

        # Dialogs
        "dialog_create_preset": "프리셋 만들기",
        "preset_name": "프리셋 이름",
        "preset_placeholder": "나만의 프리셋",
        "error_name_empty": "이름을 입력해야 합니다",
        "error_name_exists": "이미 존재하는 이름입니다",
        "btn_create_preset": "만들기",
        "btn_cancel": "취소",
        "dialog_delete_preset": "프리셋 삭제",
        "confirm_delete": "프리셋 '{name}'을(를) 삭제하시겠습니까?",
        "btn_delete_confirm": "삭제",

        # Toasts
        "toast_preset_saved": "프리셋 '{name}' 저장됨",
        "toast_preset_created": "프리셋 '{name}' 생성됨",
        "toast_preset_deleted": "프리셋 '{name}' 삭제됨",
        "toast_settings_reset": "설정이 초기화되었습니다",
        "toast_no_files": "대기열에 파일이 없습니다",
        "toast_started": "처리가 시작되었습니다",
        "toast_paused": "처리가 일시 정지되었습니다",
        "toast_resumed": "처리가 재개되었습니다",
        "toast_stopped": "처리가 중지되었습니다",
        # Buy Me a Coffee
        "bmc_support": "후원",

        # Tooltips
        "tip_max_clip_size": "한 번에 처리할 프레임 수입니다. 클수록 품질이 좋아질 수 있지만 VRAM을 더 많이 사용합니다.\n\n권장: 60 이상. 모델 컴파일을 끄더라도 최소 60을 유지하세요.\n참고: 60 (안전), 90 (균형), 180 (최고 품질, 24GB+ VRAM 필요).\n기본값: 60",
        "tip_temporal_overlap": "처리 클립 간 겹치는 프레임 수로, 경계의 깜빡임을 줄여줍니다.\n높을수록 부드럽지만 약간 느려집니다. 20을 넘으면 효과 차이가 거의 없습니다.\n\n클립 크기별 권장값:\n- 클립 60 → 오버랩 6-8\n- 클립 90 → 오버랩 8-12\n- 클립 180 → 오버랩 15-20\n기본값: 8",
        "tip_enable_crossfade": "클립 경계를 부드럽게 연결하여 깜빡임을 줄여줍니다. 이미 처리된 프레임을 재활용하므로 추가 GPU 부담이 전혀 없습니다.\n\n권장: 항상 켜기.\n기본값: 켜기",
        "tip_fp16_mode": "반정밀도 계산으로 VRAM 사용량을 줄이고 속도도 빨라지는 경우가 많습니다. 최신 GPU에서 화질 차이는 거의 없습니다.\n\n권장: RTX 20 시리즈 이상에서 켜기.\n기본값: 켜기",
        "tip_compile_basicvsrpp": "복원 모델을 TensorRT 서브엔진으로 컴파일하여 큰 속도 향상을 얻습니다 (약 2-3배).\n첫 컴파일에 15-60분이 소요됩니다. 다른 모든 앱(브라우저 포함)을 닫고, 컴파일 중 PC를 사용하지 마세요.\n엔진은 캐시되어 이후 실행 시 자동으로 재사용됩니다.\n\n엔진 VRAM: 약 1.9 GB (클립 60), 약 5.4 GB (클립 180).\n처리 시 피크 VRAM: 약 7.6 GB (클립 60), 약 14.7 GB (클립 180).\n컴파일 없이: 약 6 GB (클립 60), 약 10.4 GB (클립 180).\n\nVRAM 부족 시 이 설정을 끄거나 클립 크기를 줄이세요.\n\n권장: 클립 크기 60-90에서 켜기.\n기본값: 켜기",
        "tip_denoise_strength": "복원된 영역의 노이즈와 입자를 줄여줍니다. 높을수록 부드럽지만 세부 디테일이 줄어들 수 있습니다.\n\n없음: 노이즈 제거 안 함. 낮음/중간: 좋은 시작점. 높음: 강력한 평활화.\n기본값: 없음",
        "tip_denoise_step": "노이즈 제거를 적용하는 시점:\n- 1차 복원 후: 업스케일링(2차 복원) 전에 적용. 256x256 해상도에서 처리.\n- 2차 복원 후: 업스케일링 후, 최종 출력 직전에 적용. 전체 해상도에서 처리.\n\n기본값: 1차 복원 후",
        "tip_secondary_restoration": "복원된 영역을 256x256에서 512 또는 1024 픽셀로 확대하는 선택적 추가 처리입니다. 클로즈업이나 4K 영상에서 선명도가 향상됩니다.\n\nRTX Super Res 권장 — 빠르고, 무료이며, 추가 설정이 필요 없습니다.\nTopaz TVAI는 별도 구매 및 설치가 필요합니다.",
        "tip_secondary_tvai": "최고 품질이지만 가장 느립니다. Topaz Video AI 별도 설치 필요 (유료).\n권장 모델: iris-2.",
        "tip_secondary_rtx": "매우 빠르고 무료입니다. 일부 영상에서 깜빡임이 발생할 수 있습니다 — 짧은 클립으로 먼저 테스트하세요.",
        "tip_tvai_ffmpeg_path": "Topaz Video AI에 포함된 ffmpeg.exe의 전체 경로.\n\n기본 위치:\nC:\\Program Files\\Topaz Labs LLC\\Topaz Video\\ffmpeg.exe",
        "tip_tvai_model": "업스케일에 사용할 Topaz AI 모델.\n\niris-2: 권장 기본값, 균형 잡힌 품질.\niris-3, prob-4, nyx-1: 여러 모델을 시도해서 가장 좋은 결과를 찾아보세요.\n기본값: iris-2",
        "tip_tvai_scale": "복원 영역의 확대 배율.\n1x = 확대 없음 (256px). 2x = 512px. 4x = 1024px.\n높을수록 선명하지만 파일이 크고 느려집니다.\n\n기본값: 4x",
        "tip_tvai_workers": "동시에 실행할 Topaz 업스케일 작업 수. 많을수록 빠르지만 CPU/GPU 사용량이 증가합니다.\n\n기본값: 2",
        "tip_rtx_scale": "복원 영역의 확대 배율.\n2x = 512px. 4x = 1024px.\n높을수록 선명하지만 느려집니다.\n\n기본값: 4x",
        "tip_rtx_quality": "업스케일 품질. 높을수록 결과가 좋지만 느려집니다.\n\n기본값: High",
        "tip_rtx_denoise": "RTX 하드웨어로 노이즈를 제거합니다. None으로 건너뛰기.\n\n기본값: Medium",
        "tip_rtx_deblur": "RTX 하드웨어로 흐린 부분을 선명하게 합니다. None으로 건너뛰기.\n\n기본값: None",
        "tip_detection_model": "복원이 필요한 영역을 찾는 AI 모델.\nrfdetr-v5: 최신, 가장 정확 — 권장.\nLada YOLO 모델은 2D 애니메이션에 더 적합할 수 있습니다.\n\n기본값: rfdetr-v5",
        "tip_detection_score_threshold": "AI가 복원 대상으로 표시하기 위해 필요한 확신도.\n낮은 값 = 더 많은 영역 감지 (오탐 가능성 있음).\n높은 값 = 더 적은 영역 감지 (놓칠 가능성 있음).\n\n기본값: 0.25 (대부분의 영상에 적합)",
        "tip_codec": "출력 동영상 형식. 현재는 HEVC (H.265)만 지원.\nHEVC는 작은 파일 크기로 우수한 화질을 제공합니다.",
        "tip_encoder_cq": "동영상 품질 수준 (고정 품질 모드). 숫자가 낮을수록 화질이 좋지만 파일이 커집니다.\n\n18-22: 고화질 (권장).\n22-28: 균형.\n28 이상: 작은 파일, 낮은 화질.\n기본값: 22",
        "tip_encoder_custom_args": "고급 인코더 매개변수 (쉼표로 구분된 key=value 형식).\n잘 모르겠으면 비워두세요.\n\n예: lookahead=32",
        "tip_working_directory": "인코딩 중 생성되는 임시 파일을 저장할 폴더.\n빠른 SSD를 사용하면 속도가 향상됩니다. 비워두면 출력 폴더를 사용합니다.\n\n기본값: 출력 폴더와 동일",
        "tip_output_location": "처리된 동영상을 저장할 폴더.\n비워두면 원본 파일 옆에 저장됩니다.",
        "tip_output_pattern": "출력 파일의 이름 템플릿.\n{original}은 입력 파일명(확장자 제외)의 자리 표시자입니다.\n\n예: {original}_restored.mp4 → my_video_restored.mp4",

        # Preset button tooltips
        "tip_preset_reset": "저장된 값으로 초기화",
        "tip_preset_delete": "프리셋 삭제",
        "tip_preset_save": "프리셋 저장",
        "tip_preset_create": "새 프리셋 만들기",

        # Engine compilation / first run warnings
        "engine_first_run_title": "첫 실행은 느릴 수 있습니다",
        "engine_first_run_body": "일부 TensorRT 엔진을 GPU에 맞게 컴파일해야 합니다. 첫 실행 시 이는 정상이며, 15-60분이 소요될 수 있습니다.\n\n다른 모든 앱(브라우저, 게임 등)을 닫고, 컴파일 중 PC를 사용하지 마세요. 앱이 응답하지 않는 것처럼 보일 수 있습니다 — 닫지 마세요.\n\n엔진은 캐시되어 이후 실행 시 자동으로 재사용됩니다.",
        "engine_first_run_missing": "누락된 엔진:",
        "engine_name_rfdetr": "RF-DETR (감지)",
        "engine_name_yolo": "YOLO (감지)",
        "engine_name_basicvsrpp": "BasicVSR++ (복원)",
        "engine_basicvsrpp_risky_title": "BasicVSR++ 컴파일 경고",
        "engine_basicvsrpp_risky_body": "BasicVSR++ TensorRT 컴파일이 GPU VRAM 부족으로 위험할 수 있습니다.\n\nGPU VRAM (약): {vram_gb} GB\n요청된 클립 크기: {requested_clip}\n안전 추정 최대: {safe_clip}\n\n엔진 VRAM: 약 1.9 GB (클립 60), 약 5.4 GB (클립 180).\n처리 시 피크 VRAM: 약 7.6 GB (클립 60), 약 14.7 GB (클립 180).\n\n컴파일에 15-60분이 소요되며 VRAM 부족으로 실패할 수 있습니다.\n다른 모든 앱(브라우저 포함)을 닫고, 컴파일 중 PC를 사용하지 마세요.\n계속하시겠습니까?",
        # About dialog
        "dialog_about_title": "Jasna 정보",
        "dialog_about_version": "버전 {version}",
        "dialog_about_description": "JAV 모자이크 복원 도구",
        "dialog_about_credit": "Lada에서 영감을 받음",
        "btn_close": "닫기",

        # Language change dialog
        "dialog_language_changed": "언어가 변경되었습니다",
        "dialog_language_restart": "언어 변경을 완전히 적용하려면 애플리케이션을 다시 시작하세요.",

        # App messages
        "error_cannot_start": "처리를 시작할 수 없습니다:",
        "error_invalid_tvai": "잘못된 TVAI 구성",

        # Settings panel
        "dialog_select_tvai_ffmpeg": "Topaz Video AI ffmpeg.exe 선택",
        "placeholder_encoder_args": "예: lookahead=32",

        # Wizard check labels
        "wizard_window_title": "Jasna - 시스템 점검",
        "wizard_check_ffmpeg": "FFmpeg",
        "wizard_check_ffprobe": "FFprobe",
        "wizard_check_mkvmerge": "MKVmerge",
        "wizard_check_gpu": "NVIDIA GPU",
        "wizard_check_cuda": "CUDA 런타임",
        "wizard_check_hags": "하드웨어 가속 GPU 스케줄링",
        "wizard_not_checked": "확인되지 않음",
        "wizard_not_callable": "호출 불가: {path}",
        "wizard_found_version": "발견: {path} ({version})",
        "wizard_found_no_major": "발견: {path} (주 버전을 감지할 수 없음)",
        "wizard_found_bad_major": "발견: {path} (major={major}, 예상=8)",
        "wizard_found_major": "발견: {path} (major={major})",
        "wizard_no_cuda": "CUDA 장치 없음",
        "wizard_gpu_compute_too_low": "Compute capability 7.5 이상 필요 (GPU: {major}.{minor})",
        "wizard_cuda_version": "CUDA {version}",
        "wizard_cuda_version_compute": "CUDA {version}, compute {major}.{minor}",
        "wizard_not_available": "사용 불가",

        # Validation errors
        "error_tvai_data_dir_not_set": "환경 변수 TVAI_MODEL_DATA_DIR이 설정되지 않았습니다",
        "error_tvai_model_dir_not_set": "환경 변수 TVAI_MODEL_DIR이 설정되지 않았습니다",
        "error_tvai_data_dir_missing": "TVAI_MODEL_DATA_DIR이 존재하지 않는 디렉토리를 가리킵니다: {path}",
        "error_tvai_model_dir_missing": "TVAI_MODEL_DIR이 존재하지 않는 디렉토리를 가리킵니다: {path}",
        "error_tvai_ffmpeg_not_found": "TVAI ffmpeg를 찾을 수 없습니다: {path}",
    },

    "th": {
        # App
        "app_title": "JASNA GUI",
        "status_idle": "พร้อม",
        "status_processing": "กำลังประมวลผล",
        "status_paused": "หยุดชั่วคราว",
        "status_completed": "เสร็จสิ้น",
        "status_error": "ข้อผิดพลาด",

        # Header
        "btn_help": "ช่วยเหลือ",
        "btn_about": "เกี่ยวกับ",
        "btn_system_check": "ตรวจสอบระบบ",
        "language": "ภาษา",

        # Queue Panel
        "btn_add_files": "📁 เพิ่มไฟล์",
        "queue_empty": "ลากและวางไฟล์ที่นี่\nหรือใช้ปุ่มด้านบน",
        "items_queued": "{count} รายการในคิว",
        "btn_clear": "🗑 ล้าง",
        "btn_clear_completed": "✓ ล้างที่เสร็จแล้ว",
        "output_location": "ตำแหน่งเอาต์พุต",
        "output_pattern": "รูปแบบชื่อไฟล์เอาต์พุต",
        "output_pattern_placeholder": "{original}_restored.mp4",
        "same_as_input": "เหมือนกับอินพุต",
        "select_video_files": "เลือกไฟล์วิดีโอ",
        "select_folder": "เลือกโฟลเดอร์",
        "select_output_folder": "เลือกโฟลเดอร์เอาต์พุต",

        # Job Status
        "job_pending": "รอดำเนินการ",
        "job_processing": "กำลังประมวลผล",
        "job_completed": "เสร็จสิ้น",
        "completed_in": "เสร็จใน",
        "job_error": "ข้อผิดพลาด",
        "job_paused": "หยุดชั่วคราว",
        "job_skipped": "ข้าม",

        # Settings Panel
        "preset": "พรีเซ็ต:",
        "btn_create": "+",
        "btn_save": "💾",
        "btn_delete": "🗑",
        "btn_reset": "↺",

        # Sections
        "section_basic": "การประมวลผลพื้นฐาน",
        "section_advanced": "การประมวลผลขั้นสูง",
        "section_secondary": "การฟื้นฟูขั้นที่สอง",
        "section_encoding": "การเข้ารหัส",

        # Basic Processing
        "max_clip_size": "ขนาดคลิปสูงสุด",
        "detection_model": "โมเดลตรวจจับ",
        "detection_threshold": "ค่าความเชื่อมั่นการตรวจจับ",
        "fp16_mode": "โหมด FP16",
        "compile_basicvsrpp": "คอมไพล์ BasicVSR++",
        "file_conflict": "ไฟล์ซ้ำ",
        "file_conflict_auto_rename": "เปลี่ยนชื่ออัตโนมัติ",
        "file_conflict_overwrite": "เขียนทับ",
        "file_conflict_skip": "ข้าม",
        "file_conflict_overwrite_warning": "ไฟล์ที่มีอยู่จะถูกแทนที่อย่างถาวร",
        "tip_file_conflict": "เมื่อไฟล์เอาต์พุตมีอยู่แล้วจะทำอย่างไร",
        "conflict_tooltip": "ไฟล์เอาต์พุตมีอยู่แล้ว",
        "renamed_output": "ไฟล์เอาต์พุตมีอยู่แล้ว เปลี่ยนชื่อเป็น {filename}",

        # Advanced Processing
        "temporal_overlap": "เฟรมซ้อนทับ",
        "enable_crossfade": "เปิดใช้ครอสเฟด",
        "denoise_strength": "ระดับลดสัญญาณรบกวน",
        "denoise_step": "ลดสัญญาณรบกวนหลังจาก",
        "denoise_none": "ไม่มี",
        "denoise_low": "ต่ำ",
        "denoise_medium": "ปานกลาง",
        "denoise_high": "สูง",
        "after_primary": "หลังขั้นแรก",
        "after_secondary": "หลังขั้นที่สอง",
        "working_directory": "ไดเรกทอรีทำงาน",
        "working_directory_placeholder": "ไม่บังคับ ใช้โฟลเดอร์เอาต์พุตเดียวกัน",
        "dialog_select_working_directory": "เลือกไดเรกทอรีทำงาน",

        # Secondary Restoration
        "secondary_none": "ไม่มี",
        "secondary_tvai": "Topaz TVAI",
        "secondary_tvai_hint": "คุณภาพดีที่สุด ช้าที่สุด",
        "secondary_rtx_super_res": "RTX Super Res",
        "secondary_rtx_hint": "เร็ว อาจกะพริบ",
        "rtx_scale": "อัตราขยาย",
        "rtx_quality": "คุณภาพ",
        "rtx_denoise": "ลดสัญญาณรบกวน",
        "rtx_deblur": "ลดความเบลอ",
        "recommended": "แนะนำ",
        "batch_size": "ขนาดแบตช์",
        "compile_model": "คอมไพล์โมเดล",
        "ffmpeg_path": "เส้นทาง FFmpeg",
        "model": "โมเดล",
        "scale": "อัตราขยาย",
        "workers": "เวิร์กเกอร์",

        # Encoding
        "codec": "ตัวแปลงสัญญาณ",
        "quality_cq": "คุณภาพ (CQ)",
        "custom_args": "อาร์กิวเมนต์กำหนดเอง",

        # Control Bar
        "btn_start": "▶ เริ่ม",
        "btn_pause": "⏸ หยุดชั่วคราว",
        "btn_resume": "▶ ดำเนินต่อ",
        "btn_stop": "⏹ หยุด",
        "progress": "ความคืบหน้า",
        "time_remaining": "เหลือ",
        "no_file_processing": "ไม่มีไฟล์กำลังประมวลผล",
        "queue_label": "คิว",
        "logs_btn": ">_ บันทึก",

        # Log Panel
        "logs": "บันทึก",
        "btn_export": "ส่งออก",
        "btn_toggle_logs": "บันทึก ▼",
        "filter_all": "ทั้งหมด",
        "filter_debug": "Debug",
        "filter_info": "Info",
        "filter_warn": "Warn",
        "filter_error": "Error",
        "system_output": "เอาต์พุตระบบ",
        "filter_all_levels": "ทุกระดับ",
        "filter_errors_only": "เฉพาะข้อผิดพลาด",
        "filter_warnings_plus": "คำเตือน+",
        "filter_info_plus": "ข้อมูล+",

        # Wizard
        "wizard_title": "ตรวจสอบระบบ",
        "wizard_subtitle": "กำลังตรวจสอบ dependencies ที่จำเป็น...",
        "wizard_checking": "กำลังตรวจสอบ...",
        "wizard_found": "พบ: {path}",
        "wizard_not_found": "ไม่พบใน PATH",
        "wizard_all_passed": "✓ ผ่านทุกรายการ! พร้อมใช้งานแล้ว",
        "wizard_some_failed": "⚠ dependencies บางอย่างขาดหายไป ตรวจสอบ README สำหรับคำแนะนำการติดตั้ง",
        "btn_get_started": "เริ่มต้นใช้งาน",
        "btn_continue_anyway": "ดำเนินการต่อ",
        "btn_ok": "ตกลง",

        # Dialogs
        "dialog_create_preset": "สร้างพรีเซ็ต",
        "preset_name": "ชื่อพรีเซ็ต",
        "preset_placeholder": "พรีเซ็ตของฉัน",
        "error_name_empty": "ชื่อต้องไม่ว่างเปล่า",
        "error_name_exists": "ชื่อนี้มีอยู่แล้ว",
        "btn_create_preset": "สร้าง",
        "btn_cancel": "ยกเลิก",
        "dialog_delete_preset": "ลบพรีเซ็ต",
        "confirm_delete": "ลบพรีเซ็ต '{name}'?",
        "btn_delete_confirm": "ลบ",

        # Toasts
        "toast_preset_saved": "บันทึกพรีเซ็ต '{name}' แล้ว",
        "toast_preset_created": "สร้างพรีเซ็ต '{name}' แล้ว",
        "toast_preset_deleted": "ลบพรีเซ็ต '{name}' แล้ว",
        "toast_settings_reset": "รีเซ็ตการตั้งค่าแล้ว",
        "toast_no_files": "ไม่มีไฟล์ในคิว",
        "toast_started": "เริ่มประมวลผลแล้ว",
        "toast_paused": "หยุดประมวลผลชั่วคราว",
        "toast_resumed": "ดำเนินการประมวลผลต่อ",
        "toast_stopped": "หยุดประมวลผลแล้ว",
        # Buy Me a Coffee
        "bmc_support": "สนับสนุน",

        # Tooltips
        "tip_max_clip_size": "จำนวนเฟรมที่ประมวลผลต่อครั้ง ค่ามากขึ้นอาจให้คุณภาพดีขึ้นแต่ใช้ VRAM มากขึ้น\n\nแนะนำ: 60 ขึ้นไป แม้ต้องปิดการคอมไพล์โมเดลก็ควรใช้ 60 เป็นอย่างน้อย\nแนวทาง: 60 (ปลอดภัย), 90 (สมดุลดี), 180 (คุณภาพสูงสุด ต้องมี VRAM 24 GB+)\nค่าเริ่มต้น: 60",
        "tip_temporal_overlap": "เฟรมซ้อนทับระหว่างคลิปเพื่อลดการกะพริบที่รอยต่อ\nค่ามาก = การเปลี่ยนผ่านราบรื่นขึ้นแต่ช้าลงเล็กน้อย เกิน 20 แทบไม่ต่างกัน\n\nค่าแนะนำตามขนาดคลิป:\n- คลิป 60 → ซ้อนทับ 6-8\n- คลิป 90 → ซ้อนทับ 8-12\n- คลิป 180 → ซ้อนทับ 15-20\nค่าเริ่มต้น: 8",
        "tip_enable_crossfade": "ผสานรอยต่อของคลิปให้ราบรื่นเพื่อลดการกะพริบ ใช้เฟรมที่ประมวลผลแล้วจึงไม่มีภาระ GPU เพิ่ม\n\nแนะนำ: เปิดเสมอ\nค่าเริ่มต้น: เปิด",
        "tip_fp16_mode": "ใช้การคำนวณ half-precision เพื่อลดการใช้ VRAM และมักจะเร็วขึ้น ไม่มีความแตกต่างด้านคุณภาพบน GPU รุ่นใหม่\n\nแนะนำ: เปิดสำหรับ RTX 20-series ขึ้นไป\nค่าเริ่มต้น: เปิด",
        "tip_compile_basicvsrpp": "คอมไพล์โมเดลฟื้นฟูเป็น TensorRT sub-engine เพื่อเพิ่มความเร็วอย่างมาก (ประมาณ 2-3 เท่า)\nการคอมไพล์ครั้งแรกใช้เวลา 15-60 นาที ปิดแอปอื่นทั้งหมด (รวมเบราว์เซอร์) และอย่าใช้ PC ระหว่างคอมไพล์\nEngine จะถูกแคชและนำกลับมาใช้ในครั้งต่อไปโดยอัตโนมัติ\n\nVRAM ของ Engine: ~1.9 GB (คลิป 60), ~5.4 GB (คลิป 180)\nVRAM สูงสุดขณะประมวลผล: ~7.6 GB (คลิป 60), ~14.7 GB (คลิป 180)\nไม่คอมไพล์: ~6 GB (คลิป 60), ~10.4 GB (คลิป 180)\n\nหาก VRAM ไม่พอ ให้ปิดตัวเลือกนี้หรือลดขนาดคลิป\n\nแนะนำ: เปิดพร้อมขนาดคลิป 60-90\nค่าเริ่มต้น: เปิด",
        "tip_denoise_strength": "ลดสัญญาณรบกวนและเม็ดในพื้นที่ที่ฟื้นฟูแล้ว ค่ามาก = ราบรื่นขึ้นแต่อาจสูญเสียรายละเอียด\n\nไม่มี: ไม่ลดสัญญาณรบกวน ต่ำ/ปานกลาง: จุดเริ่มต้นที่ดี สูง: ปรับเรียบอย่างหนัก\nค่าเริ่มต้น: ไม่มี",
        "tip_denoise_step": "เวลาที่จะลดสัญญาณรบกวนในขั้นตอนการประมวลผล:\n- หลังขั้นแรก: ก่อนการขยาย (การฟื้นฟูขั้นที่สอง) ลดสัญญาณรบกวนที่ 256x256\n- หลังขั้นที่สอง: หลังการขยาย ก่อนเอาต์พุตสุดท้าย ลดสัญญาณรบกวนที่ความละเอียดเต็ม\n\nค่าเริ่มต้น: หลังขั้นแรก",
        "tip_secondary_restoration": "ขั้นตอนเสริมที่ขยายพื้นที่ฟื้นฟูจาก 256x256 เป็น 512 หรือ 1024 พิกเซล ช่วยเพิ่มความคมชัดโดยเฉพาะสำหรับภาพใกล้และวิดีโอ 4K\n\nRTX Super Res แนะนำ — เร็ว ฟรี ไม่ต้องติดตั้งเพิ่ม\nTopaz TVAI ต้องซื้อและติดตั้งแยกต่างหาก",
        "tip_secondary_tvai": "คุณภาพดีที่สุดแต่ช้าที่สุด ต้องติดตั้ง Topaz Video AI แยกต่างหาก (เสียเงิน)\nโมเดลแนะนำ: iris-2",
        "tip_secondary_rtx": "เร็วมากและฟรี บางวิดีโออาจมีการกะพริบ — ทดสอบกับคลิปสั้น ๆ ก่อน",
        "tip_tvai_ffmpeg_path": "เส้นทางเต็มไปยัง ffmpeg.exe ที่มาพร้อม Topaz Video AI\n\nตำแหน่งเริ่มต้น:\nC:\\Program Files\\Topaz Labs LLC\\Topaz Video\\ffmpeg.exe",
        "tip_tvai_model": "โมเดล Topaz AI ที่ใช้สำหรับขยาย\n\niris-2: ค่าเริ่มต้นที่ดี คุณภาพสมดุล\niris-3, prob-4, nyx-1: ลองใช้เพื่อดูว่าโมเดลไหนให้ผลลัพธ์ดีที่สุด\nค่าเริ่มต้น: iris-2",
        "tip_tvai_scale": "อัตราการขยายพื้นที่ฟื้นฟู\n1x = ไม่ขยาย (256px) 2x = 512px 4x = 1024px\nขยายมากขึ้น = คมชัดขึ้นแต่ไฟล์ใหญ่ขึ้นและช้าลง\n\nค่าเริ่มต้น: 4x",
        "tip_tvai_workers": "จำนวนงานขยาย Topaz ที่ทำงานพร้อมกัน มากขึ้น = เร็วขึ้นแต่ใช้ CPU/GPU มากขึ้น\n\nค่าเริ่มต้น: 2",
        "tip_rtx_scale": "อัตราการขยายพื้นที่ฟื้นฟู\n2x = 512px 4x = 1024px\nขยายมากขึ้น = คมชัดขึ้นแต่ช้าลง\n\nค่าเริ่มต้น: 4x",
        "tip_rtx_quality": "คุณภาพการขยาย สูงขึ้น = ผลลัพธ์ดีขึ้นแต่ช้าลง\n\nค่าเริ่มต้น: High",
        "tip_rtx_denoise": "ลดสัญญาณรบกวนด้วยฮาร์ดแวร์ RTX ตั้งเป็น None เพื่อข้าม\n\nค่าเริ่มต้น: Medium",
        "tip_rtx_deblur": "เพิ่มความคมชัดส่วนที่เบลอด้วยฮาร์ดแวร์ RTX ตั้งเป็น None เพื่อข้าม\n\nค่าเริ่มต้น: None",
        "tip_detection_model": "โมเดล AI ที่ใช้ค้นหาพื้นที่ที่ต้องฟื้นฟู\nrfdetr-v5: ล่าสุด แม่นยำที่สุด — แนะนำ\nLada YOLO อาจทำงานได้ดีกว่าสำหรับแอนิเมชัน 2D\n\nค่าเริ่มต้น: rfdetr-v5",
        "tip_detection_score_threshold": "ความมั่นใจที่ AI ต้องมีก่อนทำเครื่องหมายพื้นที่สำหรับฟื้นฟู\nค่าต่ำ = ตรวจจับมากขึ้น (อาจมีผลบวกปลอม)\nค่าสูง = ตรวจจับน้อยลง (อาจพลาดบางส่วน)\n\nค่าเริ่มต้น: 0.25 (เหมาะกับวิดีโอส่วนใหญ่)",
        "tip_codec": "รูปแบบวิดีโอเอาต์พุต รองรับเฉพาะ HEVC (H.265)\nHEVC ให้คุณภาพดีเยี่ยมด้วยขนาดไฟล์ที่เล็กลง",
        "tip_encoder_cq": "ระดับคุณภาพวิดีโอ (Constant Quality) ตัวเลขต่ำ = คุณภาพดีขึ้นแต่ไฟล์ใหญ่ขึ้น\n\n18-22: คุณภาพสูง (แนะนำ)\n22-28: สมดุล\n28+: ไฟล์เล็กลง คุณภาพต่ำลง\nค่าเริ่มต้น: 22",
        "tip_encoder_custom_args": "พารามิเตอร์ encoder ขั้นสูง ในรูปแบบ key=value คั่นด้วยจุลภาค\nเว้นว่างไว้หากไม่แน่ใจ\n\nตัวอย่าง: lookahead=32",
        "tip_working_directory": "โฟลเดอร์สำหรับไฟล์ชั่วคราวที่สร้างระหว่างการเข้ารหัส\nใช้ SSD ที่เร็วจะช่วยเพิ่มความเร็ว เว้นว่างเพื่อใช้โฟลเดอร์เอาต์พุต\n\nค่าเริ่มต้น: เหมือนกับโฟลเดอร์เอาต์พุต",
        "tip_output_location": "โฟลเดอร์ที่ใช้บันทึกวิดีโอที่ประมวลผลแล้ว\nเว้นว่างเพื่อบันทึกไว้ข้างไฟล์ต้นฉบับ",
        "tip_output_pattern": "รูปแบบชื่อไฟล์เอาต์พุต\nใช้ {original} เป็นตัวแทนชื่อไฟล์อินพุต (ไม่รวมนามสกุล)\n\nตัวอย่าง: {original}_restored.mp4 → my_video_restored.mp4",

        # Preset button tooltips
        "tip_preset_reset": "รีเซ็ตเป็นค่าที่บันทึกไว้",
        "tip_preset_delete": "ลบพรีเซ็ต",
        "tip_preset_save": "บันทึกพรีเซ็ต",
        "tip_preset_create": "สร้างพรีเซ็ตใหม่",

        # Engine compilation / first run warnings
        "engine_first_run_title": "การรันครั้งแรกอาจช้า",
        "engine_first_run_body": "TensorRT engine บางตัวต้องถูกคอมไพล์สำหรับ GPU ของคุณ นี่เป็นเรื่องปกติในการรันครั้งแรกและอาจใช้เวลา 15-60 นาที\n\nปิดแอปอื่นทั้งหมด (เบราว์เซอร์, เกม ฯลฯ) และอย่าใช้ PC ระหว่างคอมไพล์ แอปอาจดูเหมือนไม่ตอบสนอง — อย่าปิดมัน\n\nEngine จะถูกแคชและนำกลับมาใช้ในครั้งต่อไปโดยอัตโนมัติ",
        "engine_first_run_missing": "engine ที่หายไป:",
        "engine_name_rfdetr": "RF-DETR (ตรวจจับ)",
        "engine_name_yolo": "YOLO (ตรวจจับ)",
        "engine_name_basicvsrpp": "BasicVSR++ (ฟื้นฟู)",
        "engine_basicvsrpp_risky_title": "คำเตือนการคอมไพล์ BasicVSR++",
        "engine_basicvsrpp_risky_body": "การคอมไพล์ BasicVSR++ TensorRT อาจเสี่ยงกับ VRAM ของ GPU คุณ\n\nVRAM ของ GPU (โดยประมาณ): {vram_gb} GB\nขนาดคลิปที่ร้องขอ: {requested_clip}\nขนาดสูงสุดที่ปลอดภัยโดยประมาณ: {safe_clip}\n\nVRAM ของ Engine: ~1.9 GB (คลิป 60), ~5.4 GB (คลิป 180)\nVRAM สูงสุดขณะประมวลผล: ~7.6 GB (คลิป 60), ~14.7 GB (คลิป 180)\n\nการคอมไพล์ใช้เวลา 15-60 นาทีและอาจ VRAM ไม่พอ\nปิดแอปอื่นทั้งหมด (รวมเบราว์เซอร์) และอย่าใช้ PC ระหว่างคอมไพล์\nดำเนินการต่อหรือไม่?",
        # About dialog
        "dialog_about_title": "เกี่ยวกับ Jasna",
        "dialog_about_version": "เวอร์ชัน {version}",
        "dialog_about_description": "เครื่องมือฟื้นฟูโมเสก JAV",
        "dialog_about_credit": "แรงบันดาลใจจาก Lada",
        "btn_close": "ปิด",

        # Language change dialog
        "dialog_language_changed": "เปลี่ยนภาษาแล้ว",
        "dialog_language_restart": "กรุณารีสตาร์ทแอปพลิเคชันเพื่อเปลี่ยนภาษาทั้งหมด",

        # App messages
        "error_cannot_start": "ไม่สามารถเริ่มประมวลผล:",
        "error_invalid_tvai": "การกำหนดค่า TVAI ไม่ถูกต้อง",

        # Settings panel
        "dialog_select_tvai_ffmpeg": "เลือก Topaz Video AI ffmpeg.exe",
        "placeholder_encoder_args": "เช่น lookahead=32",

        # Wizard check labels
        "wizard_window_title": "Jasna - ตรวจสอบระบบ",
        "wizard_check_ffmpeg": "FFmpeg",
        "wizard_check_ffprobe": "FFprobe",
        "wizard_check_mkvmerge": "MKVmerge",
        "wizard_check_gpu": "NVIDIA GPU",
        "wizard_check_cuda": "CUDA Runtime",
        "wizard_check_hags": "Hardware Accelerated GPU Scheduling",
        "wizard_not_checked": "ยังไม่ได้ตรวจสอบ",
        "wizard_not_callable": "เรียกใช้ไม่ได้: {path}",
        "wizard_found_version": "พบ: {path} ({version})",
        "wizard_found_no_major": "พบ: {path} (ตรวจจับเวอร์ชันหลักไม่ได้)",
        "wizard_found_bad_major": "พบ: {path} (major={major}, คาดว่า=8)",
        "wizard_found_major": "พบ: {path} (major={major})",
        "wizard_no_cuda": "ไม่พบอุปกรณ์ CUDA",
        "wizard_gpu_compute_too_low": "ต้องการ Compute capability 7.5+ (GPU: {major}.{minor})",
        "wizard_cuda_version": "CUDA {version}",
        "wizard_cuda_version_compute": "CUDA {version}, compute {major}.{minor}",
        "wizard_not_available": "ไม่พร้อมใช้งาน",

        # Validation errors
        "error_tvai_data_dir_not_set": "ตัวแปรสภาพแวดล้อม TVAI_MODEL_DATA_DIR ไม่ได้ตั้งค่า",
        "error_tvai_model_dir_not_set": "ตัวแปรสภาพแวดล้อม TVAI_MODEL_DIR ไม่ได้ตั้งค่า",
        "error_tvai_data_dir_missing": "TVAI_MODEL_DATA_DIR ชี้ไปยังไดเรกทอรีที่ไม่มีอยู่: {path}",
        "error_tvai_model_dir_missing": "TVAI_MODEL_DIR ชี้ไปยังไดเรกทอรีที่ไม่มีอยู่: {path}",
        "error_tvai_ffmpeg_not_found": "ไม่พบ TVAI ffmpeg: {path}",
    },

}


LANGUAGE_NAMES = {
    "en": "English",
    "zh": "简体中文",
    "ja": "日本語",
    "ko": "한국어",
    "th": "ไทย",
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
                elif lang and lang.startswith("ko"):
                    self._current_lang = "ko"
                elif lang and lang.startswith("th"):
                    self._current_lang = "th"
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
