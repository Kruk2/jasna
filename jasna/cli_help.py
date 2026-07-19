"""Single source of truth for CLI argument help strings.

Both the CLI parser (jasna.main.build_parser) and the GUI tooltip layer read
these strings from here, so there is exactly one place to edit the wording.
Help strings keep their argparse ``%(default)s`` placeholders; the GUI strips
them when turning a help string into a tooltip.
"""

CLI_HELP: dict[str, str] = {
    "fp16": "Use FP16 where supported (restoration + TensorRT). Reduces VRAM usage and might improve performance.",
    "compile_basicvsrpp": "Compile BasicVSR++ for big performance boost (at cost of VRAM usage). Not recommended to use big clip sizes. (default: %(default)s)",
    "max_clip_size": "Maximum clip size for tracking (default: %(default)s)",
    "temporal_overlap": "Discard margin for overlap+discard clip splitting. Each split uses 2*temporal_overlap input overlap and discards temporal_overlap frames at each split boundary (default: %(default)s)",
    "enable_crossfade": "Cross-fade between clip boundaries to reduce flickering at seams. Uses frames that are already processed but otherwise discarded, so no extra GPU cost. (default: %(default)s)",
    "denoise": "Spatial denoising strength applied to restored crops. Reduces noise artifacts. (default: %(default)s)",
    "denoise_step": "When to apply denoising: after_primary (before secondary) or after_secondary (right before blend). (default: %(default)s)",
    "secondary_restoration": "Secondary restoration after primary model (default: %(default)s)",
    "vr_mode": (
        "VR180 handling: auto uses conservative studio/metadata detection; "
        "sbs processes each eye separately; sbs-fisheye also reprojects each "
        "eye for detection/restoration and preserves source projection on output. "
        "(default: %(default)s)"
    ),
    "tvai_ffmpeg_path": "Path to Topaz Video ffmpeg.exe (default: %(default)s)",
    "tvai_model": 'Topaz model name for tvai_up (e.g. "iris-2", "prob-4", "iris-3") (default: %(default)s)',
    "tvai_scale": "Topaz tvai_up scale (1=no scale). Output size is 256*scale (default: %(default)s)",
    "tvai_workers": "Number of parallel TVAI ffmpeg workers (default: %(default)s)",
    "detection_score_threshold": "Detection score threshold (default: %(default)s)",
    "max_detection_gap": "Fill detection dropouts up to N frames when the mosaic reappears at the same spot. 0 disables (default: %(default)s)",
    "min_detection_duration": "Drop detections shorter than N frames as false positives. 0 disables (default: %(default)s)",
    "codec": "Offline output video codec (HLS streaming always uses H.264). Default: %(default)s",
    "encoder_settings": 'Encoder settings, as JSON object or comma-separated key=value pairs (e.g. {"cq":22} or cq=22,rc-lookahead=32)',
    "post_export_action": "Action to run after all non-streaming exports finish.",
}


# Maps a CLI argument dest to the GUI tooltip key that reuses its help text.
GUI_TOOLTIP_KEY_BY_DEST: dict[str, str] = {
    "fp16": "fp16_mode",
    "compile_basicvsrpp": "compile_basicvsrpp",
    "max_clip_size": "max_clip_size",
    "temporal_overlap": "temporal_overlap",
    "enable_crossfade": "enable_crossfade",
    "vr_mode": "vr_mode",
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
    "post_export_action": "post_export_action",
}
