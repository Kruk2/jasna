"""CLI argument help strings for jasna.main.build_parser, keyed by argparse dest."""

CLI_HELP: dict[str, str] = {
    "machine_key": "Print this machine's Machine key for activation, then exit. No GPU is needed.",
    "cloud_workspace": (
        "Use with --machine-key or --license-code to activate a rented GPU (RunPod, Vast) "
        "with the official Linux release. Cloud must be turned on for your License code (ask where you got it). "
        "Keep the Jasna config folder on a persistent volume."
    ),
    "activation_key": (
        "Save the Activation key returned for this Machine key by the activation page. "
        "Without --input, --stream or --benchmark, activate and exit."
    ),
    "license_code": (
        "Activate online using your License code, then save the Activation key for offline use. "
        "Without --input, --stream or --benchmark, activate and exit."
    ),
    "restoration_model_name": (
        "Restoration model for video: basicvsrpp (fast), ltx (Standard Quality LTX: more detail, much slower; "
        "NVIDIA RTX only, AMD runs only --ltx-trial; recommended), ltx-4step (Good Quality: a bit more detail, about 1.5x slower than ltx), "
        "ltx-8step (High Quality: more detail, about 2.5x slower than ltx) or ltx-undistilled (Top Quality: "
        "slightly better, about 5x slower than ltx). LTX doesn't "
        "support VR side-by-side video, streaming, secondary restoration or denoise. A missing "
        "LTX model is offered for download. (default: %(default)s)"
    ),
    "restoration_model_path": (
        "Restoration model file (basicvsrpp) or folder (ltx). Default: the model in model_weights/."
    ),
    "ltx_seed": "ltx: noise seed. Different seeds give slightly different results. (default: %(default)s)",
    "ltx_fast": (
        "ltx: about 1.4x faster, slightly less detail. RTX 50-series (Blackwell) only; needs the "
        "fast model file. (default: %(default)s)"
    ),
    "ltx_trial": (
        "ltx: test the speed without the model files. Output is not restored. "
        "(default: %(default)s)"
    ),
    "ltx_auto_split": (
        "ltx: restore only mosaics at least PERCENT of the video height with LTX and smaller "
        "ones with BasicVSR++ (12 is a good start). Finds the mosaics in each video first and "
        "copies the rest unchanged, keeping the video's own codec (--cq and --encoder-settings apply "
        "only when that is --codec). Works on folders; a video "
        "that can't be split is restored whole with LTX."
    ),
    "ltx_large_canvas": (
        "ltx: restore large mosaics at 768 px instead of 512 px. More detail, about 3x slower on "
        "those; needs 12 GB of free VRAM (ignored otherwise). (default: %(default)s)"
    ),
    "experimental_ltx_less_overlap": (
        "ltx: experimental. Fewer frames shared between the chunks LTX restores, about 13%% "
        "faster when a mosaic stays on screen for more than 4 seconds. Turn it off for the older mode with more overlap. (default: %(default)s)"
    ),
    "fp16": "Use half precision. Uses less VRAM and is often faster, with no visible quality loss.",
    "compile_basicvsrpp": "Prepare BasicVSR++ for your GPU: about 3x faster, almost no extra VRAM. The first run takes 15-60 minutes. NVIDIA only. (default: %(default)s)",
    "max_clip_size": "Frames restored together. Bigger is steadier and faster but uses more VRAM; use the biggest that fits. (default: %(default)s)",
    "temporal_overlap": "Frames shared between neighboring clips, to hide flicker where they meet. 8-20 is a good range; must be less than half of --max-clip-size. (default: %(default)s)",
    "enable_crossfade": "Cross-fade between clip boundaries to reduce flickering at seams. Uses frames that are already processed but otherwise discarded, so no extra GPU cost. (default: %(default)s)",
    "denoise": "Spatial denoising strength applied to restored crops. Reduces noise artifacts. (default: %(default)s)",
    "denoise_step": "When to apply denoising: after_primary (before secondary) or after_secondary (right before blend). (default: %(default)s)",
    "secondary_restoration": "Optional second pass that makes restored areas sharper. NVIDIA only; unet-4x needs activation. (default: %(default)s)",
    "vr_mode": (
        "Side-by-side VR180 video: auto turns on by itself for VR videos; off treats every "
        "video as normal; sbs treats every video as VR; sbs-fisheye is like sbs but restores "
        "every mosaic as fisheye. (default: %(default)s)"
    ),
    "tvai_ffmpeg_path": "Path to the ffmpeg.exe of Topaz Video (Windows only). (default: %(default)s)",
    "tvai_model": 'Topaz model name for tvai_up (e.g. "iris-2", "prob-4", "iris-3") (default: %(default)s)',
    "tvai_scale": "Topaz tvai_up scale (1=no scale). Output size is 256*scale (default: %(default)s)",
    "tvai_workers": "Number of parallel TVAI ffmpeg workers (default: %(default)s)",
    "tvai_denoise": "Remove noise before the Topaz model runs. Cleaner, but slower and uses more VRAM.",
    "detection_score_threshold": "Detection threshold, 0-1. Lower finds more, higher finds fewer. Default: the selected model's recommended value.",
    "max_detection_gap": "Fill detection dropouts up to N frames when the mosaic reappears at the same spot. 0 disables (default: %(default)s)",
    "min_detection_duration": "Drop detections shorter than N frames as false positives. 0 disables (default: %(default)s)",
    "scene_detection": "Detect hard scene cuts and end all tracked mosaic clips at the cut, so no clip spans two different shots. (default: %(default)s)",
    "codec": "Offline output video codec (HLS streaming always uses H.264). Default: %(default)s",
    "cq": (
        "Video quality (CQ). Lower is better quality and bigger files. NVIDIA defaults: "
        "H.264 25, HEVC 28, AV1 35; AMD defaults: H.264 24, HEVC 25, AV1 16."
    ),
    "encoder_settings": 'Advanced encoder settings, as a JSON object or comma-separated key=value pairs (e.g. {"rc-lookahead":32} or rc-lookahead=32,bf=4)',
    "post_export_action": (
        "What to do after all videos finish: none; shutdown powers off the PC right away "
        "(no countdown); command runs --post-export-command. (default: %(default)s)"
    ),
    "post_export_video_command": (
        "Shell command to run after each successful video export. Supports "
        "{input}, {output}, {output_dir}, {output_stem}, and {output_suffix}."
    ),
}
