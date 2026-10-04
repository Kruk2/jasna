"""Single source of truth for CLI argument help strings.

Both the CLI parser (jasna.main.build_parser) and the GUI tooltip layer read
these strings from here, so there is exactly one place to edit the wording.
Help strings keep their argparse ``%(default)s`` placeholders; the GUI strips
them when turning a help string into a tooltip.
"""

CLI_HELP: dict[str, str] = {
    "machine_key": "Print this machine's Machine key for activation, then exit. No GPU is needed.",
    "cloud_workspace": (
        "CLI only, in the official Linux release running inside a supported container: "
        "with --machine-key or --license-code, create or reuse a cloud workspace identity for RunPod or Vast. "
        "Your seller must enable it for your License code. Keep the Jasna config directory on "
        "a persistent volume; copying it with the Activation key transfers the identity."
    ),
    "activation_key": (
        "Save the Activation key returned for this Machine key by the activation page or your seller. "
        "Without --input, --stream or --benchmark, activate and exit."
    ),
    "license_code": (
        "Activate online using your seller's License code, then save the Activation key for offline use. "
        "Without --input, --stream or --benchmark, activate and exit."
    ),
    "restoration_model_name": (
        "Restoration model for video: basicvsrpp (fast), ltx (best quality, much slower; NVIDIA "
        "RTX only) or ltx-undistilled (slightly better than ltx, about 2x slower). LTX doesn't "
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
        "copies the rest unchanged, keeping the video's own codec. Works on folders; a video "
        "that can't be split is restored whole with LTX."
    ),
    "ltx_large_canvas": (
        "ltx: restore large mosaics at 768 px instead of 512 px. More detail, about 3x slower on "
        "those; needs 10 GB of free VRAM (ignored otherwise). (default: %(default)s)"
    ),
    "fp16": "Use FP16 where supported (restoration + TensorRT). Reduces VRAM usage and might improve performance.",
    "compile_basicvsrpp": "Compile BasicVSR++ for big performance boost (at cost of VRAM usage). Not recommended to use big clip sizes. (default: %(default)s)",
    "max_clip_size": "Maximum clip size for tracking (default: %(default)s)",
    "temporal_overlap": "Discard margin for overlap+discard clip splitting. Each split uses 2*temporal_overlap input overlap and discards temporal_overlap frames at each split boundary (default: %(default)s)",
    "enable_crossfade": "Cross-fade between clip boundaries to reduce flickering at seams. Uses frames that are already processed but otherwise discarded, so no extra GPU cost. (default: %(default)s)",
    "denoise": "Spatial denoising strength applied to restored crops. Reduces noise artifacts. (default: %(default)s)",
    "denoise_step": "When to apply denoising: after_primary (before secondary) or after_secondary (right before blend). (default: %(default)s)",
    "secondary_restoration": "Secondary restoration after primary model (default: %(default)s)",
    "vr_mode": (
        "VR180 SBS handling: auto uses conservative studio/metadata detection and "
        "routes each mosaic region's restoration projection (raw/fisheye/gnomonic) "
        "by studio; sbs forces per-eye SBS with the same studio routing; sbs-fisheye "
        "forces fisheye conditioning for every region. Detection, tracking, and "
        "blending stay in source coordinates. (default: %(default)s)"
    ),
    "tvai_ffmpeg_path": "Path to Topaz Video ffmpeg.exe (default: %(default)s)",
    "tvai_model": 'Topaz model name for tvai_up (e.g. "iris-2", "prob-4", "iris-3") (default: %(default)s)',
    "tvai_scale": "Topaz tvai_up scale (1=no scale). Output size is 256*scale (default: %(default)s)",
    "tvai_workers": "Number of parallel TVAI ffmpeg workers (default: %(default)s)",
    "tvai_denoise": "Apply TVAI Denoise before enhancement.",
    "detection_score_threshold": "Detection score threshold. When unset, uses the selected model's recommended value (rfdetr-v6: 0.35, rfdetr-v6-large: 0.40).",
    "max_detection_gap": "Fill detection dropouts up to N frames when the mosaic reappears at the same spot. 0 disables (default: %(default)s)",
    "min_detection_duration": "Drop detections shorter than N frames as false positives. 0 disables (default: %(default)s)",
    "scene_detection": "Detect hard scene cuts and end all tracked mosaic clips at the cut, so no clip spans two different shots. (default: %(default)s)",
    "codec": "Offline output video codec (HLS streaming always uses H.264). Default: %(default)s",
    "cq": (
        "Literal encoder quality target passed unchanged. Lower values improve "
        "quality and increase file size. NVIDIA defaults: H.264 25, HEVC 28, "
        "AV1 35; AMD defaults: H.264 24, HEVC 25, AV1 32."
    ),
    "encoder_settings": 'Advanced encoder settings, as a JSON object or comma-separated key=value pairs (e.g. {"rc-lookahead":32} or rc-lookahead=32,bf=4)',
    "post_export_action": "Action to run after all non-streaming exports finish.",
    "post_export_video_command": (
        "Shell command to run after each successful video export. Supports "
        "{input}, {output}, {output_dir}, {output_stem}, and {output_suffix}."
    ),
}
