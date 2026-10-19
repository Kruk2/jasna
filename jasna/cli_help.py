"""Single source of truth for CLI argument help strings.

Both the CLI parser (jasna.main.build_parser) and the GUI tooltip layer read
these strings from here, so there is exactly one place to edit the wording.
Help strings keep their argparse ``%(default)s`` placeholders; the GUI strips
them when turning a help string into a tooltip.
"""

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
        "Restoration model for video input: basicvsrpp (fast), ltx (diffusion model, much "
        "slower, best quality; no streaming, secondary restoration or denoise) or "
        "ltx-undistilled (the same model without the 8-step speed-up, about 3x slower, looks "
        "about the same). A missing ltx model is offered for download. (default: %(default)s)"
    ),
    "restoration_model_path": (
        "Restoration model file (basicvsrpp) or folder (ltx). Default: the model in model_weights/."
    ),
    "ltx_seed": "ltx: noise seed. Another seed gives another take on the same restoration. (default: %(default)s)",
    "ltx_fast": (
        "ltx: about 1.4x faster with slightly less detail. Needs an RTX 50-series (Blackwell) GPU "
        "and the fast model file. (default: %(default)s)"
    ),
    "ltx_trial": (
        "Speed test without a license: runs LTX with placeholder weights, so the picture will look "
        "wrong. Use it to see if your GPU can run LTX and how fast. (default: %(default)s)"
    ),
    "ltx_auto_split": (
        "ltx: restore only mosaics at least PERCENT of the video height with LTX and smaller "
        "ones with BasicVSR++ (12 is a good start). Finds the mosaics in each video first and "
        "copies the rest unchanged, keeping the video's own codec. Works on folders; a video "
        "that can't be split is restored whole with LTX."
    ),
    "ltx_large_canvas": (
        "ltx: restore large mosaics at 768 px instead of 512 px. Sharper, but about 3x slower for "
        "those mosaics; needs 10 GB of free VRAM (ignored below that). (default: %(default)s)"
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
    "secondary_restoration": (
        "Secondary restoration after primary model: none, unet-4x (NVIDIA supporter model), "
        "tvai (Topaz Video), rtx-super-res (NVIDIA Maxine), or amd-upscale (AMD GPU "
        "super resolution: AMF Video SR or Real-ESRGAN on ROCm). (default: %(default)s)"
    ),
    "amd_upscale_engine": (
        "AMD upscaling engine, the upper-level class: real-esr (SRVGGNetCompact checkpoints, "
        "in-process on the ROCm/HIP device; the default), realesrgan (RRDBNet checkpoints, "
        "also in-process) or amf-sr (AMD's own Video SR 1.0/1.1 via the AMF filter sr_amf; "
        "AMF initialises over D3D11 on Windows). --amd-upscale-model lists only the weights "
        "that belong to the chosen engine. (default: %(default)s)"
    ),
    "amd_upscale_model": (
        "Super-resolution checkpoint for the rocm engine, i.e. the concrete weight of the "
        "engine above. real-esr: x4v3 (32-conv SRVGGNetCompact, fastest and the default), "
        "wdn-x4v3 (denoising variant), lsdir-c3 (16-conv 4xLSDIRCompactC3, LSDIR live-action), "
        "lsdir-v2 (16-conv 4xLSDIRCompactv2, highest-PSNR of the compact nets), "
        "hfa2k-2x (16-conv 2xHFA2kCompact, 2x-native) "
        "or auto (first match in model_weights/). realesrgan: x4plus (23-block RRDBNet, best "
        "quality), bsrnet (KAIR/BSRGAN 23-block) "
        "or auto. Ignored by amf-sr and when --amd-upscale-model-path is given. "
        "(default: %(default)s)"
    ),
    "amd_upscale_model_path": (
        "Checkpoint for the rocm engine: an RRDBNet (Real-ESRGAN x4plus / "
        "x2plus, BSRNet) or an SRVGGNetCompact (realesr-general-x4v3 / -wdn-x4v3, "
        "4xLSDIRCompactC3 / 4xLSDIRCompactv2, 2xHFA2kCompact). The architecture is derived "
        "from the checkpoint. Empty "
        "auto-detects one in model_weights/. (default: %(default)s)"
    ),
    "amd_upscale_scale": (
        "AMD upscale factor; the crop is restored at 256*scale and blended back. "
        "6x/8x keep the SR engine engaged and act as supersampling, at a "
        "proportional time cost (default: %(default)s)"
    ),
    "amd_upscale_algorithm": (
        "sr_amf algorithm: sr1-0 / sr1-1 are AMD Video SR (machine learning), "
        "bicubic / bilinear / point are the non-ML fallbacks. (default: %(default)s)"
    ),
    "amd_upscale_sharpness": (
        "Sharpening for the chosen AMD upscale engine. amf-sr: the sr_amf sharpness "
        "(-1 keeps the driver default, 0 disables, up to 2). real-esr / realesrgan: "
        "CAS sharpening of the restored patch only, applied to the mosaic region and "
        "not the whole frame (0 disables, 0.2-0.4 is a gentle boost, up to 1.0). "
        "(default: %(default)s)"
    ),
    "amd_upscale_ffmpeg_path": (
        "FFmpeg to use for AMD upscaling (needs the AMF filters). Empty uses the bundled "
        "copy or 'ffmpeg' from PATH. (default: %(default)s)"
    ),
    "amd_upscale_timeout": (
        "Seconds to wait for the AMD upscale FFmpeg process before giving up on a clip "
        "(default: %(default)s)"
    ),
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
