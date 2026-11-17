"""CLI argument help strings for jasna.cli_parser.build_parser, keyed by argparse dest."""

from jasna.session_config import DEFAULT_CODEC

CLI_HELP: dict[str, str] = {
    "license_code": (
        "Activate with your License code. On a computer with a TPM 2.0 security chip, Jasna then works offline; "
        "otherwise, and always in a Linux container or virtual machine, it checks your license online while it "
        "uses supporter models. "
        "Without --input, --stream or --benchmark, activate and exit. "
        "Instead of this option you can set the JASNA_LICENSE_CODE environment variable; Jasna uses it only on a "
        "computer that is not activated yet."
    ),
    "online": "With --license-code or JASNA_LICENSE_CODE: check the license online instead of working offline. Use it on rented or shared PCs, where an offline activation would keep one of your 3 devices.",
    "activation_status": "Show whether Jasna is activated and how many of your devices are in use, then exit.",
    "restoration_model_name": (
        "Restoration model for video: basicvsrpp (fast), ltx-basic (Basic Quality LTX: about 1.7x faster than ltx, "
        "slightly softer), ltx (Standard Quality LTX: more detail, much slower; "
        "NVIDIA RTX or AMD Radeon RX 7000/9000; recommended), ltx-4step (Good Quality: a bit more detail, about 1.5x slower than ltx), "
        "ltx-8step (High Quality: more detail, about 2.5x slower than ltx) or ltx-undistilled (Top Quality: "
        "slightly better, about 5x slower than ltx). LTX doesn't "
        "support streaming; secondary restoration and denoise work only on the "
        "BasicVSR++ parts of --ltx-auto-split. A missing "
        "LTX model is offered for download. (default: %(default)s)"
    ),
    "frame_generation": (
        "2x: double the frame rate with NVIDIA RTX Video Frame Generation for smoother motion. "
        "Needs an RTX 40-series or newer GPU. Audio and length stay the same. "
        "Restores the whole video, so it can't be combined with --segments, --ltx-auto-split or --stream. "
        "With --retarget-high-fps, 60 FPS is halved first and then doubled back. "
        "(default: %(default)s)"
    ),
    "upscale": (
        "Upscale the whole video before saving it, so the output is bigger and sharper. "
        "span: a fast AI upscaler, works on NVIDIA and AMD (NVIDIA prepares it for each new video size once, "
        "about half a minute). realplksr: a slower AI upscaler that keeps more fine detail and cleans compression, "
        "NVIDIA and AMD. rtx-super-res: NVIDIA RTX Video Super Resolution, very fast, NVIDIA only. "
        "dlss5: NVIDIA DLSS 5, reworks lighting and surfaces while it upscales, slow, Windows with an RTX 40 or 50 GPU "
        "only (RTX 30 works but is very slow); Jasna offers to download the unofficial DLSS 5 runtime (467 MB) the first time. "
        "Restores the whole video, so it can't be combined with --segments, --ltx-auto-split or --stream. "
        "(default: %(default)s)"
    ),
    "upscale_height": (
        "Output height for --upscale; the width keeps the aspect ratio. Videos already this tall are not "
        "upscaled. (default: %(default)s)"
    ),
    "upscale_strength": (
        "How much of the AI upscale to keep, 0-1. Lower mixes in a plain resize for a softer, more natural "
        "look. (default: %(default)s)"
    ),
    "upscale_rtx_quality": "--upscale rtx-super-res: upscale quality; higher is sharper and slower. (default: %(default)s)",
    "upscale_rtx_denoise": "--upscale rtx-super-res: remove noise before upscaling, none to disable. (default: %(default)s)",
    "upscale_rtx_deblur": "--upscale rtx-super-res: remove blur before upscaling, none to disable. (default: %(default)s)",
    "upscale_dlss5_preset": (
        "--upscale dlss5: the DLSS upscaling model; l and m rebuild more fine texture. (default: %(default)s)"
    ),
    "upscale_dlss5_style": "--upscale dlss5: the look of the rendered result. (default: %(default)s)",
    "upscale_dlss5_intensity": (
        "--upscale dlss5: how strongly DLSS 5 reworks lighting and surfaces, 0-2. (default: %(default)s)"
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
        "ltx: restore large mosaics at 768 px instead of 512 px. More detail; most videos have some, "
        "so a whole run can take up to about 3x longer; needs 9 GB of free VRAM, so 12 GB cards can use it (ignored otherwise). "
        "(default: %(default)s)"
    ),
    "ltx_overlap": (
        "ltx: frames shared between the chunks LTX restores. Fewer is faster when a mosaic stays on screen "
        "for more than 4 seconds; 33 is the older mode with more overlap. (default: %(default)s)"
    ),
    "ltx_vae_encoder": (
        "ltx: how LTX reads the video: full (sharpest, slowest), half (faster) or max (fastest, slightly softer). "
        "auto: half for ltx-basic, full for the others. (default: %(default)s)"
    ),
    "ltx_vae_decoder": (
        "ltx: how LTX turns its result back into video: full (sharpest, slowest), half (faster) or max "
        "(fastest, slightly softer). auto: max for ltx-basic, full for the others. (default: %(default)s)"
    ),
    "fp16": "Use half precision. Uses less VRAM and is often faster, with no visible quality loss.",
    "compile_basicvsrpp": "Prepare BasicVSR++ for your GPU: about 3x faster, almost no extra VRAM. The first run takes 15-60 minutes. NVIDIA only. (default: %(default)s)",
    "max_clip_size": "Frames restored together. Bigger is steadier and faster but uses more VRAM; use the biggest that fits. (default: %(default)s)",
    "temporal_overlap": "Frames shared between neighboring clips, to hide flicker where they meet. 8-20 is a good range; must be less than half of --max-clip-size. (default: %(default)s)",
    "denoise": "Spatial denoising strength applied to restored crops. Reduces noise artifacts. (default: %(default)s)",
    "secondary_restoration": "Optional second pass that makes restored areas sharper. UNet 4x works on NVIDIA and AMD and needs activation; TVAI and RTX Super Res require NVIDIA. With LTX only RTX Super Res works (others need --ltx-auto-split); it sharpens large mosaics. (default: %(default)s)",
    "vr_mode": (
        "VR180 video: auto turns on by itself for side-by-side and top/bottom VR videos; off "
        "treats every video as normal; sbs treats every video as side-by-side VR; sbs-fisheye "
        "is like sbs but restores every mosaic as fisheye; tb treats every video as top/bottom "
        "VR. (default: %(default)s)"
    ),
    "tvai_ffmpeg_path": "Path to the ffmpeg.exe of Topaz Video (Windows only). (default: %(default)s)",
    "tvai_model": 'Topaz model name for tvai_up (e.g. "iris-2", "prob-4", "iris-3") (default: %(default)s)',
    "tvai_scale": "Topaz tvai_up scale (1=no scale). Output size is 256*scale (default: %(default)s)",
    "tvai_workers": "Number of parallel TVAI ffmpeg workers (default: %(default)s)",
    "tvai_denoise": "Remove noise before the Topaz model runs. Cleaner, but slower and uses more VRAM.",
    "detection_score_threshold": "Detection threshold, 0-1. Lower finds more, higher finds fewer. Default: the selected model's recommended value.",
    "max_detection_gap": "Fill detection dropouts up to N frames when the mosaic reappears at the same spot. 0 disables (default: %(default)s). LTX always fills dropouts of up to 3 frames; values above 3 apply to LTX too",
    "min_detection_duration": "Drop detections shorter than N frames as false positives. 0 disables (default: %(default)s)",
    "scene_detection": "Detect hard scene cuts and end all tracked mosaic clips at the cut, so no clip spans two different shots. (default: %(default)s)",
    "codec": f"Offline output video codec (HLS streaming always uses H.264). Default: {DEFAULT_CODEC}",
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
    "benchmark": 'Run benchmarks instead of processing video',
    "input": 'Path to input video, image, or folder',
    "output": 'Path to output file, or output folder when --input is a folder',
    "output_pattern": (
        'Filename template for folder input, matching the GUI pattern behavior. Use {original} for the input '
        'stem. Default: {original}_out with each input extension. Images keep their source extension; videos '
        'use the template extension when provided.'
    ),
    "working_directory": (
        "Directory for temporary files created while assembling segment output (default: the output video's "
        'directory)'
    ),
    "log_level": 'Logging level (default: %(default)s)',
    "no_progress": 'Disable the progress bar.',
    "image_restoration_model_name": (
        'Restoration model for still-image input (image-only). Images auto-route here; no need to set '
        '--restoration-model-name.'
    ),
    "sd15_steps": 'More steps = more detail, but slower (default: %(default)s)',
    "sd15_strength": 'How much of the image is redrawn, up to 0.7. Higher = bigger changes (default: %(default)s)',
    "sd15_freeu": 'Small extra detail boost; turn off if results look over-sharpened (default: %(default)s)',
    "sd15_seed": 'Starting number for the random result. Same seed = same result (default: %(default)s)',
    "sd15_variants": (
        'Save N versions (name_v1 … name_vN), each with the next seed, so you can pick the best (default: '
        '%(default)s)'
    ),
    "rtx_scale": 'RTX Super Res upscale factor for BasicVSR++; LTX always upscales 2x (default: %(default)s)',
    "rtx_quality": 'RTX Super Res upscale quality (default: %(default)s)',
    "rtx_denoise": 'RTX Super Res denoise level, none to disable (default: %(default)s)',
    "rtx_deblur": 'RTX Super Res deblur level, none to disable (default: %(default)s)',
    "tvai_args": 'Extra options for the Topaz model, for advanced users (default: %(default)s)',
    "detection_model": (
        'Detection model name. Installed models are discovered from model_weights/; rfdetr-v6 (fast), '
        'rfdetr-vr-v1 (VR180) and lada-yolo-v4 are bundled with Jasna, rfdetr-v6-large (higher quality) and '
        'zelefans-vr-yolo-v2 are optional downloads (default: %(default)s)'
    ),
    "detection_model_path": (
        'Optional path to detection weights. If not set, uses "model_weights/<detection-model>": .onnx for '
        'RF-DETR on NVIDIA, .pt for RF-DETR on AMD and for YOLO.'
    ),
    "stream": (
        'Enable HLS streaming mode (no file output). Serves processed video via HTTP for playback in '
        'VLC/browser.'
    ),
    "stream_port": 'HTTP port for HLS streaming server (default: %(default)s)',
    "stream_segment_duration": 'HLS segment duration in seconds (default: %(default)s)',
    "no_browser": 'Do not open a browser window when starting streaming mode.',
    "lut": 'Path to a .cube color LUT (1D or 3D) applied on GPU before encoding.',
    "sharpen": (
        'Sharpen the picture on GPU before encoding, from 0 (off) to 1 (strongest). Matches the ffmpeg cas '
        'filter.'
    ),
    "retarget_high_fps": (
        'For offline exports, map 60 fps to 30 fps and 59.94 fps to 29.97 fps by processing every second '
        'frame. Other source rates are unchanged.'
    ),
    "fmp4": (
        'Write .mp4/.mov output as fragmented MP4, so the file can be played while processing runs and stays '
        'playable after an interruption. Not available with --stream or --segments.'
    ),
    "segments": (
        'Restore only selected ranges and smart-render the rest, for example 10-25,01:10-01:30. Output codec '
        'must match the H.264, HEVC, or AV1 input.'
    ),
    "post_export_command": 'Shell command to run when --post-export-action=command.',
    "benchmark_filter": "Only run benchmarks whose name contains this string (e.g. 'basicvsrpp')",
    "benchmark_video": (
        'Video path for benchmark (can be repeated). Default: test_clip1_1080p.mp4, test_clip1_2160p.mp4'
    ),
}
