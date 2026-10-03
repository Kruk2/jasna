"""Static contract checks for the opt-in unified FFmpeg/PyAV builders.

The builders require Windows toolchains and AMD hardware that CI does not
provide.  These tests therefore protect the accepted pins, patch ordering,
and artifact handoff rather than attempting a native build.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH_DIR = ROOT / "patches" / "ffmpeg"
UBUNTU_SCRIPT = ROOT / "scripts" / "build_unified_ffmpeg_pyav_ubuntu.sh"
WINDOWS_SCRIPT = ROOT / "scripts" / "build_unified_ffmpeg_pyav_windows.ps1"

FFMPEG_COMMIT = "44d082edc87381d978e8588b148116b99fefdb43"
PYAV_COMMIT = "7e3d950a8b72062502c1a60d672f8ca565313af5"
AMF_COMMIT = "c35f613aea2e5057a688c979e75b1cf24253297e"
DAV1D_COMMIT = "b546257f770768b2c88258c533da38b91a06f737"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_builders_use_the_runtime_contract_source_pins() -> None:
    ubuntu = _read(UBUNTU_SCRIPT)
    windows = _read(WINDOWS_SCRIPT)

    for source_pin in (FFMPEG_COMMIT, PYAV_COMMIT, AMF_COMMIT):
        assert source_pin in ubuntu
        assert source_pin in windows
    assert DAV1D_COMMIT not in ubuntu
    assert DAV1D_COMMIT in windows


def test_ubuntu_builder_keeps_patch_selection_narrow_and_idempotent() -> None:
    script = _read(UBUNTU_SCRIPT)

    assert "git -C \"$ffmpeg_source\" apply --check \"$patch_path\"" in script
    assert "apply --reverse --check \"$patch_path\"" in script
    assert "apply_dynamic_resolution_fix == true" in script
    assert "apply_frames_context_fix=true" in script
    assert 'apply_patch_once "$transfer_patch"' in script
    assert 'apply_patch_once "$frames_context_patch"' in script
    assert 'apply_patch_once "$dynamic_resolution_patch"' in script
    assert 'apply_patch_once "$spherical_metadata_patch"' in script
    assert 'apply_patch_once "$keyframe_reset_patch"' in script
    assert 'apply_patch_once "$host_native_encoder_patch"' in script
    assert "--apply-spherical-metadata-patch" in script
    assert script.index('apply_patch_once "$transfer_patch"') < script.index(
        'apply_patch_once "$frames_context_patch"'
    ) < script.index('apply_patch_once "$dynamic_resolution_patch"') < script.index(
        'apply_patch_once "$spherical_metadata_patch"'
    ) < script.index(
        'apply_patch_once "$keyframe_reset_patch"'
    ) < script.index(
        'apply_patch_once "$host_native_encoder_patch"'
    )

    for forbidden in (
        "--enable-encode-probe",
        "EnableEncodeProbe",
        "install_unified_runtime.py",
    ):
        assert forbidden not in script


def test_ubuntu_builder_enables_the_limited_amf_decode_surface_and_manifest() -> None:
    script = _read(UBUNTU_SCRIPT)

    for configure_flag in (
        "--enable-amf",
        "--enable-vulkan",
        "--enable-decoder=h264_amf",
        "--enable-decoder=hevc_amf",
        "--enable-decoder=aac",
        "--enable-decoder=av1_amf",
        "--enable-parser=h264",
        "--enable-parser=hevc",
        "--enable-parser=av1",
        "--enable-bsf=h264_mp4toannexb",
        "--enable-bsf=hevc_mp4toannexb",
        "--enable-demuxer=matroska",
        "--enable-muxer=matroska",
    ):
        assert configure_flag in script

    for manifest_field in (
        "FFMPEG_COMMIT=$expected_ffmpeg",
        "PYAV_COMMIT=$expected_pyav",
        "AMF_COMMIT=$expected_amf",
        "TRANSFER_PATCH_SHA256=",
        "FRAMES_CONTEXT_FIX_APPLIED=$apply_frames_context_fix",
        "DYNAMIC_RESOLUTION_FIX_APPLIED=$apply_dynamic_resolution_fix",
        "SPHERICAL_METADATA_PATCH_APPLIED=$apply_spherical_metadata_patch",
        "KEYFRAME_RESET_PATCH_SHA256=",
        "HOST_NATIVE_ENCODER_PATCH_SHA256=",
        "WHEEL_SHA256=",
        "AMF_INTEROP_BRIDGE_SHA256=",
        "AMF_INTEROP_BRIDGE_SOURCE_SHA256=",
        "FFMPEG_BIN=$install_dir/bin",
        "FFMPEG_LIB=$install_dir/lib",
    ):
        assert manifest_field in script
    assert '"$repo_root/scripts/build_amf_surface_probe.py"' in script
    assert "--rocm-include" in script
    assert "--spirv-headers" in script
    assert "spirv/unified1/spirv.h" in script


def test_linux_amf_keyframe_reset_patch_is_opt_in_and_fail_closed() -> None:
    patch = _read(PATCH_DIR / "0005-amfdec-reset-state-at-keyframes.patch")

    assert '"reset_on_keyframe"' in patch
    assert "AV_OPT_TYPE_BOOL" in patch
    assert "{ .i64 = 0 }" in patch
    assert "AV_PKT_FLAG_KEY" in patch
    assert "Drain(ctx->decoder)" in patch
    assert "Terminate(ctx->decoder)" in patch
    assert "Init(ctx->decoder, AMF_SURFACE_UNKNOWN" in patch
    assert "AMF_ASSIGN_PROPERTY_BOOL" in patch
    assert '+DEFINE_AMF_DECODER(hevc, HEVC, "hevc_mp4toannexb")' in patch


def test_linux_amf_host_native_encoder_patch_is_opt_in_and_fail_closed() -> None:
    patch = _read(PATCH_DIR / "0006-amfenc-wrap-contiguous-host-input.patch")

    assert '"host_zero_copy"' in patch
    assert "AV_OPT_TYPE_BOOL" in patch
    assert "{.i64 = 0 }" in patch
    assert "CreateSurfaceFromHostNative" in patch
    assert "frame->buf[0] && frame->buf[1]" in patch
    assert "frame->linesize[0] == frame->linesize[1]" in patch
    assert "frame->data[1] == expected_uv" in patch
    assert "amf_store_attached_frame_ref" in patch


def test_builders_include_the_product_smart_render_cli_surface() -> None:
    ubuntu = _read(UBUNTU_SCRIPT)
    windows = _read(WINDOWS_SCRIPT)

    for script in (ubuntu, windows):
        for configure_flag in (
            "--fatal-warnings",
            "--enable-protocol=file",
            "--enable-protocol=pipe",
            "--enable-demuxer=concat",
            "--enable-demuxer=matroska",
            "--enable-demuxer=mov",
            "--enable-demuxer=mpegts",
            "--enable-demuxer=nut",
            "--enable-muxer=framemd5",
            "--enable-muxer=matroska",
            "--enable-muxer=mov",
            "--enable-muxer=mp4",
            "--enable-muxer=mpegts",
            "--enable-muxer=null",
            "--enable-muxer=nut",
            "--enable-encoder=av1_amf",
            "--enable-encoder=h264_amf",
            "--enable-encoder=hevc_amf",
            "--enable-encoder=rawvideo",
            "--enable-bsf=av1_metadata",
            "--enable-bsf=dump_extradata",
            "--enable-bsf=h264_mp4toannexb",
            "--enable-bsf=hevc_mp4toannexb",
            "--enable-bsf=setts",
        ):
            assert configure_flag in script
def test_windows_builder_pins_and_packages_dav1d() -> None:
    script = _read(WINDOWS_SCRIPT)

    assert "[string] $Dav1dSource" in script
    assert DAV1D_COMMIT in script
    assert 'Assert-GitPin $Dav1dSource $expectedDav1d "dav1d"' in script
    assert "-m mesonbuild.mesonmain setup" in script
    assert "-Denable_tools=false" in script
    assert "-Denable_tests=false" in script
    assert "-Denable_examples=false" in script
    assert "Copy-Item -LiteralPath $dav1dDll" in script
    assert '"DAV1D_DLL_SHA256=' in script


def test_windows_builder_applies_conditional_patches_and_normalizes_targets() -> None:
    script = _read(WINDOWS_SCRIPT)

    assert "$useFramesContextFix = [bool]$ApplyFramesContextFix -or [bool]$ApplyDynamicResolutionFix" in script
    assert "$useDynamicResolutionFix = [bool]$ApplyDynamicResolutionFix" in script
    assert "$useSphericalMetadataPatch = [bool]$ApplySphericalMetadataPatch" in script
    assert "Invoke-GitPatchOnce $FfmpegSource $transferPatch" in script
    assert "Invoke-GitPatchOnce $FfmpegSource $framesContextPatch" in script
    assert "Invoke-GitPatchOnce $FfmpegSource $dynamicResolutionPatch" in script
    assert "Invoke-GitPatchOnce $FfmpegSource $sphericalMetadataPatch" in script
    assert "libavutil\\hwcontext_amf.c" in script
    assert "libavcodec\\amfdec.c" in script
    assert "libavformat\\matroskaenc.c" in script
    assert script.index("Invoke-GitPatchOnce $FfmpegSource $transferPatch") < script.index(
        "Invoke-GitPatchOnce $FfmpegSource $framesContextPatch"
    ) < script.index("Invoke-GitPatchOnce $FfmpegSource $dynamicResolutionPatch") < script.index(
        "Invoke-GitPatchOnce $FfmpegSource $sphericalMetadataPatch"
    )

    for forbidden in (
        "EnableEncodeProbe",
        "build_amf_surface_probe.py",
        "install_unified_runtime.py",
    ):
        assert forbidden not in script


def test_windows_builder_keeps_expected_decode_configuration_and_manifest() -> None:
    script = _read(WINDOWS_SCRIPT)

    for configure_flag in (
        "--enable-libdav1d",
        "--enable-decoder=h264",
        "--enable-decoder=hevc",
        "--enable-decoder=libdav1d",
        "--enable-decoder=h264_amf --enable-decoder=hevc_amf --enable-decoder=av1_amf",
        "--enable-parser=h264 --enable-parser=hevc --enable-parser=av1",
        "--enable-bsf=h264_mp4toannexb --enable-bsf=hevc_mp4toannexb",
        "--enable-filter=hwdownload --enable-filter=format --enable-filter=scale",
    ):
        assert configure_flag in script
    assert "--enable-decoder=av1 " not in script

    for manifest_field in (
        '"FFMPEG_COMMIT=$expectedFfmpeg"',
        '"PYAV_COMMIT=$expectedPyAv"',
        '"AMF_COMMIT=$expectedAmf"',
        '"DAV1D_COMMIT=$expectedDav1d"',
        '"FRAMES_CONTEXT_FIX_APPLIED=$($useFramesContextFix.ToString().ToLower())"',
        '"DYNAMIC_RESOLUTION_FIX_APPLIED=$($useDynamicResolutionFix.ToString().ToLower())"',
        '"SPHERICAL_METADATA_PATCH_APPLIED=$($useSphericalMetadataPatch.ToString().ToLower())"',
        '"WINDOWS_SOFTWARE_FALLBACK_DECODERS_ENABLED=true"',
        '"WINDOWS_AV1_SOFTWARE_DECODER=libdav1d"',
        '"WHEEL_SHA256=',
        '"FFMPEG_BIN=$(Join-Path $installDir',
    ):
        assert manifest_field in script


def test_transfer_patch_advertises_only_the_active_context_format() -> None:
    patch = _read(PATCH_DIR / "0001-amf-transfer-use-context-sw-format.patch")

    assert "fmts = av_malloc_array(2, sizeof(*fmts));" in patch
    assert "fmts[0] = ctx->sw_format;" in patch
    assert "fmts[1] = AV_PIX_FMT_NONE;" in patch
    assert "if (ctx->sw_format == supported_transfer_formats[i])" in patch
    assert "-        fmts[i] = supported_transfer_formats[i];" in patch


def test_frames_context_patch_replaces_the_reference_instead_of_mutating_it() -> None:
    patch = _read(PATCH_DIR / "0002-amfdec-replace-stale-frames-context.patch")

    for required in (
        "amf_replace_frames_context",
        "av_hwframe_ctx_alloc(avctx->hw_device_ctx)",
        "avctx->hw_frames_ctx = new_frames_ref;",
        "av_buffer_unref(&old_frames_ref);",
        "amf_ensure_frames_context",
        "frame->hw_frames_ctx = av_buffer_ref(avctx->hw_frames_ctx);",
    ):
        assert required in patch


def test_dynamic_resolution_patch_restarts_from_unknown_dimensions() -> None:
    patch = _read(PATCH_DIR / "0003-amfdec-fix-dynamic-resolution-reinit.patch")

    for added_line in (
        "+            res = ctx->decoder->pVtbl->Terminate(ctx->decoder);",
        "+            avctx->width = 0;",
        "+            avctx->height = 0;",
        "+            ctx->dimensions_initialized = 0;",
        "+            res = ctx->decoder->pVtbl->Init(ctx->decoder, AMF_SURFACE_UNKNOWN,",
    ):
        assert added_line in patch


def test_spherical_metadata_patch_prefers_coded_side_data_before_tag_fallback() -> None:
    patch = _read(PATCH_DIR / "0004-matroska-projection-tag-spherical.patch")

    assert "if (sd) {" in patch
    assert 'av_dict_get(metadata, "projection", NULL, 0)' in patch
    assert 'av_strcasecmp(tag->value, "equirectangular")' in patch
    assert "spherical = &tag_spherical;" in patch
    assert "projection=equirectangular" not in patch
    assert patch.index("if (sd) {") < patch.index(
        'av_dict_get(metadata, "projection", NULL, 0)'
    )
