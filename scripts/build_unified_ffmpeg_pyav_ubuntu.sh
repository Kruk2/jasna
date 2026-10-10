#!/usr/bin/env bash

# Build one Linux AMD FFmpeg/PyAV ABI unit from the accepted source pins.
#
# The output is deliberately only a build artifact. Installation and runtime
# selection remain separate responsibilities of the existing installer and
# runtime contract, respectively.

set -euo pipefail

export PATH=/usr/bin:/bin:${PATH:-}
unset PKG_CONFIG_SYSROOT_DIR

usage() {
  cat <<'EOF'
Usage:
  build_unified_ffmpeg_pyav_ubuntu.sh \
    --ffmpeg-source PATH --pyav-source PATH --amf-source PATH \
    --output-root PATH [--python PYTHON] [--jobs N] \
    [--vulkan-headers PATH] [--spirv-headers PATH] [--rocm-include PATH] \
    [--apply-frames-context-fix] \
    [--apply-dynamic-resolution-fix] \
    [--apply-spherical-metadata-patch]

Build a shared FFmpeg and PyAV wheel from the accepted FFmpeg, PyAV, and AMF
source commits. Source checkouts must be disposable: this script applies the
selected FFmpeg patches in place, but accepts a patch that is already applied.

--apply-dynamic-resolution-fix also applies the frames-context fix it depends
on. The spherical metadata patch is opt-in and enables its Matroska metadata
fallback without changing the default decoder-only patch set.
EOF
}

require_option_value() {
  if (($# < 2)); then
    printf 'Option %s requires a value.\n' "$1" >&2
    exit 2
  fi
}

ffmpeg_source=
pyav_source=
amf_source=
output_root=
vulkan_headers=
spirv_headers=
rocm_include=/opt/rocm/include
apply_frames_context_fix=false
apply_dynamic_resolution_fix=false
apply_spherical_metadata_patch=false
python=python3
jobs=1
if command -v nproc >/dev/null 2>&1; then
  jobs=$(nproc)
elif command -v getconf >/dev/null 2>&1; then
  jobs=$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')
fi

while (($#)); do
  case "$1" in
    --ffmpeg-source)
      require_option_value "$@"
      ffmpeg_source=$2
      shift 2
      ;;
    --pyav-source)
      require_option_value "$@"
      pyav_source=$2
      shift 2
      ;;
    --amf-source)
      require_option_value "$@"
      amf_source=$2
      shift 2
      ;;
    --output-root)
      require_option_value "$@"
      output_root=$2
      shift 2
      ;;
    --vulkan-headers)
      require_option_value "$@"
      vulkan_headers=$2
      shift 2
      ;;
    --spirv-headers)
      require_option_value "$@"
      spirv_headers=$2
      shift 2
      ;;
    --rocm-include)
      require_option_value "$@"
      rocm_include=$2
      shift 2
      ;;
    --python)
      require_option_value "$@"
      python=$2
      shift 2
      ;;
    --jobs)
      require_option_value "$@"
      jobs=$2
      shift 2
      ;;
    --apply-frames-context-fix)
      apply_frames_context_fix=true
      shift
      ;;
    --apply-dynamic-resolution-fix)
      apply_dynamic_resolution_fix=true
      shift
      ;;
    --apply-spherical-metadata-patch)
      apply_spherical_metadata_patch=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

[[ -n $ffmpeg_source && -n $pyav_source && -n $amf_source && -n $output_root ]] || {
  usage >&2
  exit 2
}
[[ $jobs =~ ^[1-9][0-9]*$ ]] || {
  printf '%s\n' '--jobs must be a positive integer.' >&2
  exit 2
}

ffmpeg_source=$(realpath "$ffmpeg_source")
pyav_source=$(realpath "$pyav_source")
amf_source=$(realpath "$amf_source")
mkdir -p "$output_root"
output_root=$(realpath "$output_root")

readonly expected_ffmpeg=44d082edc87381d978e8588b148116b99fefdb43
readonly expected_pyav=7e3d950a8b72062502c1a60d672f8ca565313af5
readonly expected_amf=c35f613aea2e5057a688c979e75b1cf24253297e
readonly repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
readonly transfer_patch="$repo_root/patches/ffmpeg/0001-amf-transfer-use-context-sw-format.patch"
readonly frames_context_patch="$repo_root/patches/ffmpeg/0002-amfdec-replace-stale-frames-context.patch"
readonly dynamic_resolution_patch="$repo_root/patches/ffmpeg/0003-amfdec-fix-dynamic-resolution-reinit.patch"
readonly spherical_metadata_patch="$repo_root/patches/ffmpeg/0004-matroska-projection-tag-spherical.patch"
readonly keyframe_reset_patch="$repo_root/patches/ffmpeg/0005-amfdec-reset-state-at-keyframes.patch"
readonly host_native_encoder_patch="$repo_root/patches/ffmpeg/0006-amfenc-wrap-contiguous-host-input.patch"

assert_pin() {
  local path=$1 expected=$2 label=$3 actual
  actual=$(git -C "$path" rev-parse HEAD)
  [[ $actual == "$expected" ]] || {
    printf '%s source must be pinned at %s; observed %s\n' \
      "$label" "$expected" "$actual" >&2
    exit 1
  }
}

apply_patch_once() {
  local patch_path=$1
  if git -C "$ffmpeg_source" apply --check "$patch_path" 2>/dev/null; then
    git -C "$ffmpeg_source" apply "$patch_path"
  elif ! git -C "$ffmpeg_source" apply --reverse --check "$patch_path" 2>/dev/null; then
    printf 'FFmpeg source is neither clean nor already patched for %s\n' \
      "$patch_path" >&2
    exit 1
  fi
}

assert_pin "$ffmpeg_source" "$expected_ffmpeg" FFmpeg
assert_pin "$pyav_source" "$expected_pyav" PyAV
assert_pin "$amf_source" "$expected_amf" AMF

if [[ $apply_dynamic_resolution_fix == true ]]; then
  apply_frames_context_fix=true
fi

apply_patch_once "$transfer_patch"
if [[ $apply_frames_context_fix == true ]]; then
  apply_patch_once "$frames_context_patch"
fi
if [[ $apply_dynamic_resolution_fix == true ]]; then
  apply_patch_once "$dynamic_resolution_patch"
fi
if [[ $apply_spherical_metadata_patch == true ]]; then
  apply_patch_once "$spherical_metadata_patch"
fi
apply_patch_once "$keyframe_reset_patch"
apply_patch_once "$host_native_encoder_patch"

amf_include=$amf_source
if [[ ! -f $amf_include/AMF/core/Factory.h ]]; then
  published_headers=$amf_source/amf/public/include
  [[ -f $published_headers/core/Factory.h ]] || {
    printf 'AMF headers were not found below %s\n' "$amf_source" >&2
    exit 1
  }
  amf_include=$output_root/amf-include
  mkdir -p "$amf_include/AMF"
  cp -a "$published_headers/." "$amf_include/AMF/"
fi

vulkan_include=
if [[ -n $vulkan_headers ]]; then
  vulkan_headers=$(realpath "$vulkan_headers")
  if [[ -f $vulkan_headers/include/vulkan/vulkan.h ]]; then
    vulkan_include=$vulkan_headers/include
  elif [[ -f $vulkan_headers/vulkan/vulkan.h ]]; then
    vulkan_include=$vulkan_headers
  else
    printf 'Vulkan headers were not found below %s\n' "$vulkan_headers" >&2
    exit 1
  fi
elif [[ -f /usr/include/vulkan/vulkan.h ]]; then
  vulkan_include=/usr/include
fi
[[ -n $vulkan_include ]] || {
  printf '%s\n' 'Vulkan headers are required; pass --vulkan-headers PATH.' >&2
  exit 1
}

spirv_include=
if [[ -n $spirv_headers ]]; then
  spirv_headers=$(realpath "$spirv_headers")
  if [[ -f $spirv_headers/include/spirv/unified1/spirv.h ]]; then
    spirv_include=$spirv_headers/include
  elif [[ -f $spirv_headers/spirv/unified1/spirv.h ]]; then
    spirv_include=$spirv_headers
  else
    printf 'SPIR-V headers were not found below %s\n' "$spirv_headers" >&2
    exit 1
  fi
elif [[ -f /usr/include/spirv/unified1/spirv.h ]]; then
  spirv_include=/usr/include
fi
[[ -n $spirv_include ]] || {
  printf '%s\n' 'SPIR-V headers are required; pass --spirv-headers PATH.' >&2
  exit 1
}
rocm_include=$(realpath "$rocm_include")
[[ -f $rocm_include/hip/hip_runtime_api.h ]] || {
  printf 'ROCm HIP headers were not found below %s\n' "$rocm_include" >&2
  exit 1
}

extra_cflags="-I$amf_include -I$vulkan_include -I$spirv_include"
build_dir=$output_root/ffmpeg-build
install_dir=$output_root/ffmpeg-install
wheel_dir=$output_root/wheels
mkdir -p "$build_dir" "$install_dir" "$wheel_dir"

configure_args=(
  "--prefix=$install_dir"
  --enable-shared
  --disable-static
  --disable-debug
  --disable-doc
  --disable-autodetect
  --disable-everything
  --fatal-warnings
  --enable-ffmpeg
  --enable-ffprobe
  --enable-avcodec
  --enable-avformat
  --enable-avfilter
  --enable-avdevice
  --enable-swscale
  --enable-swresample
  --enable-amf
  --enable-vulkan
  --enable-protocol=file
  --enable-protocol=pipe
  --enable-demuxer=concat
  --enable-demuxer=matroska
  --enable-demuxer=mov
  --enable-demuxer=mpegts
  --enable-demuxer=nut
  --enable-muxer=framemd5
  --enable-muxer=matroska
  --enable-muxer=mov
  --enable-muxer=mp4
  --enable-muxer=mpegts
  --enable-muxer=null
  --enable-muxer=nut
  --enable-decoder=h264_amf
  --enable-decoder=hevc_amf
  --enable-decoder=av1_amf
  --enable-decoder=aac
  --enable-encoder=h264_amf
  --enable-encoder=hevc_amf
  --enable-encoder=av1_amf
  --enable-encoder=rawvideo
  --enable-parser=h264
  --enable-parser=hevc
  --enable-parser=av1
  --enable-bsf=av1_metadata
  --enable-bsf=dump_extradata
  --enable-bsf=h264_mp4toannexb
  --enable-bsf=hevc_mp4toannexb
  --enable-bsf=setts
  --enable-filter=hwdownload
  --enable-filter=format
  --enable-encoder=wrapped_avframe
  "--extra-cflags=$extra_cflags"
)

cd "$build_dir"
"$ffmpeg_source/configure" "${configure_args[@]}"
make -j"$jobs"
make install

for required_output in "$install_dir/bin/ffmpeg" "$install_dir/bin/ffprobe"; do
  [[ -x $required_output ]] || {
    printf 'FFmpeg install did not produce %s\n' "$required_output" >&2
    exit 1
  }
done

export PKG_CONFIG_PATH="$install_dir/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$install_dir/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$pyav_source"
"$python" setup.py bdist_wheel
wheel=$(find "$pyav_source/dist" -maxdepth 1 -type f -name 'av-*.whl' \
  -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
[[ -n $wheel ]] || {
  printf '%s\n' 'PyAV build produced no wheel.' >&2
  exit 1
}
cp -f "$wheel" "$wheel_dir/"
copied_wheel=$wheel_dir/$(basename "$wheel")

bridge_dir=$output_root/amf-interop-bridge
"$python" "$repo_root/scripts/build_amf_surface_probe.py" \
  --pyav-source "$pyav_source" \
  --amf-include "$amf_include" \
  --ffmpeg-include "$install_dir/include" \
  --ffmpeg-lib "$install_dir/lib" \
  --vulkan-include "$vulkan_include" \
  --rocm-include "$rocm_include" \
  --output-dir "$bridge_dir"
mapfile -t bridges < <(find "$bridge_dir" -maxdepth 1 -type f \
  -name '_jasna_amf_surface_probe.*.so' -print)
if ((${#bridges[@]} != 1)); then
  printf 'AMF interop bridge build produced %d extension modules below %s\n' \
    "${#bridges[@]}" "$bridge_dir" >&2
  exit 1
fi
bridge=${bridges[0]}

cat >"$output_root/build-manifest.txt" <<EOF
FFMPEG_COMMIT=$expected_ffmpeg
PYAV_COMMIT=$expected_pyav
AMF_COMMIT=$expected_amf
TRANSFER_PATCH_SHA256=$(sha256sum "$transfer_patch" | awk '{print $1}')
FRAMES_CONTEXT_FIX_APPLIED=$apply_frames_context_fix
FRAMES_CONTEXT_PATCH_SHA256=$(sha256sum "$frames_context_patch" | awk '{print $1}')
DYNAMIC_RESOLUTION_FIX_APPLIED=$apply_dynamic_resolution_fix
DYNAMIC_RESOLUTION_PATCH_SHA256=$(sha256sum "$dynamic_resolution_patch" | awk '{print $1}')
SPHERICAL_METADATA_PATCH_APPLIED=$apply_spherical_metadata_patch
SPHERICAL_METADATA_PATCH_SHA256=$(sha256sum "$spherical_metadata_patch" | awk '{print $1}')
KEYFRAME_RESET_PATCH_SHA256=$(sha256sum "$keyframe_reset_patch" | awk '{print $1}')
HOST_NATIVE_ENCODER_PATCH_SHA256=$(sha256sum "$host_native_encoder_patch" | awk '{print $1}')
VULKAN_INCLUDE=$vulkan_include
WHEEL=$copied_wheel
WHEEL_SHA256=$(sha256sum "$copied_wheel" | awk '{print $1}')
AMF_INTEROP_BRIDGE=$bridge
AMF_INTEROP_BRIDGE_SHA256=$(sha256sum "$bridge" | awk '{print $1}')
AMF_INTEROP_BRIDGE_SOURCE_SHA256=$(sha256sum "$repo_root/scripts/amf_surface_probe.pyx" | awk '{print $1}')
FFMPEG_BIN=$install_dir/bin
FFMPEG_LIB=$install_dir/lib
EOF

printf 'FFmpeg SDK: %s\nPyAV wheel: %s\nAMF bridge: %s\nManifest: %s\n' \
  "$install_dir" "$copied_wheel" "$bridge" "$output_root/build-manifest.txt"
