#!/usr/bin/env bash
# Rebuild the Linux AMD colour kernels as committed HIP code objects.
#
# Usage: scripts/build_hip_code_objects.sh [architecture] [kernel-name ...]
#   scripts/build_hip_code_objects.sh
#   scripts/build_hip_code_objects.sh gfx1100 yuv_to_rgb
set -euo pipefail

architecture="${1:-gfx1100}"
if [ "$#" -gt 0 ]; then
    shift
fi

media_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/jasna/media"
if [ "$#" -gt 0 ]; then
    sources=()
    for name in "$@"; do
        sources+=("$media_dir/${name%.cu}.cu")
    done
else
    sources=(
        "$media_dir/yuv_to_rgb.cu"
        "$media_dir/rgb_to_yuv.cu"
    )
fi

for source in "${sources[@]}"; do
    destination="${source%.cu}.${architecture}.hsaco"
    echo "hipcc $(basename "$source") -> $(basename "$destination")"
    hipcc --genco --no-gpu-bundle-output --offload-arch="$architecture" -O3 -std=c++17 \
        -include hip/hip_runtime.h "$source" -o "$destination"
done

echo
for source in "${sources[@]}"; do
    destination="${source%.cu}.${architecture}.hsaco"
    printf '%-36s %8s bytes\n' \
        "$(basename "$destination")" \
        "$(stat -c%s "$destination")"
done
