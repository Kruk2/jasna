#!/usr/bin/env bash
# Rebuild every jasna/media/*.cu into its committed AMD .hipfb offload bundle.
#
# Usage: scripts/build_hipfbs.sh [kernel-name ...]
#   scripts/build_hipfbs.sh              # all kernels
#   scripts/build_hipfbs.sh cas          # just jasna/media/cas.cu
#
# Needs ROCm's hipcc (HIPCC=, default /opt/rocm/bin/hipcc). The sources are the
# CUDA ones: hip_runtime.h is force-included and cuda_fp16.h resolves to
# hip_fp16.h. ARCHS must match cuda_kernel.HIP_KERNEL_ARCHITECTURES; other
# GPUs fall back to the Torch path.
set -euo pipefail

HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
BUNDLER="$(dirname "$HIPCC")/../llvm/bin/clang-offload-bundler"
MEDIA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/jasna/media"
ARCHS=(gfx1030 gfx1100 gfx1101 gfx1102 gfx1103 gfx1150 gfx1151 gfx1200 gfx1201 gfx908 gfx90a gfx942 gfx950)

OFFLOAD=()
for arch in "${ARCHS[@]}"; do
    OFFLOAD+=("--offload-arch=$arch")
done

SHIM_DIR="$(mktemp -d)"
trap 'rm -rf "$SHIM_DIR"' EXIT
echo '#include <hip/hip_fp16.h>' > "$SHIM_DIR/cuda_fp16.h"

if [ "$#" -gt 0 ]; then
    sources=()
    for name in "$@"; do
        sources+=("$MEDIA_DIR/${name%.cu}.cu")
    done
else
    sources=("$MEDIA_DIR"/*.cu)
fi

for source in "${sources[@]}"; do
    bundle="${source%.cu}.hipfb"
    echo "hipcc $(basename "$source") -> $(basename "$bundle")"
    "$HIPCC" --genco "${OFFLOAD[@]}" -O3 -std=c++17 -fuse-cuid=none \
        -include hip/hip_runtime.h -I "$SHIM_DIR" -o "$bundle" "$source"
done

echo
for source in "${sources[@]}"; do
    bundle="${source%.cu}.hipfb"
    printf '%-24s %8s bytes  %s\n' \
        "$(basename "$bundle")" \
        "$(stat -c%s "$bundle")" \
        "$("$BUNDLER" --list --type=o --input="$bundle" | grep -oE 'gfx[0-9a-f]+' | sort | tr '\n' ' ')"
done
