#!/usr/bin/env bash
# Rebuild every jasna/media/*.cu into its committed AMD .hipfb offload bundle.
#
# Usage: scripts/build_hipfbs.sh [kernel-name ...]
#   scripts/build_hipfbs.sh              # all kernels
#   scripts/build_hipfbs.sh cas          # just jasna/media/cas.cu
#
# Needs ROCm's hipcc (HIPCC=, default /opt/rocm/bin/hipcc) and a Python with
# Torch (PYTHON=, default python3) to read the targets from
# accelerator.HIP_KERNEL_ARCHITECTURES; other GPUs fall back to the Torch path.
# The sources are the CUDA ones: hip_runtime.h is force-included and
# cuda_fp16.h resolves to hip_fp16.h.
set -euo pipefail

HIPCC="${HIPCC:-/opt/rocm/bin/hipcc}"
BUNDLER="$(dirname "$HIPCC")/../llvm/bin/clang-offload-bundler"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MEDIA_DIR="$REPO_DIR/jasna/media"
ARCH_LIST="$(cd "$REPO_DIR" && "${PYTHON:-python3}" -c \
    'from jasna.accelerator import HIP_KERNEL_ARCHITECTURES; print(*sorted(HIP_KERNEL_ARCHITECTURES))')"
read -r -a ARCHS <<< "$ARCH_LIST"

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
