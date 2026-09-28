#!/usr/bin/env bash
# Rebuild every jasna/media/*.cu into its committed .fatbin.
#
# Usage: scripts/build_fatbins.sh [kernel-name ...]
#   scripts/build_fatbins.sh              # all kernels
#   scripts/build_fatbins.sh cas          # just jasna/media/cas.cu
#
# CUDA 13 rejects host compilers newer than GCC 15, hence CCBIN. PTX is
# embedded for compute_75 only, so architectures newer than the SASS list
# still load through the driver's JIT.
set -euo pipefail

CCBIN="${CCBIN:-g++-15}"
MEDIA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/jasna/media"

GENCODE=(-gencode "arch=compute_75,code=[compute_75,sm_75]")
for arch in 80 86 87 88 89 90 100 103 110 120 121; do
    GENCODE+=(-gencode "arch=compute_${arch},code=sm_${arch}")
done

if [ "$#" -gt 0 ]; then
    sources=()
    for name in "$@"; do
        sources+=("$MEDIA_DIR/${name%.cu}.cu")
    done
else
    sources=("$MEDIA_DIR"/*.cu)
fi

for source in "${sources[@]}"; do
    fatbin="${source%.cu}.fatbin"
    echo "nvcc $(basename "$source") -> $(basename "$fatbin")"
    nvcc -ccbin "$CCBIN" -std=c++17 -O3 -fatbin "${GENCODE[@]}" \
        -o "$fatbin" "$source"
done

echo
for source in "${sources[@]}"; do
    fatbin="${source%.cu}.fatbin"
    printf '%-24s %8s bytes  %s\n' \
        "$(basename "$fatbin")" \
        "$(stat -c%s "$fatbin")" \
        "$(cuobjdump "$fatbin" | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' ')"
done
