#!/usr/bin/env bash
# Release benchmark suite. Needs an exclusive GPU — anything else on the card
# makes the numbers incomparable. Archive the CSVs in benchmarks/ as
# <date>_<topic>.csv next to a short .md write-up (see benchmarks/*.md).
#
# Usage: scripts/run_benchmarks.sh <output-dir> [--codecs] [--scan]
#
#   default    one input per resolution (H.264 8-bit) + the 8K VR clip, ~8 min
#   --codecs   also the HEVC 10-bit and AV1 encodings, ~7 min more. Only worth
#              it when decode, encode or pixel-format code changed: H.264 is
#              nearly flat across decode backends while those two spread ~19%.
#   --scan     also the GUI every-frame mosaic scan (detection only), ~5 min
set -eu
PY="${PYTHON:-$HOME/.virtualenvs/jasna-linux/bin/python}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?usage: run_benchmarks.sh <output-dir> [--codecs] [--scan]}"
shift
mkdir -p "$OUT"
cd "$ROOT"

CLIPS="$ROOT/assets/benchmark"
SUITE=(
    "$CLIPS/SONE-610_bench_720p_h264_8bit.mp4"
    "$CLIPS/SONE-610_bench_1080p_h264_8bit.mp4"
    "$CLIPS/SONE-610_bench_2160p_h264_8bit.mp4"
    "$CLIPS/vr1_8k_vr_hevc_8bit_60fps.mp4"
)
WITH_SCAN=0
for arg in "$@"; do
    case "$arg" in
        --codecs)
            SUITE+=(
                "$CLIPS/SONE-610_bench_1080p_hevc_10bit.mp4"
                "$CLIPS/SONE-610_bench_2160p_hevc_10bit.mp4"
                "$CLIPS/SONE-610_bench_2160p_av1_10bit.mp4"
            )
            ;;
        --scan) WITH_SCAN=1 ;;
        *) echo "unknown option: $arg" >&2; exit 1 ;;
    esac
done

echo "### E2E (${#SUITE[@]} clips, vali, medians of 3)"
"$PY" -u scripts/benchmark_decode_backends.py \
    --clips "${SUITE[@]}" --backends vali --repeats 3 --csv "$OUT/e2e.csv"

if [[ "$WITH_SCAN" == 1 ]]; then
    echo "### SCAN (${#SUITE[@]} clips, vali)"
    "$PY" -u scripts/benchmark_scan_backends.py \
        --clips "${SUITE[@]}" --backends vali --csv "$OUT/scan.csv"
fi

echo "### ALL BENCHMARKS DONE -> $OUT"
