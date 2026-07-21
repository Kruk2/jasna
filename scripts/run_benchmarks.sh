#!/usr/bin/env bash
# Full release benchmark suite. Run between releases and archive the CSVs in
# benchmarks/ as <date>_v<version>_<topic>.csv (see benchmarks/*.md).
#
# Usage: scripts/run_benchmarks.sh <output-dir>
set -eu
PY="${PYTHON:-$HOME/.virtualenvs/jasna-linux/bin/python}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?usage: run_benchmarks.sh <output-dir>}"
mkdir -p "$OUT"
VR="$ROOT/assets/benchmark/vr1_8k_vr_hevc_8bit_60fps.mp4"
cd "$ROOT"

echo "### E2E (all clips, vali)"
"$PY" -u scripts/benchmark_decode_backends.py --csv "$OUT/e2e.csv"
echo "### E2E VR (8K, vali)"
"$PY" -u scripts/benchmark_decode_backends.py --clips "$VR" --no-warmup --csv "$OUT/e2e_vr.csv"
echo "### SCAN (all clips, vali)"
"$PY" -u scripts/benchmark_scan_backends.py --csv "$OUT/scan.csv"
echo "### SCAN VR (8K, vali)"
"$PY" -u scripts/benchmark_scan_backends.py --clips "$VR" --no-warmup --csv "$OUT/scan_vr.csv"
echo "### ALL BENCHMARKS DONE"
