#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../data"
CURRENT_CSV="$DATA_DIR/all_benchmark_data.csv"
ORIGINAL_CSV="$DATA_DIR/all_benchmark_data.original.csv"
TILEGYM_CSV="$DATA_DIR/all_benchmark_data_tilegym.csv"

restore_original_csv() {
    if [[ -f "$ORIGINAL_CSV" ]]; then
        mv -f "$ORIGINAL_CSV" "$CURRENT_CSV"
    fi
}

if [[ -f "$ORIGINAL_CSV" ]]; then
    echo "Refusing to continue: backup already exists at $ORIGINAL_CSV" >&2
    exit 1
fi

trap restore_original_csv EXIT

cd "$SCRIPT_DIR"

if [[ -f "$CURRENT_CSV" ]]; then
    mv "$CURRENT_CSV" "$ORIGINAL_CSV"
fi

sudo nvidia-smi -i 0 -lgc 1650
python -c "from cuda.tile._compile import _get_compiler_version_string; print(_get_compiler_version_string())"
python benchmark_cross_entropy.py
python benchmark_jsd.py
python benchmark_layer_norm.py
python benchmark_geglu.py
python benchmark_fused_linear_jsd.py

if [[ ! -f "$CURRENT_CSV" ]]; then
    echo "Benchmark did not generate $CURRENT_CSV" >&2
    exit 1
fi

mv -f "$CURRENT_CSV" "$TILEGYM_CSV"
restore_original_csv
trap - EXIT