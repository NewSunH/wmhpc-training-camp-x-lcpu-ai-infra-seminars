#!/usr/bin/env bash
# Run baseline/candidate in ABBA order inside one existing Slurm allocation.
#
# Usage:
#   bash r19_pair_bench.sh BASE_PY CANDIDATE_PY PROBE OUT_DIR
#
# BASE_PY and CANDIDATE_PY are absolute interpreters from isolated virtualenvs
# containing their corresponding FlashKDA extension.  The probe is executed
# as a fresh process each time, so Python cannot retain the other extension.
# The shell script itself does not submit Slurm work; invoke it under one
# `srun --gres=gpu:1 ...` allocation.

set -euo pipefail

BASE_PY=${1:?baseline python}
CANDIDATE_PY=${2:?candidate python}
PROBE=${3:?r19_probe.py path}
OUT_DIR=${4:?output directory}

mkdir -p "$OUT_DIR"
COMMON_ARGS=(--warmup "${R19_WARMUP:-10}" --iters "${R19_ITERS:-30}" \
  --repeats "${R19_REPEATS:-3}")

run_one() {
  local label=$1 py=$2 idx=$3
  local module_root
  module_root=$("$py" -c 'import pathlib, flash_kda; print(pathlib.Path(flash_kda.__file__).resolve().parent.parent)')
  local out="$OUT_DIR/${idx}_${label}.json"
  echo "=== R19 ${idx} ${label} ==="
  env PYTHONPATH="$module_root:$module_root/tests${PYTHONPATH:+:$PYTHONPATH}" \
    "$py" "$PROBE" --label "$label" --json "$out" "${COMMON_ARGS[@]}"
}

# ABBA reduces monotonic thermal/clock drift without mixing extension modules.
run_one baseline "$BASE_PY" 00
run_one candidate "$CANDIDATE_PY" 01
run_one candidate "$CANDIDATE_PY" 02
run_one baseline "$BASE_PY" 03

"$BASE_PY" "$(dirname "$PROBE")/r19_summarize.py" \
  "$OUT_DIR/00_baseline.json" "$OUT_DIR/01_candidate.json" \
  "$OUT_DIR/02_candidate.json" "$OUT_DIR/03_baseline.json" \
  --output "$OUT_DIR/summary.json"
