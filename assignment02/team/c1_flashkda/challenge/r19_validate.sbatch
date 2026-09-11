#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=c1-r19-final
set -euo pipefail
R19_ROOT=${1:?experiment directory}
R19_BASE=${2:?baseline checkout}
R19_CAND=${3:?candidate checkout}
R19_PY=${4:?assignment Python}
export PATH="/usr/local/cuda/bin:$(dirname "$R19_PY"):$PATH"
OUT="$R19_ROOT/validation_${SLURM_JOB_ID}"
mkdir -p "$OUT"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "$OUT/device.csv"
cd "$R19_CAND"
export PYTHONPATH="$PWD:$PWD/tests"
# Existing reference tests across all H/state/tail/varlen/batch combinations.
# Long-sequence cases (131072 or 1048576 tokens) are a separate stress workload;
# the full T8192 matrix
# and long-tail sequence shapes in the normal suite run here.
"$R19_PY" -m pytest tests/test_fwd_full.py -q -x -k 'not long' \
  --junitxml="$OUT/pytest.xml" > "$OUT/pytest.log" 2>&1
for name in ${!FLASH_KDA_C1_@}; do unset "$name"; done
idx=0
for which in baseline candidate candidate baseline; do
  if [[ $which == baseline ]]; then cd "$R19_BASE"; else cd "$R19_CAND"; fi
  export PYTHONPATH="$PWD:$PWD/tests"
  "$R19_PY" "$R19_ROOT/challenge/r19_probe.py" --label "$which" \
    --skip-exact --warmup 30 --iters 100 --repeats 5 \
    --json "$OUT/${idx}_${which}.json" > "$OUT/${idx}_${which}.log" 2>&1
  echo "validated benchmark: $idx $which"
  idx=$((idx + 1))
done
"$R19_PY" "$R19_ROOT/challenge/r19_summarize.py" \
  "$OUT/0_baseline.json" "$OUT/1_candidate.json" \
  "$OUT/2_candidate.json" "$OUT/3_baseline.json" --output "$OUT/summary.json"
# Optional profiling is diagnostic; event measurements above are authoritative.
if command -v ncu >/dev/null; then
  for which in baseline candidate; do
    if [[ $which == baseline ]]; then cd "$R19_BASE"; else cd "$R19_CAND"; fi
    export PYTHONPATH="$PWD:$PWD/tests:$R19_ROOT/challenge"
    ncu --set basic --clock-control none --kernel-name-base function \
      --kernel-name 'regex:_flash_kda_fwd_recurrence' --launch-count 1 --csv \
      "$R19_PY" "$R19_ROOT/challenge/r19_profile.py" \
      > "$OUT/ncu_${which}.csv" 2> "$OUT/ncu_${which}.err" || echo "NCU failed for $which; see log"
  done
fi
