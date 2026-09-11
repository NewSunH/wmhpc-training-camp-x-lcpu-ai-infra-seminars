#!/usr/bin/env bash
# R20 boundary benchmark and precision job.  Each benchmark invocation is a
# fresh Python process so that baseline/candidate extension modules cannot
# share an import or allocator state.  The four processes are ABBA paired.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=c1-r20-bench

set -euo pipefail

R20_ROOT=${1:?experiment directory}
R20_BASE=${2:?baseline checkout}
R20_CAND=${3:?candidate checkout}
R20_PY=${4:?assignment Python}
R20_CHALLENGE=${5:?challenge scripts directory}
R20_REFERENCE=${6:?vendored FP32 reference}

mkdir -p "$R20_ROOT"
OUT="$R20_ROOT/bench_${SLURM_JOB_ID}"
mkdir -p "$OUT"
export PATH="/usr/local/cuda/bin:$(dirname "$R20_PY"):$PATH"

{
  date -Is
  printf 'hostname='; hostname
  printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
  printf 'node=%s\n' "${SLURM_JOB_NODELIST:-unknown}"
  printf 'python=%s\n' "$R20_PY"
  printf 'python_version='; "$R20_PY" --version
  printf 'torch='; "$R20_PY" -c 'import torch; print(torch.__version__)'
  printf 'torch_cuda='; "$R20_PY" -c 'import torch; print(torch.version.cuda)'
  printf 'cuda_compiler='; nvcc --version | tail -1
  printf 'baseline_so_sha256='; find "$R20_BASE" -maxdepth 3 -type f -name '*.so' -print -quit | xargs -r sha256sum | awk '{print $1}'
  printf 'candidate_so_sha256='; find "$R20_CAND" -maxdepth 3 -type f -name '*.so' -print -quit | xargs -r sha256sum | awk '{print $1}'
  printf 'reference_sha256='; sha256sum "$R20_REFERENCE" | awk '{print $1}'
} > "$OUT/environment.txt" 2>&1
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$OUT/device.csv"
nvidia-smi -L > "$OUT/device.list"

run_probe() {
  local label=$1 checkout=$2 output=$3 exact=${4:-}
  for name in ${!FLASH_KDA_C1_@}; do unset "$name"; done
  cd "$checkout"
  export PYTHONPATH="$checkout:$checkout/tests:$R20_CHALLENGE"
  echo "starting $label -> $output" | tee "$OUT/${label}_${output##*/}.log"
  if [[ "$exact" == exact ]]; then
    "$R20_PY" "$R20_CHALLENGE/r20_probe.py" --label "$label" --native-exact \
      --json "$output" >> "$OUT/${label}_${output##*/}.log" 2>&1
  else
    "$R20_PY" "$R20_CHALLENGE/r20_probe.py" --label "$label" \
      --json "$output" >> "$OUT/${label}_${output##*/}.log" 2>&1
  fi
}

run_probe baseline "$R20_BASE" "$OUT/0_baseline.json"
run_probe candidate "$R20_CAND" "$OUT/1_candidate.json" exact
run_probe candidate "$R20_CAND" "$OUT/2_candidate.json"
run_probe baseline "$R20_BASE" "$OUT/3_baseline.json"

cd "$R20_CAND"
export PYTHONPATH="$R20_CAND:$R20_CAND/tests:$R20_CHALLENGE"
for name in ${!FLASH_KDA_C1_@}; do unset "$name"; done
"$R20_PY" "$R20_CHALLENGE/r20_summarize.py" \
  "$OUT/0_baseline.json" "$OUT/1_candidate.json" \
  "$OUT/2_candidate.json" "$OUT/3_baseline.json" \
  --output "$OUT/summary.json" > "$OUT/summary.log" 2>&1

"$R20_PY" "$R20_CHALLENGE/r20_precision.py" \
  --reference "$R20_REFERENCE" --json "$OUT/precision.json" \
  > "$OUT/precision.log" 2>&1

printf 'completed_at=%s\n' "$(date -Is)" > "$OUT/status.txt"
