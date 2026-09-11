#!/usr/bin/env bash
# R20: four long-sequence correctness stress tests for the C1 candidate.
# The test file is authoritative; this job only isolates the long cases and
# records enough provenance to make a later report table reproducible.
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=c1-r20-long

set -euo pipefail

R20_ROOT=${1:?experiment directory}
R20_CAND=${2:?candidate checkout}
R20_PY=${3:?assignment Python}

mkdir -p "$R20_ROOT"
OUT="$R20_ROOT/long_${SLURM_JOB_ID}"
mkdir -p "$OUT"

export PATH="/usr/local/cuda/bin:$(dirname "$R20_PY"):$PATH"
export PYTHONPATH="$R20_CAND:$R20_CAND/tests"

{
  date -Is
  printf 'hostname='; hostname
  printf 'job_id=%s\n' "${SLURM_JOB_ID:-unknown}"
  printf 'node=%s\n' "${SLURM_JOB_NODELIST:-unknown}"
  printf 'candidate=%s\n' "$R20_CAND"
  printf 'python=%s\n' "$R20_PY"
  printf 'python_version='; "$R20_PY" --version
  printf 'torch='; "$R20_PY" -c 'import torch; print(torch.__version__)'
  printf 'torch_cuda='; "$R20_PY" -c 'import torch; print(torch.version.cuda)'
  printf 'cuda_compiler='; nvcc --version | tail -1
  printf 'candidate_git='; git -C "$R20_CAND" rev-parse HEAD 2>/dev/null || true
  printf 'candidate_so_sha256='; find "$R20_CAND" -maxdepth 3 -type f -name '*.so' -print -quit | xargs -r sha256sum | awk '{print $1}'
} > "$OUT/environment.txt" 2>&1
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv > "$OUT/device.csv"
nvidia-smi -L > "$OUT/device.list"

cd "$R20_CAND"
set +e
"$R20_PY" -m pytest tests/test_fwd_full.py -k long -vv \
  --junitxml="$OUT/pytest.xml" > "$OUT/pytest.log" 2>&1
status=$?
set -e
printf 'pytest_exit_status=%s\n' "$status" > "$OUT/status.txt"
date -Is >> "$OUT/status.txt"
exit "$status"
