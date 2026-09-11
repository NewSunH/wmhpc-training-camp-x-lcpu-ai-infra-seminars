#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=c1-r20-mma
set -euo pipefail
R20_ROOT=${1:?experiment root}
R20_CUTLASS=${2:?pinned CUTLASS directory}
R20_PY=${3:?assignment Python}
OUT="$R20_ROOT/mma_${SLURM_JOB_ID}"
mkdir -p "$OUT"
export PATH="/usr/local/cuda/bin:$PATH"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv > "$OUT/device.csv"
nvcc --version > "$OUT/nvcc.txt"
sha256sum "$R20_ROOT/challenge/r20_mma_compare.cu" \
  "$R20_ROOT/challenge/tcgen05_sm103_gemm.cu" > "$OUT/source.sha256"
nvcc -O3 -DNDEBUG -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda \
  --default-stream per-thread -arch=sm_103a -Xptxas=-v \
  -I"$R20_CUTLASS/include" -I"$R20_CUTLASS/tools/util/include" \
  -I"$R20_CUTLASS/examples/cute/tutorial/blackwell" \
  "$R20_ROOT/challenge/r20_mma_compare.cu" -o "$OUT/mma_compare" \
  > "$OUT/build.log" 2>&1
"$OUT/mma_compare" > "$OUT/run.log" 2>&1
cuobjdump --dump-sass "$OUT/mma_compare" > "$OUT/sass.txt"
"$R20_PY" "$R20_ROOT/challenge/r20_algebra_probe.py" \
  --output "$OUT/algebra.json" > "$OUT/algebra.log" 2>&1
date -Is > "$OUT/completed.txt"
