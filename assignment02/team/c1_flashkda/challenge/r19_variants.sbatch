#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=c1-r19b
set -euo pipefail
R19_ROOT=${1:?isolated experiment directory}
R19_BASE=${2:?previous baseline checkout}
R19_PY=${3:?assignment Python}
R19_CUTLASS=${4:?pinned CUTLASS directory}
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$(dirname "$R19_PY"):$PATH"
export NVCC_THREADS=4 MAX_JOBS=2 FLASH_KDA_CUDA_ARCHS=103a
mkdir -p "$R19_ROOT/results"
cd "$R19_ROOT"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
for name in ${!FLASH_KDA_C1_@}; do unset "$name"; done
[[ -e src/cutlass ]] || ln -s "$R19_CUTLASS" src/cutlass
[[ -e FlashKDA ]] || ln -s src FlashKDA
nvcc -std=c++17 -O3 -arch=sm_103a --expt-relaxed-constexpr --expt-extended-lambda \
  -I"$R19_CUTLASS/include" -I"$R19_CUTLASS/tools/util/include" \
  challenge/r19_value_major_probe.cu -o value_major_probe \
  > results/value_major_build.log 2>&1
./value_major_probe | tee results/value_major_map.json
run_bench() {
  local path=$1 label=$2 tag=$3
  cd "$path"
  export PYTHONPATH="$PWD:$PWD/tests"
  "$R19_PY" "$R19_ROOT/challenge/r19_probe.py" --label "$label" \
    --skip-exact --quick --warmup 20 --iters 50 --repeats 3 \
    --json "$R19_ROOT/results/${tag}_${label}.json" \
    > "$R19_ROOT/results/${tag}_${label}.log" 2>&1
}
for variant in reg5 early vm8 vm5; do
  for name in ${!FLASH_KDA_C1_@}; do unset "$name"; done
  case "$variant" in
    reg5) export FLASH_KDA_C1_REGISTER_STATE=1 FLASH_KDA_C1_REGISTER_STATE_BLOCKS=5;;
    early) export FLASH_KDA_C1_EARLY_OUTPUT=1;;
    vm8) export FLASH_KDA_C1_REGISTER_STATE=1 FLASH_KDA_C1_REGISTER_STATE_BLOCKS=8 FLASH_KDA_C1_VALUE_MAJOR_STATE=1;;
    vm5) export FLASH_KDA_C1_REGISTER_STATE=1 FLASH_KDA_C1_REGISTER_STATE_BLOCKS=5 FLASH_KDA_C1_VALUE_MAJOR_STATE=1;;
  esac
  cd "$R19_ROOT"
  if [[ -e "$R19_ROOT/$variant" ]]; then
    echo "Refusing to reuse existing variant checkout: $R19_ROOT/$variant" >&2
    exit 2
  fi
  cp -a src "$variant"
  cd "$R19_ROOT/$variant"
  export FLASH_KDA_VERSION_SUFFIX="+r19.$variant"
  "$R19_PY" setup.py build_ext --inplace > "$R19_ROOT/results/build_$variant.log" 2>&1
  export PYTHONPATH="$PWD:$PWD/tests"
  if ! "$R19_PY" "$R19_ROOT/challenge/r19_probe.py" --label "$variant" \
    --skip-bench --json "$R19_ROOT/results/exact_$variant.json" \
    > "$R19_ROOT/results/exact_$variant.log" 2>&1; then
    echo "exactness failed: $variant"
    continue
  fi
  echo "built and exact: $variant"
  run_bench "$R19_BASE" baseline "${variant}_0"
  run_bench "$R19_ROOT/$variant" "$variant" "${variant}_1"
  run_bench "$R19_ROOT/$variant" "$variant" "${variant}_2"
  run_bench "$R19_BASE" baseline "${variant}_3"
  echo "ABBA complete: $variant"
done
