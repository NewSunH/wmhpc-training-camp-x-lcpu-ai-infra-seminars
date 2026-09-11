#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --job-name=c1-r19
set -euo pipefail
R19_ROOT=${1:?isolated experiment directory}
R19_PY=${2:?assignment Python}
R19_CUTLASS=${3:?pinned CUTLASS directory}
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$(dirname "$R19_PY"):$PATH"
export NVCC_THREADS=4 MAX_JOBS=2 FLASH_KDA_CUDA_ARCHS=103a
mkdir -p "$R19_ROOT/results"
cd "$R19_ROOT"
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv
"$R19_PY" -c 'import torch; print(torch.__version__, torch.version.cuda)'
# Never inherit prior R1--R18 experimental flags.
for name in ${!FLASH_KDA_C1_@}; do unset "$name"; done
[[ -e src/cutlass ]] || ln -s "$R19_CUTLASS" src/cutlass
[[ -e FlashKDA ]] || ln -s src FlashKDA
nvcc -std=c++17 -O3 -arch=sm_103a --expt-relaxed-constexpr --expt-extended-lambda \
  -I"$R19_CUTLASS/include" -I"$R19_CUTLASS/tools/util/include" \
  challenge/r19_state_register_map.cu -o state_register_map \
  > results/map_build.log 2>&1
./state_register_map | tee results/map.json
for variant in baseline reg8; do
  if [[ -e "$R19_ROOT/$variant" ]]; then
    echo "Refusing to reuse existing variant checkout: $R19_ROOT/$variant" >&2
    exit 2
  fi
  cp -a src "$variant"
  cd "$R19_ROOT/$variant"
  export FLASH_KDA_C1_REGISTER_STATE=0 FLASH_KDA_C1_REGISTER_STATE_BLOCKS=8
  if [[ $variant == reg8 ]]; then export FLASH_KDA_C1_REGISTER_STATE=1; fi
  export FLASH_KDA_VERSION_SUFFIX="+r19.$variant"
  "$R19_PY" setup.py build_ext --inplace > "$R19_ROOT/results/build_$variant.log" 2>&1
  export PYTHONPATH="$PWD:$PWD/tests"
  "$R19_PY" "$R19_ROOT/challenge/r19_probe.py" --label "$variant" \
    --skip-bench --json "$R19_ROOT/results/exact_$variant.json" \
    > "$R19_ROOT/results/exact_$variant.log" 2>&1
  echo "built and exact: $variant"
  cd "$R19_ROOT"
done
idx=0
for variant in baseline reg8 reg8 baseline; do
  cd "$R19_ROOT/$variant"
  export PYTHONPATH="$PWD:$PWD/tests"
  export FLASH_KDA_C1_REGISTER_STATE=0
  if [[ $variant == reg8 ]]; then export FLASH_KDA_C1_REGISTER_STATE=1; fi
  "$R19_PY" "$R19_ROOT/challenge/r19_probe.py" --label "$variant" \
    --skip-exact --quick --warmup 20 --iters 50 --repeats 3 \
    --json "$R19_ROOT/results/pair_${idx}_${variant}.json" \
    > "$R19_ROOT/results/pair_${idx}_${variant}.log" 2>&1
  echo "paired benchmark complete: $idx $variant"
  idx=$((idx + 1))
done
