#!/usr/bin/env bash
# Build the validated R19 configuration in the explicitly selected checkout.
# Usage: bash r19_build_optimized.sh /path/to/isolated/FlashKDA /path/to/python
# Requires B300 CUDA/PyTorch build dependencies and pinned CUTLASS 5c149f5.
set -euo pipefail
R19_CHECKOUT=${1:?path to the FlashKDA checkout to build}
R19_PYTHON=${2:?Python with CUDA-enabled PyTorch}
cd "$R19_CHECKOUT"
[[ -f setup.py && -f csrc/smxx/fwd_kernel2.cuh ]] || {
  echo "Expected a FlashKDA source checkout" >&2
  exit 2
}
# Retain the user's general build environment, but exclude older C1 ablations.
for r19_flag in ${!FLASH_KDA_C1_@}; do unset "$r19_flag"; done
export FLASH_KDA_CUDA_ARCHS=103a
export FLASH_KDA_VERSION_SUFFIX=+c1.r19.optimized
export FLASH_KDA_C1_REGISTER_STATE=1
export FLASH_KDA_C1_REGISTER_STATE_BLOCKS=8
export FLASH_KDA_C1_VALUE_MAJOR_STATE=1
export FLASH_KDA_C1_PRELOAD_REGISTER_STATE=1
export FLASH_KDA_C1_WARP_STATE_SYNC=1
export FLASH_KDA_C1_EARLY_OUTPUT=1
export MAX_JOBS=${MAX_JOBS:-2} NVCC_THREADS=${NVCC_THREADS:-4}
"$R19_PYTHON" setup.py build_ext --inplace --force
