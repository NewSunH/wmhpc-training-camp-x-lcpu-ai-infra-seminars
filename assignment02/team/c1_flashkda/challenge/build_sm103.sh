#!/usr/bin/env bash
set -euo pipefail

# Build the CUTLASS CuTe tcgen05 example for B300/SM103a.
# Usage:
#   CUTLASS_ROOT=/path/to/cutlass ./build_sm103.sh [M N K]

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cutlass_root=${CUTLASS_ROOT:?set CUTLASS_ROOT to the pinned CUTLASS checkout}
arch=${CUDA_ARCH:-sm_103a}
out=${OUT:-"${script_dir}/tcgen05_sm103"}

nvcc -std=c++17 -O3 -arch="${arch}" --expt-relaxed-constexpr \
  -I"${cutlass_root}/include" \
  -I"${cutlass_root}/tools/util/include" \
  -I"${cutlass_root}/examples/common" \
  -I"${script_dir}" \
  -o "${out}" "${script_dir}/tcgen05_sm103_gemm.cu"

"${out}" "${1:-128}" "${2:-256}" "${3:-64}"
