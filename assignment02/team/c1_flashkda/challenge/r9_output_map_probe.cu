#include <cuda_runtime.h>
#include <cutlass/bfloat16.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include "../FlashKDA/csrc/smxx/fwd_kernel2.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace cute;
using BF16 = cutlass::bfloat16_t;
using OutputLayout = typename K2Layouts<128, 16>::ValueSliceVOLayout;

constexpr int kRows = 16;
constexpr int kCols = 64;
constexpr int kTailRows = 7;
constexpr int kWarps = 2;
constexpr int kBlocksPerWarp = 2;

// The shared-memory STSM path is the oracle.  Each register fragment element
// receives a unique raw BF16 bit pattern, so the comparison is about layout,
// not arithmetic or BF16 rounding.
template <class OutputLayoutT>
__global__ void r9_map_probe(BF16* direct_out, BF16* explicit_out,
                             BF16* explicit_tail_out, uint16_t* oracle_out) {
  extern __shared__ __align__(128) BF16 smem[];

  auto mma = make_tiled_mma(
      MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
      Layout<Shape<_1, _1>>{}, Tile<_16, _16, _16>{});
  const int lane_id = int(threadIdx.x) & 31;
  const int warp_id = int(threadIdx.x) >> 5;
  auto thr_mma = mma.get_slice(lane_id);

  Tensor out_tile = make_tensor(make_smem_ptr(smem), OutputLayoutT{});
  Tensor c_ref = local_tile(out_tile, make_shape(Int<16>{}, Int<16>{}),
                            make_coord(0, 0));
  auto tCrC_ref = thr_mma.partition_C(c_ref);
  auto store = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, BF16>{}, mma);
  auto thr_store = store.get_slice(lane_id);

  auto global_layout = make_layout(make_shape(Int<16>{}, Int<64>{}),
                                    make_stride(Int<64>{}, Int<1>{}));
  Tensor g_out = make_tensor(make_gmem_ptr(direct_out), global_layout);
  Tensor g_explicit = make_tensor(make_gmem_ptr(explicit_out), global_layout);
  Tensor g_explicit_tail = make_tensor(make_gmem_ptr(explicit_tail_out), global_layout);

  if (warp_id < kWarps) {
    #pragma unroll
    for (int block = 0; block < kBlocksPerWarp; ++block) {
      auto frag = make_fragment_like<BF16>(thr_mma.make_fragment_C(tCrC_ref));
      #pragma unroll
      for (int j = 0; j < size(frag); ++j) {
        // All values are raw BF16 bit patterns.  They are never used in an
        // arithmetic instruction, only copied through the layout under test.
        uint16_t code = uint16_t(1 + warp_id * 2048 + block * 1024 + lane_id * 32 + j);
        frag(j) = BF16::bitcast(code);
      }

      Tensor s_block = local_tile(out_tile,
          make_shape(Int<16>{}, Int<16>{}),
          make_coord(0, warp_id * 2 + block));
      copy(store, thr_store.retile_S(frag), thr_store.partition_D(s_block));

      // Candidate A: use the same C-copy source/destination map with a
      // row-major global tile.  This is the family of mappings that failed
      // in R8; R9 measures it against the shared-memory oracle without any
      // application data or recurrence state.
      Tensor g_block = local_tile(g_out,
          make_shape(Int<16>{}, Int<16>{}),
          make_coord(0, warp_id * 2 + block));
      auto src = thr_store.retile_S(frag);
      auto dst = thr_store.partition_D(g_block);
      #pragma unroll
      for (int j = 0; j < size(src); ++j) {
        auto src_coord = idx2crd(j, shape(src));
        auto dst_coord = idx2crd(j, shape(dst));
        dst(dst_coord) = src(src_coord);

        // Candidate B: explicit inverse of the observed K_INTER map.  The
        // fragment has eight values per lane.  Lanes are grouped in fours;
        // fragment pairs select the row half and the 8-column sub-group.
        const int row = (lane_id / 4) + (((j / 2) & 1) * 8);
        const int local_col = ((j / 4) * 8) + ((lane_id & 3) * 2) + (j & 1);
        const int col = (warp_id * kBlocksPerWarp + block) * 16 + local_col;
        g_explicit(row, col) = src(src_coord);
        if (row < kTailRows) g_explicit_tail(row, col) = src(src_coord);
      }
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    #pragma unroll
    for (int row = 0; row < kRows; ++row) {
      #pragma unroll
      for (int col = 0; col < kCols; ++col) {
        oracle_out[row * kCols + col] = out_tile(row, col).storage;
      }
    }
  }
}

static void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    std::exit(2);
  }
}

int main() {
  BF16* d_direct = nullptr;
  BF16* d_explicit = nullptr;
  BF16* d_explicit_tail = nullptr;
  uint16_t* d_oracle = nullptr;
  check_cuda(cudaMalloc(&d_direct, kRows * kCols * sizeof(BF16)), "cudaMalloc direct");
  check_cuda(cudaMalloc(&d_explicit, kRows * kCols * sizeof(BF16)), "cudaMalloc explicit");
  check_cuda(cudaMalloc(&d_explicit_tail, kRows * kCols * sizeof(BF16)), "cudaMalloc explicit tail");
  check_cuda(cudaMalloc(&d_oracle, kRows * kCols * sizeof(uint16_t)), "cudaMalloc oracle");
  check_cuda(cudaMemset(d_direct, 0, kRows * kCols * sizeof(BF16)), "memset direct");
  check_cuda(cudaMemset(d_explicit, 0, kRows * kCols * sizeof(BF16)), "memset explicit");
  check_cuda(cudaMemset(d_explicit_tail, 0, kRows * kCols * sizeof(BF16)), "memset explicit tail");
  check_cuda(cudaMemset(d_oracle, 0, kRows * kCols * sizeof(uint16_t)), "memset oracle");

  constexpr size_t smem_bytes = cosize_v<OutputLayout> * sizeof(BF16);
  r9_map_probe<OutputLayout><<<1, kWarps * 32, smem_bytes>>>(
      d_direct, d_explicit, d_explicit_tail, d_oracle);
  check_cuda(cudaGetLastError(), "r9_map_probe launch");
  check_cuda(cudaDeviceSynchronize(), "r9_map_probe sync");

  std::vector<BF16> h_direct(kRows * kCols);
  std::vector<BF16> h_explicit(kRows * kCols);
  std::vector<BF16> h_explicit_tail(kRows * kCols);
  std::vector<uint16_t> h_oracle(kRows * kCols);
  check_cuda(cudaMemcpy(h_direct.data(), d_direct,
                        h_direct.size() * sizeof(BF16), cudaMemcpyDeviceToHost),
             "copy direct");
  check_cuda(cudaMemcpy(h_explicit.data(), d_explicit,
                        h_explicit.size() * sizeof(BF16), cudaMemcpyDeviceToHost),
             "copy explicit");
  check_cuda(cudaMemcpy(h_explicit_tail.data(), d_explicit_tail,
                        h_explicit_tail.size() * sizeof(BF16), cudaMemcpyDeviceToHost),
             "copy explicit tail");
  check_cuda(cudaMemcpy(h_oracle.data(), d_oracle,
                        h_oracle.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost),
             "copy oracle");

  int mismatches = 0;
  int unwritten = 0;
  int explicit_mismatches = 0;
  int explicit_unwritten = 0;
  int tail_mismatches = 0;
  int tail_outside_writes = 0;
  for (int i = 0; i < kRows * kCols; ++i) {
    uint16_t got = h_direct[i].storage;
    if (got == 0) ++unwritten;
    if (got != h_oracle[i]) ++mismatches;
    uint16_t explicit_code = h_explicit[i].storage;
    if (explicit_code == 0) ++explicit_unwritten;
    if (explicit_code != h_oracle[i]) ++explicit_mismatches;
    uint16_t tail_code = h_explicit_tail[i].storage;
    if (i < kTailRows * kCols) {
      if (tail_code != h_oracle[i]) ++tail_mismatches;
    } else if (tail_code != 0) {
      ++tail_outside_writes;
    }
  }
  std::printf("R9 shared-oracle/direct-map comparison: mismatches=%d unwritten=%d total=%d\n",
              mismatches, unwritten, kRows * kCols);
  std::printf("R9 explicit-inverse-map comparison: mismatches=%d unwritten=%d total=%d\n",
              explicit_mismatches, explicit_unwritten, kRows * kCols);
  std::printf("R9 explicit-map tail rows=%d: mismatches=%d outside_writes=%d\n",
              kTailRows, tail_mismatches, tail_outside_writes);
  std::printf("oracle row 0:");
  for (int col = 0; col < kCols; ++col) std::printf(" %04x", h_oracle[col]);
  std::printf("\ndirect row 0:");
  for (int col = 0; col < kCols; ++col) std::printf(" %04x", h_direct[col].storage);
  std::printf("\n");
  std::printf("explicit row 0:");
  for (int col = 0; col < kCols; ++col) std::printf(" %04x", h_explicit[col].storage);
  std::printf("\n");
  for (int row = 1; row < kRows; ++row) {
    std::printf("oracle row %d:", row);
    for (int col = 0; col < kCols; ++col)
      std::printf(" %04x", h_oracle[row * kCols + col]);
    std::printf("\n");
  }

  cudaFree(d_direct);
  cudaFree(d_explicit);
  cudaFree(d_explicit_tail);
  cudaFree(d_oracle);
  // A non-zero mismatch count is the expected finding for the naive map; the
  // diagnostic itself completed successfully, so return zero for Slurm.
  return 0;
}
