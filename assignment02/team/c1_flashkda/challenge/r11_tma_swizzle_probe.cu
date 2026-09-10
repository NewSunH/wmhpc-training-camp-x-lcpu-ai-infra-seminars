#include <cuda_runtime.h>
#include <cutlass/bfloat16.h>
#include <cute/tensor.hpp>
#include <cute/algorithm/copy.hpp>
#include <cute/arch/copy_sm90_desc.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace cute;
using BF16 = cutlass::bfloat16_t;

// R11-B microkernel candidate.  TMA requires the first tile mode to be the
// unit-stride (major) global-memory dimension.  K2's output is therefore
// represented as [D, CHUNK] = [128, 16] for this probe; the actual tensor
// remains the usual [token, value] row-major output.
using PreSwizzleLayout = Layout<
    Shape<Shape<_32, _4>, _16>,
    Stride<Stride<_1, _512>, _32>>;
using SwizzledOutputLayout = ComposedLayout<
    Swizzle<3, 4, 3>,
    smem_ptr_flag_bits<sizeof_bits<BF16>::value>,
    PreSwizzleLayout>;
using TmaSwizzledOutputLayout = decltype(composition(
    SwizzledOutputLayout{}.layout_a(),
    SwizzledOutputLayout{}.offset(),
    prepend(SwizzledOutputLayout{}.layout_b())));
using TmaPlainOutputLayout = decltype(prepend(PreSwizzleLayout{}));

template <class TmaStore>
__global__ void r11_tma_store_probe(CUTE_GRID_CONSTANT TmaStore const tma_store, BF16* out) {
  extern __shared__ __align__(128) BF16 smem[];
  constexpr int kRows = 16;
  constexpr int kCols = 128;
  auto s = make_tensor(make_smem_ptr(smem), TmaPlainOutputLayout{});
  auto s_logical = make_tensor(make_smem_ptr(smem), PreSwizzleLayout{});
  // Reconstruct the descriptor-owned gmem tensor.  TMA copy traits require
  // the descriptor's coordinate-aware gmem pointer rather than a raw pointer.
  auto g_full = tma_store.get_tma_tensor(
      make_shape(Int<1>{}, Int<kCols>{}, Int<kRows>{}));

  // Fill through the logical (row,col) coordinates.  The unique bit pattern
  // makes a TMA layout permutation visible without arithmetic or rounding.
  for (int linear = int(threadIdx.x); linear < kRows * kCols;
       linear += int(blockDim.x)) {
    int row = linear / kCols;
    int col = linear % kCols;
    s_logical(col, row) = BF16::bitcast(uint16_t(1 + linear));
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    auto cta = tma_store.get_slice(Int<0>{});
    cute::copy(tma_store, cta.partition_S(s), cta.partition_D(g_full));
    tma_store_arrive();
    tma_store_wait<0>();
  }
}

static void check_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
    std::exit(2);
  }
}

int main() {
  constexpr int kRows = 16;
  constexpr int kCols = 128;
  using GmemLayout = Layout<Shape<_1, _128, _16>, Stride<_128, _1, _128>>;
  print("logical smem: "); print(SwizzledOutputLayout{}); print("\n");
  print("tma smem: "); print(TmaSwizzledOutputLayout{}); print("\n");
  BF16* d_out = nullptr;
  check_cuda(cudaMalloc(&d_out, kRows * kCols * sizeof(BF16)), "cudaMalloc");
  check_cuda(cudaMemset(d_out, 0, kRows * kCols * sizeof(BF16)), "cudaMemset");

  auto g = make_tensor(make_gmem_ptr(d_out), GmemLayout{});
  auto tma_store = make_tma_copy(SM90_TMA_STORE{}, g, TmaPlainOutputLayout{});
  constexpr size_t smem_bytes = cosize_v<TmaSwizzledOutputLayout> * sizeof(BF16);
  r11_tma_store_probe<<<1, 128, smem_bytes>>>(tma_store, d_out);
  check_cuda(cudaGetLastError(), "launch");
  check_cuda(cudaDeviceSynchronize(), "sync");

  std::vector<BF16> h_out(kRows * kCols);
  check_cuda(cudaMemcpy(h_out.data(), d_out,
                        h_out.size() * sizeof(BF16), cudaMemcpyDeviceToHost),
             "copy");
  int mismatches = 0;
  for (int i = 0; i < kRows * kCols; ++i) {
    uint16_t expected = uint16_t(1 + i);
    if (h_out[i].storage != expected) ++mismatches;
  }
  std::printf("R11-B swizzled TMA store: mismatches=%d total=%d cosize=%zu bytes=%zu\n",
              mismatches, kRows * kCols, size_t(cosize_v<SwizzledOutputLayout>),
              smem_bytes);
  std::printf("row0:");
  for (int col = 0; col < 16; ++col) std::printf(" %04x", h_out[col].storage);
  std::printf("\n");
  cudaFree(d_out);
  return mismatches == 0 ? 0 : 1;
}
