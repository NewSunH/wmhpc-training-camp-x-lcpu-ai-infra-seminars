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
using L = K2Layouts<128, 16>;
using SLayout = typename L::VOLayout;
using TmaLayout = typename L::TMAVOLayout;

template <class TmaStore>
__global__ void probe(CUTE_GRID_CONSTANT TmaStore const tma_store) {
  extern __shared__ __align__(128) BF16 smem[];
  auto s = make_tensor(make_smem_ptr(smem), TmaLayout{});
  auto s_logical = make_tensor(make_smem_ptr(smem), SLayout{});
  for (int linear = int(threadIdx.x); linear < 16 * 128; linear += int(blockDim.x)) {
    int row = linear / 128;
    int col = linear % 128;
    s_logical(row, col) = BF16::bitcast(uint16_t(1 + linear));
  }
  __syncthreads();
  tma_store_fence();
  __syncthreads();
  if (threadIdx.x == 0) {
    auto g = tma_store.get_tma_tensor(make_shape(Int<1>{}, Int<16>{}, Int<128>{}));
    auto cta = tma_store.get_slice(Int<0>{});
    cute::copy(tma_store, cta.partition_S(s), cta.partition_D(g));
    tma_store_arrive();
    tma_store_wait<0>();
  }
}

static void ck(cudaError_t e, const char* w) { if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); } }

int main() {
  constexpr int T = 16, D = 128;
  using G = Layout<Shape<_1,_16,_128>,Stride<_128,_128,_1>>;
  BF16* out = nullptr;
  ck(cudaMalloc(&out, T * D * sizeof(BF16)), "malloc");
  ck(cudaMemset(out, 0, T * D * sizeof(BF16)), "memset");
  auto g = make_tensor(make_gmem_ptr(out), G{});
  auto tma = make_tma_copy(SM90_TMA_STORE{}, g, TmaLayout{});
  constexpr size_t bytes = cosize_v<SLayout> * sizeof(BF16);
  probe<<<1,128,bytes>>>(tma);
  ck(cudaGetLastError(), "launch");
  ck(cudaDeviceSynchronize(), "sync");
  std::vector<BF16> h(T*D);
  ck(cudaMemcpy(h.data(), out, h.size()*sizeof(BF16), cudaMemcpyDeviceToHost), "copy");
  int bad = 0;
  for (int i=0;i<T*D;++i) if (h[i].storage != uint16_t(1+i)) ++bad;
  std::printf("R11-B baseline TMA: mismatches=%d total=%d smem_bytes=%zu\n", bad, T*D, bytes);
  cudaFree(out);
  return bad == 0 ? 0 : 1;
}
