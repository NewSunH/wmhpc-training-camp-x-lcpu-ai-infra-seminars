// Matched BF16 128x128x16 GEMM probe for C1 question 2.
// Reuse the pinned CUTLASS-derived tcgen05 tutorial; its BSD license is retained
// in tcgen05_sm103_gemm.cu. This is a standalone GEMM, not FlashKDA recurrence.
// Build: nvcc -O3 -DNDEBUG -std=c++17 --expt-relaxed-constexpr --expt-extended-lambda
//   -arch=sm_103a -I<CUTLASS>/include -I<CUTLASS>/tools/util/include
//   -I<CUTLASS>/examples/cute/tutorial/blackwell r20_mma_compare.cu -o <binary>
// Also pass --default-stream per-thread for graph capture.
#include "example_utils.hpp"
// The tutorial's debug check synchronizes the device, which is illegal during
// graph capture. Keep launch-error checks and synchronize explicitly outside it.
#undef CUTE_CHECK_LAST
#define CUTE_CHECK_LAST() CUTE_CHECK_ERROR(cudaGetLastError())
#define C1_TCGEN05_QUIET
#define C1_TCGEN05_STATE
#define main c1_old_tutorial_main
#include "tcgen05_sm103_gemm.cu"
#undef main
#include <cute/arch/mma_sm80.hpp>
#include <cutlass/bfloat16.h>
#include <algorithm>
#include <cmath>
#include <vector>

using BF16 = cutlass::bfloat16_t;

__global__ void sm80_state_gemm(const BF16* a, const BF16* b, float* d) {
  __shared__ __align__(128) BF16 sa[128 * 16];
  __shared__ __align__(128) BF16 sb[128 * 16];
  auto layout = make_layout(make_shape(_128{}, _16{}), make_stride(_16{}, _1{}));
  auto sA = make_tensor(make_smem_ptr(sa), layout);
  auto sB = make_tensor(make_smem_ptr(sb), layout);
  auto gA = make_tensor(make_gmem_ptr(a + blockIdx.x * 128 * 16), layout);
  auto gB = make_tensor(make_gmem_ptr(b), layout);
  cooperative_copy<128>(threadIdx.x, gA, sA);
  cooperative_copy<128>(threadIdx.x, gB, sB);
  __syncthreads();
  auto mma = make_tiled_mma(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                             Layout<Shape<_4, _1>>{}, Tile<_128, _128, _16>{});
  auto thr = mma.get_slice(threadIdx.x);
  auto rA = thr.partition_fragment_A(sA);
  auto rB = thr.partition_fragment_B(sB);
  copy(thr.partition_A(sA), rA);
  copy(thr.partition_B(sB), rB);
  auto gD = make_tensor(make_gmem_ptr(d + blockIdx.x * 128 * 128),
                       make_layout(make_shape(_128{}, _128{}), make_stride(_128{}, _1{})));
  auto rC = thr.make_fragment_C(thr.partition_C(gD));
  clear(rC);
  gemm(mma, rA, rB, rC);
  copy(rC, thr.partition_C(gD));
}

template <class Launch>
cudaGraphExec_t capture(Launch launch, cudaStream_t stream) {
  // Initialize attributes and lazy runtime state outside stream capture.
  launch();
  CUTE_CHECK_ERROR(cudaDeviceSynchronize());
  cudaGraph_t graph;
  CUTE_CHECK_ERROR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  // The tcgen05 tutorial uses the default stream. Capture on a per-thread
  // default stream via --default-stream per-thread at compilation.
  for (int i = 0; i < 100; ++i) launch();
  CUTE_CHECK_ERROR(cudaStreamEndCapture(stream, &graph));
  size_t nodes = 0;
  CUTE_CHECK_ERROR(cudaGraphGetNodes(graph, nullptr, &nodes));
  if (nodes != 100) {
    fprintf(stderr, "Expected 100 captured kernels, got %zu\n", nodes);
    std::exit(3);
  }
  cudaGraphExec_t executable;
  CUTE_CHECK_ERROR(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  CUTE_CHECK_ERROR(cudaGraphDestroy(graph));
  return executable;
}

float time_graph(cudaGraphExec_t graph, cudaStream_t stream) {
  for (int i = 0; i < 5; ++i) CUTE_CHECK_ERROR(cudaGraphLaunch(graph, stream));
  CUTE_CHECK_ERROR(cudaStreamSynchronize(stream));
  cudaEvent_t start, stop;
  CUTE_CHECK_ERROR(cudaEventCreate(&start));
  CUTE_CHECK_ERROR(cudaEventCreate(&stop));
  CUTE_CHECK_ERROR(cudaEventRecord(start, stream));
  for (int i = 0; i < 20; ++i) CUTE_CHECK_ERROR(cudaGraphLaunch(graph, stream));
  CUTE_CHECK_ERROR(cudaEventRecord(stop, stream));
  CUTE_CHECK_ERROR(cudaEventSynchronize(stop));
  float ms;
  CUTE_CHECK_ERROR(cudaEventElapsedTime(&ms, start, stop));
  CUTE_CHECK_ERROR(cudaEventDestroy(start));
  CUTE_CHECK_ERROR(cudaEventDestroy(stop));
  return ms * 1000.f / (100 * 20);
}

int main() {
  cudaDeviceProp props;
  CUTE_CHECK_ERROR(cudaGetDeviceProperties(&props, 0));
  printf("device=%s capability=%d.%d\n", props.name, props.major, props.minor);
  printf("BF16xBF16->FP32; tile=128x128x16; common B across CTAs; "
         "100 nodes/graph; 5 warmup graphs; 20 timed graphs/slot; ABBA\n");
  for (int blocks : {1, 12, 96}) {
    int m = 128 * blocks;
    std::vector<BF16> ha(m * 16), hb(128 * 16);
    for (int i = 0; i < int(ha.size()); ++i) ha[i] = BF16(float((i * 17 + 3) % 31 - 15) / 16.f);
    for (int i = 0; i < int(hb.size()); ++i) hb[i] = BF16(float((i * 13 + 5) % 29 - 14) / 16.f);
    BF16 *a, *b;
    float *c, *d80, *d100;
    CUTE_CHECK_ERROR(cudaMalloc(&a, ha.size() * sizeof(BF16)));
    CUTE_CHECK_ERROR(cudaMalloc(&b, hb.size() * sizeof(BF16)));
    CUTE_CHECK_ERROR(cudaMalloc(&c, m * 128 * sizeof(float)));
    CUTE_CHECK_ERROR(cudaMalloc(&d80, m * 128 * sizeof(float)));
    CUTE_CHECK_ERROR(cudaMalloc(&d100, m * 128 * sizeof(float)));
    CUTE_CHECK_ERROR(cudaMemcpy(a, ha.data(), ha.size() * sizeof(BF16), cudaMemcpyHostToDevice));
    CUTE_CHECK_ERROR(cudaMemcpy(b, hb.data(), hb.size() * sizeof(BF16), cudaMemcpyHostToDevice));
    CUTE_CHECK_ERROR(cudaMemset(c, 0, m * 128 * sizeof(float)));
    auto la = make_layout(make_shape(m, 16), make_stride(16, _1{}));
    auto lb = make_layout(make_shape(128, 16), make_stride(16, _1{}));
    auto lc = make_layout(make_shape(m, 128), make_stride(128, _1{}));
    auto launch80 = [&]() { sm80_state_gemm<<<blocks, 128>>>(a, b, d80); CUTE_CHECK_LAST(); };
    auto launch100 = [&]() {
      gemm_host_f16xf16_f32_f32_tnt(a, la, b, lb, c, lc, d100, lc, _1{}, _0{});
    };
    launch80(); launch100();
    CUTE_CHECK_ERROR(cudaDeviceSynchronize());
    std::vector<float> x(m * 128), y(m * 128);
    CUTE_CHECK_ERROR(cudaMemcpy(x.data(), d80, x.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUTE_CHECK_ERROR(cudaMemcpy(y.data(), d100, y.size() * sizeof(float), cudaMemcpyDeviceToHost));
    int bad80 = 0, bad100 = 0;
    for (int i = 0; i < m; ++i) for (int j = 0; j < 128; ++j) {
      float gold = 0;
      for (int k = 0; k < 16; ++k) gold += float(ha[i * 16 + k]) * float(hb[j * 16 + k]);
      bad80 += x[i * 128 + j] != gold;
      bad100 += y[i * 128 + j] != gold;
    }
    printf("blocks=%d elements=%d mismatch_sm80=%d mismatch_tcgen05=%d\n", blocks, m * 128, bad80, bad100);
    if (bad80 || bad100) return 2;
    auto stream = cudaStreamPerThread;
    auto g80 = capture(launch80, stream), g100 = capture(launch100, stream);
    const float a0 = time_graph(g80, stream), b0 = time_graph(g100, stream);
    const float b1 = time_graph(g100, stream), a1 = time_graph(g80, stream);
    printf("blocks=%d sm80_us=[%.6f,%.6f] tcgen05_us=[%.6f,%.6f] tcgen05_latency_reduction_pct=%.4f\n",
           blocks, a0, a1, b0, b1, 100.f * (1.f - (b0+b1)/(a0+a1)));
    CUTE_CHECK_ERROR(cudaGraphExecDestroy(g80)); CUTE_CHECK_ERROR(cudaGraphExecDestroy(g100));
    CUTE_CHECK_ERROR(cudaFree(a)); CUTE_CHECK_ERROR(cudaFree(b)); CUTE_CHECK_ERROR(cudaFree(c));
    CUTE_CHECK_ERROR(cudaFree(d80)); CUTE_CHECK_ERROR(cudaFree(d100));
  }
}
