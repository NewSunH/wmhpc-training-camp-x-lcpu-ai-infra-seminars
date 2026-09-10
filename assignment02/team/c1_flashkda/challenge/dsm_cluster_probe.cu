// R17 DSM/cluster feasibility probe for the B300 V-split idea.
//
// The two kernels model the data movement that a value-split K2 would need:
// each of two CTAs consumes one half of a [tile, D] input.  `duplicate` has
// both CTAs load the complete read-only tile into their own shared memory;
// `dsm` has CTA 0 load once and CTA 1 read CTA 0's shared memory through the
// cluster DSM mapping.  This is deliberately independent of FlashKDA's
// TMA/barrier ABI: it answers whether the proposed sharing primitive itself
// has a useful latency/bandwidth regime before an intrusive K2 rewrite.

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cutlass/cluster_launch.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace cg = cooperative_groups;

__device__ __forceinline__ unsigned cluster_rank(cg::cluster_group const& c) {
  return c.block_rank();
}

template <bool DSM>
__global__
void dsm_vsplit_probe(float const* __restrict__ in,
                      float* __restrict__ out,
                      int elems,
                      int iters) {
  extern __shared__ float smem[];
  auto cluster = cg::this_cluster();
  const unsigned rank = cluster_rank(cluster);
  const int cluster_id = static_cast<int>(blockIdx.x / 2);
  float accum = 0.0f;

  for (int rep = 0; rep < iters; ++rep) {
    // The duplicate path represents two ordinary V-split CTAs.  The DSM
    // path represents a producer CTA and a consumer CTA sharing one tile.
    if constexpr (!DSM) {
      for (int i = static_cast<int>(threadIdx.x); i < elems; i += blockDim.x) {
        smem[i] = in[cluster_id * elems + i];
      }
    } else if (rank == 0) {
      for (int i = static_cast<int>(threadIdx.x); i < elems; i += blockDim.x) {
        smem[i] = in[cluster_id * elems + i];
      }
    }
    __syncthreads();
    cluster.sync();

    float const* tile = smem;
    if constexpr (DSM) {
      if (rank != 0) tile = cluster.map_shared_rank(smem, 0);
    }
    // Each CTA consumes a disjoint half, as in R3--R11 V-split.  The access
    // loop is intentionally simple; the experiment isolates DSM movement,
    // not MMA throughput.
    int begin = rank * (elems / 2);
    int end = (rank + 1) * (elems / 2);
    for (int i = begin + static_cast<int>(threadIdx.x); i < end; i += blockDim.x) {
      accum += tile[i];
    }
    cluster.sync();
  }

  smem[threadIdx.x] = accum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) smem[threadIdx.x] += smem[threadIdx.x + stride];
    __syncthreads();
  }

  if (threadIdx.x == 0) out[cluster_id * 2 + static_cast<int>(rank)] = smem[0];
}

static void check(cudaError_t status, char const* what) {
  if (status != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(status) << std::endl;
    std::exit(2);
  }
}

struct Result {
  float duplicate_ms;
  float dsm_ms;
  bool duplicate_ok;
  bool dsm_ok;
};

static float run(bool dsm, float const* in, float* out, int elems, int clusters,
                 int iters, int warmup, int repeats, bool& ok) {
  dim3 grid(static_cast<unsigned>(clusters * 2), 1, 1);
  dim3 block(128, 1, 1);
  size_t smem_bytes = static_cast<size_t>(elems) * sizeof(float);
  auto launch = [&]() {
    cudaLaunchAttribute cluster_attr{};
    cluster_attr.id = cudaLaunchAttributeClusterDimension;
    cluster_attr.val.clusterDim.x = 2;
    cluster_attr.val.clusterDim.y = 1;
    cluster_attr.val.clusterDim.z = 1;
    cudaLaunchConfig_t config{};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_bytes;
    config.stream = 0;
    config.attrs = &cluster_attr;
    config.numAttrs = 1;
    if (dsm) {
      check(cudaFuncSetAttribute(
                reinterpret_cast<const void*>(dsm_vsplit_probe<true>),
                cudaFuncAttributeNonPortableClusterSizeAllowed, 1),
            "cluster size attribute");
      check(cudaFuncSetAttribute(
                reinterpret_cast<const void*>(dsm_vsplit_probe<true>),
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes)),
            "max dynamic shared memory attribute");
      void* args[] = {&in, &out, &elems, &iters};
      check(cudaLaunchKernelExC(&config, reinterpret_cast<const void*>(dsm_vsplit_probe<true>), args),
            "DSM kernel launch");
    } else {
      check(cudaFuncSetAttribute(
                reinterpret_cast<const void*>(dsm_vsplit_probe<false>),
                cudaFuncAttributeNonPortableClusterSizeAllowed, 1),
            "cluster size attribute");
      check(cudaFuncSetAttribute(
                reinterpret_cast<const void*>(dsm_vsplit_probe<false>),
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_bytes)),
            "max dynamic shared memory attribute");
      void* args[] = {&in, &out, &elems, &iters};
      check(cudaLaunchKernelExC(&config, reinterpret_cast<const void*>(dsm_vsplit_probe<false>), args),
            "duplicate kernel launch");
    }
    check(cudaGetLastError(), "kernel launch");
  };
  std::fprintf(stderr, "run %s elems=%d: warmup launch\n", dsm ? "dsm" : "dup", elems);
  for (int i = 0; i < warmup; ++i) launch();
  check(cudaDeviceSynchronize(), "warmup synchronize");
  std::fprintf(stderr, "run %s elems=%d: warmup done\n", dsm ? "dsm" : "dup", elems);
  std::vector<float> h(static_cast<size_t>(clusters) * 2);
  ok = true;
  cudaEvent_t start{}, stop{};
  check(cudaEventCreate(&start), "event create");
  check(cudaEventCreate(&stop), "event create");
  std::vector<float> samples;
  samples.reserve(repeats);
  for (int r = 0; r < repeats; ++r) {
    std::fprintf(stderr, "run %s elems=%d: sample %d\n", dsm ? "dsm" : "dup", elems, r);
    check(cudaEventRecord(start), "event record");
    launch();
    check(cudaEventRecord(stop), "event record");
    check(cudaEventSynchronize(stop), "event synchronize");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
    samples.push_back(ms / static_cast<float>(iters));
  }
  check(cudaMemcpy(h.data(), out, h.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy output");
  for (int c = 0; c < clusters; ++c) {
    float expected = 0.0f;
    for (int i = 0; i < elems; ++i) {
      size_t idx = static_cast<size_t>(c) * elems + i;
      expected += static_cast<float>(iters) * (0.25f + static_cast<float>(idx % 17) * 0.03125f);
    }
    float got = h[2 * c] + h[2 * c + 1];
    if (std::abs(got - expected) > std::max(1.0f, std::abs(expected)) * 1e-4f) ok = false;
  }
  std::sort(samples.begin(), samples.end());
  float median = samples[samples.size() / 2];
  check(cudaEventDestroy(start), "event destroy");
  check(cudaEventDestroy(stop), "event destroy");
  return median;
}

int main(int argc, char** argv) {
  int clusters = argc > 1 ? std::atoi(argv[1]) : 32;
  int iters = argc > 2 ? std::atoi(argv[2]) : 200;
  int repeats = argc > 3 ? std::atoi(argv[3]) : 7;
  std::vector<int> sizes = {1024, 2048, 4096, 8192, 16384};
  int max_elems = *std::max_element(sizes.begin(), sizes.end());
  float* d_in = nullptr;
  float* d_out = nullptr;
  check(cudaMalloc(&d_in, static_cast<size_t>(clusters * max_elems) * sizeof(float)), "malloc input");
  check(cudaMalloc(&d_out, static_cast<size_t>(clusters * 2) * sizeof(float)), "malloc output");
  std::vector<float> h_in(static_cast<size_t>(clusters * max_elems));
  for (size_t i = 0; i < h_in.size(); ++i) h_in[i] = 0.25f + static_cast<float>(i % 17) * 0.03125f;
  check(cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice), "copy input");
  std::cout << "clusters=" << clusters << " iters=" << iters << " repeats=" << repeats << " block=128 cluster=(2,1,1)\n";
  std::cout << "elems,duplicate_ms,dsm_ms,speedup,duplicate_exact,dsm_exact\n";
  for (int elems : sizes) {
    bool dup_ok = false, dsm_ok = false;
    float dup = run(false, d_in, d_out, elems, clusters, iters, 2, repeats, dup_ok);
    float dsm = run(true, d_in, d_out, elems, clusters, iters, 2, repeats, dsm_ok);
    std::cout << elems << "," << dup << "," << dsm << "," << (dup / dsm)
              << "," << dup_ok << "," << dsm_ok << "\n";
  }
  check(cudaFree(d_in), "free input");
  check(cudaFree(d_out), "free output");
}
