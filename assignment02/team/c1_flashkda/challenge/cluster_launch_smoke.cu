#include <cuda_runtime.h>
#include <cutlass/cluster_launch.hpp>
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>

__global__ void smoke(float* p) {
  if (threadIdx.x == 0) p[blockIdx.x] = float(blockIdx.x + 1);
}

namespace cg = cooperative_groups;
__global__ void group_smoke(float* p) {
  auto c = cg::this_cluster();
  unsigned rank = c.block_rank();
  c.sync();
  if (threadIdx.x == 0) p[blockIdx.x] = float(rank + 1);
}

int main() {
  float* p = nullptr;
  cudaError_t st = cudaMalloc(&p, 2 * sizeof(float));
  if (st != cudaSuccess) { std::fprintf(stderr, "malloc %s\n", cudaGetErrorString(st)); return 2; }
  st = cudaFuncSetAttribute((const void*)smoke, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  std::fprintf(stderr, "attr=%s\n", cudaGetErrorString(st));
  auto conf = cutlass::ClusterLauncher::make_cluster_launch_config({2,1,1}, {2,1,1}, {128,1,1}, 0, 0, false);
  void* args[] = {&p};
  std::fprintf(stderr, "before launch\n");
  st = cudaLaunchKernelExC(&conf.launch_config, (const void*)smoke, args);
  std::fprintf(stderr, "launch=%s\n", cudaGetErrorString(st));
  st = cudaDeviceSynchronize();
  std::fprintf(stderr, "sync=%s\n", cudaGetErrorString(st));
  st = cudaMemset(p, 0, 2 * sizeof(float));
  void* args2[] = {&p};
  st = cudaLaunchKernelExC(&conf.launch_config, (const void*)group_smoke, args2);
  std::fprintf(stderr, "group launch=%s\n", cudaGetErrorString(st));
  st = cudaDeviceSynchronize();
  std::fprintf(stderr, "group sync=%s\n", cudaGetErrorString(st));
  cudaFree(p);
}
