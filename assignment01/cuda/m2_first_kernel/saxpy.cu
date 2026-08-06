#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include <stdexcept>
#include <iostream>

#define CUDA_CHECK(call)                                    \
    do                                                      \
    {                                                       \
        cudaError_t err_ = (call);                          \
        if ((err_) != (cudaSuccess))                        \
        {                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", \
                    cudaGetErrorName(err_),                 \
                    __FILE__,                               \
                    __LINE__,                               \
                    cudaGetErrorString(err_));              \
            exit(1);                                        \
        }                                                   \
    } while (0)

#define CUDA_CHECK_KERNEL()                  \
    do                                       \
    {                                        \
        CUDA_CHECK(cudaGetLastError());      \
        CUDA_CHECK(cudaDeviceSynchronize()); \
    } while (0)

struct GpuTimer
{
    cudaEvent_t start_, stop_;
    GpuTimer()
    {
        CUDA_CHECK(cudaEventCreate(&start_));
        CUDA_CHECK(cudaEventCreate(&stop_));
    }
    ~GpuTimer()
    {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    float stop_ms()
    {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return ms;
    }
};

__global__ void saxpy(const float *x,
                      float *y,
                      const float a,
                      size_t n)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)
        y[i] = a * x[i] + y[i];
}

int main(int argc, char *argv[])
{
    size_t n;
    if (argc != 2)
    {
        throw std::invalid_argument("ERROR: invalid argument.");
    }
    n = static_cast<size_t>(std::stoul(argv[1]));
    size_t bytes = n * sizeof(float);

    if (n == 0)
    {
        printf("SUM=0\n");
        return 0;
    }

    float *h_x = (float *)malloc(bytes);
    float *h_y = (float *)malloc(bytes);

    float a = 2.0f;
    for (int i = 0; i < n; i++)
    {
        h_x[i] = ((i % 2048) - 1024) * 0.5f;
        h_y[i] = (i % 1024) - 512;
    }

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    GpuTimer timer;
    timer.start();
    saxpy<<<blocks, threads>>>(d_x, d_y, a, n);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));
    float kernel_time = timer.stop_ms();

    double sum = 0.0;
    for (int i = 0; i < n; i++)
    {
        sum += h_y[i];
    }
    printf("SUM=%.0f, n=%zu\n, t=%.1f", sum, n, kernel_time);
    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);
}