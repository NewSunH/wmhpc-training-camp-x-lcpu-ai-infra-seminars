#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

__device__ __forceinline__ uint32_t mix(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// A producer writes the workspace needed by the dependent grid.  The work
// after the trigger is deliberately independent of the workspace and models
// K1 work that could, in principle, overlap K2's independent preamble.
__global__ void producer_kernel(uint32_t* workspace, uint32_t* tail,
                                int words, int tail_iters, int trigger) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < words; i += stride) {
        workspace[i] = mix(static_cast<uint32_t>(i) + 0x1234U);
    }
    __syncthreads();
    // Every producer thread fences its own stores before block 0 announces
    // readiness.  The PDL trigger itself is only a scheduling signal; it is
    // not a memory fence.
    __threadfence();
    if (threadIdx.x == 0 && trigger) {
        cudaTriggerProgrammaticLaunchCompletion();
    }
    // Keep every CTA alive after its trigger, so the experiment can expose
    // overlap if the scheduler has resources.  This data is not consumed by
    // the secondary kernel.
    uint32_t x = static_cast<uint32_t>(tid) ^ 0x9e3779b9U;
    for (int i = 0; i < tail_iters; ++i) {
        x = mix(x + static_cast<uint32_t>(i));
    }
    if (tid < gridDim.x * blockDim.x) tail[tid] = x;
}

// The consumer performs an independent preamble before waiting on the
// producer.  It then reads the producer's workspace and writes an exact
// integer checksum.  cudaGridDependencySynchronize is required by PDL.
__global__ void consumer_kernel(const uint32_t* workspace, uint32_t* output,
                                int words, int preamble_iters, int use_pdl) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t x = static_cast<uint32_t>(tid) ^ 0x31415926U;
    for (int i = 0; i < preamble_iters; ++i) {
        x = mix(x + static_cast<uint32_t>(i));
    }
    if (use_pdl) cudaGridDependencySynchronize();

    uint32_t acc = x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < words; i += stride) acc ^= workspace[i];
    atomicXor(output, acc);
}

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

struct Result {
    float ms = 0.0f;
    uint32_t output = 0;
};

Result run_once(int blocks, int threads, int words, int tail_iters,
                int preamble_iters, bool pdl, int warmup, int repeats) {
    uint32_t *d_ws = nullptr, *d_tail = nullptr, *d_out = nullptr;
    check(cudaMalloc(&d_ws, static_cast<size_t>(words) * sizeof(uint32_t)),
          "cudaMalloc workspace");
    check(cudaMalloc(&d_tail, static_cast<size_t>(blocks * threads) * sizeof(uint32_t)),
          "cudaMalloc tail");
    check(cudaMalloc(&d_out, sizeof(uint32_t)), "cudaMalloc output");
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
          "cudaStreamCreate");

    cudaLaunchAttribute attr{};
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = pdl ? 1 : 0;
    cudaLaunchConfig_t config{};
    config.gridDim = dim3(blocks);
    config.blockDim = dim3(threads);
    config.stream = stream;
    config.attrs = &attr;
    config.numAttrs = 1;

    auto enqueue = [&]() {
        check(cudaMemsetAsync(d_out, 0, sizeof(uint32_t), stream), "memset");
        producer_kernel<<<blocks, threads, 0, stream>>>(
            d_ws, d_tail, words, tail_iters, pdl ? 1 : 0);
        check(cudaGetLastError(), "producer launch");
        if (pdl) {
            check(cudaLaunchKernelEx(&config, consumer_kernel, d_ws, d_out,
                                     words, preamble_iters, 1),
                  "PDL consumer launch");
        } else {
            consumer_kernel<<<blocks, threads, 0, stream>>>(
                d_ws, d_out, words, preamble_iters, 0);
            check(cudaGetLastError(), "serialized consumer launch");
        }
    };

    for (int i = 0; i < warmup; ++i) {
        enqueue();
        check(cudaStreamSynchronize(stream), "warmup sync");
    }
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop), "event create stop");
    std::vector<float> samples;
    samples.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        check(cudaEventRecord(start, stream), "event record start");
        enqueue();
        check(cudaEventRecord(stop, stream), "event record stop");
        check(cudaEventSynchronize(stop), "event sync stop");
        float ms = 0.0f;
        check(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
        samples.push_back(ms);
    }
    uint32_t output = 0;
    check(cudaMemcpy(&output, d_out, sizeof(output), cudaMemcpyDeviceToHost),
          "copy output");
    std::sort(samples.begin(), samples.end());
    Result result;
    result.ms = samples[samples.size() / 2];
    result.output = output;
    check(cudaEventDestroy(start), "event destroy start");
    check(cudaEventDestroy(stop), "event destroy stop");
    check(cudaStreamDestroy(stream), "stream destroy");
    check(cudaFree(d_ws), "free workspace");
    check(cudaFree(d_tail), "free tail");
    check(cudaFree(d_out), "free output");
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    int blocks = argc > 1 ? std::atoi(argv[1]) : 96;
    int threads = argc > 2 ? std::atoi(argv[2]) : 128;
    int words = argc > 3 ? std::atoi(argv[3]) : (1 << 20);
    int tail_iters = argc > 4 ? std::atoi(argv[4]) : 2000;
    int preamble_iters = argc > 5 ? std::atoi(argv[5]) : 1000;
    int warmup = argc > 6 ? std::atoi(argv[6]) : 5;
    int repeats = argc > 7 ? std::atoi(argv[7]) : 15;

    cudaDeviceProp prop{};
    int device = 0;
    check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
    std::printf("device=%s cc=%d.%d blocks=%d threads=%d words=%d tail=%d preamble=%d\n",
                prop.name, prop.major, prop.minor, blocks, threads, words,
                tail_iters, preamble_iters);
    Result serial = run_once(blocks, threads, words, tail_iters, preamble_iters,
                             false, warmup, repeats);
    Result pdl = run_once(blocks, threads, words, tail_iters, preamble_iters,
                          true, warmup, repeats);
    std::printf("serialized_ms=%.6f pdl_ms=%.6f speedup=%.3f%% serial_out=0x%08x pdl_out=0x%08x exact=%s\n",
                serial.ms, pdl.ms, 100.0f * (serial.ms - pdl.ms) / serial.ms,
                serial.output, pdl.output, serial.output == pdl.output ? "yes" : "no");
    std::printf("{\"device\":\"%s\",\"cc\":\"%d.%d\",\"blocks\":%d,\"threads\":%d,"
                "\"words\":%d,\"tail_iters\":%d,\"preamble_iters\":%d,"
                "\"serialized_ms\":%.6f,\"pdl_ms\":%.6f,\"speedup_pct\":%.6f,"
                "\"serial_out\":\"0x%08x\",\"pdl_out\":\"0x%08x\",\"exact\":%s}\n",
                prop.name, prop.major, prop.minor, blocks, threads, words,
                tail_iters, preamble_iters, serial.ms, pdl.ms,
                100.0f * (serial.ms - pdl.ms) / serial.ms,
                serial.output, pdl.output,
                serial.output == pdl.output ? "true" : "false");
    return serial.output == pdl.output ? 0 : 3;
}
