// Verify the R19 state C-fragment -> projection B-fragment conversion
// against the actual STSM_T/LDSM_N shared-memory path, without arithmetic.
#include "../FlashKDA/csrc/smxx/fwd_kernel1.cuh"
#include "../FlashKDA/csrc/smxx/fwd_kernel2.cuh"
#include <cstdio>
#include <cstdlib>

using BF16 = cutlass::bfloat16_t;
using L = K2Layouts<128, 16>;

__global__ void state_register_map(unsigned* errors) {
    __shared__ __align__(128) BF16 state[128 * 128];
    auto s = make_tensor(make_smem_ptr(state), L::StateSmemLayout{});
    auto st = make_tensor(make_smem_ptr(state), L::TransposedStateSmemLayout{});
    auto mma = make_tiled_mma(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                             Layout<Shape<_1, _1>>{}, Tile<_16, _16, _16>{});
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    auto thr = mma.get_slice(lane);
    auto store = make_tiled_copy_C(Copy_Atom<SM90_U16x8_STSM_T, BF16>{}, mma);
    auto load_c = make_tiled_copy_C(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
    auto load_b = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto ts = store.get_slice(lane);
    auto tc = load_c.get_slice(lane);
    auto tb = load_b.get_slice(lane);
    unsigned mismatches_c = 0, mismatches_b = 0;
    for (int k = 0; k < 8; ++k) {
        for (int v = 0; v < 2; ++v) {
            auto c_tile = local_tile(st, make_shape(_16{}, _16{}), make_coord(k, warp * 2 + v));
            auto b_tile = local_tile(s, make_shape(_16{}, _16{}), make_coord(warp * 2 + v, k));
            auto c = make_fragment_like<BF16>(thr.make_fragment_C(thr.partition_C(c_tile)));
            #pragma unroll
            for (int j = 0; j < size(c); ++j) {
                // Unique finite raw bits across all 16,384 state elements.
                c(j) = BF16::bitcast(1 + (((k * 8 + warp * 2 + v) * 32 + lane) * 8 + j));
            }
            copy(store, ts.retile_S(c), ts.partition_D(c_tile));
            __syncwarp();
            auto restored = make_fragment_like(c);
            copy(load_c, tc.partition_S(c_tile), tc.retile_D(restored));
            auto b = make_fragment_like<BF16>(thr.partition_fragment_B(b_tile));
            copy(load_b, tb.partition_S(b_tile), tb.retile_D(b));
            uint32_t const* src = reinterpret_cast<uint32_t const*>(&c(0));
            uint32_t const* oracle = reinterpret_cast<uint32_t const*>(&b(0));
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                uint32_t transposed;
                SM75_U32x1_MOVM_T::copy(src[j], transposed);
                mismatches_b += transposed != oracle[j];
            }
            #pragma unroll
            for (int j = 0; j < size(c); ++j) {
                mismatches_c += c(j).storage != restored(j).storage;
            }
        }
    }
    atomicAdd(errors, mismatches_c);
    atomicAdd(errors + 1, mismatches_b);
}

static void check(cudaError_t code) {
    if (code != cudaSuccess) {
        fprintf(stderr, "%s\n", cudaGetErrorString(code));
        std::exit(2);
    }
}

int main() {
    unsigned *device_errors, errors[2];
    check(cudaMalloc(&device_errors, sizeof(errors)));
    check(cudaMemset(device_errors, 0, sizeof(errors)));
    state_register_map<<<1, 128>>>(device_errors);
    check(cudaGetLastError());
    check(cudaMemcpy(errors, device_errors, sizeof(errors), cudaMemcpyDeviceToHost));
    check(cudaFree(device_errors));
    printf("{\"state_elements\":16384,\"c_mismatches\":%u,\"b_packed_mismatches\":%u}\n",
           errors[0], errors[1]);
    return errors[0] || errors[1] ? 1 : 0;
}
