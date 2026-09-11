// R19 value-major state: exact register-map and paired-arithmetic probe.
// Build from challenge/ with CUDA 12.9+ (B300: sm_103a):
// nvcc -std=c++17 -O3 --use_fast_math --expt-relaxed-constexpr \
//   --expt-extended-lambda -arch=sm_103a \
//   -I../FlashKDA/cutlass/include \
//   -I../FlashKDA/cutlass/examples/common \
//   -I../FlashKDA/cutlass/tools/util/include \
//   r19_value_major_probe.cu -o /tmp/r19_value_major_probe
// Run: /tmp/r19_value_major_probe [seed_count=64]
// Mapping uses actual CuTe copy atoms; arithmetic compares original
// KR^T @ U with U^T @ KR, including the final BF16 update and STSM_N.
// The arithmetic kernel uses 46 KiB static shared memory: each warp has
// only one 16x16 FP32 reference tile and one 16x16 BF16 reference tile.
#include "../FlashKDA/csrc/smxx/fwd_kernel1.cuh"
#include "../FlashKDA/csrc/smxx/fwd_kernel2.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

using BF16 = cutlass::bfloat16_t;
using Layouts = K2Layouts<128, 16>;
constexpr unsigned kSeed = 0x6b39d27au;
constexpr int kThreads = 128;
constexpr int kWarps = 4;
constexpr int kCounters = 5;

enum Counter {
    StateCToB = 0,
    UToA = 1,
    KRToB = 2,
    AccumulatorBits = 3,
    UpdatedStateBits = 4,
};

__device__ __forceinline__ unsigned mix_bits(unsigned x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

// All arithmetic inputs are finite, nonzero, normal BF16 numbers. Both
// signs and several exponent distributions exercise cancellation/rounding.
__device__ __forceinline__ BF16 input_value(unsigned seed, unsigned index,
                                            unsigned salt, int profile) {
    const unsigned h = mix_bits(seed ^ salt ^ (index * 0x9e3779b9u));
    const unsigned span = profile == 0 ? 3u : (profile == 1 ? 9u : 21u);
    const unsigned exponent = (profile == 0 ? 126u : (profile == 1 ? 122u : 112u))
        + ((h >> 8) % span);
    return BF16::bitcast(uint16_t(((h >> 16) & 0x8000u) |
                                 (exponent << 7) | (h & 0x7fu)));
}

__device__ __forceinline__ float decay_value(unsigned seed, int key) {
    // Positive, nonzero FP32 decay <= 1, with bits below BF16 precision.
    return __uint_as_float(0x3e800000u |
        (mix_bits(seed ^ (unsigned(key) * 0x85ebca6bu)) & 0x00ffffffu));
}

template <class Source, class Destination>
__device__ __forceinline__ void swap_middle_words(Source const& src,
                                                 Destination& dst) {
    static_assert(decltype(size(src))::value == 8);
    static_assert(decltype(size(dst))::value == 8);
    uint32_t const* s = reinterpret_cast<uint32_t const*>(&src(0));
    uint32_t* d = reinterpret_cast<uint32_t*>(&dst(0));
    d[0] = s[0]; d[1] = s[2]; d[2] = s[1]; d[3] = s[3];
}

template <class Source, class Destination>
__device__ __forceinline__ void transpose_words(Source const& src,
                                               Destination& dst) {
    uint32_t const* s = reinterpret_cast<uint32_t const*>(&src(0));
    uint32_t* d = reinterpret_cast<uint32_t*>(&dst(0));
    #pragma unroll
    for (int r = 0; r < 4; ++r) SM75_U32x1_MOVM_T::copy(s[r], d[r]);
}

template <class Left, class Right>
__device__ __forceinline__ unsigned different_words(Left const& a, Right const& b) {
    uint32_t const* ap = reinterpret_cast<uint32_t const*>(&a(0));
    uint32_t const* bp = reinterpret_cast<uint32_t const*>(&b(0));
    unsigned n = 0;
    #pragma unroll
    for (int r = 0; r < 4; ++r) n += ap[r] != bp[r];
    return n;
}

__global__ void register_map_probe(unsigned* errors) {
    __shared__ __align__(128) BF16 state[128 * 128];
    __shared__ __align__(128) BF16 u[16 * 128];
    __shared__ __align__(128) BF16 kr[16 * 128];
    auto s = make_tensor(make_smem_ptr(state), Layouts::StateSmemLayout{});
    auto u_nt = make_tensor(make_smem_ptr(u), Layouts::MMALayout{});
    auto u_t = make_tensor(make_smem_ptr(u), Layouts::TransposedMMALayout{});
    auto kr_nt = make_tensor(make_smem_ptr(kr), Layouts::MMALayout{});
    auto kr_t = make_tensor(make_smem_ptr(kr), Layouts::TransposedMMALayout{});
    for (int i = threadIdx.x; i < 128 * 128; i += kThreads)
        s(i / 128, i % 128) = BF16::bitcast(uint16_t(i + 1));
    for (int i = threadIdx.x; i < 16 * 128; i += kThreads) {
        u_nt(i / 128, i % 128) = BF16::bitcast(uint16_t(i + 1));
        kr_nt(i / 128, i % 128) = BF16::bitcast(uint16_t(i + 4097));
    }
    __syncthreads();

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    auto mma = make_tiled_mma(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                             Layout<Shape<_1, _1>>{}, Tile<_16, _16, _16>{});
    auto thr = mma.get_slice(lane);
    auto lc = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto lb = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto lat = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
    auto lbt = make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
    auto tc = lc.get_slice(lane);
    auto tb = lb.get_slice(lane);
    auto ta_t = lat.get_slice(lane);
    auto tb_t = lbt.get_slice(lane);
    unsigned bad_state = 0, bad_u = 0, bad_kr = 0;

    for (int key_block = 0; key_block < 8; ++key_block) {
        for (int value_block = 0; value_block < 2; ++value_block) {
            auto tile = local_tile(s, make_shape(_16{}, _16{}),
                                  make_coord(warp * 2 + value_block, key_block));
            auto c = make_fragment_like<BF16>(thr.make_fragment_C(thr.partition_C(tile)));
            auto b = make_fragment_like<BF16>(thr.partition_fragment_B(tile));
            auto candidate = make_fragment_like(b);
            copy(lc, tc.partition_S(tile), tc.retile_D(c));
            copy(lb, tb.partition_S(tile), tb.retile_D(b));
            swap_middle_words(c, candidate);
            bad_state += different_words(candidate, b);
        }
        auto kr_tile = local_tile(kr_t, make_shape(_16{}, _16{}),
                                  make_coord(key_block, 0));
        auto a = make_fragment_like<BF16>(thr.partition_fragment_A(kr_tile));
        auto b = make_fragment_like<BF16>(thr.partition_fragment_B(kr_tile));
        auto candidate = make_fragment_like(b);
        copy(lat, ta_t.partition_S(kr_tile), ta_t.retile_D(a));
        copy(lbt, tb_t.partition_S(kr_tile), tb_t.retile_D(b));
        swap_middle_words(a, candidate);
        bad_kr += different_words(candidate, b);
    }
    for (int value_block = 0; value_block < 2; ++value_block) {
        auto tile = local_tile(u_nt, make_shape(_16{}, _16{}),
                               make_coord(0, warp * 2 + value_block));
        auto tile_t = local_tile(u_t, make_shape(_16{}, _16{}),
                                 make_coord(warp * 2 + value_block, 0));
        auto c = make_fragment_like<BF16>(thr.make_fragment_C(thr.partition_C(tile)));
        auto b = make_fragment_like<BF16>(thr.partition_fragment_B(tile_t));
        auto a = make_fragment_like<BF16>(thr.partition_fragment_A(tile_t));
        auto candidate = make_fragment_like(a);
        copy(lc, tc.partition_S(tile), tc.retile_D(c));
        // Exactly the existing Phase 4 U conversion, followed by the proposal.
        transpose_words(c, b);
        swap_middle_words(b, candidate);
        copy(lat, ta_t.partition_S(tile_t), ta_t.retile_D(a));
        bad_u += different_words(candidate, a);
    }
    atomicAdd(errors + StateCToB, bad_state);
    atomicAdd(errors + UToA, bad_u);
    atomicAdd(errors + KRToB, bad_kr);
}

__global__ void paired_arithmetic_probe(unsigned* errors, int seed_count) {
    __shared__ __align__(128) BF16 state[128 * 128];
    __shared__ __align__(128) BF16 u[16 * 128];
    __shared__ __align__(128) BF16 kr[16 * 128];
    __shared__ float ref_acc[kWarps][16 * 16];
    __shared__ BF16 ref_update[kWarps][16 * 16];
    auto s = make_tensor(make_smem_ptr(state), Layouts::StateSmemLayout{});
    auto st = make_tensor(make_smem_ptr(state), Layouts::TransposedStateSmemLayout{});
    auto u_nt = make_tensor(make_smem_ptr(u), Layouts::MMALayout{});
    auto u_t = make_tensor(make_smem_ptr(u), Layouts::TransposedMMALayout{});
    auto kr_nt = make_tensor(make_smem_ptr(kr), Layouts::MMALayout{});
    auto kr_t = make_tensor(make_smem_ptr(kr), Layouts::TransposedMMALayout{});
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    auto mma = make_tiled_mma(MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                             Layout<Shape<_1, _1>>{}, Tile<_16, _16, _16>{});
    auto thr = mma.get_slice(lane);
    auto lc = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
    auto lct = make_tiled_copy_C(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
    auto sc = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, BF16>{}, mma);
    auto lat = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
    auto lbt = make_tiled_copy_B(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
    auto tc = lc.get_slice(lane);
    auto tc_t = lct.get_slice(lane);
    auto ts = sc.get_slice(lane);
    auto ta_t = lat.get_slice(lane);
    auto tb_t = lbt.get_slice(lane);
    // Use CuTe's actual C-coordinate partition for the comparison, rather
    // than assuming the candidate's lane/element inverse map is correct.
    auto coords = thr.partition_C(make_identity_tensor(make_shape(_16{}, _16{})));
    unsigned bad_acc = 0, bad_update = 0;

    for (int iteration = 0; iteration < seed_count; ++iteration) {
        const unsigned seed = kSeed + unsigned(iteration) * 0x9e3779b9u;
        const int profile = iteration % 3;
        for (int i = threadIdx.x; i < 128 * 128; i += kThreads)
            s(i / 128, i % 128) = input_value(seed, i, 0x243f6a88u, profile);
        for (int i = threadIdx.x; i < 16 * 128; i += kThreads) {
            u_nt(i / 128, i % 128) = input_value(seed, i, 0x85a308d3u, profile);
            kr_nt(i / 128, i % 128) = input_value(seed, i, 0x13198a2eu, profile);
        }
        __syncthreads();
        for (int key_block = 0; key_block < 8; ++key_block) {
            auto kr_tile = local_tile(kr_t, make_shape(_16{}, _16{}),
                                      make_coord(key_block, 0));
            auto kr_a = make_fragment_like<BF16>(thr.partition_fragment_A(kr_tile));
            auto kr_b = make_fragment_like<BF16>(thr.partition_fragment_B(kr_tile));
            copy(lat, ta_t.partition_S(kr_tile), ta_t.retile_D(kr_a));
            copy(lbt, tb_t.partition_S(kr_tile), tb_t.retile_D(kr_b));
            for (int value_block = 0; value_block < 2; ++value_block) {
                const int vb = warp * 2 + value_block;
                auto u_tile = local_tile(u_nt, make_shape(_16{}, _16{}), make_coord(0, vb));
                auto u_tile_t = local_tile(u_t, make_shape(_16{}, _16{}), make_coord(vb, 0));
                auto old_tile = local_tile(st, make_shape(_16{}, _16{}), make_coord(key_block, vb));
                auto new_tile = local_tile(s, make_shape(_16{}, _16{}), make_coord(vb, key_block));
                auto u_c = make_fragment_like<BF16>(thr.make_fragment_C(thr.partition_C(u_tile)));
                auto u_b = make_fragment_like<BF16>(thr.partition_fragment_B(u_tile_t));
                auto u_a = make_fragment_like<BF16>(thr.partition_fragment_A(u_tile_t));
                copy(lc, tc.partition_S(u_tile), tc.retile_D(u_c));
                transpose_words(u_c, u_b);
                swap_middle_words(u_b, u_a);
                auto old_acc = thr.make_fragment_C(thr.partition_C(old_tile));
                auto new_acc = thr.make_fragment_C(thr.partition_C(new_tile));
                auto old_state = make_fragment_like<BF16>(old_acc);
                auto new_state = make_fragment_like<BF16>(new_acc);
                copy(lct, tc_t.partition_S(old_tile), tc_t.retile_D(old_state));
                copy(lc, tc.partition_S(new_tile), tc.retile_D(new_state));
                clear(old_acc);
                clear(new_acc);
                gemm(thr, kr_a(_, _, Int<0>{}), u_b(_, _, Int<0>{}), old_acc);
                gemm(thr, u_a(_, _, Int<0>{}), kr_b(_, _, Int<0>{}), new_acc);
                #pragma unroll
                for (int a = 0; a < 2; ++a) {
                    #pragma unroll
                    for (int d = 0; d < 2; ++d) {
                        auto c0 = make_coord(make_coord(a, 0), 0, d);
                        auto c1 = make_coord(make_coord(a, 1), 0, d);
                        const float g0 = decay_value(seed, key_block * 16 + lane / 4);
                        const float g1 = decay_value(seed, key_block * 16 + lane / 4 + 8);
                        old_state(c0) = BF16(bf16_to_f32(old_state(c0)) * g0 + old_acc(c0));
                        old_state(c1) = BF16(bf16_to_f32(old_state(c1)) * g1 + old_acc(c1));
                        const float g = decay_value(seed, key_block * 16 + 2 * (lane % 4) + a + 8 * d);
                        new_state(c0) = BF16(bf16_to_f32(new_state(c0)) * g + new_acc(c0));
                        new_state(c1) = BF16(bf16_to_f32(new_state(c1)) * g + new_acc(c1));
                    }
                }
                #pragma unroll
                for (int j = 0; j < size(old_acc); ++j) {
                    const auto c = coords(j);
                    const int row = int(get<0>(c)), col = int(get<1>(c));
                    ref_acc[warp][row * 16 + col] = old_acc(j);
                    ref_update[warp][row * 16 + col] = old_state(j);
                }
                // Also validate the proposed final-state publication atom.
                copy(sc, ts.retile_S(new_state), ts.partition_D(new_tile));
                __syncwarp();
                #pragma unroll
                for (int j = 0; j < size(new_acc); ++j) {
                    const auto c = coords(j);
                    const int value = int(get<0>(c)), key = int(get<1>(c));
                    bad_acc += __float_as_uint(new_acc(j)) !=
                        __float_as_uint(ref_acc[warp][key * 16 + value]);
                    bad_update += s(vb * 16 + value, key_block * 16 + key).storage !=
                        ref_update[warp][key * 16 + value].storage;
                }
                // Protect the per-warp reference scratch before its reuse.
                __syncwarp();
            }
        }
        // All warps must finish reading inputs before the next seed replaces them.
        __syncthreads();
    }
    atomicAdd(errors + AccumulatorBits, bad_acc);
    atomicAdd(errors + UpdatedStateBits, bad_update);
}

static void check(cudaError_t status, char const* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(2);
    }
}

int main(int argc, char** argv) {
    int seed_count = 64;
    if (argc > 2) {
        std::fprintf(stderr, "usage: %s [seed_count=64]\n", argv[0]);
        return 2;
    }
    if (argc == 2) {
        char* end = nullptr;
        const long n = std::strtol(argv[1], &end, 10);
        if (!end || *end || n < 1 || n > 4096) {
            std::fprintf(stderr, "seed_count must be an integer in [1,4096]\n");
            return 2;
        }
        seed_count = int(n);
    }
    unsigned* device_errors = nullptr;
    unsigned errors[kCounters]{};
    cudaFuncAttributes attributes{};
    check(cudaFuncGetAttributes(&attributes, paired_arithmetic_probe), "kernel attributes");
    check(cudaMalloc(&device_errors, sizeof(errors)), "cudaMalloc");
    check(cudaMemset(device_errors, 0, sizeof(errors)), "cudaMemset");
    register_map_probe<<<1, kThreads>>>(device_errors);
    check(cudaGetLastError(), "register_map_probe launch");
    check(cudaDeviceSynchronize(), "register_map_probe execution");
    paired_arithmetic_probe<<<1, kThreads>>>(device_errors, seed_count);
    check(cudaGetLastError(), "paired_arithmetic_probe launch");
    check(cudaMemcpy(errors, device_errors, sizeof(errors), cudaMemcpyDeviceToHost), "read counters");
    check(cudaFree(device_errors), "cudaFree");
    std::printf("{\"seed\":%u,\"seed_count\":%d,\"static_shared_bytes\":%zu,"
                "\"state_c_to_b_packed_comparisons\":8192,"
                "\"u_b_to_a_packed_comparisons\":1024,"
                "\"kr_a_to_b_packed_comparisons\":4096,"
                "\"fp32_comparisons\":%u,\"bf16_update_comparisons\":%u,"
                "\"state_c_to_b_packed_mismatches\":%u,"
                "\"u_b_to_a_packed_mismatches\":%u,"
                "\"kr_a_to_b_packed_mismatches\":%u,"
                "\"fp32_accumulator_bit_mismatches\":%u,"
                "\"bf16_state_update_bit_mismatches\":%u}\n",
                kSeed, seed_count, attributes.sharedSizeBytes,
                unsigned(seed_count) * 16384u, unsigned(seed_count) * 16384u,
                errors[StateCToB], errors[UToA], errors[KRToB],
                errors[AccumulatorBits], errors[UpdatedStateBits]);
    for (unsigned e : errors) if (e != 0) return 1;
    return 0;
}
