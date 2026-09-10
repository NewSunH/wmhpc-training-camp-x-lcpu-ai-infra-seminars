#pragma once

// TMA_DISABLE_ALL: when defined, disable load/store warps entirely
// and let MMA warps work without pipeline synchronization
// #define TMA_DISABLE_ALL

#include "utils.cuh"

// R7 probe: keep the recurrence/state update intact while disabling the
// output materialization path.  This is intentionally opt-in at compile time
// so the production kernel and its Python API remain unchanged.
#ifndef C1_K2_STATE_ONLY
#define C1_K2_STATE_ONLY 0
#endif

// R8 probe: keep the output accumulator in FP32 through both output GEMMs,
// then round once to BF16 immediately before the shared-memory epilogue.
// The default remains the R4/R7 ordering for exact rollback.
#ifndef C1_K2_OUT_FP32_ACCUM
#define C1_K2_OUT_FP32_ACCUM 0
#endif

// R8 probe: fuse conversion of the second output GEMM with the BF16 add.
// This preserves the original "round GEMM term, then BF16 add" ordering.
#ifndef C1_K2_FUSE_OUT_ADD
#define C1_K2_FUSE_OUT_ADD 0
#endif

// R10 probe: write the final BF16 output fragment directly to global memory
// using the lane/fragment inverse map established by the R9 standalone
// oracle.  The default remains the shared-memory + TMA output pipeline.
#ifndef C1_K2_DIRECT_OUTPUT
#define C1_K2_DIRECT_OUTPUT 0
#endif

// R10-C probe: pack each adjacent BF16 pair from the explicit map into one
// 32-bit global transaction.  This is opt-in and only valid for the same
// 16x16 fragment shape proven by the scalar R10-A path.
#ifndef C1_K2_DIRECT_OUTPUT_VEC
#define C1_K2_DIRECT_OUTPUT_VEC 0
#endif

// R11-A probe: when direct output is selected, omit the shared-memory output
// ring from SharedStorageK2.  The default and the R10 direct path retain the
// original storage layout unless this separate opt-in is enabled.
#ifndef C1_K2_COMPACT_DIRECT_STORAGE
#define C1_K2_COMPACT_DIRECT_STORAGE 0
#endif

// R11-B probe: use a 128-byte-swizzled shared-memory output tile followed by
// a matching TMA store.  This is opt-in; the default/R10 paths are unchanged.
#ifndef C1_K2_TMA_SWIZZLED_OUTPUT
#define C1_K2_TMA_SWIZZLED_OUTPUT 0
#endif

// R12-B probe: use the existing CuTe STSM_N atom to write directly into the
// TMA-compatible swizzled tile.  This is deliberately separate from the R11
// scalar inverse-map path, so either candidate can be benchmarked independently.
#ifndef C1_K2_FUSED_TMA_EPILOGUE
#define C1_K2_FUSED_TMA_EPILOGUE 0
#endif

// C1 prototype: fuse K1's per-tile preparation into K2.  K2 recomputes the
// six intermediates from q/k/g instead of reading K1's global workspace.  The
// default remains the two-kernel implementation; this route is deliberately
// restricted to the full (non-value-split) K2 until its resource/performance
// trade-off is understood.
#ifndef C1_K1_K2_FUSED_WS
#define C1_K1_K2_FUSED_WS 0
#endif

template <int D, int CHUNK = 16>
struct K2Layouts {
    static constexpr int kValueSliceD = D / 2;
    using MMALayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<D>{}),
        LayoutLeft{}
    ));
    using TransposedMMALayout = decltype(tile_to_shape(
        GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<D>{}, Int<CHUNK>{}),
        LayoutRight{}
    ));
    using VOLayout = MMALayout;
    using TransposedVOLayout = TransposedMMALayout;
    using BetaSmemLayout = Layout<Shape<Int<32>>, Stride<Int<1>>>;
    using StateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<D>{}, Int<D>{}),
        LayoutLeft{}
    ));
    using TransposedStateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<D>{}, Int<D>{}),
        LayoutRight{}
    ));
    using GTotalLayout = Layout<Shape<Int<D>>, Stride<Int<1>>>;
    using LMLayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<CHUNK>{}),
        LayoutLeft{}
    ));

    using TMABetaSmemLayout = BetaSmemLayout;  // 1D TMA, no dummy dim
    using TMAVOLayout = decltype(composition(
        VOLayout{}.layout_a(),
        VOLayout{}.offset(),
        prepend(VOLayout{}.layout_b())
    ));
    // C1 R4: one value-split CTA owns exactly half of the value dimension.
    // These layouts are used for the V/output TMA descriptors and for the
    // compact per-CTA shared-memory buffers.
    using ValueSliceVOLayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<CHUNK>{}, Int<kValueSliceD>{}),
        LayoutLeft{}
    ));
    using TMAValueSliceVOLayout = decltype(composition(
        ValueSliceVOLayout{}.layout_a(),
        ValueSliceVOLayout{}.offset(),
        prepend(ValueSliceVOLayout{}.layout_b())
    ));

    // The TMA descriptor uses the unit-stride value dimension as its first
    // global-memory mode.  The shared tile is therefore logically [D,CHUNK]
    // and uses the SM90 128-byte epilogue swizzle.  For value-split CTAs the
    // same construction is instantiated with D/2.
    using SwizzledOutputPreLayout = Layout<
        Shape<Shape<Int<D / 4>, Int<4>>, Int<CHUNK>>,
        Stride<Stride<Int<1>, Int<D * CHUNK / 4>>, Int<D / 4>>>;
    using SwizzledOutputLayout = ComposedLayout<
        Swizzle<3, 4, 3>,
        smem_ptr_flag_bits<sizeof_bits<cute::bfloat16_t>::value>,
        SwizzledOutputPreLayout>;
    using TMAFullSwizzledOutputLayout = decltype(composition(
        SwizzledOutputLayout{}.layout_a(),
        SwizzledOutputLayout{}.offset(),
        prepend(SwizzledOutputLayout{}.layout_b())
    ));
    using SwizzledValueSlicePreLayout = Layout<
        Shape<Shape<Int<kValueSliceD / 4>, Int<4>>, Int<CHUNK>>,
        Stride<Stride<Int<1>, Int<kValueSliceD * CHUNK / 4>>, Int<kValueSliceD / 4>>>;
    using SwizzledValueSliceLayout = ComposedLayout<
        Swizzle<3, 4, 3>,
        smem_ptr_flag_bits<sizeof_bits<cute::bfloat16_t>::value>,
        SwizzledValueSlicePreLayout>;
    using TMASwizzledValueSliceOutputLayout = decltype(composition(
        SwizzledValueSliceLayout{}.layout_a(),
        SwizzledValueSliceLayout{}.offset(),
        prepend(SwizzledValueSliceLayout{}.layout_b())
    ));
    using TMAStateSmemLayout = decltype(composition(
        StateSmemLayout{}.layout_a(),
        StateSmemLayout{}.offset(),
        prepend(StateSmemLayout{}.layout_b())
    ));
    // State is indexed [value, key].  A split CTA therefore needs 64x128 of
    // it: all key columns, but only the value rows that it updates and reads.
    using ValueSliceStateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_K_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<kValueSliceD>{}, Int<D>{}),
        LayoutLeft{}
    ));
    using TransposedValueSliceStateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_MN_INTER_Atom<cute::bfloat16_t>{},
        make_shape(Int<D>{}, Int<kValueSliceD>{}),
        LayoutRight{}
    ));
    using TMAValueSliceStateSmemLayout = decltype(composition(
        ValueSliceStateSmemLayout{}.layout_a(),
        ValueSliceStateSmemLayout{}.offset(),
        prepend(ValueSliceStateSmemLayout{}.layout_b())
    ));
    using TMALMLayout = decltype(composition(
        LMLayout{}.layout_a(),
        LMLayout{}.offset(),
        prepend(LMLayout{}.layout_b())
    ));
    using TMAGTotalSmemLayout = decltype(prepend(GTotalLayout{}));

    // FP32 state layout (K_SW32 atom, same 8x8 atom structure as K_INTER bf16)
    using FP32StateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_K_SW32_Atom<float>{},
        make_shape(Int<D>{}, Int<D>{}),
        LayoutLeft{}
    ));
    using TMAFP32StateSmemLayout = decltype(composition(
        FP32StateSmemLayout{}.layout_a(),
        FP32StateSmemLayout{}.offset(),
        prepend(FP32StateSmemLayout{}.layout_b())
    ));
    using FP32ValueSliceStateSmemLayout = decltype(tile_to_shape(
        GMMA::Layout_K_SW32_Atom<float>{},
        make_shape(Int<kValueSliceD>{}, Int<D>{}),
        LayoutLeft{}
    ));
    using TMAFP32ValueSliceStateSmemLayout = decltype(composition(
        FP32ValueSliceStateSmemLayout{}.layout_a(),
        FP32ValueSliceStateSmemLayout{}.offset(),
        prepend(FP32ValueSliceStateSmemLayout{}.layout_b())
    ));
};

template <class Layouts, int InputStages, int OutputStages, bool ValueSplit = false>
struct SharedStorageK2 {
    using BF16 = cutlass::bfloat16_t;
    using VOLayout = typename Layouts::VOLayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using StateSmemLayout = typename Layouts::StateSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;
    using MMALayout = typename Layouts::MMALayout;
    using ValueVOLayout = std::conditional_t<
        ValueSplit, typename Layouts::ValueSliceVOLayout, VOLayout>;
    using ValueStateSmemLayout = std::conditional_t<
        ValueSplit, typename Layouts::ValueSliceStateSmemLayout, StateSmemLayout>;
    using ValueOutputLayout = std::conditional_t<
        ValueSplit, typename Layouts::SwizzledValueSliceLayout,
        typename Layouts::SwizzledOutputLayout>;

    alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<ValueStateSmemLayout>> state_acc;

    struct InputStorage {
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<ValueVOLayout>> v;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<BetaSmemLayout>> beta;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_decayed;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> q_decayed;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_restored;
#if C1_K1_K2_FUSED_WS
        // K1's inverse-decay vector is needed only while forming L/Mqk.  It
        // is a full [CHUNK,D] tile, so it cannot fit in the 16x16 INV tile.
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<MMALayout>> k_inv;
#endif
        alignas(128) cute::ArrayEngine<float, cute::cosize_v<GTotalLayout>> g_total;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> INV;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> Mqk;
#if C1_K1_K2_FUSED_WS
        // K1/K2 workspace-fusion temporary.  During fused preparation this
        // holds L=K_decayed @ K_inv before being overwritten by the final INV.
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<LMLayout>> L;
#endif
    };

    struct OutputStorage {
        using StorageLayout = std::conditional_t<
            ((C1_K2_TMA_SWIZZLED_OUTPUT != 0) || (C1_K2_FUSED_TMA_EPILOGUE != 0)),
            ValueOutputLayout, ValueVOLayout>;
        alignas(128) cute::ArrayEngine<BF16, cute::cosize_v<StorageLayout>> out;
    };

    // Direct output writes the completed fragment from the MMA warp straight
    // to global memory, so it has no consumer for the output ring.  Keep a
    // tiny aligned placeholder in the union so the member remains well-formed
    // for all template instantiations; the real ring is still used by every
    // default/non-direct build.  The following input/state members determine
    // the union size in the compact variant.
    struct EmptyOutputStorage {
        alignas(128) cute::ArrayEngine<BF16, 1> out;

        // Preserve the original source expression
        // shared_storage.output[stage].out.begin() in the discarded direct
        // output path without allocating an output ring.
        __host__ __device__ EmptyOutputStorage& operator[](int) { return *this; }
    };
    using OutputStorageArray = std::conditional_t<
        (C1_K2_COMPACT_DIRECT_STORAGE != 0 && C1_K2_DIRECT_OUTPUT != 0),
        EmptyOutputStorage,
        OutputStorage[OutputStages]>;

    // Anonymous union: pipeline buffers share space with fp32 state conversion buffer.
    // FP32 state load/store happens before/after the pipeline loop, so no overlap.
    union {
        struct {
            InputStorage input[InputStages];
            OutputStorageArray output;
        };
        alignas(128) char state_fp32_buf[cute::cosize_v<ValueStateSmemLayout> * sizeof(float)];
    };

    typename cutlass::PipelineTmaAsync<InputStages>::SharedStorage load_pipeline;
    typename cutlass::PipelineAsync<OutputStages>::SharedStorage store_pipeline;
    alignas(16) cutlass::arch::ClusterTransactionBarrier state_acc_tma_barrier;
};

// ==================== Kernel 2: Recurrence ====================
template <
    class TmaLoadQ,
    class TmaLoadK,
    class TmaLoadG,
    class TmaLoadV,
    class TmaLoadVSlice,
    class TmaLoadBeta,
    class TmaLoadWsKD, class TmaLoadWsQD, class TmaLoadWsKR,
    class TmaLoadWsGT, class TmaLoadWsINV, class TmaLoadWsMqk,
    class TmaLoadState,
    class TmaLoadStateSlice,
    class TmaStoreState,
    class TmaStoreStateSlice,
    class TmaStoreOut,
    class TmaStoreOutSlice,
    class TmaStoreOutSwizzled,
    class TmaStoreOutSwizzledSlice,
    int CHUNK,
    int D,
    int InputStages,
    int OutputStages,
    int NumThreads,
    bool HasStateIn = true,
    bool HasStateOut = true,
    bool StateFP32 = false,
    bool IsVarlen = true,
    bool ValueSplit = false
>
__global__ void __launch_bounds__(NumThreads) _flash_kda_fwd_recurrence(
    CUTE_GRID_CONSTANT TmaLoadQ const tma_load_q,
    CUTE_GRID_CONSTANT TmaLoadK const tma_load_k,
    CUTE_GRID_CONSTANT TmaLoadG const tma_load_g,
    CUTE_GRID_CONSTANT TmaLoadV const tma_load_v,
    CUTE_GRID_CONSTANT TmaLoadVSlice const tma_load_v_slice,
    CUTE_GRID_CONSTANT TmaLoadBeta const tma_load_beta,
    CUTE_GRID_CONSTANT TmaLoadWsKD const tma_load_ws_kd,
    CUTE_GRID_CONSTANT TmaLoadWsQD const tma_load_ws_qd,
    CUTE_GRID_CONSTANT TmaLoadWsKR const tma_load_ws_kr,
    CUTE_GRID_CONSTANT TmaLoadWsGT const tma_load_ws_gt,
    CUTE_GRID_CONSTANT TmaLoadWsINV const tma_load_ws_inv,
    CUTE_GRID_CONSTANT TmaLoadWsMqk const tma_load_ws_mqk,
    CUTE_GRID_CONSTANT TmaLoadState const tma_load_initial_state,
    CUTE_GRID_CONSTANT TmaLoadStateSlice const tma_load_initial_state_slice,
    CUTE_GRID_CONSTANT TmaStoreState const tma_store_final_state,
    CUTE_GRID_CONSTANT TmaStoreStateSlice const tma_store_final_state_slice,
    CUTE_GRID_CONSTANT TmaStoreOut const tma_store_out,
    CUTE_GRID_CONSTANT TmaStoreOutSlice const tma_store_out_slice,
    CUTE_GRID_CONSTANT TmaStoreOutSwizzled const tma_store_out_swizzled,
    CUTE_GRID_CONSTANT TmaStoreOutSwizzledSlice const tma_store_out_swizzled_slice,
    cutlass::bfloat16_t* out_raw_ptr,
    void* final_state_raw_ptr,
    float scale,
    float const* A_log_ptr,
    float const* dt_bias_ptr,
    float gate_scale,
    int T_total,
    int H,
    int N,
    int64_t const* cu_seqlens,
    int total_tiles
) {
    using BF16 = cutlass::bfloat16_t;
    using FP16 = cutlass::half_t;
    using Layouts = K2Layouts<D, CHUNK>;
    using MMALayout = typename Layouts::MMALayout;
    using TransposedMMALayout = typename Layouts::TransposedMMALayout;
    using VOLayout = typename Layouts::VOLayout;
    using TransposedVOLayout = typename Layouts::TransposedVOLayout;
    using BetaSmemLayout = typename Layouts::BetaSmemLayout;
    using StateSmemLayout = typename Layouts::StateSmemLayout;
    using TransposedStateSmemLayout = typename Layouts::TransposedStateSmemLayout;
    using GTotalLayout = typename Layouts::GTotalLayout;
    using LMLayout = typename Layouts::LMLayout;
    using TMAVOLayout = typename Layouts::TMAVOLayout;
    using TMABetaSmemLayout = typename Layouts::TMABetaSmemLayout;
    using TMAStateSmemLayout = typename Layouts::TMAStateSmemLayout;
    using ValueSliceStateSmemLayout = typename Layouts::ValueSliceStateSmemLayout;
    using TransposedValueSliceStateSmemLayout = typename Layouts::TransposedValueSliceStateSmemLayout;
    using TMAValueSliceStateSmemLayout = typename Layouts::TMAValueSliceStateSmemLayout;
    using ValueSliceVOLayout = typename Layouts::ValueSliceVOLayout;
    using TMAValueSliceVOLayout = typename Layouts::TMAValueSliceVOLayout;
    using SwizzledOutputLayout = typename Layouts::SwizzledOutputLayout;
    using SwizzledValueSliceLayout = typename Layouts::SwizzledValueSliceLayout;
    using TMAFullSwizzledOutputLayout = typename Layouts::TMAFullSwizzledOutputLayout;
    using TMASwizzledValueSliceOutputLayout = typename Layouts::TMASwizzledValueSliceOutputLayout;
    using FP32StateSmemLayout = typename Layouts::FP32StateSmemLayout;
    using TMAFP32StateSmemLayout = typename Layouts::TMAFP32StateSmemLayout;
    using FP32ValueSliceStateSmemLayout = typename Layouts::FP32ValueSliceStateSmemLayout;
    using TMAFP32ValueSliceStateSmemLayout = typename Layouts::TMAFP32ValueSliceStateSmemLayout;
    using TMALMLayout = typename Layouts::TMALMLayout;
    using TMAGTotalSmemLayout = typename Layouts::TMAGTotalSmemLayout;
    // Raw q/k/g tiles are loaded using K1's row-major TMA destination before
    // their storage is reused as K2's MMA-interleaved intermediates.
    using K1RawLayouts = K1Layouts<D, CHUNK>;
    using QKLayout = typename K1RawLayouts::QKLayout;
    using GLayout = typename K1RawLayouts::GLayout;
    using TMAQKLayout = typename K1RawLayouts::TMAQKLayout;
    constexpr int kWarpSize = 32;
    // Baseline: 4 MMA warps + TMA load/store warps (192 threads).  The opt-in
    // prototype uses 2 MMA warps + TMA load/store warps (128 threads) and
    // maps two CTAs to disjoint 64-column value slices of a head.
    constexpr int kComputeThreads = ValueSplit ? 64 : 128;
    static_assert(!ValueSplit || NumThreads == kComputeThreads + 2 * kWarpSize);
    constexpr bool kStateOnly = C1_K2_STATE_ONLY != 0;
    constexpr bool kOutFp32Accum = C1_K2_OUT_FP32_ACCUM != 0;
    constexpr bool kFuseOutAdd = C1_K2_FUSE_OUT_ADD != 0;
    constexpr bool kDirectOutput = C1_K2_DIRECT_OUTPUT != 0;
    constexpr bool kDirectOutputVec = C1_K2_DIRECT_OUTPUT_VEC != 0;
    constexpr bool kFusedTmaEpilogue = C1_K2_FUSED_TMA_EPILOGUE != 0;
    constexpr bool kTmaSwizzledOutput = (C1_K2_TMA_SWIZZLED_OUTPUT != 0) || kFusedTmaEpilogue;
    constexpr bool kFusedWorkspace = C1_K1_K2_FUSED_WS != 0;
    static_assert(!kFusedWorkspace || !ValueSplit,
                  "K1/K2 fused workspace prototype currently supports full K2 only");
    static_assert(!(kTmaSwizzledOutput && kDirectOutput),
                  "R11/R12 swizzled TMA and R10 direct output are exclusive");

    // Transaction bytes: v + beta plus either six workspace intermediates or
    // the three raw K1 inputs used by the fused recompute path.
    constexpr uint32_t kValueTmaElements = ValueSplit
        ? uint32_t(cute::cosize_v<ValueSliceVOLayout>)
        : uint32_t(cute::cosize_v<VOLayout>);
    constexpr uint32_t kTmaTransactionBytes =
#ifndef TMA_DISABLE_ALL
        kValueTmaElements * uint32_t(sizeof(BF16)) +
        uint32_t(32) * uint32_t(sizeof(BF16)) +  // beta (bf16, sigmoid fused)
        (kFusedWorkspace
            ? uint32_t(cute::cosize_v<MMALayout>) * uint32_t(sizeof(BF16)) * 3
            : uint32_t(cute::cosize_v<MMALayout>) * uint32_t(sizeof(BF16)) * 3 +
              uint32_t(cute::cosize_v<GTotalLayout>) * uint32_t(sizeof(float)) +
              uint32_t(cute::cosize_v<LMLayout>) * uint32_t(sizeof(BF16)) * 2) +
#endif
        0u;

    // --- shared memory
    extern __shared__ __align__(128) unsigned char shared_mem[];
    using SharedStorageT = SharedStorageK2<Layouts, InputStages, OutputStages, ValueSplit>;
    SharedStorageT& shared_storage = *reinterpret_cast<SharedStorageT*>(shared_mem);

    // --- warp specialization
    int warp_id = threadIdx.x / kWarpSize;
    WarpRole warp_role = WarpRole::NonParticipant;
    if (warp_id < kComputeThreads / kWarpSize) {
        warp_role = WarpRole::MMA;
    } else if (warp_id < kComputeThreads / kWarpSize + 1) {
        warp_role = WarpRole::LOAD_QKG;
    } else if (warp_id < kComputeThreads / kWarpSize + 2) {
        warp_role = WarpRole::STORE;
    }

#ifndef TMA_DISABLE_ALL
    using LoadPipelineState = cutlass::PipelineState<InputStages>;
    using LoadPipeline = cutlass::PipelineTmaAsync<InputStages>;
    LoadPipeline load_pipeline = make_load_pipeline<InputStages>(
        shared_storage.load_pipeline,
        kTmaTransactionBytes,
        warp_role, 1, kComputeThreads
    );
    using StorePipelineState = cutlass::PipelineState<OutputStages>;
    using StorePipeline = cutlass::PipelineAsync<OutputStages>;
    StorePipeline store_pipeline = make_store_pipeline<OutputStages>(
        shared_storage.store_pipeline,
        warp_role, kComputeThreads, 1
    );
#endif

    // --- per-block sequence info
    int seq_idx  = blockIdx.x;
    int head_idx = blockIdx.y;
    int value_slice_idx = 0;
    // State/V/output buffers are compact in the split kernel, so MMA tiles
    // are always addressed relative to the CTA-local 64-column slice.
    constexpr int value_block_base = 0;
    if constexpr (ValueSplit) {
        head_idx = blockIdx.y / 2;
        value_slice_idx = blockIdx.y & 1;
    }
    int64_t bos, eos;
    int tile_base;

    if constexpr (IsVarlen) {
        bos = cu_seqlens[seq_idx];
        eos = cu_seqlens[seq_idx + 1];
        // Compute tile_base via linear scan (no host-precomputed table)
        tile_base = 0;
        for (int i = 0; i < seq_idx; i++) {
            tile_base += (int(cu_seqlens[i + 1] - cu_seqlens[i]) + CHUNK - 1) / CHUNK;
        }
    } else {
        int T_seq = T_total / N;
        bos = seq_idx * T_seq;
        eos = bos + T_seq;
        tile_base = seq_idx * ((T_seq + CHUNK - 1) / CHUNK);
    }
    int seq_len  = int(eos - bos);
    int t_tiles  = (seq_len + CHUNK - 1) / CHUNK;
    bool lane_predicate = cute::elect_one_sync();

    // --- Load initial state
#ifndef TMA_DISABLE_ALL
    if constexpr (HasStateIn && !StateFP32) {
        // BF16 state: TMA load directly into state_acc
        if (warp_role == WarpRole::LOAD_QKG && lane_predicate) {
            using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;
            constexpr uint32_t kStateTransactionBytes = (ValueSplit
                ? cute::cosize_v<ValueSliceStateSmemLayout>
                : cute::cosize_v<StateSmemLayout>) * sizeof(BF16);

            shared_storage.state_acc_tma_barrier.init(1);
            cutlass::arch::fence_barrier_init();  // generic init -> visible to async proxy (TMA complete-tx)
            shared_storage.state_acc_tma_barrier.arrive_and_expect_tx(kStateTransactionBytes);

            if constexpr (ValueSplit) {
                Tensor g_init = tma_load_initial_state_slice.get_tma_tensor(make_shape(N * H, D, D));
                auto init_off = g_init.layout()(seq_idx * H + head_idx, value_slice_idx * (D / 2), 0);
                Tensor g_init_tile = make_tensor(g_init.data() + init_off,
                    make_layout(make_shape(Int<1>{}, Int<D / 2>{}, Int<D>{}), stride(g_init.layout())));
                Tensor s_state = make_tensor(
                    make_smem_ptr(shared_storage.state_acc.begin()), TMAValueSliceStateSmemLayout{});
                auto cta_tma_load_state = tma_load_initial_state_slice.get_slice(Int<0>{});
                cute::copy(
                    tma_load_initial_state_slice.with(reinterpret_cast<BarrierType&>(shared_storage.state_acc_tma_barrier)),
                    cta_tma_load_state.partition_S(g_init_tile),
                    cta_tma_load_state.partition_D(s_state)
                );
            } else {
                Tensor g_init = tma_load_initial_state.get_tma_tensor(make_shape(N * H, D, D));
                auto init_off = g_init.layout()(seq_idx * H + head_idx, 0, 0);
                Tensor g_init_tile = make_tensor(g_init.data() + init_off,
                    make_layout(make_shape(Int<1>{}, Int<D>{}, Int<D>{}), stride(g_init.layout())));
                Tensor s_state = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), TMAStateSmemLayout{});
                auto cta_tma_load_state = tma_load_initial_state.get_slice(Int<0>{});
                cute::copy(
                    tma_load_initial_state.with(reinterpret_cast<BarrierType&>(shared_storage.state_acc_tma_barrier)),
                    cta_tma_load_state.partition_S(g_init_tile),
                    cta_tma_load_state.partition_D(s_state)
                );
            }
        }
        __syncthreads();
        shared_storage.state_acc_tma_barrier.wait(0);
        cutlass::arch::fence_view_async_shared();
    } else if constexpr (HasStateIn && StateFP32) {
        // FP32 state: TMA load fp32 into pipeline buffer, then convert to bf16 in state_acc
        if (warp_role == WarpRole::LOAD_QKG && lane_predicate) {
            using BarrierType = cutlass::arch::ClusterTransactionBarrier::ValueType;
            constexpr uint32_t kFP32StateTransactionBytes = (ValueSplit
                ? cute::cosize_v<ValueSliceStateSmemLayout>
                : cute::cosize_v<StateSmemLayout>) * sizeof(float);

            shared_storage.state_acc_tma_barrier.init(1);
            cutlass::arch::fence_barrier_init();  // generic init -> visible to async proxy (TMA complete-tx)
            shared_storage.state_acc_tma_barrier.arrive_and_expect_tx(kFP32StateTransactionBytes);

            if constexpr (ValueSplit) {
                Tensor g_init = tma_load_initial_state_slice.get_tma_tensor(make_shape(N * H, D, D));
                auto init_off = g_init.layout()(seq_idx * H + head_idx, value_slice_idx * (D / 2), 0);
                Tensor g_init_tile = make_tensor(g_init.data() + init_off,
                    make_layout(make_shape(Int<1>{}, Int<D / 2>{}, Int<D>{}), stride(g_init.layout())));
                Tensor s_fp32 = make_tensor(
                    make_smem_ptr(reinterpret_cast<float*>(shared_storage.state_fp32_buf)),
                    TMAFP32ValueSliceStateSmemLayout{});
                auto cta_tma_load_state = tma_load_initial_state_slice.get_slice(Int<0>{});
                cute::copy(
                    tma_load_initial_state_slice.with(reinterpret_cast<BarrierType&>(shared_storage.state_acc_tma_barrier)),
                    cta_tma_load_state.partition_S(g_init_tile),
                    cta_tma_load_state.partition_D(s_fp32)
                );
            } else {
                Tensor g_init = tma_load_initial_state.get_tma_tensor(make_shape(N * H, D, D));
                auto init_off = g_init.layout()(seq_idx * H + head_idx, 0, 0);
                Tensor g_init_tile = make_tensor(g_init.data() + init_off,
                    make_layout(make_shape(Int<1>{}, Int<D>{}, Int<D>{}), stride(g_init.layout())));
                Tensor s_fp32 = make_tensor(
                    make_smem_ptr(reinterpret_cast<float*>(shared_storage.state_fp32_buf)),
                    TMAFP32StateSmemLayout{});
                auto cta_tma_load_state = tma_load_initial_state.get_slice(Int<0>{});
                cute::copy(
                    tma_load_initial_state.with(reinterpret_cast<BarrierType&>(shared_storage.state_acc_tma_barrier)),
                    cta_tma_load_state.partition_S(g_init_tile),
                    cta_tma_load_state.partition_D(s_fp32)
                );
            }
        }
        __syncthreads();
        shared_storage.state_acc_tma_barrier.wait(0);
        cutlass::arch::fence_view_async_shared();

        // All threads: convert fp32 -> bf16 with layout transformation
        smem_cvt_fp32_to_bf16<
            std::conditional_t<ValueSplit, FP32ValueSliceStateSmemLayout, FP32StateSmemLayout>,
            std::conditional_t<ValueSplit, ValueSliceStateSmemLayout, StateSmemLayout>,
            ValueSplit ? D / 2 : D, D, NumThreads>(
            reinterpret_cast<float*>(shared_storage.state_fp32_buf),
            shared_storage.state_acc.begin(),
            threadIdx.x);
        __syncthreads();
    } else {
        // No state in: zero-initialize state_acc
        {
            BF16* buf = shared_storage.state_acc.begin();
            constexpr int kTotal = ValueSplit
                ? cute::cosize_v<ValueSliceStateSmemLayout>
                : cute::cosize_v<StateSmemLayout>;
            for (int i = threadIdx.x; i < kTotal; i += NumThreads) {
                buf[i] = BF16(0);
            }
        }
        // generic writes -> visible to async proxy (TMA state store covers t_tiles==0)
        cutlass::arch::fence_view_async_shared();
        __syncthreads();
    }
#endif

#ifndef TMA_DISABLE_ALL
    __syncthreads();

    // --- LOAD warp: issue TMA loads for v, beta, and workspace intermediates
    if (warp_role == WarpRole::LOAD_QKG && lane_predicate) {
        Tensor g_q = tma_load_q.get_tma_tensor(make_shape(H, T_total, D));
        Tensor g_k = tma_load_k.get_tma_tensor(make_shape(H, T_total, D));
        Tensor g_gate = tma_load_g.get_tma_tensor(make_shape(H, T_total, D));
        Tensor g_v = tma_load_v.get_tma_tensor(make_shape(H, T_total, D));
        Tensor g_beta = tma_load_beta.get_tma_tensor(make_shape(H * T_total));

        // Workspace gmem tensors
        auto g_ws_kd = tma_load_ws_kd.get_tma_tensor(make_shape(H * total_tiles, CHUNK, D));
        auto g_ws_qd = tma_load_ws_qd.get_tma_tensor(make_shape(H * total_tiles, CHUNK, D));
        auto g_ws_kr = tma_load_ws_kr.get_tma_tensor(make_shape(H * total_tiles, CHUNK, D));
        auto g_ws_gt = tma_load_ws_gt.get_tma_tensor(make_shape(H * total_tiles, D));
        auto g_ws_inv = tma_load_ws_inv.get_tma_tensor(make_shape(H * total_tiles, CHUNK, CHUNK));
        auto g_ws_mqk = tma_load_ws_mqk.get_tma_tensor(make_shape(H * total_tiles, CHUNK, CHUNK));

        LoadPipelineState load_write = cutlass::make_producer_start_state<LoadPipeline>();
        auto cta_tma_load_beta = tma_load_beta.get_slice(Int<0>{});
        auto cta_ws_kd = tma_load_ws_kd.get_slice(Int<0>{});
        auto cta_ws_qd = tma_load_ws_qd.get_slice(Int<0>{});
        auto cta_ws_kr = tma_load_ws_kr.get_slice(Int<0>{});
        auto cta_ws_gt = tma_load_ws_gt.get_slice(Int<0>{});
        auto cta_ws_inv = tma_load_ws_inv.get_slice(Int<0>{});
        auto cta_ws_mqk = tma_load_ws_mqk.get_slice(Int<0>{});
        auto cta_tma_load_q = tma_load_q.get_slice(Int<0>{});
        auto cta_tma_load_k = tma_load_k.get_slice(Int<0>{});
        auto cta_tma_load_g = tma_load_g.get_slice(Int<0>{});

        for (int t = 0; t < t_tiles; ++t) {
            load_pipeline.producer_acquire(load_write);
            using LoadBarrierType = typename LoadPipeline::ProducerBarrierType;
            LoadBarrierType* tma_barrier = load_pipeline.producer_get_barrier(load_write);
            int stage = load_write.index();
            int ws_idx = head_idx * total_tiles + tile_base + t;

            // TMA load V.  In the split kernel descriptor and shared-memory
            // tile are 16x64, so the two CTAs do not duplicate value traffic.
            if constexpr (ValueSplit) {
                auto v_off = g_v.layout()(head_idx, int(bos) + t * CHUNK, value_slice_idx * (D / 2));
                Tensor g_v_tile = make_tensor(g_v.data() + v_off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D / 2>{}), stride(g_v.layout())));
                Tensor s_v_tile = make_tensor(
                    make_smem_ptr(shared_storage.input[stage].v.begin()), TMAValueSliceVOLayout{});
                auto cta_tma_load_v = tma_load_v_slice.get_slice(Int<0>{});
                cute::copy(tma_load_v_slice.with(*tma_barrier),
                    cta_tma_load_v.partition_S(g_v_tile), cta_tma_load_v.partition_D(s_v_tile));
            } else {
                auto v_off = g_v.layout()(head_idx, int(bos) + t * CHUNK, 0);
                Tensor g_v_tile = make_tensor(g_v.data() + v_off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_v.layout())));
                Tensor s_v_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].v.begin()), TMAVOLayout{});
                auto cta_tma_load_v = tma_load_v.get_slice(Int<0>{});
                cute::copy(tma_load_v.with(*tma_barrier),
                    cta_tma_load_v.partition_S(g_v_tile), cta_tma_load_v.partition_D(s_v_tile));
            }

            // TMA load beta (1D)
            int beta_linear = head_idx * T_total + (int(bos) + t * CHUNK);
            int beta_aligned = beta_linear & ~7;
            auto beta_off = g_beta.layout()(beta_aligned);
            Tensor g_beta_tile = make_tensor(g_beta.data() + beta_off, BetaSmemLayout{});
            Tensor s_beta_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].beta.begin()), TMABetaSmemLayout{});
            cute::copy(tma_load_beta.with(*tma_barrier),
                cta_tma_load_beta.partition_S(g_beta_tile), cta_tma_load_beta.partition_D(s_beta_tile));

            if constexpr (kFusedWorkspace) {
                // Reuse the three full-size K2 input buffers for raw q, k and
                // gate.  The MMA warps turn them in-place into q/k-decayed,
                // k-restored, INV and Mqk after the TMA barrier completes.
                auto q_off = g_q.layout()(head_idx, int(bos) + t * CHUNK, 0);
                auto q_tile = make_tensor(g_q.data() + q_off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_q.layout())));
                auto k_tile = make_tensor(g_k.data() + q_off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_k.layout())));
                auto g_tile = make_tensor(g_gate.data() + q_off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_gate.layout())));
                Tensor s_q = make_tensor(make_smem_ptr(shared_storage.input[stage].k_decayed.begin()), TMAQKLayout{});
                Tensor s_k = make_tensor(make_smem_ptr(shared_storage.input[stage].q_decayed.begin()), TMAQKLayout{});
                Tensor s_g = make_tensor(make_smem_ptr(shared_storage.input[stage].k_restored.begin()), TMAQKLayout{});
                cute::copy(tma_load_q.with(*tma_barrier), cta_tma_load_q.partition_S(q_tile), cta_tma_load_q.partition_D(s_q));
                cute::copy(tma_load_k.with(*tma_barrier), cta_tma_load_k.partition_S(k_tile), cta_tma_load_k.partition_D(s_k));
                cute::copy(tma_load_g.with(*tma_barrier), cta_tma_load_g.partition_S(g_tile), cta_tma_load_g.partition_D(s_g));
            } else {
                // TMA load workspace: k_decayed
                {
                    auto off = g_ws_kd.layout()(ws_idx, 0, 0);
                    Tensor g_tile = make_tensor(g_ws_kd.data() + off,
                        make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_ws_kd.layout())));
                    Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].k_decayed.begin()), TMAVOLayout{});
                    cute::copy(tma_load_ws_kd.with(*tma_barrier), cta_ws_kd.partition_S(g_tile), cta_ws_kd.partition_D(s_tile));
                }
                // q_decayed
                {
                    auto off = g_ws_qd.layout()(ws_idx, 0, 0);
                    Tensor g_tile = make_tensor(g_ws_qd.data() + off,
                        make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_ws_qd.layout())));
                    Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].q_decayed.begin()), TMAVOLayout{});
                    cute::copy(tma_load_ws_qd.with(*tma_barrier), cta_ws_qd.partition_S(g_tile), cta_ws_qd.partition_D(s_tile));
                }
                // k_restored
                {
                    auto off = g_ws_kr.layout()(ws_idx, 0, 0);
                    Tensor g_tile = make_tensor(g_ws_kr.data() + off,
                        make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_ws_kr.layout())));
                    Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].k_restored.begin()), TMAVOLayout{});
                    cute::copy(tma_load_ws_kr.with(*tma_barrier), cta_ws_kr.partition_S(g_tile), cta_ws_kr.partition_D(s_tile));
                }
                // g_total
                {
                    auto off = g_ws_gt.layout()(ws_idx, 0);
                    Tensor g_tile = make_tensor(g_ws_gt.data() + off,
                        make_layout(make_shape(Int<1>{}, Int<D>{}), stride(g_ws_gt.layout())));
                    Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].g_total.begin()), TMAGTotalSmemLayout{});
                    cute::copy(tma_load_ws_gt.with(*tma_barrier), cta_ws_gt.partition_S(g_tile), cta_ws_gt.partition_D(s_tile));
                }
                // INV
                {
                    auto off = g_ws_inv.layout()(ws_idx, 0, 0);
                    Tensor g_tile = make_tensor(g_ws_inv.data() + off,
                        make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<CHUNK>{}), stride(g_ws_inv.layout())));
                    Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].INV.begin()), TMALMLayout{});
                    cute::copy(tma_load_ws_inv.with(*tma_barrier), cta_ws_inv.partition_S(g_tile), cta_ws_inv.partition_D(s_tile));
                }
                // Mqk
                {
                    auto off = g_ws_mqk.layout()(ws_idx, 0, 0);
                    Tensor g_tile = make_tensor(g_ws_mqk.data() + off,
                        make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<CHUNK>{}), stride(g_ws_mqk.layout())));
                    Tensor s_tile = make_tensor(make_smem_ptr(shared_storage.input[stage].Mqk.begin()), TMALMLayout{});
                    cute::copy(tma_load_ws_mqk.with(*tma_barrier), cta_ws_mqk.partition_S(g_tile), cta_ws_mqk.partition_D(s_tile));
                }
            }

            ++load_write;
        }
        load_pipeline.producer_tail(load_write);
    }
#endif

    // --- MMA warps
    if (warp_role == WarpRole::MMA) {
        cutlass::arch::NamedBarrier compute_barrier(kComputeThreads, 0);
#ifndef TMA_DISABLE_ALL
        LoadPipelineState load_read;
        StorePipelineState out_write = cutlass::make_producer_start_state<StorePipeline>();
#endif
        int compute_tid = threadIdx.x;

        for (int t = 0; t < t_tiles; ++t) {
#ifndef TMA_DISABLE_ALL
            if constexpr (!kStateOnly && !kDirectOutput) {
                store_pipeline.producer_acquire(out_write);
            }
            load_pipeline.consumer_wait(load_read);
            int load_stage = load_read.index();
            int out_stage = (kStateOnly || kDirectOutput) ? 0 : out_write.index();
#else
            constexpr int load_stage = 0;
            constexpr int out_stage = 0;
#endif

            Tensor v_tile = make_tensor(
                make_smem_ptr(shared_storage.input[load_stage].v.begin()),
                std::conditional_t<ValueSplit, ValueSliceVOLayout, VOLayout>{});
            Tensor beta_tile = make_tensor(make_smem_ptr(shared_storage.input[load_stage].beta.begin()), BetaSmemLayout{});
            int beta_smem_offset = (head_idx * T_total + int(bos) + t * CHUNK) & 7;
            Tensor out_tile = make_tensor(
                make_smem_ptr(shared_storage.output[out_stage].out.begin()),
                std::conditional_t<ValueSplit, ValueSliceVOLayout, VOLayout>{});

            Tensor k_decayed = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_decayed.begin()), MMALayout{});
            Tensor q_decayed = make_tensor(make_smem_ptr(shared_storage.input[load_stage].q_decayed.begin()), MMALayout{});
            Tensor k_restored = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_restored.begin()), MMALayout{});
            Tensor g_total = make_tensor(make_smem_ptr(shared_storage.input[load_stage].g_total.begin()), GTotalLayout{});
            Tensor INV = make_tensor(make_smem_ptr(shared_storage.input[load_stage].INV.begin()), LMLayout{});
            Tensor Mqk = make_tensor(make_smem_ptr(shared_storage.input[load_stage].Mqk.begin()), LMLayout{});

            if constexpr (kFusedWorkspace) {
                // K1/K2 workspace fusion prototype.  The producer loaded raw
                // q/k/g into the three full-size buffers that normally carry
                // k_decayed/q_decayed/k_restored.  Reconstruct K1's prepare
                // stages here, in-place, before entering the unchanged K2
                // recurrence.  This removes both the K1 workspace stores and
                // the K2 workspace loads from the global-memory path.
                Tensor q_raw = make_tensor(
                    make_smem_ptr(shared_storage.input[load_stage].k_decayed.begin()), QKLayout{});
                Tensor k_raw = make_tensor(
                    make_smem_ptr(shared_storage.input[load_stage].q_decayed.begin()), QKLayout{});
                Tensor g_raw = make_tensor(
                    make_smem_ptr(shared_storage.input[load_stage].k_restored.begin()), GLayout{});
                Tensor qd_out = make_tensor(
                    make_smem_ptr(shared_storage.input[load_stage].q_decayed.begin()), MMALayout{});
                Tensor kd_out = make_tensor(
                    make_smem_ptr(shared_storage.input[load_stage].k_decayed.begin()), MMALayout{});
                Tensor kr_out = make_tensor(
                    make_smem_ptr(shared_storage.input[load_stage].k_restored.begin()), MMALayout{});
                Tensor ki_out = make_tensor(
                    make_smem_ptr(shared_storage.input[load_stage].k_inv.begin()), MMALayout{});
                Tensor L = make_tensor(
                    make_smem_ptr(shared_storage.input[load_stage].L.begin()), LMLayout{});
                Tensor L_fp16 = make_tensor(
                    make_smem_ptr(reinterpret_cast<FP16*>(shared_storage.input[load_stage].L.begin())), LMLayout{});

                // Match K1's per-row L2 normalization and BF16 rounding.
                constexpr int ELEMS_PER_THREAD = 8;
                constexpr int THREADS_PER_ROW = D / ELEMS_PER_THREAD;
                // 128 compute threads cover the 16 rows in two disjoint
                // passes; the warp-shuffle reduction remains identical to
                // K1's 16-thread row groups.
                for (int norm_pass = 0; norm_pass < 2; ++norm_pass) {
                    int my_row = (compute_tid / THREADS_PER_ROW) + norm_pass * 8;
                    int my_col = (compute_tid % THREADS_PER_ROW) * ELEMS_PER_THREAD;
                    float q_vals[ELEMS_PER_THREAD], k_vals[ELEMS_PER_THREAD];
                    float q_sq = 0.0f, k_sq = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
                        float qv = bf16_to_f32(q_raw(my_row, my_col + i));
                        float kv = bf16_to_f32(k_raw(my_row, my_col + i));
                        q_vals[i] = qv;
                        k_vals[i] = kv;
                        q_sq += qv * qv;
                        k_sq += kv * kv;
                    }
                    #pragma unroll
                    for (int delta = 8; delta >= 1; delta >>= 1) {
                        q_sq += __shfl_xor_sync(0xFFFFFFFF, q_sq, delta);
                        k_sq += __shfl_xor_sync(0xFFFFFFFF, k_sq, delta);
                    }
                    float q_inv_norm = rsqrtf(q_sq + 1e-6f);
                    float k_inv_norm = rsqrtf(k_sq + 1e-6f);
                    #pragma unroll
                    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
                        q_raw(my_row, my_col + i) = BF16(q_vals[i] * q_inv_norm);
                        k_raw(my_row, my_col + i) = BF16(k_vals[i] * k_inv_norm);
                    }
                }
                compute_barrier.arrive_and_wait();

                // Fused gate activation + cumulative sum.  The raw gate
                // buffer is overwritten with BF16 cumulative values, while
                // g_total retains the FP32 terminal cumulative exponent.
                if (compute_tid < D) {
                    int col = compute_tid;
                    float dt = dt_bias_ptr[head_idx * D + col];
                    float sum = 0.0f;
                    float a_log_exp = expf(A_log_ptr[head_idx]);
                    #pragma unroll
                    for (int row = 0; row < CHUNK; ++row) {
                        int global_row = t * CHUNK + row;
                        float gv = (global_row < seq_len)
                            ? bf16_to_f32(g_raw(row, col)) + dt : 0.0f;
                        gv = a_log_exp * gv;
                        gv = gate_scale * sigmoid_tanh_approx_f32(gv);
                        sum += gv;
                        g_raw(row, col) = BF16(sum);
                    }
                    g_total(col) = ex2_approx_ftz_f32(sum);
                }
                compute_barrier.arrive_and_wait();

                // Match K1's decay layout.  K2 has four MMA warps (128
                // compute threads), so each thread performs two of K1's
                // eight warp groups.  The two passes have disjoint output
                // elements and therefore need no intermediate barrier.
                static_assert(D % 64 == 0);
                static_assert(CHUNK % 8 == 0);
                constexpr int N_M = CHUNK / 8;
                constexpr int N_N = D / 64;
                constexpr int N_TILES = N_M * N_N;
                for (int pass = 0; pass < 2; ++pass) {
                    int fused_tid = compute_tid + pass * kComputeThreads;
                    int lane = fused_tid % 32;
                    int fused_warp = fused_tid / 32;
                    int g = lane / 4;
                    int tt = lane % 4;
                    float reg_g[N_TILES][2];
                    BF16 reg_q[N_TILES][2];
                    BF16 reg_k[N_TILES][2];
                    float reg_gt[N_TILES][2];

                    #pragma unroll
                    for (int m_blk = 0; m_blk < CHUNK; m_blk += 8) {
                        #pragma unroll
                        for (int n_blk = 0; n_blk < D; n_blk += 64) {
                            int tile_idx = (m_blk / 8) * N_N + (n_blk / 64);
                            int row = m_blk + ((fused_warp + g) % 8);
                            int col_base = n_blk + g * 8;
                            int col_tile = col_base / 8;
                            Tensor tile_g = local_tile(g_raw, make_shape(_1{}, _8{}), make_coord(row, col_tile));
                            Tensor tile_q = local_tile(q_raw, make_shape(_1{}, _8{}), make_coord(row, col_tile));
                            Tensor tile_k = local_tile(k_raw, make_shape(_1{}, _8{}), make_coord(row, col_tile));
                            Tensor tile_gt = local_tile(g_total, make_shape(_8{}), make_coord(col_tile));
                            Tensor s_g = local_tile(tile_g, make_shape(_1{}, _2{}), make_coord(0, tt));
                            Tensor s_q = local_tile(tile_q, make_shape(_1{}, _2{}), make_coord(0, tt));
                            Tensor s_k = local_tile(tile_k, make_shape(_1{}, _2{}), make_coord(0, tt));
                            Tensor s_gt = local_tile(tile_gt, make_shape(_2{}), make_coord(tt));
                            Tensor r_g = make_tensor_like<float>(s_g);
                            Tensor r_q = make_tensor_like<BF16>(s_q);
                            Tensor r_k = make_tensor_like<BF16>(s_k);
                            Tensor r_gt = make_tensor_like<float>(s_gt);
                            cute::copy(AutoVectorizingCopy{}, s_g, r_g);
                            cute::copy(AutoVectorizingCopy{}, s_q, r_q);
                            cute::copy(AutoVectorizingCopy{}, s_k, r_k);
                            cute::copy(AutoVectorizingCopy{}, s_gt, r_gt);
                            #pragma unroll
                            for (int v = 0; v < 2; ++v) {
                                reg_g[tile_idx][v] = r_g(0, v);
                                reg_q[tile_idx][v] = r_q(0, v);
                                reg_k[tile_idx][v] = r_k(0, v);
                                reg_gt[tile_idx][v] = r_gt(v);
                            }
                        }
                    }
                    #pragma unroll
                    for (int m_blk = 0; m_blk < CHUNK; m_blk += 8) {
                        #pragma unroll
                        for (int n_blk = 0; n_blk < D; n_blk += 64) {
                            int tile_idx = (m_blk / 8) * N_N + (n_blk / 64);
                            int row = m_blk + ((fused_warp + g) % 8);
                            int col_base = n_blk + g * 8;
                            int col_tile = col_base / 8;
                            Tensor tile_qd = local_tile(qd_out, make_shape(_1{}, _8{}), make_coord(row, col_tile));
                            Tensor tile_kd = local_tile(kd_out, make_shape(_1{}, _8{}), make_coord(row, col_tile));
                            Tensor tile_kr = local_tile(kr_out, make_shape(_1{}, _8{}), make_coord(row, col_tile));
                            Tensor tile_ki = local_tile(ki_out, make_shape(_1{}, _8{}), make_coord(row, col_tile));
                            Tensor s_qd = local_tile(tile_qd, make_shape(_1{}, _2{}), make_coord(0, tt));
                            Tensor s_kd = local_tile(tile_kd, make_shape(_1{}, _2{}), make_coord(0, tt));
                            Tensor s_kr = local_tile(tile_kr, make_shape(_1{}, _2{}), make_coord(0, tt));
                            Tensor s_ki = local_tile(tile_ki, make_shape(_1{}, _2{}), make_coord(0, tt));
                            Tensor r_qd = make_tensor_like<BF16>(s_qd);
                            Tensor r_kd = make_tensor_like<BF16>(s_kd);
                            Tensor r_ki = make_tensor_like<BF16>(s_ki);
                            Tensor r_kr = make_tensor_like<BF16>(s_kr);
                            #pragma unroll
                            for (int v = 0; v < 2; ++v) {
                                float gv = reg_g[tile_idx][v];
                                BF16 qv = reg_q[tile_idx][v];
                                BF16 kv = reg_k[tile_idx][v];
                                BF16 exp_cumsum = BF16(ex2_approx_ftz_f32(gv));
                                r_qd(0, v) = qv * exp_cumsum * BF16(scale);
                                r_kd(0, v) = kv * exp_cumsum;
                                BF16 inv_cumsum = BF16(ex2_approx_ftz_f32(-gv));
                                r_ki(0, v) = kv * inv_cumsum;
                                r_kr(0, v) = kv * inv_cumsum * BF16(reg_gt[tile_idx][v]);
                            }
                            cute::copy(AutoVectorizingCopy{}, r_qd, s_qd);
                            cute::copy(AutoVectorizingCopy{}, r_kd, s_kd);
                            cute::copy(AutoVectorizingCopy{}, r_ki, s_ki);
                            cute::copy(AutoVectorizingCopy{}, r_kr, s_kr);
                        }
                    }
                }
                compute_barrier.arrive_and_wait();

                // Form K1's L and Mqk, then run the same triangular inverse
                // and Neumann-series routine.  L is a dedicated temporary;
                // INV is first used as k_inv and then overwritten in-place by
                // the final inverse.
                if (compute_tid < 32) {
                    mma_m16n16_bf16bf16fp16_1warp(kd_out, ki_out, L_fp16, compute_tid);
                }
                if (compute_tid >= 32 && compute_tid < 64) {
                    mma_m16n16_bf16bf16bf16_1warp(qd_out, ki_out, Mqk, compute_tid - 32);
                }
                compute_barrier.arrive_and_wait();

                Tensor INV_fp16 = make_tensor(
                    make_smem_ptr(reinterpret_cast<FP16*>(shared_storage.input[load_stage].INV.begin())), LMLayout{});
                for (int inv_pass = 0; inv_pass < 2; ++inv_pass) {
                    int inv_tid = compute_tid + inv_pass * kComputeThreads;
                    const int col_block_size = 8;
                    int block_idx = inv_tid / (CHUNK * col_block_size);
                    int i = (inv_tid / col_block_size) % CHUNK;
                    int j = inv_tid % col_block_size + block_idx * col_block_size;
                    if (i <= j) {
                        L_fp16(i, j) = FP16::bitcast(0);
                    } else {
                        L_fp16(i, j) = L_fp16(i, j) * FP16(sigmoid_tanh_approx_f32(float(beta_tile(beta_smem_offset + i))));
                    }
                    if (i < j) Mqk(i, j) = BF16::bitcast(0);
                    FP16 x = L_fp16(i, j);
                    INV_fp16(i, j) = (i == j ? FP16(1.0f) - x : -x);
                }
                compute_barrier.arrive_and_wait();
                // The helper is warp-specialized and owns one 16x16 tile;
                // its own guard limits work to the first 32 threads.
                if (compute_tid < 32) {
                    neumann_inv_fused_1warp(L_fp16, INV_fp16, INV, compute_tid);
                }
                cutlass::arch::fence_view_async_shared();
                compute_barrier.arrive_and_wait();
            }

            Tensor s_acc = make_tensor(
                make_smem_ptr(shared_storage.state_acc.begin()),
                std::conditional_t<ValueSplit, ValueSliceStateSmemLayout, StateSmemLayout>{});
            Tensor s_acc_T = make_tensor(
                make_smem_ptr(shared_storage.state_acc.begin()),
                std::conditional_t<ValueSplit, TransposedValueSliceStateSmemLayout, TransposedStateSmemLayout>{});

            // Fused MMA: v_sub, v_beta, U=INV@v, out=q@s, out+=Mqk@U, s_acc_update
            // Each warp handles TWO 16x16 column blocks (N=128 / 4 warps = 32 = 2 x 16)
            // U stays in registers via SM75_U32x1_MOVM_T (no smem round-trip)
            {
            Tensor k_restored_t = make_tensor(make_smem_ptr(shared_storage.input[load_stage].k_restored.begin()), TransposedMMALayout{});

            constexpr int PREFETCH = 1;

            auto mma = make_tiled_mma(
                MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>{},
                Layout<Shape<_1,_1>>{},
                Tile<_16,_16,_16>{}
            );

            const int warp_id = compute_tid / 32;
            const int lane_id = compute_tid % 32;
            const int group_id = (lane_id / 4) % 8;

            ThrMMA thr_mma = mma.get_slice(lane_id);

            // A copy: K_INTER → LDSM_N (for k_decayed, q_decayed, INV, Mqk)
            auto smem_tiled_copy_A = make_tiled_copy_A(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_copy_A   = smem_tiled_copy_A.get_thread_slice(lane_id);

            // A copy: MN_INTER → LDSM_T (for k_restored_t in Phase 7)
            auto smem_tiled_copy_A_T = make_tiled_copy_A(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
            auto smem_thr_copy_A_T   = smem_tiled_copy_A_T.get_thread_slice(lane_id);

            // B copy: K_INTER → LDSM_N
            auto smem_tiled_copy_B = make_tiled_copy_B(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_copy_B   = smem_tiled_copy_B.get_thread_slice(lane_id);

            // C load/store
            auto smem_tiled_load_C  = make_tiled_copy_C(Copy_Atom<SM75_U32x4_LDSM_N, BF16>{}, mma);
            auto smem_thr_load_C    = smem_tiled_load_C.get_slice(lane_id);
            auto smem_tiled_store_C = make_tiled_copy_C(Copy_Atom<SM90_U32x4_STSM_N, BF16>{}, mma);
            auto smem_thr_store_C   = smem_tiled_store_C.get_slice(lane_id);

            // C load/store transposed (for Phase 6 state access via s_acc_T)
            auto smem_tiled_load_C_T  = make_tiled_copy_C(Copy_Atom<SM75_U16x8_LDSM_T, BF16>{}, mma);
            auto smem_thr_load_C_T    = smem_tiled_load_C_T.get_slice(lane_id);
            auto smem_tiled_store_C_T = make_tiled_copy_C(Copy_Atom<SM90_U16x8_STSM_T, BF16>{}, mma);
            auto smem_thr_store_C_T   = smem_tiled_store_C_T.get_slice(lane_id);

            Tensor A_ref = local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            Tensor B_ref = local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));
            Tensor C_ref = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0));

            Tensor tCrAi_k = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_k_view = smem_thr_copy_A.retile_D(tCrAi_k);
            auto tCrA_k = thr_mma.partition_fragment_A(A_ref);

            Tensor tCrAi_q = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_q_view = smem_thr_copy_A.retile_D(tCrAi_q);
            auto tCrA_q = thr_mma.partition_fragment_A(A_ref);

            Tensor tCrBi = make_fragment_like<BF16>(thr_mma.partition_fragment_B(B_ref));
            auto tCrBi_view = smem_thr_copy_B.retile_D(tCrBi);
            auto tCrB = thr_mma.partition_fragment_B(B_ref);

            auto tCrC_ref = thr_mma.partition_C(C_ref);

            using AccFragT = decltype(thr_mma.make_fragment_C(tCrC_ref));
            using SFragT = decltype(make_fragment_like<BF16>(thr_mma.make_fragment_C(tCrC_ref)));
            using AFragT = decltype(thr_mma.partition_fragment_A(A_ref));
            using BFragT_u = decltype(thr_mma.partition_fragment_B(B_ref));

            AccFragT u_acc[2], out_acc[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) { u_acc[i] = thr_mma.make_fragment_C(tCrC_ref); clear(u_acc[i]); }
            #pragma unroll
            for (int i = 0; i < 2; ++i) { out_acc[i] = thr_mma.make_fragment_C(tCrC_ref); clear(out_acc[i]); }

            // ======== Phase 1: Dual GEMM k@s and q@s (k-loop, 2 blocks per warp) ========
            constexpr int K_BLOCKS = decltype(cute::size<1>(k_decayed))::value / 16;

            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0))), tCrAi_k_view);
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, 0))), tCrAi_q_view);
            copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(value_block_base + warp_id * 2, 0))), tCrBi_view);

            #pragma unroll
            for (int k = 0; k < K_BLOCKS; ++k) {
                cute::transform(tCrAi_k, tCrA_k, cute::identity{});
                cute::transform(tCrAi_q, tCrA_q, cute::identity{});
                cute::transform(tCrBi, tCrB, cute::identity{});

                copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                    local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(value_block_base + warp_id * 2 + 1, k))), tCrBi_view);

                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), u_acc[0]);
                gemm(thr_mma, tCrA_q(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), out_acc[0]);

                cute::transform(tCrBi, tCrB, cute::identity{});

                if (k + 1 < K_BLOCKS) {
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                        local_tile(k_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_k_view);
                    copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(
                        local_tile(q_decayed, make_shape(Int<16>{}, Int<16>{}), make_coord(0, k + 1))), tCrAi_q_view);
                    copy(smem_tiled_copy_B, smem_thr_copy_B.partition_S(
                        local_tile(s_acc, make_shape(Int<16>{}, Int<16>{}), make_coord(value_block_base + warp_id * 2, k + 1))), tCrBi_view);
                }

                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), u_acc[1]);
                gemm(thr_mma, tCrA_q(_,_,Int<0>{}), tCrB(_,_,Int<0>{}), out_acc[1]);
            }

            // ======== Phase 2: Cast out (keep in regs), load v/INV/beta ========
            SFragT out_bf16[2];
            if constexpr (!kOutFp32Accum) {
                #pragma unroll
                for (int i = 0; i < 2; ++i)
                    cute::transform(out_acc[i], out_bf16[i], [] __device__ (float x) { return BF16(x); });
            }

            SFragT v_bf16[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                Tensor v_block = local_tile(v_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, value_block_base + warp_id * 2 + i));
                copy(smem_tiled_load_C, smem_thr_load_C.partition_S(v_block), smem_thr_load_C.retile_D(v_bf16[i]));
            }

            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(INV), tCrAi_k_view);
            cute::transform(tCrAi_k, tCrA_k, cute::identity{});

            BF16 beta0 = BF16(sigmoid_tanh_approx_f32(float(beta_tile(beta_smem_offset + group_id))));
            BF16 beta1 = BF16(sigmoid_tanh_approx_f32(float(beta_tile(beta_smem_offset + group_id + 8))));

            // ======== Phase 3: u = (v - u) * beta; u = INV @ u (per block) ========
            SFragT u_bf16[2];
            uint32_t u_b_regs[4];

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                cute::transform(u_acc[i], u_bf16[i], [] __device__ (float x) { return BF16(x); });

                #pragma unroll
                for (int a = 0; a < 2; ++a) {
                    #pragma unroll
                    for (int d = 0; d < 2; ++d) {
                        auto c0 = make_coord(make_coord(a, 0), 0, d);
                        auto c1 = make_coord(make_coord(a, 1), 0, d);
                        u_bf16[i](c0) = (v_bf16[i](c0) - u_bf16[i](c0)) * beta0;
                        u_bf16[i](c1) = (v_bf16[i](c1) - u_bf16[i](c1)) * beta1;
                    }
                }

                uint32_t* u_c = reinterpret_cast<uint32_t*>(&u_bf16[i](0));
                SM75_U32x1_MOVM_T::copy(u_c[0], u_b_regs[0]);
                SM75_U32x1_MOVM_T::copy(u_c[1], u_b_regs[1]);
                SM75_U32x1_MOVM_T::copy(u_c[2], u_b_regs[2]);
                SM75_U32x1_MOVM_T::copy(u_c[3], u_b_regs[3]);

                auto tCrB_u_tmp = thr_mma.partition_fragment_B(B_ref);
                uint32_t* b_dst = reinterpret_cast<uint32_t*>(&tCrB_u_tmp(0));
                b_dst[0] = u_b_regs[0]; b_dst[1] = u_b_regs[1];
                b_dst[2] = u_b_regs[2]; b_dst[3] = u_b_regs[3];

                clear(u_acc[i]);
                gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_tmp(_,_,Int<0>{}), u_acc[i]);

                cute::transform(u_acc[i], u_bf16[i], [] __device__ (float x) { return BF16(x); });
            }

            // ======== Phase 4: Load Mqk, MOVM_T → tCrB_u_arr, Mqk@U + add out ========
            copy(smem_tiled_copy_A, smem_thr_copy_A.partition_S(Mqk), tCrAi_k_view);
            cute::transform(tCrAi_k, tCrA_k, cute::identity{});

            BFragT_u tCrB_u_arr[2];

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                uint32_t* u_c = reinterpret_cast<uint32_t*>(&u_bf16[i](0));
                SM75_U32x1_MOVM_T::copy(u_c[0], u_b_regs[0]);
                SM75_U32x1_MOVM_T::copy(u_c[1], u_b_regs[1]);
                SM75_U32x1_MOVM_T::copy(u_c[2], u_b_regs[2]);
                SM75_U32x1_MOVM_T::copy(u_c[3], u_b_regs[3]);

                tCrB_u_arr[i] = thr_mma.partition_fragment_B(B_ref);
                uint32_t* b_dst = reinterpret_cast<uint32_t*>(&tCrB_u_arr[i](0));
                b_dst[0] = u_b_regs[0]; b_dst[1] = u_b_regs[1];
                b_dst[2] = u_b_regs[2]; b_dst[3] = u_b_regs[3];

                if constexpr (kOutFp32Accum) {
                    // The phase-1 q@s result is still in out_acc.  Let the
                    // second GEMM accumulate into it, avoiding an early BF16
                    // round and a separate BF16 add fragment.
                    gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_arr[i](_,_,Int<0>{}), out_acc[i]);
                    cute::transform(out_acc[i], out_bf16[i], [] __device__ (float x) { return BF16(x); });
                } else {
                    clear(out_acc[i]);
                    gemm(thr_mma, tCrA_k(_,_,Int<0>{}), tCrB_u_arr[i](_,_,Int<0>{}), out_acc[i]);

                    if constexpr (kFuseOutAdd) {
                        cute::transform(out_bf16[i], out_acc[i], out_bf16[i],
                            [] __device__ (BF16 c, float a) { return c + BF16(a); });
                    } else {
                        SFragT gemm_bf16;
                        cute::transform(out_acc[i], gemm_bf16, [] __device__ (float x) { return BF16(x); });
                        cute::transform(out_bf16[i], gemm_bf16, out_bf16[i], [] __device__ (BF16 c, BF16 a) { return c + a; });
                    }
                }
            }

            // ======== Phase 5: Store final out ========
            // R7 state-only probe: phases 1--4 still execute exactly as in
            // K2, but no output tile is written to shared memory and no STORE
            // warp output transaction is issued.  State update remains below.
            if constexpr (!kStateOnly) {
                if constexpr (kFusedTmaEpilogue) {
                    // R12-B: use the hardware STSM_N copy atom directly. The
                    // destination is logically [value,token], matching the
                    // swizzled TMA source layout; no scalar inverse-map loop
                    // or intermediate row-major tile is emitted.
                    Tensor fused_out = make_tensor(
                        make_smem_ptr(shared_storage.output[out_stage].out.begin()),
                        std::conditional_t<ValueSplit, SwizzledValueSliceLayout,
                                           SwizzledOutputLayout>{});
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        auto out_block = local_tile(
                            fused_out, make_shape(Int<16>{}, Int<16>{}),
                            make_coord(warp_id * 2 + i, 0));
                        copy(smem_tiled_store_C,
                             smem_thr_store_C.retile_S(out_bf16[i]),
                             smem_thr_store_C.partition_D(out_block));
                    }
                } else if constexpr (kTmaSwizzledOutput) {
                    // R11-B: materialize the register fragment into the
                    // swizzled [value,token] tile consumed by the matching
                    // TMA descriptor.  The fragment inverse map is the same
                    // exact map audited by R9/R10; only the destination view
                    // changes from row-major output to the swizzled SMEM view.
                    Tensor swizzled_out = make_tensor(
                        make_smem_ptr(shared_storage.output[out_stage].out.begin()),
                        std::conditional_t<ValueSplit, SwizzledValueSliceLayout,
                                           SwizzledOutputLayout>{});
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        auto src = smem_thr_store_C.retile_S(out_bf16[i]);
                        #pragma unroll
                        for (int j = 0; j < size(src); ++j) {
                            auto src_coord = idx2crd(j, shape(src));
                            const int row = (lane_id / 4) + (((j / 2) & 1) * 8);
                            const int local_col = ((j / 4) * 8) +
                                ((lane_id & 3) * 2) + (j & 1);
                            const int tile_col = (warp_id * 2 + i) * 16 + local_col;
                            const int col = ValueSplit
                                ? value_slice_idx * (D / 2) + tile_col
                                : tile_col;
                            swizzled_out(col, row) = src(src_coord);
                        }
                    }
                } else if constexpr (kDirectOutput) {
                    // R9 established the exact inverse of the K_INTER C-copy
                    // map for each 16x16 block.  Keep this first production
                    // version scalar: correctness and address ownership are
                    // easier to inspect before attempting vectorized stores.
                    const int actual_len = min(CHUNK, seq_len - t * CHUNK);
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        auto src = smem_thr_store_C.retile_S(out_bf16[i]);
                        if constexpr (kDirectOutputVec) {
                            // j={0,1}, {2,3}, {4,5}, {6,7} are four
                            // contiguous BF16 pairs in the logical output.
                            // Their row/column starts are always 4-byte
                            // aligned for the K2 16x16 tile.
                            #pragma unroll
                            for (int pair = 0; pair < 4; ++pair) {
                                const int j0 = pair * 2;
                                const int j1 = j0 + 1;
                                auto c0 = idx2crd(j0, shape(src));
                                auto c1 = idx2crd(j1, shape(src));
                                const uint32_t packed =
                                    uint32_t(src(c0).storage) |
                                    (uint32_t(src(c1).storage) << 16);
                                const int row = (lane_id / 4) + (((j0 / 2) & 1) * 8);
                                const int local_col = ((j0 / 4) * 8) + ((lane_id & 3) * 2);
                                const int tile_col = (warp_id * 2 + i) * 16 + local_col;
                                const int col = ValueSplit ? value_slice_idx * (D / 2) + tile_col : tile_col;
                                if (row < actual_len) {
                                    const int64_t global_base =
                                        (bos + t * CHUNK + row) * H * D + head_idx * D;
                                    *reinterpret_cast<uint32_t*>(out_raw_ptr + global_base + col) = packed;
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int j = 0; j < size(src); ++j) {
                                auto src_coord = idx2crd(j, shape(src));
                                const int row = (lane_id / 4) + (((j / 2) & 1) * 8);
                                const int local_col = ((j / 4) * 8) + ((lane_id & 3) * 2) + (j & 1);
                                const int tile_col = (warp_id * 2 + i) * 16 + local_col;
                                const int col = ValueSplit ? value_slice_idx * (D / 2) + tile_col : tile_col;
                                if (row < actual_len) {
                                    const int64_t global_base =
                                        (bos + t * CHUNK + row) * H * D + head_idx * D;
                                    out_raw_ptr[global_base + col] = src(src_coord);
                                }
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int i = 0; i < 2; ++i) {
                        Tensor out_block = local_tile(out_tile, make_shape(Int<16>{}, Int<16>{}), make_coord(0, value_block_base + warp_id * 2 + i));
                        copy(smem_tiled_store_C, smem_thr_store_C.retile_S(out_bf16[i]), smem_thr_store_C.partition_D(out_block));
                    }
                }
            }

            // ======== Phase 6: s_acc update ========
            // s_acc[D, D] = s_acc * g_total + k_restored_t[D, 16] @ U[16, D]
            // Each warp handles columns [warp_id*32, (warp_id+1)*32] = 2 x 16x16 blocks
            // U is already in tCrB_u_arr[0..1] as B operands (from Phase 4 MOVM_T)
            constexpr int S_M_BLOCKS = decltype(cute::size<0>(k_restored_t))::value / 16;

            Tensor tCrAi_kr = make_fragment_like<BF16>(thr_mma.partition_fragment_A(A_ref));
            auto tCrAi_kr_view = smem_thr_copy_A_T.retile_D(tCrAi_kr);

            AFragT ring_A_kr[PREFETCH];
            SFragT ring_S_acc[2][PREFETCH];
            float ring_g0[PREFETCH], ring_g1[PREFETCH];

            #pragma unroll
            for (int i = 0; i < PREFETCH; ++i) {
                Tensor kr_block = local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}), make_coord(i, 0));
                copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_block), tCrAi_kr_view);
                cute::transform(tCrAi_kr, ring_A_kr[i], cute::identity{});

                #pragma unroll
                for (int bi = 0; bi < 2; ++bi) {
                    Tensor s_block = local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}), make_coord(i, value_block_base + warp_id * 2 + bi));
                    copy(smem_tiled_load_C_T, smem_thr_load_C_T.partition_S(s_block), smem_thr_load_C_T.retile_D(ring_S_acc[bi][i]));
                }

                ring_g0[i] = g_total(i * 16 + group_id);
                ring_g1[i] = g_total(i * 16 + group_id + 8);
            }

            #pragma unroll
            for (int m = 0; m < S_M_BLOCKS; ++m) {
                const int slot = m % PREFETCH;

                float g0 = ring_g0[slot];
                float g1 = ring_g1[slot];

                #pragma unroll
                for (int bi = 0; bi < 2; ++bi) {
                    clear(u_acc[bi]);
                    gemm(thr_mma, ring_A_kr[slot](_,_,Int<0>{}), tCrB_u_arr[bi](_,_,Int<0>{}), u_acc[bi]);
                }

                if (m + PREFETCH < S_M_BLOCKS) {
                    Tensor kr_next = local_tile(k_restored_t, make_shape(Int<16>{}, Int<16>{}), make_coord(m + PREFETCH, 0));
                    copy(smem_tiled_copy_A_T, smem_thr_copy_A_T.partition_S(kr_next), tCrAi_kr_view);
                    cute::transform(tCrAi_kr, ring_A_kr[slot], cute::identity{});

                    ring_g0[slot] = g_total((m + PREFETCH) * 16 + group_id);
                    ring_g1[slot] = g_total((m + PREFETCH) * 16 + group_id + 8);
                }

                #pragma unroll
                for (int bi = 0; bi < 2; ++bi) {
                    #pragma unroll
                    for (int a = 0; a < 2; ++a) {
                        #pragma unroll
                        for (int d = 0; d < 2; ++d) {
                            auto c0 = make_coord(make_coord(a, 0), 0, d);
                            auto c1 = make_coord(make_coord(a, 1), 0, d);
                            ring_S_acc[bi][slot](c0) = BF16(bf16_to_f32(ring_S_acc[bi][slot](c0)) * g0 + u_acc[bi](c0));
                            ring_S_acc[bi][slot](c1) = BF16(bf16_to_f32(ring_S_acc[bi][slot](c1)) * g1 + u_acc[bi](c1));
                        }
                    }

                    Tensor s_block = local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}), make_coord(m, value_block_base + warp_id * 2 + bi));
                    copy(smem_tiled_store_C_T, smem_thr_store_C_T.retile_S(ring_S_acc[bi][slot]), smem_thr_store_C_T.partition_D(s_block));

                    if (m + PREFETCH < S_M_BLOCKS) {
                        Tensor s_next = local_tile(s_acc_T, make_shape(Int<16>{}, Int<16>{}), make_coord(m + PREFETCH, value_block_base + warp_id * 2 + bi));
                        copy(smem_tiled_load_C_T, smem_thr_load_C_T.partition_S(s_next), smem_thr_load_C_T.retile_D(ring_S_acc[bi][slot]));
                    }
                }
            }
            }
            compute_barrier.arrive_and_wait();

#ifndef TMA_DISABLE_ALL
            cutlass::arch::fence_view_async_shared();
            if constexpr (!kStateOnly && !kDirectOutput) {
                store_pipeline.producer_commit(out_write);
            }
            load_pipeline.consumer_release(load_read);
            ++load_read;
            if constexpr (!kStateOnly && !kDirectOutput) {
                ++out_write;
            }
#endif
        }
    }

    // In the normal path the STORE warp's output-pipeline wait naturally
    // keeps its final-state TMA store behind the MMA warps.  The R7 probe
    // removes that output wait, so provide the equivalent CTA-wide ordering
    // before the STORE warp can publish state_acc.
    if constexpr (kStateOnly || kDirectOutput) {
        __syncthreads();
    }

#ifndef TMA_DISABLE_ALL
    if (warp_role == WarpRole::STORE && lane_predicate) {
        StorePipelineState out_read;
        if constexpr (!kStateOnly && !kDirectOutput) for (int t = 0; t < t_tiles; ++t) {
            store_pipeline.consumer_wait(out_read);
            int stage = out_read.index();
            int actual_len = min(CHUNK, seq_len - t * CHUNK);

            BF16* out_stage_ptr = shared_storage.output[stage].out.begin();

            if constexpr (kTmaSwizzledOutput) {
                using SwizzledOutLayout = std::conditional_t<
                    ValueSplit, SwizzledValueSliceLayout, SwizzledOutputLayout>;
                if (actual_len < CHUNK) {
                    // TMA cannot shorten a tile.  Read the swizzled SMEM
                    // view scalarly for the one partial tile.
                    Tensor s_out = make_tensor(make_smem_ptr(out_stage_ptr), SwizzledOutLayout{});
                    int value_begin = ValueSplit ? value_slice_idx * (D / 2) : 0;
                    int value_width = ValueSplit ? D / 2 : D;
                    for (int row = 0; row < actual_len; ++row) {
                        int64_t global_base = (bos + t * CHUNK + row) * H * D + head_idx * D;
                        for (int col = 0; col < value_width; ++col) {
                            out_raw_ptr[global_base + value_begin + col] = s_out(value_begin + col, row);
                        }
                    }
                } else if constexpr (ValueSplit) {
                    Tensor g_out = tma_store_out_swizzled_slice.get_tma_tensor(make_shape(H, D, T_total));
                    auto out_off = g_out.layout()(head_idx, value_slice_idx * (D / 2), int(bos) + t * CHUNK);
                    Tensor g_out_tile = make_tensor(g_out.data() + out_off,
                        make_layout(make_shape(Int<1>{}, Int<D / 2>{}, Int<CHUNK>{}), stride(g_out.layout())));
                    Tensor s_out_tile = make_tensor(make_smem_ptr(out_stage_ptr), TMASwizzledValueSliceOutputLayout{});
                    auto cta_tma_store = tma_store_out_swizzled_slice.get_slice(Int<0>{});
                    cute::copy(tma_store_out_swizzled_slice,
                               cta_tma_store.partition_S(s_out_tile),
                               cta_tma_store.partition_D(g_out_tile));
                    tma_store_arrive();
                } else {
                    Tensor g_out = tma_store_out_swizzled.get_tma_tensor(make_shape(H, D, T_total));
                    auto out_off = g_out.layout()(head_idx, 0, int(bos) + t * CHUNK);
                    Tensor g_out_tile = make_tensor(g_out.data() + out_off,
                        make_layout(make_shape(Int<1>{}, Int<D>{}, Int<CHUNK>{}), stride(g_out.layout())));
                    Tensor s_out_tile = make_tensor(make_smem_ptr(out_stage_ptr), TMAFullSwizzledOutputLayout{});
                    auto cta_tma_store = tma_store_out_swizzled.get_slice(Int<0>{});
                    cute::copy(tma_store_out_swizzled,
                               cta_tma_store.partition_S(s_out_tile),
                               cta_tma_store.partition_D(g_out_tile));
                    tma_store_arrive();
                }
            } else if constexpr (ValueSplit) {
                if (actual_len < CHUNK) {
                    // A TMA tile cannot be shortened.  The tail has no valid
                    // neighbouring sequence columns, so retain a bounded
                    // scalar path only for this one partial tile.
                    Tensor s_out = make_tensor(make_smem_ptr(out_stage_ptr), ValueSliceVOLayout{});
                    int value_begin = value_slice_idx * (D / 2);
                    for (int row = 0; row < actual_len; ++row) {
                        int64_t global_base = (bos + t * CHUNK + row) * H * D + head_idx * D;
                        for (int col = 0; col < D / 2; ++col) {
                            out_raw_ptr[global_base + value_begin + col] = s_out(row, col);
                        }
                    }
                } else {
                    Tensor g_out = tma_store_out_slice.get_tma_tensor(make_shape(H, T_total, D));
                    auto out_off = g_out.layout()(head_idx, int(bos) + t * CHUNK, value_slice_idx * (D / 2));
                    Tensor g_out_tile = make_tensor(g_out.data() + out_off,
                        make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D / 2>{}), stride(g_out.layout())));
                    Tensor s_out_tile = make_tensor(make_smem_ptr(out_stage_ptr), TMAValueSliceVOLayout{});
                    auto cta_tma_store = tma_store_out_slice.get_slice(Int<0>{});
                    cute::copy(
                        tma_store_out_slice,
                        cta_tma_store.partition_S(s_out_tile),
                        cta_tma_store.partition_D(g_out_tile)
                    );
                    tma_store_arrive();
                }
            } else if (actual_len < CHUNK) {
                // Manual store for tail tile to avoid overwriting next sequence
                // Only one thread (lane_predicate) runs here, so loop over all D
                Tensor s_out = make_tensor(make_smem_ptr(out_stage_ptr), VOLayout{});
                for (int row = 0; row < actual_len; ++row) {
                    int64_t global_base = (bos + t * CHUNK + row) * H * D + head_idx * D;
                    for (int col = 0; col < D; ++col) {
                        out_raw_ptr[global_base + col] = s_out(row, col);
                    }
                }
            } else {
                // TMA store for full tiles
                Tensor g_out = tma_store_out.get_tma_tensor(make_shape(H, T_total, D));
                auto out_off = g_out.layout()(head_idx, int(bos) + t * CHUNK, 0);
                Tensor g_out_tile = make_tensor(g_out.data() + out_off,
                    make_layout(make_shape(Int<1>{}, Int<CHUNK>{}, Int<D>{}), stride(g_out.layout())));
                Tensor s_out_tile = make_tensor(make_smem_ptr(out_stage_ptr), TMAVOLayout{});
                auto cta_tma_store = tma_store_out.get_slice(Int<0>{});
                cute::copy(
                    tma_store_out,
                    cta_tma_store.partition_S(s_out_tile),
                    cta_tma_store.partition_D(g_out_tile)
                );
                tma_store_arrive();
            }

            tma_store_wait<0>();
            store_pipeline.consumer_release(out_read);
            ++out_read;
        }

        if constexpr (HasStateOut && !StateFP32) {
            if constexpr (ValueSplit) {
                Tensor g_final = tma_store_final_state_slice.get_tma_tensor(make_shape(N * H, D, D));
                auto state_off = g_final.layout()(seq_idx * H + head_idx, value_slice_idx * (D / 2), 0);
                Tensor g_final_tile = make_tensor(g_final.data() + state_off,
                    make_layout(make_shape(Int<1>{}, Int<D / 2>{}, Int<D>{}), stride(g_final.layout())));
                Tensor s_state = make_tensor(
                    make_smem_ptr(shared_storage.state_acc.begin()), TMAValueSliceStateSmemLayout{});
                auto cta_tma_store_state = tma_store_final_state_slice.get_slice(Int<0>{});
                cute::copy(
                    tma_store_final_state_slice,
                    cta_tma_store_state.partition_S(s_state),
                    cta_tma_store_state.partition_D(g_final_tile)
                );
                tma_store_arrive();
            } else {
                // BF16 state: TMA store directly from state_acc
                Tensor g_final = tma_store_final_state.get_tma_tensor(make_shape(N * H, D, D));
                auto state_off = g_final.layout()(seq_idx * H + head_idx, 0, 0);
                Tensor g_final_tile = make_tensor(g_final.data() + state_off,
                    make_layout(make_shape(Int<1>{}, Int<D>{}, Int<D>{}), stride(g_final.layout())));
                Tensor s_state = make_tensor(make_smem_ptr(shared_storage.state_acc.begin()), TMAStateSmemLayout{});

                auto cta_tma_store_state = tma_store_final_state.get_slice(Int<0>{});
                cute::copy(
                    tma_store_final_state,
                    cta_tma_store_state.partition_S(s_state),
                    cta_tma_store_state.partition_D(g_final_tile)
                );
                tma_store_arrive();
            }
        }
    }

    if constexpr (HasStateOut && StateFP32) {
        // FP32 state: all threads sync, convert bf16->fp32, then STORE warp does TMA
        __syncthreads();  // all warps sync — pipeline smem now free

        smem_cvt_bf16_to_fp32<
            std::conditional_t<ValueSplit, ValueSliceStateSmemLayout, StateSmemLayout>,
            std::conditional_t<ValueSplit, FP32ValueSliceStateSmemLayout, FP32StateSmemLayout>,
            ValueSplit ? D / 2 : D, D, NumThreads>(
            shared_storage.state_acc.begin(),
            reinterpret_cast<float*>(shared_storage.state_fp32_buf),
            threadIdx.x);
        cutlass::arch::fence_view_async_shared();  // generic-proxy writes -> visible to async proxy (TMA)
        __syncthreads();  // conversion complete

        if (warp_role == WarpRole::STORE && lane_predicate) {
            if constexpr (ValueSplit) {
                Tensor g_final = tma_store_final_state_slice.get_tma_tensor(make_shape(N * H, D, D));
                auto state_off = g_final.layout()(seq_idx * H + head_idx, value_slice_idx * (D / 2), 0);
                Tensor g_final_tile = make_tensor(g_final.data() + state_off,
                    make_layout(make_shape(Int<1>{}, Int<D / 2>{}, Int<D>{}), stride(g_final.layout())));
                Tensor s_fp32 = make_tensor(
                    make_smem_ptr(reinterpret_cast<float*>(shared_storage.state_fp32_buf)),
                    TMAFP32ValueSliceStateSmemLayout{});
                auto cta_tma_store_state = tma_store_final_state_slice.get_slice(Int<0>{});
                cute::copy(
                    tma_store_final_state_slice,
                    cta_tma_store_state.partition_S(s_fp32),
                    cta_tma_store_state.partition_D(g_final_tile)
                );
                tma_store_arrive();
            } else {
                Tensor g_final = tma_store_final_state.get_tma_tensor(make_shape(N * H, D, D));
                auto state_off = g_final.layout()(seq_idx * H + head_idx, 0, 0);
                Tensor g_final_tile = make_tensor(g_final.data() + state_off,
                    make_layout(make_shape(Int<1>{}, Int<D>{}, Int<D>{}), stride(g_final.layout())));
                Tensor s_fp32 = make_tensor(
                    make_smem_ptr(reinterpret_cast<float*>(shared_storage.state_fp32_buf)),
                    TMAFP32StateSmemLayout{});

                auto cta_tma_store_state = tma_store_final_state.get_slice(Int<0>{});
                cute::copy(
                    tma_store_final_state,
                    cta_tma_store_state.partition_S(s_fp32),
                    cta_tma_store_state.partition_D(g_final_tile)
                );
                tma_store_arrive();
            }
        }
    }

    __syncthreads();
#endif
}
