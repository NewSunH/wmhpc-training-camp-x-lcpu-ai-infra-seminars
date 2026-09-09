# C1 challenge：SM100/SM103 `tcgen05` 可行性探针

这个目录不是 FlashKDA 的生产 kernel 改版，而是一个最小的、可运行的
CuTe `tcgen05.mma` 对照实验。它回答讨论点 2 的关键问题：能否把
FlashKDA 中的 `SM80_16x8x16` atom 直接替换成 SM100 atom？

## 结论先行

不能直接替换。当前 CUTLASS 的 `SM100_MMA_F16BF16_SS` 对单 CTA
cluster 的 `M` 维有静态约束 `M ∈ {64, 128}`，而 FlashKDA 的 K2
内层块是 `16×16×16`。此外，tcgen05 的累加器只能放在 TMEM，需要
TMEM allocator、barrier 以及 TMEM→寄存器的 epilogue；SM80 路径的
warp-local register accumulator 不能原样复用。

## 文件和复现

- `tcgen05_sm103_gemm.cu`：从 CUTLASS 固定版本的
  `examples/cute/tutorial/blackwell/01_mma_sm100.cu` 提取，并把 host
  端检查放宽到 SM103a，便于在 B300 上运行；使用 F16×F16→F32 的
  `128×256×16` tcgen05 atom。
- `example_utils.hpp`：该教程的辅助函数。
- `build_sm103.sh`：编译并运行脚本。远端 CUTLASS checkout 应为任务要求的
  `5c149f5`（实际完整 hash 为
  `5c149f52a436782210263fb2f19b354443a61c6a`）。

```bash
CUTLASS_ROOT=/path/to/cutlass ./build_sm103.sh 128 256 64
```

其中 `K=64` 是教程把四个 `K=16` 指令串起来的最小工作例。B300
（compute capability 10.3）实测输出 `Execution is successful.`，并且
与 CPU reference 的相对误差为 0。

故意把 atom 改成 `M=N=16` 再编译，会触发：

```text
SM100_MMA_F16BF16 M-mode size should be 64 or 128
UMMA_1SM M-mode size should be 64 or 128
```

完整编译输出保存在 `../results/tcgen05_m16_compile.log`；成功运行输出在
`../results/tcgen05_run_b300.log`。一次 Nsight Compute CSV 探针在
`../results/tcgen05_ncu_b300.csv`，只用于观察 tcgen05 kernel 的 grid/block
和内存流量，不把 profiler 的时间当作正式 benchmark（NCU 会显著扰动
单次 kernel 的时间）。

## 与 FlashKDA 的对应关系

FlashKDA 的 K2 是 `grid=(N,H)`、每 CTA 192 threads，四个 warp 各处理
两个 `16×16` 列块，动态 shared memory 约 98 KiB。若改成 tcgen05，至少
要把一个 MMA tile 扩成 `64/128 × N × 16`，重写 shared-memory swizzle、
CTA 划分和 TMEM accumulator 生命周期；这已经是算法/数据布局重构，
不是指令名替换。因此本 challenge 的负结果本身支持 v1 继续使用 SM80
路径，并把 SM100 专版留给一次独立的 v2 kernel 设计。
