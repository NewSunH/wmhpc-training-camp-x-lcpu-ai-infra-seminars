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

## 第五轮：state-update 微内核与 2SM 对照

第五轮没有修改 FlashKDA 默认路径，而是把 tcgen05 的硬件潜力拆成可复现的
小实验：`tcgen05_sm103_gemm.cu` 增加 `C1_TCGEN05_STATE` 编译选项，将
1SM atom 改为 `128x128x16`，并以 `C1_TCGEN05_BENCH_ITERS`/
`C1_TCGEN05_BENCH_WARMUP` 控制 CUDA-event 计时；`C1_TCGEN05_QUIET` 关闭
CuTe layout dump。`tcgen05_sm103_2sm_probe.cu` 是固定 CUTLASS checkout 的
官方 2SM TMA 教程副本，增加同样的计时接口；`C1_TCGEN05_STATE2` 再将其缩小
为 `128x128x16`、cluster `(2,1,1)`、`SW32` shared-memory layout。

B300 编译使用 CUDA 13 的 `sm_103a`：

```bash
nvcc -std=c++17 -O3 -arch=sm_103a --expt-relaxed-constexpr \\
  -DC1_TCGEN05_STATE=1 -DC1_TCGEN05_QUIET=1 \\
  -I../FlashKDA/cutlass/include -I../FlashKDA/cutlass/tools/util/include \\
  -I../FlashKDA/cutlass/examples/common -I. \\
  -o tcgen05_sm103_state tcgen05_sm103_gemm_r5.cu
```

在 `512x1024x64` 上，1SM `128x128x16` tiled probe 为 `0.039775 ms`，官方
2SM `256x256x16` TMA probe 为 `0.0335718 ms`，约快 15.6%；在真正对应
`128x128x16` 的小问题上，1SM 为 `0.0242883 ms`，2SM state2 为
`0.0199293 ms`，约快 17.9%。所有 run 的 CPU reference 相对误差均为零。
NCU 显示最小 2SM state2 为 181 registers/thread、4,224 B dynamic smem；
通用 1SM probe 为 255/8,320，官方大 2SM 为 255/32,896。小问题的低吞吐
主要受 launch、TMEM epilogue 和 global store 固定成本影响，不能外推为
FlashKDA 端到端加速。

第五轮的结论是：2SM tcgen05 在 B300 上可以表达与 state update 对齐的
`128x128x16` 形状，且微内核确实比 1SM 快；但它仍需要 TMEM allocator、
cluster barrier、TMA multicast 和新的 epilogue，和 FlashKDA 现有的
`16x16` warp-local recurrence 不是 ABI 兼容的替换。由于 probe 的寄存器和
epilogue 成本已经显著，当前证据不足以冒险接入完整 K2；默认环境继续保持
`0.0.1+baseline.r4`。

## 第六轮：alpha=1/beta=0 direct epilogue 优化

第五轮 NCU 显示最小 2SM state2 为 181 registers/thread。state-update
probe 的 host 参数固定为 `alpha=1, beta=0`，因此 C tile 和 AXPBY 在这个
窄实验中是无效工作。`tcgen05_sm103_2sm_probe.cu` 新增了编译期开关
`C1_TCGEN05_DIRECT_EPILOGUE`：只保留 TMEM→RMEM 和 RMEM→GMEM，跳过
`tDrC`、C 的 global load 以及 `axpby`。默认编译和通用 epilogue 完全不变，
该开关不能用于需要 beta 或非单位 alpha 的生产路径。

B300 的编译方式为：

```bash
nvcc -std=c++17 -O3 -arch=sm_103a --expt-relaxed-constexpr \
  -DC1_TCGEN05_STATE2 -DC1_TCGEN05_QUIET \
  -DC1_TCGEN05_DIRECT_EPILOGUE \
  -I../FlashKDA/cutlass/include -I../FlashKDA/cutlass/tools/util/include \
  -I../FlashKDA/cutlass/examples/common -I. \
  -o tcgen05_sm103_2sm_state2_r6_direct tcgen05_sm103_2sm_probe_r6.cu
```

job 24262 在 `128x128x16` 上用 200 次 event、30 次 warm-up 得到：

| variant | event (ms) | correctness | registers/thread |
| --- | ---: | --- | ---: |
| general AXPBY | 0.0212210 | exact | 181 |
| direct alpha=1,beta=0 | 0.0204546 | exact | 106 |

direct epilogue 将寄存器数减少 41.4%，event 时间减少 3.6%。NCU 的单次
replay 时间为 15.584 us 对 14.592 us；该列只用来确认结构变化，正式结论
使用 CUDA event。较大的 `512x1024x64` 工作量在 job 24265 的三次成对重复
中，direct 相对 general 分别快 2.98%、1.99% 和 2.31%，但没有达到原先
设定的 10% 微内核加速门槛。两种变体的 CPU reference 均为
`Relative error = 0`。

本轮只证明“去掉无效 AXPBY 临时量可以显著降低寄存器压力”，没有证明
完整 K2 应采用这个 epilogue。K2 需要保留 recurrence 的 state 更新和
实际的写回协议；下一步若继续，应先把这个 direct 变体用于更接近 K2 的
state-only layout，再决定是否值得做完整集成。B300 默认 FlashKDA 仍保持
`0.0.1+baseline.r4`。
