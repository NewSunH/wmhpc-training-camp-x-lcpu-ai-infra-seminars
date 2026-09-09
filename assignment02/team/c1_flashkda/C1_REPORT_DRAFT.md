# C1 FlashKDA：复现、分析与 SM100 challenge（草稿）

这份草稿记录当前已经完成的实验和仍待补齐的分析。原始日志在
`results/`，challenge 源码在 `challenge/`。

## 1. 实验环境与复现状态

| 项目 | infra5090 | infrab300 |
|---|---|---|
| GPU | RTX 5090，CC 12.0 | NVIDIA B300 SXM6，CC 10.3 |
| CUDA/driver | CUDA 13.0 / 580.142 | CUDA 13.0 / 580.126.09 |
| FlashKDA arch | `120a` | `103a` |
| PyTorch | 2.13.0+cu130 | 2.13.0+cu130 |
| Flash Linear Attention | 0.5.2 | 0.5.2 |

CUTLASS 使用任务要求的 commit：短 hash `5c149f5`，完整 hash
`5c149f52a436782210263fb2f19b354443a61c6a`。FlashKDA 源码为任务快照的
`1ce47ea`。

B300 上官方 `tests/test_fwd.py` 的 fixed 与 varlen 对拍均通过，日志见
[`test_fwd_b300.log`](results/test_fwd_b300.log)，包括：

- fixed：`T=8192,H=96,D=128`；
- varlen：`[1300,547,2048,963,271,3063]`，总长度仍为 8192；
- 对 fla/Triton 参照的误差门限测试：`Assert results: Success`。

5090 上 kernel 可以编译、运行和 benchmark；官方测试在第一处
`torch.equal` 停止。日志中的平均误差约为 `7.6e-9`，最大绝对误差为
`0.0625`，属于严格逐元素相等未满足，不能把它写成“5090 正确性通过”。
这很可能与不同架构上的舍入/FMA 顺序有关，后续应补充统一 `atol/rtol` 的
跨卡比较，而不是修改官方测试的验收条件。

## 2. 性能基线

下面 B300 的 H=96 官方形状使用任务推荐的
`warmup=30,iters=200,repeats=5`；5090 和 H=64 附加形状使用
`warmup=10,iters=50,repeats=3`，正式报告应保留这个参数差异。FlashKDA
的第一列是 bf16 state，`chunk_kda` 和 `chunk_gdn` 是 FLA 参照。

### B300（官方形状 H=96）

原始输出：[`bench_official_h96_b300.log`](results/bench_official_h96_b300.log)。

| case | FlashKDA (ms) | chunk_kda (ms) | 加速 | chunk_gdn (ms) | 加速 |
|---|---:|---:|---:|---:|---:|
| fixed | 1.0324 | 2.3723 | 2.298× | 1.3056 | 1.265× |
| varlen mixed | 0.8591 | 2.3905 | 2.783× | 1.3120 | 1.527× |
| varlen `1024×8` | 0.7003 | 2.3482 | 3.353× | 1.2618 | 1.802× |

### RTX 5090（同形状）

原始输出：[`bench_official_h96_5090.log`](results/bench_official_h96_5090.log)，
同样使用 `30/200/5`。

| case | FlashKDA (ms) | chunk_kda (ms) | 加速 | chunk_gdn (ms) | 加速 |
|---|---:|---:|---:|---:|---:|
| fixed | 2.6148 | 5.4165 | 2.071× | 3.0850 | 1.180× |
| varlen mixed | 2.3280 | 5.4629 | 2.347× | 3.1463 | 1.352× |
| varlen `1024×8` | 2.0546 | 5.4488 | 2.652× | 3.1189 | 1.518× |

H=64 的附加结果在 `bench_*_h64_*.log`；它显示形状敏感性：B300 fixed
H=64 时 FlashKDA 为 1.6077 ms，而 gdn 为 1.3337 ms（FlashKDA 不是所有
小头数形状都领先）。这可以作为“不能只报一个 speedup”的反例。

## 3. SASS 与 Nsight 证据

对两个架构生成的 `flash_kda_C` 都执行了 `cuobjdump --dump-sass`：

| SASS pattern | SM103/B300 | SM120/5090 |
|---|---:|---:|
| `HMMA.16816.F16` | 24 | 24 |
| `HMMA.16816.F32.BF16` | 1520 | 1520 |
| `WGMMA` | 0 | 0 |
| `TCGEN05` | 0 | 0 |

因此“B300 上运行”不等于“使用了 tcgen05”；当前主路径仍然是
SM80 风格 `mma.sync`（SASS 中的 HMMA）。

B300 NCU 的代表性指标（`ncu --set full`，只用于结构分析）：

- K1 prepare：block 256，grid 49152，dynamic shared memory 21.25 KiB，
  achieved occupancy 约 97%，compute throughput 约 73.6%，memory throughput
  约 71.3%。
- K2 recurrence：block 192，grid 96，dynamic shared memory 约 98.4 KiB，
  registers/thread 约 73–74，理论 occupancy 18.75%，实际约 9.37%；
  SM compute 约 20.8–21.0%，tensor-pipe active 约 19.5–20.3%，memory
  throughput 约 38%。

NCU 会改变时钟/执行开销，不能把 recurrence 的 profiled duration 当成
benchmark 时间；轻量 CSV 见 [`tcgen05_ncu_b300.csv`](results/tcgen05_ncu_b300.csv)，完整 `.ncu-rep` 仅保留在运行环境。

## 4. Challenge：只换指令路线

### 4.1 可运行的 SM103 tcgen05 对照

[`challenge/tcgen05_sm103_gemm.cu`](challenge/tcgen05_sm103_gemm.cu) 是
CUTLASS CuTe `01_mma_sm100.cu` 的最小对照版本，放宽 host 端架构检查以
允许 CC 10.3，并使用：

- F16 × F16 → F32；
- `SM100_MMA_F16BF16_SS<..., M=128, N=256>`；
- 单 CTA cluster，4 个 `K=16` 指令组成 `K=64` 的工作例；
- TMEM allocator、barrier、TMEM→register epilogue。

B300 实跑命令：

```bash
CUTLASS_ROOT=/path/to/cutlass ./challenge/build_sm103.sh 128 256 64
```

结果为 `Execution is successful.`，CPU reference 的 relative error 为 0，
见 [`tcgen05_run_b300.log`](results/tcgen05_run_b300.log)。

### 4.2 直接替换为何失败

把上面对照的 atom 参数改为 `M=N=16` 后，nvcc 在实例化阶段报：

```text
SM100_MMA_F16BF16 M-mode size should be 64 or 128
UMMA_1SM M-mode size should be 64 or 128
```

完整输出见 [`tcgen05_m16_compile.log`](results/tcgen05_m16_compile.log)。
这不是运行时调参问题，而是 tcgen05 的 tile/布局契约不允许 `16×16`。

### 4.3 当前 challenge 结论

FlashKDA K2 每个 warp 处理两个 `16×16` 列块，使用
`SM80_16x8x16_*`，累加器在寄存器中；K2 的 CTA 是 192 threads，且状态
占用约 98 KiB shared memory。tcgen05 需要至少 `M=64/128` 的大 tile，
累加器独占 TMEM，并要求重新设计：

1. K2 的 M/N tile 与 CTA/head 划分；
2. shared-memory swizzle 和 descriptor；
3. TMEM 分配、同步和 epilogue；
4. 大 tile 对 recurrence 的状态复用与并行度。

所以“只把 SM80 atom 换成 SM100 atom”没有可行的局部 patch；要做 SM100
专版必须连数据布局/算法分块一起重构。这个负结果支持 v1 保持 SM80
路径的可移植性结论，但还没有证明完整 v2 重构一定没有收益。

## 5. 尚待补齐的实验

### 5.1 已补上的 CHUNK 数值范围探针

[`challenge/chunk_range_probe.py`](challenge/chunk_range_probe.py) 按 kernel
的 `g_log2 = -5*log2(e)*sigmoid(exp(A_log)*(g+dt_bias))` 计算累计指数，
并模拟写回 bf16。B300 上 1,048,576 个独立标量、seed 0 的结果为：

| CHUNK | 累计 log2 最小值 | `exp2` 最小值 | 写回 bf16 后为 0 的比例 |
|---:|---:|---:|---:|
| 16 | -103.066 | `9.419e-32` | 0.000000 |
| 32 | -189.270 | 0（已下溢） | 0.098475 |
| 64 | -347.805 | 0（已下溢） | 0.547464 |

这是范围探针，不是完整 kernel 对拍；它已经量化地说明在当前
`lower_bound=-5` 和 bf16 workspace 下，CHUNK=32 开始出现明显指数下溢，
CHUNK=64 更严重。因而若要采用大 CHUNK，必须加入 intra-chunk rescale
或改用更高精度存储，不能只改一个常量。

### 5.2 状态精度初步验证

[`challenge/state_precision_probe.py`](challenge/state_precision_probe.py) 在
B300 上用 `T=1024,H=4,D=128` 固定 seed 对比了 bf16-state、fp32-state 和
FLA Triton 参照。FlashKDA 的两种 state 接口输出逐元素相同（output 与
final state 的 max/mean diff 都为 0）；相对 FLA 的代表性误差为：

| 比较 | max abs | mean abs |
|---|---:|---:|
| FlashKDA output vs FLA | `4.883e-4` | `1.290e-5` |
| FlashKDA final state vs FLA | `3.915e-3` | `9.136e-5` |

这不是 fp32 在内部保留了更多精度：源码在 `StateFP32` 分支明确先把
global fp32 state 转成 shared-memory bf16，递推结束后再转回 fp32。因此
该选项只改变 I/O 格式，内部 recurrent state 仍是 bf16；报告中应把它
表述为“接口精度验证”，而不是“fp32 内部状态对照”。

### 5.3 下一轮可以推进的实验

这些是下一轮可以并行推进的项目：

- CHUNK=32/64：先只在参考实现中做数值范围和逆矩阵误差扫描，再决定是否
  改 kernel；记录首次失败位置、shared memory、寄存器和耗时。
- 状态精度：固定输入种子，比较 bf16/fp32 state 的输出、最终 state 的
  max/mean error，并按 token 画误差随序列长度增长曲线。
- 并行度：评估“多 head/CTA、persistent、2-CTA”三种候选，先做 shared
  memory/register footprint 纸面上界，再选一个能保持状态依赖正确性的
  prototype。
- 5090 跨架构校验：将官方 `torch.equal` 改为只用于诊断的 allclose 统计，
  分别报告输出和 final state 的误差分布。
