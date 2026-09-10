# C1 实验结果

这里保留可用于报告复核的轻量日志。完整的 Nsight Compute `.ncu-rep` 和
两张卡的 SASS dump 体积较大，按仓库 `.gitignore` 留在运行环境，不纳入
git；报告中的结论同时记录了关键计数和指标。

- `bench_official_h96_{b300,5090}.log`：官方 `30/200/5` H96 benchmark。
- `bench_*_h64_*.log`、`bench_*_h96_*.log`：附加形状的轻量 benchmark。
- `test_fwd_{b300,5090}.log`：两卡正确性结果。
- `tcgen05_run_b300.log`：SM103 tcgen05 对照 GEMM 成功运行。
- `tcgen05_m16_compile.log`：故意使用 `16×16` tcgen05 tile 的编译失败。
- `tcgen05_ncu_b300.csv`：tcgen05 对照的轻量 NCU CSV。
- `chunk_range_b300.log`、`state_precision_b300.log`：数值范围和状态精度探针。
- `diagnose_5090_boundary.{log,json}`：第二轮 job 9875；RTX 5090 的 fixed/
  chunk-boundary/state-dtype exactness 与重复性诊断。
- `diagnose_b300_boundary_path.{log,json}`：第二轮 job 23442；显式加入 venv
  `bin/` 到 `PATH` 后，在 B300 上运行同一诊断的成功记录。
- `diagnose_b300_boundary_no_venv_path.log`：第二轮 job 23441；保留未把 venv
  `bin/` 加入 `PATH` 时 PyTorch JIT 找不到 Ninja 的失败，以记录环境修复过程。
- `k2_resource_limits_b300.log`：第二轮 job 23452；B300 的 SM 数、每 SM
  thread/shared-memory/register 上限，用于 K2 CTA 并行度的资源上界分析。
- `vsplit_reference_b300_t{16,17,64}.{log,json}`：第三轮的列切分 reference
  oracle。三个长度的 output 与 final state 均逐元素完全一致；对应 Slurm job 为
  23473、23478、23479。T=64 日志还保留了一次 Slurm client 通信告警，数值 JSON
  仍由脚本完整写出，不能把该次运行用于调度性能判断。
- vsplit_native_build_b300*.log：第三轮 native prototype 的三次构建尝试：job
  23491 的 bin/pip 不存在，job 23492 暴露 FP32 TMA layout 的 rank 错误，修复后
  job 23493 成功构建 flash-kda==0.0.1+c1vsplit。绝对远程路径已替换为
  <B300_REPO>。
- vsplit_native_smoke_b300.{log,json}：第三轮 job 23495 的 native exactness
  矩阵；五个 fixed-length/state 配置的 output 及（适用时）final state 均 exact。
- vsplit_native_ncu_summary_b300.csv：第三轮 job 23500 的 NCU details 提取；
  六次 recurrence capture 均为 block=128、grid-y=192、98,432 B dynamic smem，
  register/thread 为 70 或 72，active warps 为 8.12--8.14%。
- vsplit_{native,baseline}_bench_h96_b300.log：第三轮同一 B300 环境的官方
  30/200/5 fixed H96 event benchmark（job 23501 和 23504）；split prototype 在
  三个 state 变体均慢于 baseline。vsplit_baseline_build_b300.log 记录了 job
  23502 关闭编译开关后的基线重建，远程绝对路径已替换为占位符。
- `vsplit_sliced_build_{no_gpu_,}b300.log`：第四轮两次构建记录。job 23564
  未显式申请 GPU，故 `FLASH_KDA_CUDA_ARCHS=auto` 无法探测架构；job 23566
  在 `--gres=gpu:1` 下成功安装 `0.0.1+c1r4slice`。远程绝对路径已替换为
  `<B300_REPO>`。
- `vsplit_sliced_{,extended_}smoke_b300.json`：第四轮 job 23567 的位级
  exactness。前者复跑五个 R3 state 分支；后者额外覆盖 H96 的 97-token
  fixed sequence、两个 varlen 形状和两个 `B=2` batched 形状，所有 output/final
  state difference 均为零。
- `vsplit_sliced_ncu_summary_b300.csv`：第四轮 job 23569 的 six recurrence
  captures。block=128、grid-y=192，dynamic shared memory 为 68,608 B；相较 R3
  的 98,432 B 减少 29,824 B（30.3%）。
- `vsplit_sliced_bench_h96_b300.log`：第四轮 job 23570 的官方 fixed H96
  `30/200/5` CUDA-event benchmark。sliced prototype 比 R3 baseline 明显改善，
  但三种 state 变体仍未达到 baseline。
- `vsplit_sliced_baseline_build_b300.log`：第四轮 job 23571 的 GPU-backed
  baseline restore，关闭 `FLASH_KDA_C1_VSPLIT_K2` 后安装
  `0.0.1+baseline.r4`；远程绝对路径已替换为 `<B300_REPO>`。
- `tcgen05_state_sm103_run_24206.log`：第五轮 1SM `128x128x16` state-update
  probe；B300 上 warmup=10、event iterations=100，单 tile 为
  `0.0242883 ms`，CPU reference exact。
- `tcgen05_1sm_state_probe_run_24220.log`、`tcgen05_2sm_state2_run_24223.log`：
  同一 `128x128x16` 问题的 1SM/2SM 对照，分别为 `0.0242883 ms` 和
  `0.0199293 ms`，两者均 exact；后者使用 cluster `(2,1,1)`。
- `tcgen05_2sm_probe_run_24218.log`：官方 2SM TMA probe 在相同
  `512x1024x64` 问题上的 clean benchmark，`0.0335718 ms`，CPU reference
  exact；1SM 对照为 `tcgen05_1sm_state_probe_run_24220.log` 中的
  `0.039775 ms`。
- `tcgen05_{1sm,2sm,2sm_state2}_ncu_b300.csv`：第五轮 lightweight NCU
  details。1SM state probe 为 255 registers/thread、8,320 B dynamic smem；
  官方 2SM 为 255/32,896 B；最小 2SM state2 降至 181/4,224 B。NCU 的
  `gpu__time_duration` 只用于结构观察，正式比较使用 CUDA events。
- `tcgen05_direct_epilogue_run_24262.log`：第六轮 `128x128x16` 的 general
  AXPBY 与 direct alpha=1,beta=0 epilogue 对照；两者均 exact，event 分别为
  0.0212210 ms 和 0.0204546 ms。
- `tcgen05_direct_epilogue_large_24264.log`、
  `tcgen05_direct_epilogue_repeat_24265.log`：第六轮
  `512x1024x64` 的一次大问题和三次成对重复；direct 变体相对 general
  分别约快 0.8%（单次）以及 2.98%、1.99%、2.31%（重复）。
- `tcgen05_direct_epilogue_{base,direct}_ncu_24263.csv`：第六轮 NCU
  对照；direct epilogue 将 registers/thread 从 181 降至 106，dynamic
  shared memory 保持 4,224 B。NCU replay 时间仅用于结构观察。
- `tcgen05_r7_{base,direct}_stalls_24279.csv`：R7 预检的 barrier、
  long-scoreboard、wait/MIO、Tensor/TMA/TMEM 和 kernel-time counters。
  direct 使 barrier stall 从 9.74% 降至 7.80%、long-scoreboard 从
  25.45% 降至 20.84%，但 Tensor/TMA/TMEM 指令数量不变，单次 NCU replay
  时间受噪声影响反而略高；该结果只用于规划 R7，不替代 CUDA-event 重复。
- `r7_state_only_probe.py` 与 `r7_{stateonly,full}_t{16,8192}_h96.json`：
  R7 真实 K2 state-only 消融。state-only 保留 recurrence/final-state，
  关闭 output pipeline/store；有效对照的两个尺寸 final state 均与 reference
  位级 exact，T=8192,H=96 从 2.004288 ms 降到 1.621376 ms（19.11%）。
  `r7_stateonly_t16_h96.json` 是远端源码同步错误、实际加载 full binary 的
  早期记录（output 非零），不用于结论；`r7_stateonly_t16_h96_v2.json`
  是第一次正确关闭 output 但缺少 CTA-wide publication barrier 的失败，
  `r7_stateonly_barrier_t16_h96.json` 才是修正后的 T=16,H=96 结果。
- `r7_{stateonly,full}_t8192_h{1,4}.json`：门槛通过后的 H 扫描；H=1/H=4
  仍 exact，state-only median 分别为 0.483520/0.495936 ms，full 为
  0.655776/0.668160 ms，对应 26.27%/25.78% reduction。
- `r7_{stateonly,full}_ncu_{pipe,mem}_*.log`：B300 NCU replay 原始 CSV。
  pipe 对照显示 registers/thread 78->60、tensor-pipe 39936->26112、
  TMA-pipe 19968->18240；dynamic smem 均为 68608 B。TMA output store
  不表现为 SASS global-store 指令，故该计数器两边均为零，不能据此否定
  output 写回已被消融；NCU 仅用于结构解释，正式延迟以 CUDA events 为准。
- `r7_lowprio_state_precision_24316.log`：低优先级（Slurm `--nice=10000`）
  的 T=1024,H=4 补充实验。BF16/FP32 state 的 output 相同；BF16 final state
  相对 FP32 的 max/mean abs 为 4.117889/0.798589，而 FP32 state 相对 FLA
  reference 为 3.915220e-3/9.14e-5，说明状态精度仍是独立的误差维度。

## R8：完整 output epilogue 候选筛选

- `r8_out1_t8192_h96.json`、`r8_out3_t8192_h96.json` 以及
  `r8_out1_t16_h96_exact.json`、`r8_out3_t16_h96_exact.json`：B300 上
  output pipeline stage=1/3 的消融；stage=1 比同源码 stage=2 control
  慢约 6.1%，stage=3 的单次约 1.6% 优势不足以排除噪声。增强探针在
  T=16,H=96 对两种 stage 均确认 output/state exact。
- `r8_fp32out_t{16,8192}_h96*.json`：延后 BF16 rounding 的候选；T=16,H=96
  有 44,748 个 output 元素不一致（最大绝对差 256），因此拒绝。
- `r8_direct*.json`：fragment-to-global 直写的多次 CuTe layout 探针；最后
  direct7 虽保持非零计数，仍有 190,924 个 output 元素错误。中间两次
  UniversalCopy/layout 尝试的编译失败记录见
  `r8_direct_compile_failures.txt`；直写分支已从主源码移除，避免保留
  错误的 opt-in 路径。
- `r8_fuseadd_t{16,8192}_h*.json`、`r8_ctrlfull_*.json`：融合第二个
  output GEMM 的 BF16 转换与 add；split H=1/H=96 均 exact 且与 control
  持平，但 default H=96 比 control 慢约 1.5%，未设为默认。
- `r8_base{fuse,ctrl}_t8192_h96.json`：非 split default 路径的 fuse/control
  配对结果；用于确认融合候选不会因旧 binary 或 split 开关造成误判。
- `r8_default_smoke.json`：关闭全部 R8 开关、重建 `+baseline.r8` 后的
  T=16,H=1 output/state exact smoke。
- `r8_lowprio_baseline_b300.log`：低优先级 B300 H=96 基线复核；FlashKDA
  1.7843 ms，no-state 1.7839 ms，FP32-state 1.7350 ms，chunk_kda
  3.7064 ms，chunk_gated_delta 1.9965 ms。
- `r8_lowprio_bench_h1_b300.log`：低优先级 B300 H=1 复核；FlashKDA BF16
  state/no-state/FP32-state 为 1.2758/1.2759/1.2212 ms，两个 FLA 参考为
  0.5300/0.4293 ms。
- `r8_lowprio_varlen_h4_b300.log`：低优先级 B300 H=4 varlen 复核；不等长
  分段 BF16/no-state/FP32-state 为 0.5267/0.5242/0.5185 ms，八段 1024
  为 0.2139/0.2120/0.2177 ms。
- `r8_baseline_rebuild_24445.log`、`r8_baseline_smoke_24448.log` 和
  `r8_baseline_smoke_24448.json`：移除错误 direct probe 后，在 B300 GPU
  节点重建 `flash-kda==0.0.1+baseline.r8` 并完成 T=16,H=1 默认
  output/state exactness smoke。
- `r8_baseline_final_rebuild_24459.log`、`r8_baseline_final_smoke_24461.log`
  和 `r8_baseline_final_smoke_24461.json`：stage exactness 补测后再次恢复
  baseline 的最终构建与 smoke；结果仍为 output/state exact。
- `r8_5090_EVALUATION.md` 与 `r8_5090_existing_t8192_h{1,4,96}.json`：
  5090 SM120 架构确认和现有扩展探针。H=96 旧扩展 state 不 exact；R7/R8
  重建仍被远端 CUTLASS 子模块和 `Python.h` 依赖阻塞，不能宣称超过官方
  5090 baseline。
- `r8_5090_state2_*.json`、`r8_5090_state2_9928.log`：补齐 CUTLASS 后的
  SM120 构建尝试；编译链通过，但 Python editable import 仍加载旧 full
  binary，未形成可用的 state-only/full 对照。

R8 的完整命令、编译开关、exactness 判据、失败尝试及 R9 计划见
`C1_R8_EXECUTION_SECTION.tex`。`r7_state_only_probe.py` 同时扩展了
`output_equal_reference`、digest、差分元素数和绝对误差字段，便于后续完整
output 优化的自动筛选。
