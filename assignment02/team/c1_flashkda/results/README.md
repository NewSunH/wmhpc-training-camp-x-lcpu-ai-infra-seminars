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
