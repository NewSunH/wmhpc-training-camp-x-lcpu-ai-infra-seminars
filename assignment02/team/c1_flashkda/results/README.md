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

## R9：CuTe output fragment 映射诊断

- `../challenge/r9_output_map_probe.cu`：最小 2-warp、4 个 `16×16` block
  诊断。每个 fragment 元素使用唯一 BF16 原始 bit pattern；现有
  `SM90_U32x4_STSM_N` shared-store 作为 oracle，再测试同一
  `retile_S/partition_D` map 到 row-major global tile。
- `r9_map_build_run_24501.log`：首次构建漏掉 `--expt-extended-lambda`，保留
  nvcc 的 device-lambda 失败信息。
- `r9_map_build_run_24516.log`：加入 extended-lambda 后在 B300/SM103a 编译
  运行成功；完整 tile 1024 个位置中 direct map 错 1008 个、无空洞。随后
  `r9_map_build_run_24520.log`、`r9_map_build_run_24523.log`、
  `r9_map_build_run_24524.log` 逐步补齐完整 oracle map、显式逆映射和
  7 行 tail predicate；最终 direct map 仍错 1008/1024，而显式公式为
  0/1024 mismatch，tail 为 0 个错误、0 个越界写入。

R9 证明 R8 direct 失败来自 K_INTER/SM90 STSM swizzle 与 row-major global
layout 不同构，而不是 recurrence 数值误差或 pipeline race；同时得到一个
可执行的显式 lane/fragment→row/column 公式。R10 需要把公式改造成低寄存器
实际写回，再用真实 K2 的 output/state exactness 和 CUDA events 评估收益。

## R10：真实 K2 显式 direct-output 写回

- `r10_direct_build_24545.log`：修正远端同步路径后，B300/SM103a 上启用
  `FLASH_KDA_C1_VSPLIT_K2=1`、`FLASH_KDA_C1_DIRECT_OUTPUT=1` 的真实 K2
  构建。此前 job 24534--24543 把头文件放在 `FlashKDA/` 根目录，实际编译
  仍使用旧的 `csrc/smxx/fwd_kernel2.cuh`；那些 direct/control benchmark
  只记录为无效尝试，不用于性能结论。
- `r10_direct2_smoke_24546.{log,json}`：修正路径后的 scalar 显式
  fragment→global 基础 exactness；5 个 state I/O 组合的 output 和 final state
  difference 均为零。
- `r10_scalar_ext_build_24584.log`、`r10_scalar_ext_24585.log`：scalar direct
  的扩展 exactness 补测，覆盖跨 chunk、varlen、`B=2`、BF16/FP32 state；5 个
  case 的 output 和 final state difference 均为零。
- `r10_direct2_bench_fixed_h96_b300_24547.log`：scalar V-split direct 在
  `[8192,96,128]` 上为 BF16/no-state/FP32 `2.1088/2.0659/2.0762 ms`。
- `r10_vsplit_base2_build_24549.log`、`r10_vsplit_base2_bench_fixed_h96_b300_24550.log`：
  同源码 V-split control 为 `2.0291/2.0280/2.0505 ms`，scalar direct 未能
  减少端到端延迟。
- `r10_vec_build_24551.log`、`r10_vec_{smoke,extended}_*.{log,json}`：将每个
  fragment 的四个相邻 BF16 pair 打包为 32-bit global store。所有 exactness
  矩阵仍为零差分。
- `r10_vec_bench_fixed_h96_b300_24558.log`：V-split pair-packed direct 为
  `1.1744/1.1591/1.1553 ms`，相对 V-split control 约减少 42.1%；但 V-split
  control 本身不是当前最佳路径。
- `r10_full_directvec_build_24560.log`、`r10_full_directvec_{smoke,extended}_*.{log,json}`：
  将同一显式 map 用于 non-split 四个 MMA warp，基础/扩展 exactness 仍通过。
- `r10_full_directvec_bench_fixed_h96_b300_24564.log` 和
  `r10_full_directvec_bench_varlen_h96_b300_24565.log`：non-split
  pair-packed direct 的 fixed BF16/no-state/FP32 为 `1.6779/1.7037/1.6529 ms`；
  两个 varlen case 的 BF16-state 为 `1.4910/1.2140 ms`，均未超过默认路径。
- `r10_base_build_24541.log`、`r10_base_bench_fixed_h96_b300_24542.log`：同一
  B300 环境的 non-split default 对照为 `1.0285/1.0285/0.9979 ms`。
- `r10_full_directvec_ncu_basic_24567.csv`、`r10_full_base_ncu_basic_24569.csv`：
  NCU basic set 的 resource 对照。default 为 66--74 registers/thread，
  direct 为 78--92；两者 shared-memory configuration 均为 200,704 B/block，
  achieved occupancy 均约 9.37%。direct 的 DRAM throughput 略高，不能支持
  “减少 TMA 就降低内存压力”的假设；NCU replay 时间不作为正式 benchmark。
- `r10_direct_extended_24537.json`、`r10_direct_build_24540.log`、
  `r10_direct_bench_fixed_h96_b300_24539.log` 等早期文件保留 UV 构建和错误
  同步路径，便于审计；正式结果只采用修正路径的 `24545` 之后文件。

R10 将 R9 的映射从诊断 kernel 接入真实 K2 并完成 scalar/vector 两个写回版本。
pair-packed direct 对 V-split 有明显局部收益，但 non-split default 仍快约
14.2%，且 direct path 增加寄存器、没有减少 shared-memory 配置。因此两个宏
均保持 opt-in，不进入默认构建；后续优化应转向 output storage/epilogue
结构或 launch/TMA 固定开销，而不是继续增加普通 global-store 地址计算。

## R11：shared storage、swizzled TMA 与固定开销

- `r11_a_SUMMARY.md` 及 `r11_a_*`：删除 direct-output 专用 output ring 的
  opt-in probe。T=16/17/64/97、H=1/4/96、varlen、B=2、BF16/FP32 state
  均 exact；dynamic shared memory 仍由 union 中其他成员主导，compact path
  没有稳定性能收益。
- `r11_b_probe_24689.log`：独立 128-byte `Swizzle<3,4,3>` TMA store probe，
  2048/2048 元素 exact。`r11_b_k2probe_24702.log` 保留直接套用
  `[CHUNK,D]=[16,128]` layout 的 illegal-memory-access 负 probe；随后改用
  `[D,CHUNK]=[128,16]` 逻辑 tile 才能接入完整 K2。
- `r11_b_smoke_24758.log` 与远端 `r11_b_smoke_%j.json`、
  `r11_b_extended_%j.json`：完整 swizzled K2 的 5+5 case exactness，含
  H=96、tail、varlen、B=2、BF16/FP32 state，所有 output/state difference=0。
- `r11_remote/r11_default_smoke_24786.log` 与
  `r11_remote/r11_default_smoke_%j.json`：关闭 R11 开关后的默认路径 5-case
  exactness 回归，所有 output/state difference=0。
- `r11_remote/r11_b_rebuild_base_24761.log`、
  `r11_remote/r11_b_rebuild_swz_24768.log`、
  `r11_remote/r11_bench_base_24765.log`、`r11_remote/r11_bench_swz_24772.log`：同源 B300
  H=96/T=8192 benchmark。baseline BF16/no-state/FP32 为
  1.0272/1.0285/0.9977 ms；swizzled TMA 为 1.7623/1.7615/1.7104 ms，
  回退约 71--72%，故只保留 opt-in。
- `r11_remote/r11_bench_small_24773.log` 与 `r11_remote/r11_bench_smallb_24778.log`：H=1/4
  同源对照；baseline 与 swizzled 分别约 0.733/0.748 ms，几乎无变化，
  退化集中在 H=96 的完整 output 路径。
- `r11_c_overhead_probe.py`、`r11_c_{base,vsplit}_h*.json`、
  `r11_c_ncu_summary.csv`、`r11_c_nsys_summary.csv`：CUDA-event/NCU/NSYS
  固定开销诊断。H=1/4 的 V-split 分别快 12.5%/10.8%，H=96 慢 14.1%；
  H=96 baseline grid=(1,96,1), block=192, 98,432 B dynamic smem，V-split
  grid=(1,192,1), block=128, 68,608 B。V-split 虽降低 shared memory，CTA
  翻倍和计算 warp 减少仍使 recurrence 变慢。

R11 的完整命令、失败布局、exactness 判据与判定见
`../C1_R11_EXECUTION_SECTION.tex`。远端完整日志保存在 `results/r11_remote/`；
大型 `.ncu-rep` 仍按 `.gitignore` 留在 B300 运行环境。

## R12：output 消融与 fused TMA epilogue

- `r12_remote/r12_state_*`、`r12_remote/r12_full_*`：paired full/state-only
  探针。H=96、T=8192 的同一探针中，full 约 1.785 ms，state-only 约
  1.532 ms，output 相关阶段约占 14.2%；短 T=16 的差值只有约 1.7 us。
- `r12_remote/r12_stage1_bench_24835.log`、
  `r12_remote/r12_stage3_bench_24829.log`：output pipeline stage 消融。
  stage=1 的 BF16-state 为 1.1229 ms，stage=3 为 1.0294 ms；stage=2
  baseline 为 1.0272 ms。stage=1 稳定变慢，stage=3 没有可复现收益。
- `r12_remote/r12_fused_build_24870.log`、
  `r12_remote/r12_fused_smoke_24875.log`、
  `r12_remote/r12_fused_bench_24880.log`：R12-B
  `C1_K2_FUSED_TMA_EPILOGUE`。使用 `SM90_U32x4_STSM_N` 直接写
  swizzled shared tile；10 个基础/扩展 case 全部 exact。H=96 benchmark
  BF16/no-state/FP32 为 1.0287/1.0283/0.9978 ms，与 baseline 持平，未达
  10% 接入门槛。
- `r12_remote/r12_restore_*`：关闭 R12 开关后的默认 baseline 恢复与扩展
  exactness 回归，所有 output/state difference=0。

R12 的完整命令、stage 噪声辨析、fused layout 实现和判定见
`../C1_R12_EXECUTION_SECTION.tex`。R12 fused 路径保留为 opt-in 参考，不进入
默认构建。

## R13：CHUNK=32/64 common-exponent rescale

- `../challenge/chunk_rescale_probe.py`、`chunk_rescale_b300_v3.json` 和
  `chunk_rescale_b300_v3.log`：B300 scalar range 和 factor-product probe。
  CHUNK=32 的 balanced shift 将指数范围压到 `[-91.68,91.68]`，消除原始
  factor 约 15% zero/14% inf；CHUNK=64 仍为约 `[-171.06,171.06]`，超出
  BF16 表示范围。
- CHUNK=32 的 `k_restored` mean relative diff 约 `1.86e-8`，但现有 K1
  FP16 accumulator 的 factor-product overflow proxy 约 4.6%；CHUNK=64
  约 12.5%。这不是完整 kernel exactness 或性能结果，不能宣称 rescale
  已经加速。

R13 的完整设计、命令、数值解释和未接入原因见
`../C1_R13_EXECUTION_SECTION.tex`。默认 CHUNK=16 kernel 未改变。

## R14：K1/K2 workspace fusion

- `r14_remote/r14_build_initial.log`、`r14_build_inverse_fix.log` 和
  `r14_build_barrier_fix.log`：workspace-recompute prototype 的三次构建。
  初次 inverse BF16/FP16 alias 和 CTA-wide barrier 问题修正后，SM103a
  extension 成功编译。
- `r14_remote/r14_ws_smoke_24949.json`/`.log`：K2 直接加载 q/k/g、在
  recurrence 内重建 K1 中间量的五 case smoke。虽然编译通过，五个 case
  的 output 均不 exact，带 state 的 case 同时出现 state mismatch；未进入
  benchmark。
- 默认宏 `FLASH_KDA_C1_FUSED_WS` 保持关闭，实验新增的 `k_inv/L` 临时
  storage 条件编译，不改变默认 K2 shared-memory footprint。

R14 的完整代码改动、死锁修正、exactness 数据和失败原因见
`../C1_R14_EXECUTION_SECTION.tex`。该路线目前只能作为需要重新设计
CuTe physical layout、线程映射和 pipeline 生命周期的结构性候选。

## R17：DSM/cluster V-split primitive

- `r17_cluster_launch_smoke_b300.{log,json}`：B300 job 25001 的
  `cudaLaunchKernelExC` cluster smoke。普通 kernel 与使用
  `cooperative_groups::this_cluster()` 的 cluster kernel 均 launch/sync
  成功。
- `r17_dsm_cluster_probe_b300.{log,json}`：B300 job 25002 的 DSM/duplicate
  多规模 probe。clusters=8/16/32/64、elems=1024--16384 的 20 个组合中，
  duplicate 与 DSM exactness 全部为 true；`duplicate_ms/dsm_ms` 范围为
  0.812226--0.879043，即 DSM 比 duplicate 慢约 13.8%--23.1%。

R17 的完整命令、源码修正、结果表和接入判断见
`../C1_R17_EXECUTION_SECTION.tex`。默认生产宏未改变，DSM path 未接入
FlashKDA extension。

## R15：最小 `g_total` workspace fusion

- `r15_remote/r15_gtotal_smoke.json`、`r15_gtotal_extended.json`：只跳过
  K1 的 `g_total[D]` FP32 workspace 写回，并在 K2 从原始 BF16 gate 重建。
  基础与扩展共 10 个组合全部 output/state exact，覆盖 tail、varlen、B=2、
  BF16/FP32 state 和 no-state。
- `r15_remote/r15_gtotal_candidate_bench_v2.json`、`r15_base_bench.json`：
  同源 B300 CUDA-event median 配对。短 T=16 接近持平；长序列
  `T=8192,H=96` 候选 2.3392 ms、baseline 1.7516 ms，候选慢约 33.55%（按
  时间），说明重算 gate 的成本超过节省的一次 workspace 通路。

R15 的完整实现、命令、数值判据和接入决策见
`../C1_R15_EXECUTION_SECTION.tex`；默认 `C1_K1_K2_FUSED_GTOTAL=0`。

## R16：CHUNK=32 scaled-state reference

- `r16_scaled_state_algebra_b300.{json,log}`：float64 reference 中两个连续
  CHUNK=32 的 output/state 与 raw 版本最大相对差异分别为
  `3.40e-12` 和 `3.88e-13`，证明按 key channel 对 state 做逆缩放在代数上
  可行。
- `r16_scaled_state_bench_b300.{json,log}`：目标 `H=96,T=8192,D=128` 的
  runtime operand scaling median 约 `0.7853 ms`；这是独立 microbenchmark，
  不包含完整 K1/K2/TMA/MMA，不能宣称端到端收益。

R16 的完整设计、命令、reference 和性能边界见
`../C1_R16_EXECUTION_SECTION.tex`；默认生产 CHUNK 仍为 16。

## R18：Programmatic Dependent Launch

- `r18_pdl_launch_probe_24995.{log,jsonl}`：B300 producer/consumer 独立 probe
  的 5 个 case 全部 checksum exact；PDL 相对 serialized 的缩短范围为
  `8.120%--47.044%`，取决于 trigger 后 tail 与 consumer preamble 的长度。
- `r18_flashkda_vsplit_pdl_25000.{log,json}`：value-split opt-in 的 serial/PDL
  smoke 都通过；固定 `T=8192,H=96` 的 BF16 state 为 `1.1630/1.1634 ms`，
  PDL 比 serial 慢约 `0.034%`，未形成端到端收益。job 25006 已恢复默认无
  opt-in macro 的构建。

R18 的完整命令、PDL 语义、integrated wiring 和接入判断见
`../C1_R18_EXECUTION_SECTION.tex`；默认 `C1_K1_K2_PDL=0`。


## R19：寄存器状态与流水线优化

- `r19_25211/`：原方向 full cache exact，但 H96 BF16 延迟增加 14.92%。
- `r19_25266/`：部分缓存、value-major、early-output 对照；独立映射与 FP32/BF16 探针全部零差异。
- `r19_25281/`：preload 和同步消融；最佳 `vm8pre_ew` 的 H96 BF16 延迟降低 5.956%。
- `r19_25286/`：最终独立验证，576 tests passed，4 long-sequence stress tests deselected。H96 BF16 `1.746072 → 1.643856 ms`，延迟降低 **5.854%**，速度比 **1.06218×**。H96 no-state 延迟降低 **12.971%**。

最终候选通过六个 build-time 定义启用；`challenge/r19_build_optimized.sh` 提供强制重建入口。默认宏保持关闭。`env_*` 是进程环境，不能用来反推 `.so` 编译配置；实际 module path、SHA 与 build log 才是构建身份。报告区分延迟降低 `1-new/base` 与速度提升 `base/new-1`。

## R20：补全压力回归与历史数学证据勘误

- `r20_25541/`：同形状 MMA 初次编译与 CPU 对拍通过，Graph 捕获因教程调试宏同步失败；仅保留诊断。
- `r20_25542/`：修正捕获后的初次 MMA 对照，以及 18 组修正数学/范围探针。此 MMA 构建未关闭 CuTe 断言，正式报告改用下一组 release 数据。
- `r20_25543/`：`-O3 -DNDEBUG` release 的同 BF16 128×128×16 GEMM，1/12/96 CTA；两路径各 1,785,856 元素 CPU exact。tcgen05 教程路径是 SM80 probe 的 3.397/3.433/3.363 倍延迟，仅为独立 GEMM 结论。含 build/run、源码/binary SHA 和实际 BF16 kernel 的 SASS 摘要。18 组修正 float64 分块验证再度通过，最大 O/S 误差 3.33e-16/3.55e-15。

- `r20_25537/`：另一张 B300 的 11 个 BF16-state 边界 benchmark，4 份 ABBA raw JSON、汇总、实际计时输入/输出/状态摘要；H12 T8192 native reference 额外逐位通过。主形状降延迟 6.143%，TP8 7.958%，百万 token H1 9.438%；T16/T64 和 mixed varlen 接近持平。
- `r20_25537/precision.json`：9 个真实 FP32 token reference 精度 case（弱/随机/强 gate，T256/1024/8192），同时报告仅每 16 token 保存 BF16 state 的受控诊断。kernel 总误差和状态保存误差分开解释，不能外推为模型质量结论。

- `r20_25535/`：同一 R19 最终候选 binary 的四项超长测试全部通过，JUnit 为 `4 tests, 0 failures, 0 errors, 0 skipped`，总耗时 `71.198 s`。H=1、D=128、BF16 state，fixed T=131072/1048576，varlen `[131072]`/`[524288,524288]`。与 R19 的 576 项常规测试合计，该测试文件的 580 项均已通过。耗时包含参考实现，不是 kernel benchmark。
- 新入口 `../challenge/r20_algebra_probe.py`：正确的 KDA strict-lower `(I+L)^-1`、跨时间共享的 per-key shift 和 state 补偿，分别对照 token recurrence 和未修改的 vendored FP32 函数；CPU 数学/范围诊断不产生 GPU 性能结论。

**历史解释勘误：**上文 R13 的 product-probe 统计不再用于支持合法 common-shift 方案：旧脚本在 `[tiles,C,D]` 上沿 D 求 shift，导致 shift 随 token 改变。R16 旧 reference 保留 L 对角且求 `(I-L)^-1`，与正确 KDA 的严格下三角 `(I+L)^-1` 不一致，其 raw/scaled allclose 不能作为 KDA 正确性证据。旧文件保留以便审计。R13 独立 scalar-range 观测以及 R16 特定物化缩放操作的时间仍是相应探针的数据；后者不是完整融合 kernel 的必要开销。K1 BF16 GEMM 实际使用 FP32 累加后转换 FP16；只有 Neumann inverse 使用 FP16 累加。

详见 `../C1_R20_EXECUTION_SECTION.tex`。上述勘误不影响 R19 独立的生产 kernel 正确性和 ABBA 性能数据。
