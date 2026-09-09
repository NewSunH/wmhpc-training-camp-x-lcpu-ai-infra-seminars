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
