# R8 5090 支线评估

日期：2026-09-10

## 环境

- 节点：`gj-5090-1`
- GPU：NVIDIA GeForce RTX 5090
- CUDA capability：`(12, 0)`，编译目标应为 `sm_120a`
- Python：3.10.12
- PyTorch：2.13.0+cu130
- Slurm 作业：设备诊断 `9925`，现有扩展探针 `9927`

## 现有扩展探针

使用 `challenge/r7_state_only_probe.py`，`T=8192`、seed=2027、warmup=10、iters=30。该远端副本中的现有扩展不是本轮 R8 state-only 构建，因此结果仅作为 5090 baseline/环境对照。

| H | median CUDA-event ms | state exact | output nonzero |
|---:|---:|:---:|---:|
| 1 | 0.798144 | yes | 1,048,537 |
| 4 | 0.819392 | yes | 4,194,165 |
| 96 | 2.629152 | no | 100,659,399 |

H=96 的 state digest 与 reference 不一致，不能把该现有二进制视为正确 baseline；H=1/H=4 则 exact。既有官方 benchmark 日志（不同 harness、warmup/repeat）记录 H=96 full state 平均约 2.6148 ms，和本探针的 2.6292 ms 同量级。

## 编译阻塞

尝试使用 `FLASH_KDA_CUDA_ARCHS=120a`、`FLASH_KDA_C1_VSPLIT_K2=1`、`FLASH_KDA_C1_STATE_ONLY=1` 构建，nvcc 能识别 `compute_120a,code=sm_120a`，但远端仓库副本缺少 CUTLASS 子模块，报错 `fatal error: cutlass/bfloat16.h: No such file or directory`；登录节点环境还会缺少 `Python.h`。因此本轮没有声称 5090 上已复现 R7 state-only/full 对照，也没有修改主线 kernel。

随后从 B300 实验副本转存并解包完整 CUTLASS（压缩包约 69 MB，解包后约 191 MB），在 Slurm GPU 节点 `gj-5090-1` 重新提交构建，作业 `9928`。构建阶段通过了 `sm_120a` 编译目标并生成了三份探针结果，但 output 仍非零，说明远端 editable extension/import 路径仍加载了旧的 full-output 二进制，不能作为 R7 state-only 结果；作业最后因 exactness 检查退出码 1。

对应文件为 `r8_5090_state2_t8192_h1.json`、`r8_5090_state2_t8192_h4.json`、`r8_5090_state2_t8192_h96.json` 和 `r8_5090_state2_9928.log`。这组结果只证明依赖和架构编译链已基本打通，不用于性能结论。

## 结论

5090 架构兼容性方向明确：setup.py 的架构列表已包含 `120a`，并且 nvcc 接受该目标。下一步应在具有完整 CUTLASS 和 Python 开发头文件的 GPU 构建环境中重编译 R7/R8 二进制，再以同一探针对 full 与 state-only 做 exactness 和 timing 对照。当前可用数据只支持“官方/现有扩展约 0.80/0.82/2.63 ms”的环境基线，不支持迁移优化已超过官方的结论。
