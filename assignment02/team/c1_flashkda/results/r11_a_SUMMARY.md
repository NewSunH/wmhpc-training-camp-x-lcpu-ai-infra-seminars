# R11-A：K2 direct-output shared storage probe

## 实验目的

R10 已经把 K2 的输出路径切换为 direct global-store。R11-A 检查
`SharedStorageK2` 中原有的 output ring 是否仍被 direct-output 路径占用，
并实现一个可以单独打开的存储布局探针。默认编译路径不改变；只有设置
`FLASH_KDA_C1_COMPACT_DIRECT_STORAGE=1` 时，direct-output 实例才省略 output
ring。

## 实现

修改文件：

- `FlashKDA/csrc/smxx/fwd_kernel2.cuh`
- `FlashKDA/setup.py`

`SharedStorageK2::OutputStorageArray` 使用 `std::conditional_t`：

- 默认、非 direct-output 和所有旧路径：仍然是 `OutputStorage[OutputStages]`；
- `C1_K2_COMPACT_DIRECT_STORAGE=1 && C1_K2_DIRECT_OUTPUT=1`：替换成一个
  128-byte 对齐、仅含 1 个 BF16 元素的占位对象。

占位对象提供 `operator[]`，使原有 direct-output 分支中的模板表达式保持可
实例化；真正的 direct-output 分支不读写 output ring。该改动只影响 opt-in
模板实例，未修改主 TeX 文档。

## 构建和正确性

构建环境：B300，`FLASH_KDA_CUDA_ARCHS=103a`，PyTorch/CUDA 环境使用仓库
`.venv`。构建日志为 `r11_a_compact_build_24617.log` 和
`r11_a_compact_rebuild2_24671.log`，后者确认 NVCC 参数包含：

```text
-DC1_VSPLIT_K2=1 -DC1_K2_DIRECT_OUTPUT=1
-DC1_K2_DIRECT_OUTPUT_VEC=1 -DC1_K2_COMPACT_DIRECT_STORAGE=1
```

正确性使用 `warmup=2, iters=3` 的 exact comparison，结果见：

- `r11_a_compact_smoke2_24628.json`
- `r11_a_compact_extended2_24628.json`

覆盖 T=16/17/64/97、H=1/4/96、BF16/FP32、state/no-state、in-only、
out-only、变长和 batch=2。所有 case 均为 `output_exact=true`、
`state_exact=true`，差异为 0。

## B300 性能

统一协议为 `warmup=20, iters=100, repeats=5`，固定形状
`[T,H,D]=[8192,96,128]`。

| 构建 | BF16 state | no-state | FP32 state |
|---|---:|---:|---:|
| R11-A compact，Vsplit，`r11_a_compact_bench2_24681.log` | 2.0172 ms | 1.9917 ms | 1.9838 ms |
| 同源 direct control（compact off），`r11_a_direct_control_bench_24703.log` | 2.0179 ms | 1.9907 ms | 1.9834 ms |
| R11-A compact，非 Vsplit，`r11_a_full_compact_24707.log` | 1.6714 ms | 1.7075 ms | 1.6457 ms |

变长补充结果记录在 `r11_a_compact_bench2_24681.log`：

- `[1300,547,2048,963,271,3063]`：BF16 1.6271 ms，no-state 1.6118 ms，FP32 1.6494 ms；
- `[1024]*8`：BF16 1.3142 ms，no-state 1.2960 ms，FP32 1.3530 ms。

## Nsight Compute / 资源观察

`r11_a_compact_ncu2_24647.log` 和对应的 `.ncu-rep` 记录了 Vsplit compact
实例：128 threads/block，80 registers/thread，dynamic shared memory
64640 B（reported shared memory 65664 B，静态 0 B）。该实例与同源 compact-off
control 的 ptxas 资源信息均为约 80 registers、9 barriers；compact storage
没有带来可测的 dynamic-smem 或性能下降/上升。

原因是当前实例的 union 大小由 input/state/pipeline 等成员主导，去掉 output
ring 并未降低最终 launch 的动态 shared-memory 配额。因而本轮结果是一个
有效的负结果：direct-output 下 output ring 确实可以在类型层面移除，但它不
是当前 B300 固定形状的性能瓶颈。

## 与历史数据的解释

R10 历史日志中的 Vsplit direct-vector 固定形状约为 1.1744 ms，而本轮同源
control 与 compact 均约 2.017 ms；这说明不能把该差异归因于 compact storage。
本轮已经用同一源码、同一构建参数，仅切换 compact 宏做了 control，二者几乎
相同。历史结果应继续作为待复核的构建/环境/源码快照差异处理，而不是作为
本轮优化收益。

## 结论

R11-A 完成了 opt-in shared-storage 重构并通过 correctness matrix，但在当前
K2 布局中没有降低实际动态 shared memory，也没有超过同源 direct control。
建议主线保留该宏作为可复现实验开关，不接入默认路径；后续优化重点应放在
TMA/布局、pipeline 和实际 kernel issue/occupancy 瓶颈，而不是继续压缩这个
已经被 union 其他成员掩盖的 output ring。
