"""问题 7.7（压轴）：softmax in TileLang（FROM-SCRATCH）。

contract：
- softmax(x) 接收形状 (M, N) 的 float32 CUDA tensor，返回同形状结果，
  对每一行独立做 softmax；
- kernel 用 TileLang 自己写，一个 block 处理一行（或一小批行）；
- 为了确保数值稳定，要求行内先减最大值，再做 exp 与求和。测试里有一行
  数值巨大的输入，不稳定的实现会得到 inf/nan；
- 行宽 N 任意，可以假设 N <= 4096。TileLang 的 kernel 按形状编译，
  用 make_xxx(M, N) 针对形状生成、在 wrapper 里按形状缓存编译结果
  是常见做法（结构可以参考 7.3、7.4）；
- 归约用 T.reduce_max / T.reduce_sum，逐元素部分用 T.Parallel 加 T.exp；
- fragment 的宽度建议取不小于 N 的 2 的幂（类比 Triton 的
  next_power_of_2），不足的位置补 -inf（T.if_then_else 加 T.infinity），
  否则布局推断可能报 no available layout；
- 通过 pytest tests/test_tilelang_softmax.py 即为完成。

(Optional) 将你的实现和 torch.softmax 比较一下性能（行宽取 256/1024/4096），
Tip: elementwise + 行内归约的 kernel 大概率是带宽瓶颈，可以想想理论上限是多少。
"""
# pyright: reportInvalidTypeForm=false

import torch
import tilelang
import tilelang.language as T
from tilelang import jit

_kernel_cache = {}

@jit
def make_softmax(M, N, BLOCK_N = 32):
    threads = 256
    @T.prim_func
    def kernel(X:T.Tensor((M, N), "float32"),
               Y: T.Tensor((M, N), "float32")):
        with T.Kernel(M, threads = threads) as row:
            values = T.alloc_fragment((BLOCK_N,), "float32")
            row_max = T.alloc_fragment((1,), "float32")
            row_sum = T.alloc_fragment((1,), "float32")
            for j in T.Parallel(BLOCK_N):
                values[j] = T.if_then_else(
                    j < N,
                    X[row, j],
                    -T.infinity("float32"),)
            T.reduce_max(values, row_max, dim=0)
            for j in T.Parallel(BLOCK_N):
                values[j] = T.exp(values[j] - row_max[0])
            T.reduce_sum(values, row_sum, dim=0)
            for j in T.Parallel(BLOCK_N):
                if j < N:
                    Y[row, j] = values[j] / row_sum[0]
        return kernel



            

def softmax(x: torch.Tensor) -> torch.Tensor:

    M, N = x.shape

    if N <= 0 or N > 4096:
        raise ValueError("N must satisfy 0 < N <= 4096")

    key = (M, N)

    # 编译 + 缓存都在公开函数 softmax 里
    if key not in _kernel_cache:
        _kernel_cache[key] = tilelang.compile(
            make_softmax(M, N),
            out_idx=[1],
            target="cuda",
        )

    # 调用也在 softmax 里
    return _kernel_cache[key](x)