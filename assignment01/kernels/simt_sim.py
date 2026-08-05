"""问题 1.6（选做）：SIMT Simulator —— 一个 warp 的执行模拟器。

不需要 GPU

contract: 实现 run(program) -> (regs, cycles)
- warp 固定 32 个 lane，lane i 的寄存器初值为 i（int）；
- program 是指令列表，指令是元组，共三种：
    ("add", k)   active lanes 的 reg += k，1 cycle
    ("mul", k)   active lanes 的 reg *= k，1 cycle
    ("if_lt", t, then_prog, else_prog)
        reg < t 的 lane 走 then_prog，其余走 else_prog。
        模拟器先带 mask 执行 then_prog，再带 mask 的补集执行
        else_prog，然后汇合。某一支没有 active lane 时整支跳过、
        不计拍。嵌套指令照常计拍（divergence 的代价就在这里）。
        if_lt 这条指令本身不计拍，拍数只来自实际执行到的 add / mul。
- 返回值 regs 是 32 个 lane 的最终寄存器值（list），cycles 是总拍数。

通过 pytest tests/test_simt_sim.py 即为完成。
"""

from typing import TypeAlias, Union
from typing import cast

Add: TypeAlias = tuple[str, int]
Mul: TypeAlias = tuple[str, int]

Program: TypeAlias = list["Instruction"]

IfLt: TypeAlias = tuple[str, int, Program, Program]

Instruction: TypeAlias = Union[
    Add,
    Mul,
    IfLt,
]
Regs: TypeAlias = list[int]
Cycles: TypeAlias = int

def run(program: Program) -> tuple[Regs, Cycles]:
    # raise NotImplementedError("从这里开始写")
    regs = list(range(32))
    mask = [1] * 32
    return execute(program, mask, regs)


def execute(program: Program, mask: list[int], regs: Regs)  -> tuple[Regs, Cycles]:
    cycles = 0
    for instruction in program:
        if instruction[0] == "add":
            isExecute = 0
            for i in range(32):
                if mask[i] == 1:
                    isExecute = 1
                    regs[i] += instruction[1]
            cycles += isExecute
        if instruction[0] == "mul":
            isExecute = 0
            for i in range(32):
                if mask[i] == 1:
                    isExecute = 1
                    regs[i] *= instruction[1]
            cycles += isExecute
        if instruction[0] == "if_lt":
            inst = cast(IfLt, instruction)
            maskThen = list(mask)
            maskElse = list(mask)
            for i in range(32):
                if regs[i] < inst[1]:
                    maskElse[i] *= 0
                else:
                    maskThen[i] *= 0
            result = execute(inst[2], maskThen, regs)
            regs = result[0]
            cycles += result[1]
            result = execute(inst[3], maskElse, regs)
            regs = result[0]
            cycles += result[1]
    return (regs, cycles)
        

