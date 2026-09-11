#!/usr/bin/env python3
"""One H96/T8192 public forward call for diagnostic K2 profiling."""
import flash_kda
import torch
from r19_probe import make_call, make_inputs

inputs = make_inputs(8192, 96, 20260910)
call, output, state = make_call(flash_kda, inputs, "bf16")
call()
torch.cuda.synchronize()
