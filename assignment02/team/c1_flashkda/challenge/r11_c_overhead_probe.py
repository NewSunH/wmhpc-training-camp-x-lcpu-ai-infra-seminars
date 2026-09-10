"""R11-C fixed-overhead diagnostic for the B300 baseline.

This probe deliberately uses the public FlashKDA Python API and CUDA events.  It
does not change the kernel.  The events cover the asynchronous GPU work issued
by one ``flash_kda.fwd`` call (prepare + recurrence); CPU allocation and Python
dispatch are not treated as kernel time.  Shape/state metadata is emitted as
JSON so the result can be joined with Nsight Compute launch statistics.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import flash_kda


LOWER_BOUND = -5.0
D = 128


def make_case(seq_lens: list[int], h: int):
    total = sum(seq_lens)
    n = len(seq_lens)
    device = torch.device("cuda")
    q = F.normalize(torch.randn((1, total, h, D), dtype=torch.float32, device=device), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn((1, total, h, D), dtype=torch.float32, device=device), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn((1, total, h, D), dtype=torch.bfloat16, device=device)
    g = torch.randn((1, total, h, D), dtype=torch.bfloat16, device=device)
    beta = torch.randn((1, total, h), dtype=torch.bfloat16, device=device)
    a_log = torch.rand(h, dtype=torch.float32, device=device)
    dt_bias = torch.rand(h, D, dtype=torch.float32, device=device)
    initial = torch.arange(n * h * D * D, dtype=torch.float32, device=device).reshape(n, h, D, D).to(torch.bfloat16)
    final = torch.zeros_like(initial)
    out = torch.zeros_like(q)
    extra = {}
    if n > 1:
        cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), dim=0).tolist()), dtype=torch.long, device=device)
        extra["cu_seqlens"] = cu
    return (q, k, v, g, beta, a_log, dt_bias, initial, final, out, extra)


def invoke(case, state_mode: str):
    q, k, v, g, beta, a_log, dt_bias, initial, final, out, extra = case
    if state_mode == "bf16_state":
        initial_arg, final_arg = initial, final
    elif state_mode == "fp32_state":
        initial_arg, final_arg = initial.float(), final.float()
    elif state_mode == "no_state":
        initial_arg = final_arg = None
    else:
        raise ValueError(state_mode)
    flash_kda.fwd(
        q, k, v, g, beta, 1.0 / math.sqrt(D), out,
        A_log=a_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
        initial_state=initial_arg, final_state=final_arg, **extra,
    )


def event_once(case, state_mode: str) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    invoke(case, state_mode)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end))


def measure(seq_lens: list[int], h: int, state_mode: str, warmup: int, iters: int, repeats: int):
    case = make_case(seq_lens, h)
    for _ in range(warmup):
        invoke(case, state_mode)
    torch.cuda.synchronize()
    values = []
    host_values = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        for _ in range(iters):
            host_start = time.perf_counter_ns()
            values.append(event_once(case, state_mode))
            host_values.append((time.perf_counter_ns() - host_start) / 1e6)
    values.sort()
    host_values.sort()
    trim = values[max(0, len(values) // 20): max(1, len(values) - len(values) // 20)]
    return {
        "seq_lens": seq_lens,
        "T_total": sum(seq_lens),
        "N": len(seq_lens),
        "H": h,
        "D": D,
        "state_mode": state_mode,
        "warmup": warmup,
        "iters": iters,
        "repeats": repeats,
        "event_ms_mean": sum(values) / len(values),
        "event_ms_median": values[len(values) // 2],
        "event_ms_min": values[0],
        "event_ms_max": values[-1],
        "event_ms_trimmed_mean": sum(trim) / len(trim),
        "host_wall_ms_median": host_values[len(host_values) // 2],
        "host_minus_event_ms_median": host_values[len(host_values) // 2] - values[len(values) // 2],
    }


def parse_cases(args):
    cases = []
    if args.mode in ("fixed", "all"):
        for t in args.fixed_t:
            cases.append(([t], args.H))
    if args.mode in ("varlen", "all"):
        cases.extend([
            ([1300, 547, 2048, 963, 271, 3063], args.H),
            ([1024] * 8, args.H),
        ])
    return cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fixed", "varlen", "all"], default="all")
    p.add_argument("--H", type=int, default=96)
    p.add_argument("--fixed-t", type=int, nargs="+", default=[16, 8192])
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.cuda.init()
    result = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "protocol": "CUDA events around one flash_kda.fwd call; prepare+recurrence; host allocation excluded from event",
        "results": [],
    }
    for seq_lens, h in parse_cases(args):
        print(f"case seq_lens={seq_lens} H={h}", flush=True)
        for state_mode in ("bf16_state", "no_state", "fp32_state"):
            row = measure(seq_lens, h, state_mode, args.warmup, args.iters, args.repeats)
            result["results"].append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
