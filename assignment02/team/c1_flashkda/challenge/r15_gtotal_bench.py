#!/usr/bin/env python3
"""CUDA-event benchmark for the minimal R15 g_total fusion candidate.

The script deliberately uses the public flash_kda.fwd call and the same input
construction for every build.  It is run once with the opt-in binary and once
after rebuilding the default binary; the JSON files are therefore paired
same-source measurements rather than comparisons with a different harness.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

FLASH_ROOT = Path(__file__).resolve().parents[1] / "FlashKDA"
sys.path.insert(0, str(FLASH_ROOT))
import flash_kda  # noqa: E402

D = 128
LOWER_BOUND = -5.0
CASES = (
    ("fixed_t16_h1", [16], 1),
    ("fixed_t16_h4", [16], 4),
    ("fixed_t16_h96", [16], 96),
    ("fixed_t8192_h1", [8192], 1),
    ("fixed_t8192_h4", [8192], 4),
    ("fixed_t8192_h96", [8192], 96),
    ("varlen_17_33_h4", [17, 33], 4),
    ("batch_like_64_h4", [64], 4),
)


def make_case(seq_lens: list[int], heads: int, seed: int):
    torch.manual_seed(seed)
    total = sum(seq_lens)
    q = F.normalize(torch.randn((1, total, heads, D), device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn((1, total, heads, D), device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn((1, total, heads, D), dtype=torch.bfloat16, device="cuda")
    g = torch.randn((1, total, heads, D), dtype=torch.bfloat16, device="cuda")
    beta = torch.randn((1, total, heads), dtype=torch.bfloat16, device="cuda")
    a_log = torch.rand(heads, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(heads, D, dtype=torch.float32, device="cuda")
    state = torch.arange(len(seq_lens) * heads * D * D, dtype=torch.float32, device="cuda").reshape(len(seq_lens), heads, D, D).to(torch.bfloat16)
    out = torch.zeros_like(q)
    final = torch.zeros_like(state)
    extra = {}
    if len(seq_lens) > 1:
        extra["cu_seqlens"] = torch.tensor([0] + list(torch.cumsum(torch.tensor(seq_lens), 0).tolist()), dtype=torch.long, device="cuda")
    def run():
        flash_kda.fwd(q, k, v, g, beta, 1.0 / math.sqrt(D), out,
                      A_log=a_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
                      initial_state=state, final_state=final, **extra)
    return run


def timed(fn, warmup: int, iters: int, repeats: int):
    for _ in range(max(1, warmup)):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for start, end in zip(starts, ends):
            start.record()
            fn()
            end.record()
        torch.cuda.synchronize()
        values.extend(float(start.elapsed_time(end)) for start, end in zip(starts, ends))
    values.sort()
    return {
        "mean_ms": sum(values) / len(values),
        "median_ms": values[len(values) // 2],
        "min_ms": values[0],
        "max_ms": values[-1],
        "samples_ms": values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    result = {"seed": args.seed, "warmup": args.warmup, "iters": args.iters, "repeats": args.repeats, "cases": []}
    for index, (name, seq_lens, heads) in enumerate(CASES):
        timing = timed(make_case(seq_lens, heads, args.seed + index), args.warmup, args.iters, args.repeats)
        row = {"name": name, "seq_lens": seq_lens, "T_total": sum(seq_lens), "H": heads, "timing": timing}
        result["cases"].append(row)
        print(f"{name}: mean={timing['mean_ms']:.6f} ms median={timing['median_ms']:.6f} ms min={timing['min_ms']:.6f} ms")
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
