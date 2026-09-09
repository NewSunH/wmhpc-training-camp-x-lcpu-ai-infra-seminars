#!/usr/bin/env python3
"""Focused exactness probe for the opt-in C1 two-CTA K2 prototype."""

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
from tests.torch_ref import torch_ref  # noqa: E402


D, LOWER_BOUND = 128, -5.0
CASES = (
    ("t16_h96_bf16_inout", 16, 96, torch.bfloat16, True, True),
    ("t17_h1_fp32_inout", 17, 1, torch.float32, True, True),
    ("t64_h1_bf16_no_state", 64, 1, torch.bfloat16, False, False),
    ("t17_h1_bf16_in_only", 17, 1, torch.bfloat16, True, False),
    ("t17_h1_fp32_out_only", 17, 1, torch.float32, False, True),
)


def make_inputs(t_len: int, heads: int):
    shape = (1, t_len, heads, D)
    q = F.normalize(torch.randn(shape, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(shape, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    beta = torch.randn((1, t_len, heads), dtype=torch.bfloat16, device="cuda")
    a_log = torch.rand(heads, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(heads, D, dtype=torch.float32, device="cuda")
    return q, k, v, g, beta, a_log, dt_bias


def make_state(heads: int, dtype: torch.dtype):
    return torch.arange(heads * D * D, device="cuda", dtype=torch.float32).reshape(1, heads, D, D).to(torch.bfloat16).to(dtype)


def compare(name, t_len, heads, state_dtype, has_in, has_out):
    q, k, v, g, beta, a_log, dt_bias = make_inputs(t_len, heads)
    initial_kernel = make_state(heads, state_dtype) if has_in else None
    initial_ref = initial_kernel.clone() if has_in else None
    final_kernel = torch.zeros(1, heads, D, D, device="cuda", dtype=state_dtype) if has_out else None
    final_ref = torch.zeros_like(final_kernel) if has_out else None
    out_kernel = torch.zeros_like(q)
    out_ref = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(D)

    flash_kda.fwd(
        q, k, v, g, beta, scale, out_kernel,
        A_log=a_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
        initial_state=initial_kernel, final_state=final_kernel,
    )
    torch.cuda.synchronize()
    torch_ref(
        q, k, v, g, beta, scale, out_ref,
        A_log=a_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
        initial_state=initial_ref, final_state=final_ref,
    )
    output_exact = bool(torch.equal(out_kernel, out_ref))
    state_exact = final_kernel is None or bool(torch.equal(final_kernel, final_ref))
    return {
        "name": name,
        "T": t_len,
        "H": heads,
        "state_dtype": str(state_dtype).removeprefix("torch."),
        "has_state_in": has_in,
        "has_state_out": has_out,
        "output_exact": output_exact,
        "output_different": int((out_kernel != out_ref).sum().item()),
        "state_exact": state_exact,
        "state_different": 0 if final_kernel is None else int((final_kernel != final_ref).sum().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    results = [compare(*case) for case in CASES]
    result = {"seed": args.seed, "cases": results}
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if any(not row["output_exact"] or not row["state_exact"] for row in results):
        raise SystemExit("native V-split prototype is not exact")


if __name__ == "__main__":
    main()
