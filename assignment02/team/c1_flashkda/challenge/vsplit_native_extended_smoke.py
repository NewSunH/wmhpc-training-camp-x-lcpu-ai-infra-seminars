#!/usr/bin/env python3
"""Exactness coverage beyond the focused C1 native V-split smoke test.

The cases here exercise the coordinate paths that change when two CTAs own
disjoint value rows: fixed sequences crossing chunks, explicit varlen input,
and automatic equal-length batching.  Each result is an exact bitwise
comparison with ``tests.torch_ref``.
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
from tests.torch_ref import torch_ref  # noqa: E402


D, LOWER_BOUND = 128, -5.0


def make_inputs(batch: int, t_len: int, heads: int):
    shape = (batch, t_len, heads, D)
    q = F.normalize(torch.randn(shape, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(shape, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    beta = torch.randn((batch, t_len, heads), dtype=torch.bfloat16, device="cuda")
    a_log = torch.rand(heads, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(heads, D, dtype=torch.float32, device="cuda")
    return q, k, v, g, beta, a_log, dt_bias


def make_state(n_seq: int, heads: int, dtype: torch.dtype):
    return (
        torch.arange(n_seq * heads * D * D, device="cuda", dtype=torch.float32)
        .reshape(n_seq, heads, D, D)
        .to(torch.bfloat16)
        .to(dtype)
    )


def compare(
    name: str,
    batch: int,
    t_len: int,
    heads: int,
    state_dtype: torch.dtype,
    has_in: bool,
    has_out: bool,
    cu_seqlens: torch.Tensor | None = None,
):
    q, k, v, g, beta, a_log, dt_bias = make_inputs(batch, t_len, heads)
    n_seq = batch if cu_seqlens is None else cu_seqlens.numel() - 1
    initial_kernel = make_state(n_seq, heads, state_dtype) if has_in else None
    initial_ref = initial_kernel.clone() if has_in else None
    final_kernel = (
        torch.zeros(n_seq, heads, D, D, device="cuda", dtype=state_dtype)
        if has_out
        else None
    )
    final_ref = torch.zeros_like(final_kernel) if has_out else None
    out_kernel = torch.zeros_like(q)
    out_ref = torch.zeros_like(q)
    scale = 1.0 / math.sqrt(D)

    flash_kda.fwd(
        q, k, v, g, beta, scale, out_kernel,
        A_log=a_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
        initial_state=initial_kernel, final_state=final_kernel,
        cu_seqlens=cu_seqlens,
    )
    torch.cuda.synchronize()
    torch_ref(
        q, k, v, g, beta, scale, out_ref,
        A_log=a_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
        initial_state=initial_ref, final_state=final_ref,
        cu_seqlens=cu_seqlens,
    )
    return {
        "name": name,
        "B": batch,
        "T": t_len,
        "H": heads,
        "varlen": cu_seqlens is not None,
        "state_dtype": str(state_dtype).removeprefix("torch."),
        "has_state_in": has_in,
        "has_state_out": has_out,
        "output_exact": bool(torch.equal(out_kernel, out_ref)),
        "output_different": int((out_kernel != out_ref).sum().item()),
        "state_exact": final_kernel is None or bool(torch.equal(final_kernel, final_ref)),
        "state_different": 0 if final_kernel is None else int((final_kernel != final_ref).sum().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    varlen_short = torch.tensor([0, 17, 50, 115], dtype=torch.long, device="cuda")
    varlen_h96 = torch.tensor([0, 16, 33], dtype=torch.long, device="cuda")
    results = [
        compare("fixed_t97_h96_bf16_inout", 1, 97, 96, torch.bfloat16, True, True),
        compare("varlen_17_33_65_h4_bf16_inout", 1, 115, 4, torch.bfloat16, True, True, varlen_short),
        compare("varlen_16_17_h96_fp32_out_only", 1, 33, 96, torch.float32, False, True, varlen_h96),
        compare("batch_b2_t17_h4_bf16_inout", 2, 17, 4, torch.bfloat16, True, True),
        compare("batch_b2_t64_h4_fp32_inout", 2, 64, 4, torch.float32, True, True),
    ]
    result = {"seed": args.seed, "cases": results}
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if any(not row["output_exact"] or not row["state_exact"] for row in results):
        raise SystemExit("extended native V-split exactness test is not exact")


if __name__ == "__main__":
    main()
