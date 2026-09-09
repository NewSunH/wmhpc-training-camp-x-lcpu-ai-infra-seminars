#!/usr/bin/env python3
"""Reference-level oracle for a 64+64 value-column K2 split.

This is the correctness preflight for the R3 2-CTA/head idea.  It preserves
the official torch_ref arithmetic, but computes the V dimension in two slices.
It is not a CUDA-kernel or a performance benchmark.
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
from tests.torch_ref import (  # noqa: E402
    LOG2E,
    fp32_ex2_ftz,
    fp32_fma,
    l2_normalize_kernel_match,
    matmul_fp16acc,
    sigmoid_ext,
    torch_ref,
)


CHUNK, D, LOWER_BOUND = 16, 128, -5.0


def split_reference(q, k, v, g, beta, scale, a_log, dt_bias, initial_state, split=64):
    """Match torch_ref while independently updating V columns [0:split], [split:D]."""
    assert q.shape[0] == 1 and q.shape[2] == 1 and q.shape[-1] == D
    assert 0 < split < D
    q, k = l2_normalize_kernel_match(q), l2_normalize_kernel_match(k)
    g = g.float() + dt_bias.view(1, 1, 1, D)
    a_exp = fp32_ex2_ftz(a_log * LOG2E).view(1, 1, 1, 1)
    g = LOWER_BOUND * LOG2E * sigmoid_ext.sigmoid_tanh_fp32(a_exp * g)

    t_len = q.shape[1]
    state = initial_state.to(torch.bfloat16).clone()
    out = torch.zeros_like(q)
    scale_bf16 = torch.tensor(scale, dtype=torch.bfloat16, device="cuda")
    for t0 in range(0, t_len, CHUNK):
        actual = min(CHUNK, t_len - t0)
        q_chunk = torch.zeros(CHUNK, D, dtype=torch.bfloat16, device="cuda")
        k_chunk = torch.zeros_like(q_chunk)
        v_chunk = torch.zeros_like(q_chunk)
        g_chunk = torch.zeros(CHUNK, D, dtype=torch.float32, device="cuda")
        beta_chunk = torch.zeros(CHUNK, dtype=torch.bfloat16, device="cuda")
        q_chunk[:actual], k_chunk[:actual], v_chunk[:actual] = q[0, t0:t0 + actual, 0], k[0, t0:t0 + actual, 0], v[0, t0:t0 + actual, 0]
        g_chunk[:actual], beta_chunk[:actual] = g[0, t0:t0 + actual, 0], beta[0, t0:t0 + actual, 0]

        g_cumsum = g_chunk.cumsum(0)
        g_total = g_cumsum[-1:]
        k_decayed = k_chunk * fp32_ex2_ftz(g_cumsum).to(torch.bfloat16)
        q_decayed = q_chunk * fp32_ex2_ftz(g_cumsum).to(torch.bfloat16) * scale_bf16
        k_inv = k_chunk * fp32_ex2_ftz(-g_cumsum).to(torch.bfloat16)
        k_restored = k_inv * fp32_ex2_ftz(g_total).to(torch.bfloat16)
        l_mat = torch.mm(k_decayed, k_inv.t(), out_dtype=torch.float32).to(torch.float16)
        mqk = torch.matmul(q_decayed, k_inv.t())
        beta_f32 = sigmoid_ext.sigmoid_tanh_fp32(beta_chunk.float())
        beta_bf16, beta_f16 = beta_f32.to(torch.bfloat16).unsqueeze(-1), beta_f32.to(torch.float16).unsqueeze(-1)
        l_mat = torch.tril(l_mat, diagonal=-1) * beta_f16
        mqk = torch.tril(mqk)
        inv = torch.eye(CHUNK, dtype=torch.float16, device="cuda") - l_mat
        l2 = matmul_fp16acc(l_mat, l_mat)
        inv = inv + matmul_fp16acc(inv, l2)
        l4 = matmul_fp16acc(l2, l2)
        inv = inv + matmul_fp16acc(inv, l4)
        l8 = matmul_fp16acc(l4, l4)
        inv = (inv + matmul_fp16acc(inv, l8)).to(torch.bfloat16)

        # state is [V,K].  Each V-row owns one output column and can be updated
        # without reading another V-row; q/k-side tensors remain full width.
        updated = torch.empty_like(state[0, 0])
        chunk_out = torch.empty_like(v_chunk)
        for lo in range(0, D, split):
            hi = min(lo + split, D)
            state_v = state[0, 0, lo:hi, :]
            v_v = v_chunk[:, lo:hi] - torch.matmul(k_decayed, state_v.t())
            u_v = torch.matmul(inv, v_v * beta_bf16)
            out_v = torch.matmul(q_decayed, state_v.t()) + torch.matmul(mqk, u_v)
            delta_v = torch.mm(k_restored.t(), u_v, out_dtype=torch.float32)
            next_v = fp32_fma(delta_v, state_v.float().t(), fp32_ex2_ftz(g_total).squeeze(0).unsqueeze(-1)).to(torch.bfloat16).t()
            updated[lo:hi], chunk_out[:, lo:hi] = next_v, out_v
        state[0, 0] = updated
        out[0, t0:t0 + actual, 0] = chunk_out[:actual]
    return out, state


def stats(x, y):
    d = (x.float() - y.float()).abs()
    return {"exact": bool(torch.equal(x, y)), "different": int((x != y).sum().item()), "max_abs": float(d.max().item()), "mean_abs": float(d.mean().item())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--split", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    if args.T < 1 or not 0 < args.split < D or D % args.split:
        raise SystemExit("require T>=1 and a nontrivial divisor of D=128 for --split")
    torch.manual_seed(args.seed)
    shape = (1, args.T, 1, D)
    q = F.normalize(torch.randn(shape, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(shape, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    v, g = torch.randn(shape, dtype=torch.bfloat16, device="cuda"), torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    beta = torch.randn((1, args.T, 1), dtype=torch.bfloat16, device="cuda")
    a_log, dt_bias = torch.rand(1, dtype=torch.float32, device="cuda"), torch.rand(1, D, dtype=torch.float32, device="cuda")
    initial = torch.arange(D * D, dtype=torch.float32, device="cuda").reshape(1, 1, D, D).to(torch.bfloat16)
    full_out, full_state = torch.zeros_like(q), torch.zeros_like(initial)
    torch_ref(q, k, v, g, beta, 1 / math.sqrt(D), full_out, a_log, dt_bias, LOWER_BOUND, initial.clone(), full_state)
    split_out, split_state = split_reference(q, k, v, g, beta, 1 / math.sqrt(D), a_log, dt_bias, initial, args.split)
    torch.cuda.synchronize()
    result = {"T": args.T, "split": args.split, "seed": args.seed, "output": stats(split_out, full_out), "final_state": stats(split_state, full_state)}
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["output"]["exact"] or not result["final_state"]["exact"]:
        raise SystemExit("V-split reference is not exact")


if __name__ == "__main__":
    main()
