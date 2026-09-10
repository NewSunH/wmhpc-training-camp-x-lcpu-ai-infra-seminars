"""Reference and microbenchmark probe for a CHUNK=32 scaled-state design.

This is intentionally not a replacement for FlashKDA.  It checks the missing
semantic piece from R13: if K1 stores q/k factors with a per-key-channel
exponent shift, K2 must apply the inverse shift to the state operand for the
two direct state projections.  The first part uses float64 to verify the
algebra on one and two consecutive chunks.  The optional CUDA microbenchmark
measures the cost of the additional scaled operands at the target H=96,
T=8192 shape, but does not benchmark a complete K1/K2 launch.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


LOG2E = math.log2(math.e)
D = 128


def prefix_gate(g: torch.Tensor) -> torch.Tensor:
    # g is already in log2-gate units.  Keeping this operation separate makes
    # the reference's scaling relation easy to inspect.
    return g.cumsum(dim=0)


def chunk_step(q, k, v, g, beta, state, scaled: bool):
    """One mathematical KDA recurrence step; all inputs/outputs are float64."""
    s = prefix_gate(g)
    pos = torch.pow(torch.tensor(2.0, dtype=torch.float64, device=q.device), s)
    neg = torch.pow(torch.tensor(2.0, dtype=torch.float64, device=q.device), -s)
    last = s[-1]
    kd, qd = k * pos, q * pos / math.sqrt(D)
    ki = k * neg
    kr = k * neg * torch.pow(torch.tensor(2.0, dtype=torch.float64, device=q.device), last)

    if scaled:
        c = (s.amin(dim=0) + s.amax(dim=0)) * 0.5
        shift_pos = torch.pow(torch.tensor(2.0, dtype=torch.float64, device=q.device), -c)
        shift_neg = torch.pow(torch.tensor(2.0, dtype=torch.float64, device=q.device), c)
        kd, qd, ki = kd * shift_pos, qd * shift_pos, ki * shift_neg
        # state is [value,key], hence its columns are multiplied by 2**c.
        state_for_projection = state * shift_neg[None, :]
    else:
        c = torch.zeros(D, dtype=torch.float64, device=q.device)
        state_for_projection = state

    beta_val = torch.sigmoid(beta).to(torch.float64)
    L = torch.tril(kd @ ki.transpose(0, 1))
    L = L * beta_val[:, None]
    Mqk = torch.tril(qd @ ki.transpose(0, 1))
    inv = torch.linalg.inv(torch.eye(g.shape[0], dtype=torch.float64, device=q.device) - L)
    u = (v - kd @ state_for_projection.transpose(0, 1)) * beta_val[:, None]
    U = inv @ u
    out = qd @ state_for_projection.transpose(0, 1) + Mqk @ U
    decay = torch.pow(torch.tensor(2.0, dtype=torch.float64, device=q.device), last)
    new_state = (kr.transpose(0, 1) @ U + decay * state.transpose(0, 1)).transpose(0, 1)
    return out, new_state, c


def algebra_probe(device: torch.device, seed: int) -> dict:
    torch.manual_seed(seed)
    # Use the C1 D=128 dimension and a 32-token tile.  The gate distribution
    # is the same broad lower_bound=-5 distribution used by the range probe.
    q = torch.randn(64, D, dtype=torch.float64, device=device)
    k = torch.randn(64, D, dtype=torch.float64, device=device)
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    k = k / torch.linalg.vector_norm(k, dim=-1, keepdim=True)
    v = torch.randn(64, D, dtype=torch.float64, device=device)
    z = torch.randn(64, D, dtype=torch.float64, device=device)
    g = -5.0 * LOG2E * torch.sigmoid(z)
    beta = torch.randn(64, dtype=torch.float64, device=device)
    state0 = torch.randn(D, D, dtype=torch.float64, device=device) * 0.02

    raw_state = state0.clone()
    scaled_state = state0.clone()
    raw_out, scaled_out = [], []
    shifts = []
    for begin in (0, 32):
        ro, raw_state, _ = chunk_step(q[begin:begin + 32], k[begin:begin + 32],
                                      v[begin:begin + 32], g[begin:begin + 32],
                                      beta[begin:begin + 32], raw_state, False)
        so, scaled_state, c = chunk_step(q[begin:begin + 32], k[begin:begin + 32],
                                         v[begin:begin + 32], g[begin:begin + 32],
                                         beta[begin:begin + 32], scaled_state, True)
        raw_out.append(ro)
        scaled_out.append(so)
        shifts.append(c)
    ref_out = torch.cat(raw_out)
    test_out = torch.cat(scaled_out)
    out_delta = (test_out - ref_out).abs()
    state_delta = (scaled_state - raw_state).abs()
    return {
        "device": str(device),
        "seed": seed,
        "D": D,
        "chunk": 32,
        "chunks_tested": 2,
        "single_and_cross_chunk": True,
        "float64_output_max_abs_diff": float(out_delta.max().item()),
        "float64_output_max_rel_diff": float((out_delta / ref_out.abs().clamp_min(1e-30)).max().item()),
        "float64_state_max_abs_diff": float(state_delta.max().item()),
        "float64_state_max_rel_diff": float((state_delta / raw_state.abs().clamp_min(1e-30)).max().item()),
        "float64_allclose": bool(torch.allclose(test_out, ref_out, rtol=1e-11, atol=1e-11)
                                  and torch.allclose(scaled_state, raw_state, rtol=1e-11, atol=1e-11)),
        "shift_min": float(torch.cat(shifts).min().item()),
        "shift_max": float(torch.cat(shifts).max().item()),
    }


def event_time(fn, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    vals = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        vals.append(float(start.elapsed_time(end)))
    vals.sort()
    return {"mean_ms": sum(vals) / len(vals), "median_ms": vals[len(vals) // 2],
            "min_ms": vals[0], "samples_ms": vals}


def microbenchmark(device: torch.device, warmup: int, iters: int) -> dict:
    # Full target shape for elementwise scaling.  Projection benchmark is
    # intentionally one representative tile/head set: expanding state to all
    # 49,152 tiles would duplicate 1.6 GiB of state and measure a bad layout,
    # not the proposed K2 implementation.
    H, T, C = 96, 8192, 32
    tiles = T // C
    qd = torch.randn(H, tiles, C, D, dtype=torch.bfloat16, device=device)
    kd = torch.randn_like(qd)
    state = torch.randn(H, D, D, dtype=torch.bfloat16, device=device)
    c = torch.full((H, 1, D), -64.0, dtype=torch.float32, device=device)
    pos = torch.exp2(-c).to(torch.bfloat16)
    neg = torch.exp2(c).to(torch.bfloat16)
    qd_scaled = qd * pos[:, None, :, :]
    kd_scaled = kd * pos[:, None, :, :]
    state_scaled = state * neg

    def baseline_scale():
        return qd, kd, state

    def rescale_operands():
        return qd * pos[:, None, :, :], kd * pos[:, None, :, :], state * neg

    # Representative projection: 8 tiles/head; this retains state reuse by
    # broadcasting the same state per head and keeps the result inspectable.
    q_small = qd[:, :8].reshape(H * 8, C, D)
    s_small = state[:, None].expand(H, 8, D, D).reshape(H * 8, D, D)
    q_small_scaled = qd_scaled[:, :8].reshape(H * 8, C, D)
    s_small_scaled = state_scaled[:, None].expand(H, 8, D, D).reshape(H * 8, D, D)

    def baseline_projection():
        return torch.bmm(q_small, s_small.transpose(1, 2))

    def scaled_projection():
        return torch.bmm(q_small_scaled, s_small_scaled.transpose(1, 2))

    return {
        "target_shape": [H, T, D],
        "target_tiles": H * tiles,
        "fixed_shift_exp": -64.0,
        "full_target_elementwise_baseline_ms": event_time(baseline_scale, warmup, iters),
        "full_target_elementwise_rescaled_ms": event_time(rescale_operands, warmup, iters),
        "representative_projection_shape": [H * 8, C, D, D],
        "projection_baseline_ms": event_time(baseline_projection, warmup, iters),
        "projection_rescaled_ms": event_time(scaled_projection, warmup, iters),
        "note": "microbenchmark only; not a complete FlashKDA K1/K2 benchmark",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {"algebra": algebra_probe(device, args.seed)}
    if args.benchmark:
        if device.type != "cuda":
            raise SystemExit("--benchmark requires CUDA")
        result["benchmark"] = microbenchmark(device, args.warmup, args.iters)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
