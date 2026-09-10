#!/usr/bin/env python3
"""R7 probe for the K2 state-update/output-materialization split.

The extension is built in one of two compile-time modes:

* default: normal K2 output and final-state stores;
* ``FLASH_KDA_C1_STATE_ONLY=1``: the same recurrence and final-state store,
  but no output pipeline acquire/commit or output-store transaction.

This script intentionally measures the complete public ``flash_kda.fwd`` call
so that Kernel 1 and launch overhead remain visible.  The output buffer is
initialized to zero; in state-only mode it must remain zero, while final_state
must match the torch reference exactly.
"""

from __future__ import annotations

import argparse
import hashlib
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


D = 128
LOWER_BOUND = -5.0


def _digest(tensor: torch.Tensor) -> str:
    """Stable digest for cross-process comparison of bf16/fp32 state bytes."""
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _make_inputs(T: int, H: int, seed: int):
    torch.manual_seed(seed)
    shape = (1, T, H, D)
    q = F.normalize(torch.randn(shape, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(shape, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    beta = torch.randn((1, T, H), dtype=torch.bfloat16, device="cuda")
    a_log = torch.rand(H, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(H, D, dtype=torch.float32, device="cuda")
    initial = torch.arange(H * D * D, dtype=torch.float32, device="cuda").reshape(1, H, D, D)
    return q, k, v, g, beta, a_log, dt_bias, initial.to(torch.bfloat16)


def _run(q, k, v, g, beta, a_log, dt_bias, initial, out, final_state):
    flash_kda.fwd(
        q, k, v, g, beta, 1.0 / math.sqrt(D), out,
        A_log=a_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
        initial_state=initial, final_state=final_state,
    )


def _timed(fn, warmup: int, iters: int):
    for _ in range(max(1, warmup)):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)))
    values.sort()
    return {
        "mean_ms": sum(values) / len(values),
        "median_ms": values[len(values) // 2],
        "min_ms": values[0],
        "max_ms": values[-1],
        "samples_ms": values,
    }


def run_case(T: int, H: int, seed: int, warmup: int, iters: int):
    q, k, v, g, beta, a_log, dt_bias, initial = _make_inputs(T, H, seed)
    out = torch.zeros_like(q)
    final_state = torch.zeros_like(initial)

    # One untimed invocation provides the artifact used for exactness and the
    # digest that can be compared against a separately built baseline.
    _run(q, k, v, g, beta, a_log, dt_bias, initial.clone(), out, final_state)
    torch.cuda.synchronize()
    ref_out = torch.zeros_like(q)
    ref_state = torch.zeros_like(initial)
    torch_ref(
        q, k, v, g, beta, 1.0 / math.sqrt(D), ref_out,
        A_log=a_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
        initial_state=initial.clone(), final_state=ref_state,
    )
    state_equal_ref = bool(torch.equal(final_state, ref_state))
    output_nonzero = int(torch.count_nonzero(out).item())

    def invoke():
        _run(q, k, v, g, beta, a_log, dt_bias, initial, out, final_state)

    timing = _timed(invoke, warmup, iters)
    return {
        "T": T,
        "H": H,
        "seed": seed,
        "state_dtype": str(final_state.dtype).removeprefix("torch."),
        "state_equal_reference": state_equal_ref,
        "state_digest": _digest(final_state),
        "reference_state_digest": _digest(ref_state),
        "output_nonzero_elements": output_nonzero,
        "timing": timing,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=8192)
    ap.add_argument("--H", type=int, default=96)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    result = run_case(args.T, args.H, args.seed, args.warmup, args.iters)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if not result["state_equal_reference"]:
        raise SystemExit("R7 state-only final state is not exact")


if __name__ == "__main__":
    main()
