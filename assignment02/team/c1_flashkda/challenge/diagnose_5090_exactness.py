#!/usr/bin/env python3
"""Diagnose a non-exact FlashKDA-vs-torch_ref result without changing the kernel.

This is a reproducibility probe for C1 round 2.  It deliberately uses the
official fixed-length input convention and the official ``torch_ref``.  The
output is compact JSON plus human-readable lines, so a Slurm log is enough to
audit a run.  It is not a performance benchmark.

Example (from FlashKDA/ on an allocated GPU)::

    python ../challenge/diagnose_5090_exactness.py \
      --preset boundary --repeats 3 \
      --json ../results/diagnose_5090_boundary.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


# The script lives in challenge/, while the extension and the official
# reference live under the sibling FlashKDA/ snapshot.
FLASH_ROOT = Path(__file__).resolve().parents[1] / "FlashKDA"
if str(FLASH_ROOT) not in sys.path:
    sys.path.insert(0, str(FLASH_ROOT))

import flash_kda  # noqa: E402
from tests.torch_ref import torch_ref  # noqa: E402


D = 128
LOWER_BOUND = -5.0


@dataclass(frozen=True)
class Case:
    """One exactness experiment; T=16/17 explicitly brackets a chunk boundary."""

    name: str
    t: int
    h: int
    state_dtype: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        choices=("smoke", "boundary", "canonical"),
        default="boundary",
        help=(
            "smoke: two small cases; boundary: T=16/17/64/1024 plus canonical "
            "H and dtype checks; canonical: only T=8192,H=96,bf16"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="number of identical kernel executions per case (>=2 required)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="optional JSON summary path; parent directory must already exist",
    )
    return parser.parse_args()


def preset_cases(name: str) -> list[Case]:
    # H=1 tests whether the discrepancy depends on the number of K2 CTAs.
    # fp32 state changes the API storage type but not the current bf16 recurrence.
    if name == "smoke":
        return [Case("one_chunk_h96", 16, 96, "bf16"), Case("two_chunks_h96", 17, 96, "bf16")]
    if name == "canonical":
        return [Case("official_fixed_h96", 8192, 96, "bf16")]
    return [
        Case("one_chunk_h96", 16, 96, "bf16"),
        Case("two_chunks_h96", 17, 96, "bf16"),
        Case("four_chunks_h96", 64, 96, "bf16"),
        Case("sixtyfour_chunks_h96", 1024, 96, "bf16"),
        Case("official_fixed_h1", 8192, 1, "bf16"),
        Case("official_fixed_h96_bf16", 8192, 96, "bf16"),
        Case("official_fixed_h96_fp32", 8192, 96, "fp32"),
    ]


def state_tensor(h: int, dtype: torch.dtype) -> torch.Tensor:
    # This is exactly the structured initial state used by tests/test_fwd.py.
    return torch.arange(h * D * D, dtype=torch.float32, device="cuda").reshape(1, h, D, D).to(torch.bfloat16).to(dtype)


def inputs_for(case: Case, seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    shape = (1, case.t, case.h, D)
    q = F.normalize(torch.randn(shape, dtype=torch.float32, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(shape, dtype=torch.float32, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    beta = torch.randn((1, case.t, case.h), dtype=torch.bfloat16, device="cuda")
    a_log = torch.rand(case.h, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(case.h, D, dtype=torch.float32, device="cuda")
    dtype = torch.bfloat16 if case.state_dtype == "bf16" else torch.float32
    return q, k, v, g, beta, a_log, dt_bias, state_tensor(case.h, dtype)


def tensor_stats(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    """Return scalar evidence while keeping the large tensors on the GPU."""
    different = actual != expected
    diff = (actual.float() - expected.float()).abs()
    count = int(different.sum().item())
    stats: dict[str, Any] = {
        "equal": bool(torch.equal(actual, expected)),
        "different_elements": count,
        "total_elements": actual.numel(),
        "different_fraction": count / actual.numel(),
        "mean_abs": float(diff.mean().item()),
        "max_abs": float(diff.max().item()),
    }
    ref_abs_mean = expected.float().abs().mean()
    stats["mean_relative_to_mean_abs_ref"] = float((diff.mean() / (ref_abs_mean + 1e-8)).item())

    # For bf16 outputs, test whether every mismatch is no more than one adjacent
    # representable bf16 step from the reference.  This is a diagnostic, not an
    # error tolerance: nextafter is evaluated in the same dtype as the tensor.
    if actual.dtype == torch.bfloat16:
        pos_inf = torch.full_like(expected, float("inf"))
        neg_inf = torch.full_like(expected, float("-inf"))
        upward = (torch.nextafter(expected, pos_inf).float() - expected.float()).abs()
        downward = (torch.nextafter(expected, neg_inf).float() - expected.float()).abs()
        one_ulp = torch.maximum(upward, downward)
        beyond = different & (diff > one_ulp)
        stats["mismatches_beyond_one_bf16_ulp"] = int(beyond.sum().item())

    # A few values make the mismatch concrete without writing a giant tensor.
    if count:
        indices = torch.nonzero(different, as_tuple=False)[:4].cpu().tolist()
        samples = []
        for index in indices:
            key = tuple(index)
            samples.append(
                {
                    "index": index,
                    "actual": float(actual[key].float().item()),
                    "reference": float(expected[key].float().item()),
                    "abs_difference": float(diff[key].item()),
                }
            )
        stats["first_mismatches"] = samples
    return stats


def run_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros_like(q)
    final = torch.zeros_like(initial_state)
    torch_ref(
        q, k, v, g, beta, 1.0 / math.sqrt(D), out,
        A_log=a_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
        initial_state=initial_state.clone(), final_state=final,
    )
    torch.cuda.synchronize()
    return out, final


def run_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.zeros_like(q)
    final = torch.zeros_like(initial_state)
    flash_kda.fwd(
        q, k, v, g, beta, 1.0 / math.sqrt(D), out,
        A_log=a_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
        initial_state=initial_state.clone(), final_state=final,
    )
    torch.cuda.synchronize()
    return out, final


def execute_case(case: Case, seed: int, repeats: int) -> dict[str, Any]:
    q, k, v, g, beta, a_log, dt_bias, initial_state = inputs_for(case, seed)
    started = time.perf_counter()
    ref_out_0, ref_state_0 = run_reference(q, k, v, g, beta, a_log, dt_bias, initial_state)
    ref_out_1, ref_state_1 = run_reference(q, k, v, g, beta, a_log, dt_bias, initial_state)
    reference_repeatable = {
        "output_equal": bool(torch.equal(ref_out_0, ref_out_1)),
        "final_state_equal": bool(torch.equal(ref_state_0, ref_state_1)),
    }

    kernel_runs = [run_kernel(q, k, v, g, beta, a_log, dt_bias, initial_state) for _ in range(repeats)]
    first_out, first_state = kernel_runs[0]
    kernel_repeatable = [
        {
            "repeat": index,
            "output_equal_to_repeat_0": bool(torch.equal(first_out, out)),
            "final_state_equal_to_repeat_0": bool(torch.equal(first_state, state)),
        }
        for index, (out, state) in enumerate(kernel_runs[1:], start=1)
    ]
    result = {
        "case": asdict(case),
        "reference_repeatability": reference_repeatable,
        "kernel_repeatability": kernel_repeatable,
        "output_vs_reference": tensor_stats(first_out, ref_out_0),
        "final_state_vs_reference": tensor_stats(first_state, ref_state_0),
        "wall_seconds_including_reference": time.perf_counter() - started,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def environment() -> dict[str, Any]:
    prop = torch.cuda.get_device_properties(0)
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device_name": prop.name,
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "flash_kda_module": str(Path(flash_kda.__file__).resolve()),
        "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
    }


def main() -> None:
    args = parse_args()
    if args.repeats < 2:
        raise SystemExit("--repeats must be at least 2 so kernel repeatability is tested")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; run inside a Slurm GPU allocation")
    summary = {
        "purpose": "FlashKDA exactness/repeatability diagnostic; not a benchmark",
        "seed": args.seed,
        "preset": args.preset,
        "repeats": args.repeats,
        "environment": environment(),
        "results": [execute_case(case, args.seed, args.repeats) for case in preset_cases(args.preset)],
    }
    print("\n=== compact summary ===")
    for item in summary["results"]:
        output = item["output_vs_reference"]
        final = item["final_state_vs_reference"]
        print(
            f"{item['case']['name']}: "
            f"out_equal={output['equal']} out_diff={output['different_elements']}/{output['total_elements']} "
            f"out_max={output['max_abs']:.7g}; "
            f"state_equal={final['equal']} state_diff={final['different_elements']}/{final['total_elements']} "
            f"state_max={final['max_abs']:.7g}"
        )
    if args.json is not None:
        with args.json.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"wrote JSON: {args.json}")


if __name__ == "__main__":
    main()
