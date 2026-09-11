#!/usr/bin/env python3
"""R19 register-resident-state correctness and CUDA-event probe.

This file is deliberately usable from two independently installed FlashKDA
checkouts.  The caller runs it once with the baseline Python environment and
once with the candidate Python environment; both processes create identical
inputs from ``seed``.  Keeping the process boundary makes extension imports
unambiguous while the surrounding Slurm job can keep both runs on one GPU
allocation.

The benchmark measures the public ``flash_kda.fwd`` call, including K1, K2,
workspace allocation, and launch overhead.  Exactness uses the vendored
``tests.torch_ref.torch_ref`` implementation.  No reference output is used
inside timed iterations.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


D = 128
LOWER_BOUND = -5.0


def digest(t: torch.Tensor) -> str:
    raw = t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def file_digest(path: str | None) -> str | None:
    if not path:
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_inputs(T: int, H: int, seed: int, nseq: int = 1):
    torch.manual_seed(seed)
    shape = (1, T, H, D)
    q = F.normalize(torch.randn(shape, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(torch.randn(shape, device="cuda"), p=2, dim=-1).to(torch.bfloat16)
    v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    beta = torch.randn((1, T, H), dtype=torch.bfloat16, device="cuda")
    a_log = torch.rand(H, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(H, D, dtype=torch.float32, device="cuda")
    state_n = nseq if nseq > 1 else 1
    initial = torch.arange(state_n * H * D * D, dtype=torch.float32, device="cuda")
    initial = initial.reshape(state_n, H, D, D).to(torch.bfloat16)
    cu = None
    if nseq > 1:
        # The caller chooses equal segments for deterministic, compact probes.
        assert T % nseq == 0
        step = T // nseq
        cu = torch.arange(0, T + 1, step, dtype=torch.long, device="cuda")
    return q, k, v, g, beta, a_log, dt_bias, initial, cu


def make_call(flash_kda, inp, state_mode: str):
    """Prepare one reusable public-API call outside the timed region.

    The benchmark keeps allocations made by ``flash_kda.fwd``
    itself (notably its workspace ABI), but excludes Python-side tensor clone,
    zero-fill and argument construction.  ``out`` and final state are legal
    output buffers and are overwritten by the public API on every invocation.
    """
    q, k, v, g, beta, a_log, dt_bias, initial, cu = inp
    kwargs = {} if cu is None else {"cu_seqlens": cu}
    if state_mode == "none":
        state_in = None
        state_out = None
    elif state_mode == "bf16":
        state_in = initial.clone()
        state_out = torch.zeros_like(initial)
    elif state_mode == "fp32":
        state_in = initial.float()
        state_out = torch.zeros_like(state_in)
    else:
        raise ValueError(state_mode)
    out = torch.zeros_like(q)

    def call():
        flash_kda.fwd(
            q, k, v, g, beta, 1.0 / math.sqrt(D), out,
            A_log=a_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
            initial_state=state_in, final_state=state_out, **kwargs,
        )

    return call, out, state_out


def timed(fn, warmup: int, iters: int, repeats: int):
    for _ in range(max(1, warmup)):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        samples.extend(float(a.elapsed_time(b)) for a, b in zip(starts, ends))
    samples.sort()
    return {
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "mean_ms": sum(samples) / len(samples),
        "min_ms": samples[0],
        "max_ms": samples[-1],
    }


def exact_case(flash_kda, T: int, H: int, seed: int, state_mode: str, nseq: int = 1):
    from torch_ref import torch_ref

    inp = make_inputs(T, H, seed, nseq)
    call, out, state = make_call(flash_kda, inp, state_mode)
    call()
    q, k, v, g, beta, a_log, dt_bias, initial, cu = inp
    kwargs = {} if cu is None else {"cu_seqlens": cu}
    ref_out = torch.zeros_like(q)
    if state_mode == "none":
        ref_state = None
        ref_initial = None
    elif state_mode == "bf16":
        ref_initial = initial.clone()
        ref_state = torch.zeros_like(initial)
    else:
        ref_initial = initial.float()
        ref_state = torch.zeros_like(ref_initial)
    torch_ref(
        q, k, v, g, beta, 1.0 / math.sqrt(D), ref_out,
        A_log=a_log, dt_bias=dt_bias, lower_bound=LOWER_BOUND,
        initial_state=ref_initial, final_state=ref_state, **kwargs,
    )
    torch.cuda.synchronize()
    od = (out.float() - ref_out.float()).abs()
    result = {
        "T": T, "H": H, "nseq": nseq, "seed": seed,
        "state_mode": state_mode,
        "output_equal": bool(torch.equal(out, ref_out)),
        "output_different_elements": int(torch.count_nonzero(out != ref_out).item()),
        "output_max_abs_diff": float(od.max().item()),
        "output_mean_abs_diff": float(od.mean().item()),
        "output_digest": digest(out), "reference_output_digest": digest(ref_out),
    }
    if state is not None:
        sd = (state.float() - ref_state.float()).abs()
        result.update({
            "state_equal": bool(torch.equal(state, ref_state)),
            "state_different_elements": int(torch.count_nonzero(state != ref_state).item()),
            "state_max_abs_diff": float(sd.max().item()),
            "state_mean_abs_diff": float(sd.mean().item()),
            "state_digest": digest(state), "reference_state_digest": digest(ref_state),
        })
    else:
        result["state_equal"] = None
    return result


def bench_case(flash_kda, T: int, H: int, seed: int, state_mode: str, nseq: int,
               warmup: int, iters: int, repeats: int):
    inp = make_inputs(T, H, seed, nseq)
    call, _, _ = make_call(flash_kda, inp, state_mode)
    return {
        "T": T, "H": H, "nseq": nseq, "seed": seed, "state_mode": state_mode,
        "timing": timed(call, warmup, iters, repeats),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--label", default="unknown")
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--skip-exact", action="store_true")
    ap.add_argument("--quick", action="store_true", help="only T8192 H96 bf16 timing")
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    # Import only after the caller has selected the isolated Python/module path.
    import flash_kda
    extension = importlib.import_module("flash_kda_C")

    exact_specs = [
        (16, 1, "bf16", 1), (17, 4, "bf16", 1), (64, 12, "bf16", 1),
        (97, 96, "bf16", 1), (17, 4, "fp32", 1), (64, 4, "none", 1),
        (64, 4, "bf16", 2),
    ]
    bench_specs = [
        (8192, 96, "bf16", 1), (8192, 12, "bf16", 1), (8192, 4, "bf16", 1),
        (8192, 96, "fp32", 1), (8192, 96, "none", 1),
        (8192, 96, "bf16", 8),
    ]
    result = {
        "label": args.label,
        "seed": args.seed,
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "cuda_version": torch.version.cuda,
        "timing_config": {"warmup": args.warmup, "iters": args.iters, "repeats": args.repeats},
        "environment_note": "env_* records runtime values, not compiled flags; use extension SHA and build log for build identity",
        "env_early_output": os.getenv("FLASH_KDA_C1_EARLY_OUTPUT", "unset"),
        "env_register_state": os.getenv("FLASH_KDA_C1_REGISTER_STATE", "unset"),
        "env_register_state_blocks": os.getenv("FLASH_KDA_C1_REGISTER_STATE_BLOCKS", "unset"),
        "flash_kda_module": str(Path(flash_kda.__file__).resolve()),
        "extension_module": str(Path(extension.__file__).resolve()),
        "extension_sha256": file_digest(extension.__file__),
        "exactness": [], "benchmarks": [],
    }
    if not args.skip_exact:
        for i, (T, H, mode, nseq) in enumerate(exact_specs):
            result["exactness"].append(exact_case(flash_kda, T, H, args.seed + i, mode, nseq))
    bad = [x for x in result["exactness"] if not x["output_equal"] or x.get("state_equal") is False]
    if bad:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        raise SystemExit(f"exactness failure in {len(bad)} case(s)")
    if args.quick:
        bench_specs = bench_specs[:1]
    if not args.skip_bench:
        for i, (T, H, mode, nseq) in enumerate(bench_specs):
            result["benchmarks"].append(bench_case(
                flash_kda, T, H, args.seed + 100 + i, mode, nseq,
                args.warmup, args.iters, args.repeats,
            ))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    bad = [x for x in result["exactness"] if not x["output_equal"] or x.get("state_equal") is False]
    if bad:
        raise SystemExit(f"exactness failure in {len(bad)} case(s)")


if __name__ == "__main__":
    main()
