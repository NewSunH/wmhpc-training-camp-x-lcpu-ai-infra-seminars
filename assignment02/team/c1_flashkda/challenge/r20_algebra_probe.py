"""CPU mathematical/range audit of CHUNK=16/32/64 KDA.

This does not benchmark GPU performance or reproduce tensor-core rounding.
It deliberately preserves the historical R13/R16 scripts.  R13's product
probe selected a shift along the key axis instead of across time; R16's
reference included L's diagonal and inverted I-L rather than I+L.

The float64 token recurrence below transcribes fla_kda_ref/naive.py:59-63.
We also execute that vendored function unchanged in its native FP32 dtype.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
DTYPE = torch.float64


def stats(actual: torch.Tensor, expected: torch.Tensor) -> dict:
    a, b = actual.double(), expected.double()
    finite = torch.isfinite(a) & torch.isfinite(b)
    delta = a - b
    if not bool(finite.all()):
        return {"nonfinite_elements": int((~finite).sum()),
                "max_abs": None, "mean_abs": None, "relative_l2": None}
    return {
        "nonfinite_elements": 0,
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "relative_l2": float(torch.linalg.vector_norm(delta) /
                             torch.linalg.vector_norm(b).clamp_min(1e-30)),
    }


def load_vendored_recurrent():
    """Execute only the unmodified recurrent definition, which needs no einops."""
    path = ROOT / "fla_kda_ref/naive.py"
    source = path.read_text()
    node = next(n for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == "naive_recurrent_kda")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["naive_recurrent_kda"], hashlib.sha256(source.encode()).hexdigest()


def make_inputs(tokens: int, dim: int, profile: str, seed: int):
    rng = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(tokens, dim, dtype=DTYPE, generator=rng)
    k = torch.randn(tokens, dim, dtype=DTYPE, generator=rng)
    # BF16-representable inputs, but the diagnostic arithmetic stays float64.
    q = (q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)).bfloat16().double()
    k = (k / torch.linalg.vector_norm(k, dim=-1, keepdim=True)).bfloat16().double()
    v = torch.randn(tokens, dim, dtype=DTYPE, generator=rng).bfloat16().double()
    raw_g = torch.randn(tokens, dim, dtype=DTYPE, generator=rng).bfloat16().double()
    a_log = torch.rand(1, dtype=DTYPE, generator=rng)
    dt_bias = torch.rand(dim, dtype=DTYPE, generator=rng)
    offset = {"normal": 0.0, "weak": -8.0, "strong": 8.0}[profile]
    # A_log is per head; dt_bias per key. Both remain fixed across time.
    gate = -5.0 * torch.sigmoid(a_log.exp() * (raw_g + dt_bias + offset))
    beta = torch.sigmoid(torch.randn(tokens, dtype=DTYPE, generator=rng))
    state = torch.randn(dim, dim, dtype=DTYPE, generator=rng) * 0.02
    return q, k, v, gate, beta, state


def token_recurrent(q, k, v, gate, beta, initial):
    """Float64 transcription of the vendored KDA token recurrence (K,V state)."""
    state = initial.clone()
    outputs = []
    scale = q.shape[-1] ** -0.5
    for t in range(q.shape[0]):
        state = state * gate[t].exp()[:, None]
        residual = v[t] - (k[t, :, None] * state).sum(dim=0)
        state = state + torch.outer(beta[t] * k[t], residual)
        outputs.append((q[t] * scale) @ state)
    return torch.stack(outputs), state


def neumann_inverse(strict_lower):
    """Finite doubling polynomial for (I+L)^-1; L is strictly lower triangular."""
    n = strict_lower.shape[0]
    inverse = torch.eye(n, dtype=DTYPE) - strict_lower
    power = strict_lower @ strict_lower
    exponent = 2
    while exponent < n:
        inverse = inverse + inverse @ power
        exponent *= 2
        if exponent < n:
            power = power @ power
    return inverse


def chunk_recurrent(q, k, v, gate, beta, initial, chunk, scaled, neumann=False):
    state = initial.clone()
    outputs, inverse_errors, prefix_ranges, shift_ranges = [], [], [], []
    scale = q.shape[-1] ** -0.5
    for begin in range(0, len(q), chunk):
        stop = min(begin + chunk, len(q))
        prefix = gate[begin:stop].cumsum(dim=0)
        # prefix=[time,key]: shift=[key], common across every time in this chunk.
        shift = ((prefix.amin(dim=0) + prefix.amax(dim=0)) * 0.5
                 if scaled else torch.zeros(q.shape[-1], dtype=DTYPE))
        positive, negative = (prefix - shift).exp(), (-prefix + shift).exp()
        kd = k[begin:stop] * positive
        qd = q[begin:stop] * positive * scale
        ki = k[begin:stop] * negative
        projected_state = shift.exp()[:, None] * state
        strict_lower = torch.tril(kd @ ki.T, diagonal=-1) * beta[begin:stop, None]
        causal_qk = torch.tril(qd @ ki.T)
        identity = torch.eye(stop - begin, dtype=DTYPE)
        exact_inverse = torch.linalg.solve_triangular(identity + strict_lower,
                                                      identity, upper=False)
        polynomial_inverse = neumann_inverse(strict_lower)
        inverse_errors.append(float((exact_inverse - polynomial_inverse).abs().max()))
        inverse = polynomial_inverse if neumann else exact_inverse
        innovation = beta[begin:stop, None] * (v[begin:stop] - kd @ projected_state)
        u = inverse @ innovation
        outputs.append(qd @ projected_state + causal_qk @ u)
        # State update uses the original state and stable differences of prefixes.
        restored_key = k[begin:stop] * (prefix[-1:] - prefix).exp()
        state = prefix[-1].exp()[:, None] * state + restored_key.T @ u
        prefix_ranges.append((float(prefix.min()), float(prefix.max())))
        shift_ranges.append((float(shift.min()), float(shift.max())))
    return torch.cat(outputs), state, {
        "max_neumann_vs_triangular_inverse_abs": max(inverse_errors),
        "prefix_natural_log_min": min(x[0] for x in prefix_ranges),
        "prefix_natural_log_max": max(x[1] for x in prefix_ranges),
        "shift_natural_log_min": min(x[0] for x in shift_ranges),
        "shift_natural_log_max": max(x[1] for x in shift_ranges),
    }


def exp_ftz_bf16(exponent_natural):
    # Range model: exp in FP64, round to FP32, flush FP32 subnormals, cast BF16.
    # This models representability, not ex2.approx.ftz instruction accuracy.
    value = exponent_natural.exp().float()
    value = torch.where(value.abs() < torch.finfo(torch.float32).tiny, 0.0, value)
    return value.bfloat16()


def finite_stats(tensor):
    return {"elements": tensor.numel(),
            "zero_fraction": float((tensor == 0).double().mean()),
            "inf_fraction": float(torch.isinf(tensor).double().mean()),
            "nan_fraction": float(torch.isnan(tensor).double().mean())}


def factor_storage_diagnostic(q, k, gate, beta, chunk):
    """Corrected BF16-factor -> FP32 dot -> FP16 L-store range model.

    Upper-triangular nonfinites are counted separately because production
    overwrites them before the inverse; they do not imply a causal failure.
    """
    prefix = gate[:chunk].cumsum(0)
    shift = (prefix.amin(0) + prefix.amax(0)) * 0.5
    results = {}
    for name, c in [("raw", torch.zeros_like(shift)), ("scaled", shift)]:
        kd = k[:chunk].bfloat16() * exp_ftz_bf16(prefix - c)
        ki = k[:chunk].bfloat16() * exp_ftz_bf16(-prefix + c)
        dot = kd.float() @ ki.float().T
        stored = dot.half()
        below = torch.tril(torch.ones(chunk, chunk, dtype=torch.bool), diagonal=-1)
        masked = torch.where(below, stored, 0.0)
        # Production then applies beta in FP16 to the strictly lower entries.
        masked = masked * beta[:chunk, None].half()
        results[name] = {
            "kd": finite_stats(kd), "ki": finite_stats(ki),
            "fp32_dot_nonfinite_causal": int((~torch.isfinite(dot[below])).sum()),
            "fp32_dot_nonfinite_masked": int((~torch.isfinite(dot[~below])).sum()),
            "fp16_store_nonfinite_causal": int((~torch.isfinite(stored[below])).sum()),
            "fp16_store_nonfinite_masked": int((~torch.isfinite(stored[~below])).sum()),
            "after_mask_and_beta_nonfinite": int((~torch.isfinite(masked)).sum()),
        }
    return results


def range_probe(chunk, dim, profile, seed, tiles):
    rng = torch.Generator(device="cpu").manual_seed(seed + 10000)
    raw = torch.randn(tiles, chunk, dim, dtype=DTYPE, generator=rng).bfloat16().double()
    a_log = torch.rand(tiles, 1, 1, dtype=DTYPE, generator=rng)
    dt_bias = torch.rand(tiles, 1, dim, dtype=DTYPE, generator=rng)
    offset = {"normal": 0.0, "weak": -8.0, "strong": 8.0}[profile]
    gate = -5.0 * torch.sigmoid(a_log.exp() * (raw + dt_bias + offset))
    prefix = gate.cumsum(dim=1)
    shift = (prefix.amin(dim=1, keepdim=True) + prefix.amax(dim=1, keepdim=True)) * 0.5
    shifted = prefix - shift
    return {
        "chunk": chunk, "profile": profile, "seed": seed, "tiles": tiles,
        "dim": dim, "prefix_log2_min": float(prefix.min() / math.log(2)),
        "prefix_log2_max": float(prefix.max() / math.log(2)),
        "shifted_log2_min": float(shifted.min() / math.log(2)),
        "shifted_log2_max": float(shifted.max() / math.log(2)),
        "raw_positive": finite_stats(exp_ftz_bf16(prefix)),
        "raw_negative": finite_stats(exp_ftz_bf16(-prefix)),
        "scaled_positive": finite_stats(exp_ftz_bf16(shifted)),
        "scaled_negative": finite_stats(exp_ftz_bf16(-shifted)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--tokens", type=int, default=263)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--range-tiles", type=int, default=64)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_device("cpu")
    vendored, vendor_sha = load_vendored_recurrent()
    cases, ranges = [], []
    for seed in args.seeds:
        for profile in ["normal", "weak", "strong"]:
            q, k, v, gate, beta, initial = make_inputs(args.tokens, args.dim, profile, seed)
            expected_o, expected_s = token_recurrent(q, k, v, gate, beta, initial)
            vo, vs = vendored(q[None, :, None], k[None, :, None], v[None, :, None],
                              gate[None, :, None], beta[None, :, None],
                              initial_state=initial[None, None], output_final_state=True)
            vendor_stats = {"output": stats(vo[0, :, 0], expected_o),
                            "state": stats(vs[0, 0], expected_s)}
            for chunk in [16, 32, 64]:
                case = {"chunk": chunk, "profile": profile, "seed": seed,
                        "tokens": args.tokens, "dim": args.dim,
                        "chunks": math.ceil(args.tokens / chunk),
                        "tail_tokens": args.tokens % chunk,
                        "vendored_native_fp32_vs_float64": vendor_stats}
                for route, scaled, neumann in [("raw_triangular", False, False),
                                                ("scaled_triangular", True, False),
                                                ("scaled_neumann", True, True)]:
                    o, s, diagnostics = chunk_recurrent(q, k, v, gate, beta, initial,
                                                        chunk, scaled, neumann)
                    case[route] = {"output": stats(o, expected_o), "state": stats(s, expected_s),
                                   **diagnostics}
                    for what in ["output", "state"]:
                        assert case[route][what]["nonfinite_elements"] == 0
                        assert case[route][what]["max_abs"] < 1e-11, (case, route, what)
                        assert case[route][what]["relative_l2"] < 1e-10, (case, route, what)
                case["factor_storage_model"] = factor_storage_diagnostic(q, k, gate, beta, chunk)
                cases.append(case)
                ranges.append(range_probe(chunk, args.dim, profile, seed, args.range_tiles))
                print(f"PASS seed={seed} gate={profile} C={chunk} T={args.tokens} "
                      f"output={case['scaled_neumann']['output']['max_abs']:.3e} "
                      f"state={case['scaled_neumann']['state']['max_abs']:.3e}", flush=True)
    result = {
        "device": "cpu", "torch": torch.__version__, "arithmetic": "float64",
        "vendored_reference_sha256": vendor_sha,
        "cases": cases, "range_cases": ranges,
        "acceptance": {"float64_output_and_state_max_abs": 1e-11,
                       "float64_output_and_state_relative_l2": 1e-10,
                       "passed_cases": len(cases)},
        "limits": ["No GPU timing or tensor-core instruction emulation.",
                   "Float64 tests establish algebraic agreement on the tested inputs, not BF16 accuracy.",
                   "The unchanged vendored recurrence converts inputs to FP32; the float64 oracle transcribes its formula.",
                   "Range simulation models FP32 flush-to-zero and BF16 conversion, not approximate exp2 instruction error.",
                   "Upper-triangular nonfinites overwritten by the causal mask do not by themselves imply output failure.",
                   "This samples fixed per-head A_log and per-key dt_bias over time, not trained model inputs."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Wrote {args.output}: {len(cases)} algebra cases, {len(ranges)} range cases.", flush=True)


if __name__ == "__main__":
    main()
