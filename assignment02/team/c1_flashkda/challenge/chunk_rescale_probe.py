"""Probe a balanced exponent rescale for CHUNK=32/64.

The production kernel stores ``exp2(cumsum(g))`` in BF16.  With the current
``lower_bound=-5`` this underflows for larger chunks.  A common per-tile shift
``c`` can preserve the algebra used by K1:

    qd = q * 2**(s-c), kd = k * 2**(s-c)
    ki = k * 2**(-s+c)
    kr = k * 2**(-s+c) * 2**(s_last-c)

The factors in every K1 GEMM cancel the shift, and ``kr`` is unchanged.  The
state decay still needs the unshifted ``2**s_last``.  This script only probes
the representation; it deliberately does not change the production launch or
workspace ABI.  It reports whether a common shift fits both positive and
negative exponent tensors in BF16, and estimates the BF16 product error on a
small random tile.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


LOG2E = math.log2(math.e)
BF16_MIN_EXP = -133.0  # includes subnormal range used by the BF16 converter
BF16_MAX_EXP = 127.0


def exp2_ftz(x: torch.Tensor) -> torch.Tensor:
    """Match the kernel's ``ex2.approx.ftz.f32`` range for this probe."""
    value = torch.special.exp2(x)
    return torch.where(value.abs() < torch.finfo(torch.float32).tiny,
                       torch.zeros_like(value), value)


def balanced_shift(x: torch.Tensor) -> torch.Tensor:
    """Choose c halfway between the min and max prefix exponents per tile."""
    # For the current negative gate, max is normally zero and min is the final
    # cumulative exponent.  Taking both extrema also makes this valid for a
    # future gate range that is not monotonic.
    x_min = x.amin(dim=-1, keepdim=True)
    x_max = x.amax(dim=-1, keepdim=True)
    return (x_min + x_max) * 0.5


def bf16_stats(values: torch.Tensor) -> dict[str, float | int]:
    cast = values.to(torch.bfloat16)
    finite = torch.isfinite(cast)
    return {
        "zero_fraction": float((cast == 0).float().mean().item()),
        "inf_fraction": float(torch.isinf(cast).float().mean().item()),
        "finite_fraction": float(finite.float().mean().item()),
        "min_finite": float(cast[finite].float().min().item()) if finite.any() else math.nan,
        "max_finite": float(cast[finite].float().max().item()) if finite.any() else math.nan,
    }


def tile_product_error(chunk: int, seed: int, device: torch.device) -> dict[str, float | int]:
    """Compare BF16 K1 factors with/without rescale on several tiles.

    The matrix products are performed in FP32 after BF16 conversion.  This is
    not a bitwise model of the SM80 MMA instruction; it is a diagnostic for the
    representation error introduced before the MMA.  A full kernel would need
    an exactness and performance pass after changing the workspace ABI.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + chunk)
    tiles, d = 8, 128
    # Generate gate prefixes with the same broad distribution as the existing
    # B300 scalar range probe, but retain D channels per token for GEMM probes.
    g = torch.randn((tiles, chunk, d), dtype=torch.bfloat16,
                    device=device, generator=generator)
    dt = torch.rand((tiles, d), dtype=torch.float32, device=device,
                    generator=generator)
    a_log = torch.rand((tiles, 1, d), dtype=torch.float32,
                       device=device, generator=generator)
    z = torch.exp(a_log) * (g.float() + dt[:, None, :])
    gate = -5.0 * LOG2E * torch.sigmoid(z)
    prefix = gate.cumsum(dim=1)
    shift = balanced_shift(prefix)

    k = torch.randn((tiles, chunk, d), dtype=torch.bfloat16,
                    device=device, generator=generator)
    q = torch.randn((tiles, chunk, d), dtype=torch.bfloat16,
                    device=device, generator=generator)
    # The unscaled production representation.
    pos = exp2_ftz(prefix)
    neg = exp2_ftz(-prefix)
    # Common shift: qd/kd gets (s-c), ki gets (-s+c).
    pos_rs = exp2_ftz(prefix - shift)
    neg_rs = exp2_ftz(-prefix + shift)

    def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a.to(torch.bfloat16).float(),
                            b.to(torch.bfloat16).float().transpose(-1, -2))

    # L and Mqk are the products whose shift cancellation is exact in real
    # arithmetic.  kr is the restored key used by the state update.
    l_rs = matmul(k * pos_rs, k * neg_rs)
    mqk_rs = matmul(q * pos_rs, k * neg_rs)
    last = prefix[:, -1:, :]
    last_rs = prefix[:, -1:, :] - shift
    kr_rs = k * neg_rs * exp2_ftz(last_rs)

    # Use float64 and *non-FTZ* factors as the algebraic reference.  The raw
    # BF16 representation is intentionally allowed to contain zero/inf, so a
    # direct raw-vs-rescaled comparison would mostly measure those failures.
    pos_ideal = torch.pow(torch.tensor(2.0, dtype=torch.float64, device=device), prefix.double())
    neg_ideal = torch.pow(torch.tensor(2.0, dtype=torch.float64, device=device), -prefix.double())
    l_ideal = torch.matmul((k.double() * pos_ideal),
                           (k.double() * neg_ideal).transpose(-1, -2))
    mqk_ideal = torch.matmul((q.double() * pos_ideal),
                             (k.double() * neg_ideal).transpose(-1, -2))
    kr_ideal = k.double() * neg_ideal * torch.pow(
        torch.tensor(2.0, dtype=torch.float64, device=device), last.double())

    def error(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, int]:
        delta = (a.float() - b.float()).abs()
        finite = torch.isfinite(a) & torch.isfinite(b)
        if not finite.any():
            return math.nan, math.nan, int(delta.numel())
        d = delta[finite]
        scale = b.float().abs()[finite].clamp_min(1e-12)
        return float(d.max().item()), float((d / scale).mean().item()), int((~finite).sum().item())

    l_abs, l_rel, l_nonfinite = error(l_rs, l_ideal)
    mqk_abs, mqk_rel, mqk_nonfinite = error(mqk_rs, mqk_ideal)
    kr_abs, kr_rel, kr_nonfinite = error(kr_rs, kr_ideal)

    # K1's L/Mqk helper accumulates into FP16.  This is a key limitation of a
    # common exponent shift: although the BF16 *inputs* become finite, the
    # individual products can exceed the FP16 accumulator range before the
    # shifted factors cancel in exact arithmetic.  This count is a conservative
    # proxy (it casts each product to FP16) rather than a replacement for a
    # tensor-core instruction-level test.
    l_terms_fp16 = ((k * pos_rs).to(torch.bfloat16).float()[..., :, None, :] *
                    (k * neg_rs).to(torch.bfloat16).float()[..., None, :, :]).to(torch.float16)
    mqk_terms_fp16 = ((q * pos_rs).to(torch.bfloat16).float()[..., :, None, :] *
                      (k * neg_rs).to(torch.bfloat16).float()[..., None, :, :]).to(torch.float16)
    l_term_overflow = int(torch.isinf(l_terms_fp16).sum().item())
    mqk_term_overflow = int(torch.isinf(mqk_terms_fp16).sum().item())
    return {
        "chunk": chunk,
        "tiles": tiles,
        "d": d,
        "raw_prefix_min": float(prefix.min().item()),
        "raw_prefix_max": float(prefix.max().item()),
        "shift_min": float(shift.min().item()),
        "shift_max": float(shift.max().item()),
        "raw_positive_bf16": bf16_stats(pos),
        "raw_negative_bf16": bf16_stats(neg),
        "rescaled_positive_bf16": bf16_stats(pos_rs),
        "rescaled_negative_bf16": bf16_stats(neg_rs),
        "L_max_abs_diff": l_abs,
        "L_mean_relative_diff": l_rel,
        "L_nonfinite_elements": l_nonfinite,
        "Mqk_max_abs_diff": mqk_abs,
        "Mqk_mean_relative_diff": mqk_rel,
        "Mqk_nonfinite_elements": mqk_nonfinite,
        "k_restored_max_abs_diff": kr_abs,
        "k_restored_mean_relative_diff": kr_rel,
        "k_restored_nonfinite_elements": kr_nonfinite,
        "L_fp16_term_overflow_elements": l_term_overflow,
        "Mqk_fp16_term_overflow_elements": mqk_term_overflow,
    }


def range_case(chunk: int, samples: int, seed: int, device: torch.device) -> dict:
    # Deliberately use the default CUDA generator, matching the historical
    # chunk_range_probe.py stream.  Each chunk is grouped from the same scalar
    # sample vector, so the three rows are comparable rather than three
    # unrelated random draws.
    torch.manual_seed(seed)
    n = (samples // chunk) * chunk
    g = torch.randn(n, dtype=torch.bfloat16, device=device)
    dt = torch.rand(n, dtype=torch.float32, device=device)
    a_log = torch.rand(n, dtype=torch.float32, device=device)
    z = torch.exp(a_log) * (g.float() + dt)
    gate = -5.0 * LOG2E * torch.sigmoid(z)
    prefix = gate[:n].reshape(-1, chunk).cumsum(dim=-1)
    shift = balanced_shift(prefix)
    raw_pos_exp = prefix
    raw_neg_exp = -prefix
    rs_pos_exp = prefix - shift
    rs_neg_exp = -prefix + shift
    # A common shift is representable when both intervals fit in the BF16
    # exponent range.  The check includes subnormals conservatively.
    common_fit = bool(
        rs_pos_exp.min() >= BF16_MIN_EXP and rs_pos_exp.max() <= BF16_MAX_EXP
        and rs_neg_exp.min() >= BF16_MIN_EXP and rs_neg_exp.max() <= BF16_MAX_EXP
    )
    return {
        "chunk": chunk,
        "samples": n,
        "raw_prefix_exp_range": [float(prefix.min().item()), float(prefix.max().item())],
        "balanced_shift_range": [float(shift.min().item()), float(shift.max().item())],
        "rescaled_positive_exp_range": [float(rs_pos_exp.min().item()), float(rs_pos_exp.max().item())],
        "rescaled_negative_exp_range": [float(rs_neg_exp.min().item()), float(rs_neg_exp.max().item())],
        "common_shift_fits_bf16_exponent": common_fit,
        "raw_positive_bf16": bf16_stats(exp2_ftz(raw_pos_exp)),
        "raw_negative_bf16": bf16_stats(exp2_ftz(raw_neg_exp)),
        "rescaled_positive_bf16": bf16_stats(exp2_ftz(rs_pos_exp)),
        "rescaled_negative_bf16": bf16_stats(exp2_ftz(rs_neg_exp)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1 << 20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "device": str(device),
        "seed": args.seed,
        "bf16_exponent_range": [BF16_MIN_EXP, BF16_MAX_EXP],
        "cases": [range_case(c, args.samples, args.seed, device) for c in (16, 32, 64)],
        "product_probes": [tile_product_error(c, args.seed, device) for c in (32, 64)],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.json:
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
