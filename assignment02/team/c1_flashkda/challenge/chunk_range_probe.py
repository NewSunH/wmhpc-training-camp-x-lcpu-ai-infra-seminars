"""Estimate gate-cumsum exponent range for CHUNK=16/32/64.

This is a scalar range probe, not a replacement FlashKDA kernel.  It follows
the kernel's base-2 representation: g_log2 = lower_bound * log2(e) * sigmoid(...)
and then converts exp2(cumsum(g_log2)) to bf16, where the state/workspace is
stored by the production kernel.
"""

import argparse
import math
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=1 << 20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Match benchmark distributions: g is bf16 normal, dt_bias and A_log are
    # float32 uniform.  Each scalar is an independent head/channel draw.
    g_bf16 = torch.randn(args.samples, dtype=torch.bfloat16, device=device)
    dt_bias = torch.rand(args.samples, dtype=torch.float32, device=device)
    a_log = torch.rand(args.samples, dtype=torch.float32, device=device)
    z = torch.exp(a_log) * (g_bf16.float() + dt_bias)
    sigmoid = torch.sigmoid(z)
    lower_bound = -5.0
    g_log2 = lower_bound * 1.4426950408889634 * sigmoid

    print(f"device={device} samples={args.samples} seed={args.seed}")
    print("bf16 min normal ~= 2^-126; bf16 min subnormal ~= 2^-133")
    for chunk in (16, 32, 64):
        n = (g_log2.numel() // chunk) * chunk
        x = g_log2[:n].reshape(-1, chunk)
        cumsum_log2 = torch.cumsum(x, dim=-1)
        values = torch.exp2(cumsum_log2)
        values_bf16 = values.to(torch.bfloat16)
        zeros = (values_bf16 == 0).float().mean().item()
        print(
            f"CHUNK={chunk:2d} log2_range=[{cumsum_log2.min().item():.3f}, "
            f"{cumsum_log2.max().item():.3f}] "
            f"exp2_range=[{values.min().item():.3e}, {values.max().item():.3e}] "
            f"bf16_zero_fraction={zeros:.6f}"
        )


if __name__ == "__main__":
    main()
