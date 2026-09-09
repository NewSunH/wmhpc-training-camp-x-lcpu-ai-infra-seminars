"""Compare FlashKDA bf16/fp32 state storage against the FLA Triton reference."""

import math
import torch
import torch.nn.functional as F
import flash_kda
from fla.ops.kda import chunk_kda


@torch.inference_mode()
def main():
    torch.manual_seed(0)
    B, T, H, D = 1, 1024, 4, 128
    lower_bound = -5.0
    q = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32, device="cuda"), dim=-1).bfloat16()
    k = F.normalize(torch.randn(B, T, H, D, dtype=torch.float32, device="cuda"), dim=-1).bfloat16()
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    g = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda")
    beta = torch.randn(B, T, H, dtype=torch.bfloat16, device="cuda")
    A_log = torch.rand(H, dtype=torch.float32, device="cuda")
    dt_bias = torch.rand(H, D, dtype=torch.float32, device="cuda")
    init_bf16 = torch.randn(B, H, D, D, dtype=torch.float32, device="cuda").bfloat16()

    out_bf16 = torch.zeros_like(q)
    state_bf16 = torch.zeros_like(init_bf16)
    flash_kda.fwd(q, k, v, g, beta, 1 / math.sqrt(D), out_bf16,
                  A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
                  initial_state=init_bf16.clone(), final_state=state_bf16)

    out_fp32 = torch.zeros_like(q)
    state_fp32 = torch.zeros(B, H, D, D, dtype=torch.float32, device="cuda")
    flash_kda.fwd(q, k, v, g, beta, 1 / math.sqrt(D), out_fp32,
                  A_log=A_log, dt_bias=dt_bias, lower_bound=lower_bound,
                  initial_state=init_bf16.float().clone(), final_state=state_fp32)

    ref = chunk_kda(q=q, k=k, v=v, g=g, beta=beta, scale=1 / math.sqrt(D),
                    initial_state=init_bf16.float(), output_final_state=True,
                    use_gate_in_kernel=True, use_qk_l2norm_in_kernel=True,
                    use_beta_sigmoid_in_kernel=True, A_log=A_log, dt_bias=dt_bias,
                    lower_bound=lower_bound, transpose_state_layout=True)
    out_ref, state_ref = ref[0], ref[1]

    def stats(name, x, y):
        d = (x.float() - y.float()).abs()
        denom = y.float().abs().clamp_min(1e-6)
        print(f"{name}: max_abs={d.max().item():.6e} mean_abs={d.mean().item():.6e} "
              f"max_rel={(d / denom).max().item():.6e}")

    print(f"shape=[{B},{T},{H},{D}] seed=0")
    stats("bf16_state vs fp32_state output", out_bf16, out_fp32)
    stats("bf16_state vs fp32_state final", state_bf16, state_fp32)
    stats("bf16_state vs FLA output", out_bf16, out_ref)
    stats("fp32_state vs FLA output", out_fp32, out_ref)
    stats("bf16_state vs FLA final", state_bf16, state_ref)
    stats("fp32_state vs FLA final", state_fp32, state_ref)


if __name__ == "__main__":
    main()
