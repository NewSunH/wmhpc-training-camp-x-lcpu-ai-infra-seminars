#!/usr/bin/env python3
"""C1 BF16 error study against vendored FP32 token recurrence.

The state-only diagnostic rounds the FP32 recurrence state to BF16 after each
16-token block. It isolates boundary-state rounding in a reference model; it
does not emulate the kernel's other BF16 operands/FP16 inverse/approximate gates.
Actual kernel error includes those sources as well. This is not a model-quality
test or a proof for arbitrary learned inputs.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path

import torch

from r19_probe import file_digest, make_call


def metrics(actual, reference):
    x, y = actual.float(), reference.float()
    d = (x - y).abs()
    return dict(finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
                max_abs=float(d.max()), mean_abs=float(d.mean()),
                p99_abs=float(torch.quantile(d.flatten(), 0.99)),
                relative_l2=float(torch.linalg.vector_norm(x - y) /
                                  torch.linalg.vector_norm(y).clamp_min(1e-30)))


def boundary_round_reference(q, k, v, g, beta, initial, *, round_state):
    # The same token update/order as fla_kda_ref.naive_recurrent_kda, H=1.
    q = q.float() * (128 ** -0.5)
    k, v, g, beta = [x.float() for x in (k, v, g, beta)]
    state = initial.float().clone()
    out = torch.zeros_like(v)
    for i in range(v.shape[1]):
        ki, vi = k[:, i], v[:, i]
        state = state * g[:, i, ..., None].exp()
        state = state + torch.einsum(
            'b h k, b h v -> b h k v', beta[:, i, ..., None] * ki,
            vi - (ki[..., None] * state).sum(-2))
        out[:, i] = torch.einsum('b h k, b h k v -> b h v', q[:, i], state)
        if round_state and (i + 1) % 16 == 0:
            state = state.to(torch.bfloat16).float()
    return out, state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference', required=True, type=Path, help='vendored naive.py')
    ap.add_argument('--json', required=True, type=Path)
    args = ap.parse_args()
    spec = importlib.util.spec_from_file_location('c1_vendored_naive', args.reference)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    import flash_kda
    extension = importlib.import_module('flash_kda_C')
    torch.backends.cuda.matmul.allow_tf32 = False
    result = dict(extension_sha256=file_digest(extension.__file__),
                  reference_sha256=file_digest(str(args.reference)),
                  device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  note=__doc__, cases=[])
    for gate_name in ('weak', 'random', 'strong'):
        torch.manual_seed(20260911)
        shape = (1, 8192, 1, 128)
        q, k = [torch.randn(shape, device='cuda').to(torch.bfloat16) for _ in range(2)]
        v = torch.randn(shape, device='cuda').to(torch.bfloat16)
        raw_g = torch.randn(shape, device='cuda').to(torch.bfloat16)
        if gate_name != 'random':
            raw_g.fill_(-8 if gate_name == 'weak' else 8)
        raw_beta = torch.randn((1, 8192, 1), device='cuda').to(torch.bfloat16)
        a_log = torch.zeros(1, device='cuda')
        dt_bias = torch.zeros((1, 128), device='cuda')
        initial = (0.1 * torch.randn((1, 1, 128, 128), device='cuda')).to(torch.bfloat16)
        qn, kn = [x.float() * torch.rsqrt(x.float().square().sum(-1, keepdim=True) + 1e-6)
                  for x in (q, k)]
        gate = -5 * raw_g.float().sigmoid()
        beta = raw_beta.float().sigmoid()
        for T in (256, 1024, 8192):
            inp = [x[:, :T].contiguous() for x in (q, k, v, raw_g, raw_beta)]
            inp += [a_log, dt_bias, initial, None]
            fn, out, state = make_call(flash_kda, inp, 'bf16')
            fn()
            ri = initial.transpose(-1, -2).contiguous().float()
            tensors = [x[:, :T] for x in (qn, kn, v.float(), gate, beta)]
            ro, rs = ref.naive_recurrent_kda(*tensors, initial_state=ri,
                                             output_final_state=True)
            # First validate the diagnostic's unrounded implementation against
            # the pinned vendor, then change exactly one boundary operation.
            if T == 256:
                vo, vs = boundary_round_reference(*tensors, ri, round_state=False)
                if not torch.equal(vo, ro) or not torch.equal(vs, rs):
                    raise RuntimeError('Unrounded diagnostic differs from vendored FP32 recurrence')
            bo, bs = boundary_round_reference(*tensors, ri, round_state=True)
            row = dict(T=T, H=1, D=128, seed=20260911, gate=gate_name,
                       mean_log_decay=float(gate[:, :T].mean()),
                       kernel_output=metrics(out, ro),
                       kernel_state=metrics(state.transpose(-1, -2), rs),
                       boundary_round_output=metrics(bo, ro),
                       boundary_round_state=metrics(bs, rs))
            if any(not row[key]['finite'] for key in (
                    'kernel_output', 'kernel_state', 'boundary_round_output', 'boundary_round_state')):
                raise RuntimeError(f'Nonfinite precision result: {gate_name}, T={T}')
            result['cases'].append(row)
            args.json.write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
