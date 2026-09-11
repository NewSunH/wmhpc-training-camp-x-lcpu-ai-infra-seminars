#!/usr/bin/env python3
"""C1 boundary benchmarks, with input/output hashes outside the timed region.

Run in separate baseline/candidate processes in ABBA order. The public API is
timed with CUDA events, as in R19; this is neither kernel-only nor CPU wall time.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch

from r19_probe import digest, exact_case, file_digest, make_call, make_inputs, timed


def specs():
    rows = [(t, 96, None) for t in (16, 64, 256, 1024, 8192, 32768)]
    rows += [(8192, h, None) for h in (12, 64)]
    rows += [(8192, 96, [1300, 547, 2048, 963, 271, 3063])]
    rows += [(t, 1, None) for t in (131072, 1048576)]
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--json', required=True, type=Path)
    ap.add_argument('--label', required=True)
    ap.add_argument('--native-exact', action='store_true')
    args = ap.parse_args()
    import flash_kda
    extension = importlib.import_module('flash_kda_C')
    result = dict(label=args.label, torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, device=torch.cuda.get_device_name(),
                  extension_module=extension.__file__,
                  extension_sha256=file_digest(extension.__file__),
                  benchmarks=[], exactness=[])
    if args.native_exact:
        # The TP8 shape was timed but not checked at this T/seed in R19.
        check = exact_case(flash_kda, 8192, 12, 20260911 + 6, 'bf16')
        result['exactness'].append(check)
        if not check['output_equal'] or not check['state_equal']:
            args.json.write_text(json.dumps(result, indent=2) + '\n')
            raise RuntimeError('TP8 native-reference mismatch')
    for i, (T, H, lengths) in enumerate(specs()):
        seed = 20260911 + i
        inp = list(make_inputs(T, H, seed))
        if lengths:
            inp[-1] = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()),
                                   device='cuda', dtype=torch.long)
            inp[-2] = torch.arange(len(lengths) * H * 128 * 128,
                                   device='cuda', dtype=torch.float32).reshape(
                                       len(lengths), H, 128, 128).to(torch.bfloat16)
        fn, out, state = make_call(flash_kda, inp, 'bf16')
        fn()
        torch.cuda.synchronize()
        finite = bool(torch.isfinite(out).all() and torch.isfinite(state).all())
        if not finite:
            raise RuntimeError(f'nonfinite output/state T={T}, H={H}')
        hashes = [digest(x) if x is not None else None for x in inp]
        row = dict(T=T, H=H, nseq=len(lengths) if lengths else 1,
                   seq_lens=lengths, seed=seed, state_mode='bf16',
                   input_digests=hashes, output_digest=digest(out),
                   state_digest=digest(state), finite=finite)
        cfg = dict(warmup=3, iters=10, repeats=3) if T >= 131072 else dict(
            warmup=30, iters=100, repeats=5)
        row['timing_config'] = cfg
        row['timing'] = timed(fn, **cfg)
        result['benchmarks'].append(row)
        args.json.write_text(json.dumps(result, indent=2) + '\n')
        print(f'{args.label}: T={T} H={H} lengths={lengths}: '
              f'{row["timing"]["median_ms"]:.6f} ms', flush=True)
        del fn, inp, out, state
    args.json.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
