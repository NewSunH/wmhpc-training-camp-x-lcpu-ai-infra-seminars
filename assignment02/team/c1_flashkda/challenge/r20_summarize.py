#!/usr/bin/env python3
"""ABBA summary requiring equal finite outputs on identical timed inputs."""
import argparse
import json
from pathlib import Path

from r19_summarize import summarize


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('runs', nargs=4, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    args = ap.parse_args()
    result = summarize(args.runs)
    runs = [json.loads(p.read_text()) for p in args.runs]
    for i, row in enumerate(result['cases']):
        cases = [run['benchmarks'][i] for run in runs]
        keys = ('input_digests', 'output_digest', 'state_digest', 'seq_lens', 'timing_config')
        if any(not c['finite'] for c in cases):
            raise ValueError('Nonfinite output/state')
        if any(c[k] != cases[0][k] for c in cases for k in keys):
            raise ValueError(f'Input/output/shape/timing mismatch in case {i}')
        row.update(seq_lens=cases[0]['seq_lens'],
                   timing_config=cases[0]['timing_config'],
                   cross_binary_bitwise_equal=True,
                   output_digest=cases[0]['output_digest'],
                   state_digest=cases[0]['state_digest'])
    result['native_reference_checks'] = [c for r in runs for c in r['exactness']]
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for c in result['cases']:
        print(f'T={c["T"]} H={c["H"]} nseq={c["nseq"]}: '
              f'{c["baseline_ms"]:.6f} -> {c["candidate_ms"]:.6f} ms '
              f'({c["latency_reduction_percent"]:+.3f}% latency reduction)')


if __name__ == '__main__':
    main()
