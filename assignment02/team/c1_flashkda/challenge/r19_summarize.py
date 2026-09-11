#!/usr/bin/env python3
"""Summarize four explicit ABBA JSON files, validating sample and binary identity."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics


def summarize(paths: list[Path]) -> dict:
    if len(paths) != 4:
        raise ValueError("Exactly four files in A,B,B,A order are required")
    runs = [json.loads(p.read_text()) for p in paths]
    hashes = [r["extension_sha256"] for r in runs]
    if hashes[0] != hashes[3] or hashes[1] != hashes[2] or hashes[0] == hashes[1]:
        raise ValueError("ABBA must use two distinct, stable extension binaries")
    for run in runs:
        for case in run.get("exactness", []):
            if not case["output_equal"] or case.get("state_equal") is False:
                raise ValueError("Cannot summarize a run with failed exactness")
    keys = ("T", "H", "nseq", "state_mode", "seed")
    cases = runs[0]["benchmarks"]
    if not cases or any(len(r["benchmarks"]) != len(cases) for r in runs):
        raise ValueError("All runs must contain the same nonempty benchmark matrix")
    rows = []
    for i, case in enumerate(cases):
        medians = []
        for run in runs:
            other = run["benchmarks"][i]
            if any(case[k] != other[k] for k in keys):
                raise ValueError("Shape, state mode, or seed differs across paired runs")
            samples = other["timing"]["samples_ms"]
            if not samples or any(not math.isfinite(x) or x <= 0 for x in samples):
                raise ValueError("Samples must be finite, positive event times")
            median = statistics.median(samples)
            if not math.isclose(median, other["timing"]["median_ms"], rel_tol=1e-9):
                raise ValueError("Stored median differs from raw sample median")
            medians.append(median)
        base = statistics.median([medians[0], medians[3]])
        candidate = statistics.median([medians[1], medians[2]])
        rows.append({**{k: case[k] for k in keys},
                     "baseline_ms": base, "candidate_ms": candidate,
                     "baseline_run_medians_ms": [medians[0], medians[3]],
                     "candidate_run_medians_ms": [medians[1], medians[2]],
                     "latency_reduction_percent": 100 * (1 - candidate / base),
                     "speedup_ratio": base / candidate,
                     "speedup_percent": 100 * (base / candidate - 1)})
    return {"inputs": [str(p) for p in paths],
            "aggregate": "median of the two run medians for each variant",
            "baseline_extension_sha256": hashes[0],
            "candidate_extension_sha256": hashes[1], "cases": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs=4, metavar="JSON")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = summarize(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for case in result["cases"]:
        print(f"H={case['H']} nseq={case['nseq']} {case['state_mode']}: "
              f"{case['baseline_ms']:.6f} -> {case['candidate_ms']:.6f} ms, "
              f"latency reduction {case['latency_reduction_percent']:.3f}%")


if __name__ == "__main__":
    main()
