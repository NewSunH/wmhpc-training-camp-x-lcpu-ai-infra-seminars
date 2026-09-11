# R19 final validation — B300 job 25286

- Device: NVIDIA B300 SXM6 AC
- Toolchain: CUDA 13.0, PyTorch 2.13.0+cu130
- Candidate macros: register-state cache with 8 key blocks, value-major state layout, loop-external preload, warp state sync, and early output commit.
- Correctness: `tests/test_fwd_full.py -k 'not long'`: 576 passed, 4 deselected (four long-sequence stress cases with T=131072 or 1048576).
- Benchmark: public `flash_kda.fwd`, CUDA events, warmup 30, 100 iterations, 5 repeats, ABBA order.
- NCU is diagnostic only; event timings are authoritative.

See `summary.json` for shape-level medians. The H=96 BF16 state case is 1.643856 ms candidate versus 1.746072 ms baseline (5.85% lower latency); the H=96 no-state case is 1.518976 ms versus 1.745360 ms (12.97% lower). State and output exactness remain unchanged.
