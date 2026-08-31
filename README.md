# TPU Roofline Explorer

An interactive, SRAM-aware roofline model for TPU-style systolic accelerators.
The project combines a browser-based architecture workbench with a reusable
Python CLI model.

## What it models

- GEMM operation count for `C[M,N] = A[M,K] @ B[K,N]`
- Tiled HBM traffic for activations, weights, and outputs
- SRAM-constrained weight and activation reuse
- Systolic MXU padding plus fill/drain utilization
- Compute-bound, HBM-bound, and estimated latency
- Arithmetic intensity and the roofline operating point
- Posit-(4,1), FP2, INT4, INT8, BF16, and FP32 storage precision

## Development branches

- `master` is the stable, tested explorer.
- `full-system-modeling` is the integration branch for the next model revision.
- The integration branch returns to `master` only after the web UI, TypeScript
  model, Python model, lint checks, and both test suites pass.

The full-system revision is planned to add independent activation, weight,
output, and accumulator precision; a four-stage Posit/FP MAC pipeline
(decode, multiply, accumulate, encode) with quire finalization cost; 64 KiB
SRAM banking and explicit A/B/C allocation; loop-order and dataflow controls;
HBM, SRAM, NoC, and compute overlap; multicast and multiple TPU cores/MXUs;
tile pipeline overlap; compiler transformations and fusion; host and launch
overhead; and energy/power estimates. These are analytical approximations, not
cycle-accurate TPU claims, and each coefficient will remain visible in the UI.

## Interactive UI

The interface exposes architecture and workload presets plus editable controls
for peak TOPS, HBM bandwidth, SRAM capacity, matrix dimensions, MXU dimensions,
reuse factors, and precision. The roofline graph and all metrics update live.

```bash
npm ci
npm run dev
```

Production validation:

```bash
npm run lint
npm test
```

## Python model

The Python implementation lives in `python/roofline.py`.

```bash
cd python
python roofline.py \
  --precision bf16 \
  --weight-reuse 8 \
  --activation-reuse 4 \
  --csv results.csv \
  --plot roofline.png

python -m unittest -v
```

Custom example:

```bash
python roofline.py \
  --m 32 --n 11008 --k 4096 \
  --precision 'posit-(4,1)' \
  --peak-tops 131.072 \
  --hbm-bandwidth-gbs 900 \
  --sram-mib 64 \
  --weight-reuse 8 \
  --activation-reuse 4 \
  --mxu-rows 256 --mxu-cols 256 \
  --plot custom-roofline.png
```

## Core equations

```text
Operations = 2*M*N*K
Arithmetic intensity = operations / estimated HBM bytes
Compute latency = operations / (peak compute * MXU utilization)
HBM latency = estimated HBM bytes / HBM bandwidth
Estimated latency = max(compute latency, HBM latency)
```

The `max` latency is a lower-bound roofline estimate that assumes perfect
compute/HBM overlap. SRAM residency is a transparent capacity heuristic rather
than a cache simulator. See `docs/PORTFOLIO_NOTES.md` for interpretation and
limitations.

## Repository structure

```text
app/                         Web application entry point
components/roofline-explorer.tsx
lib/roofline.ts              TypeScript analytical model
python/roofline.py           Python CLI analytical model
python/test_roofline.py      Python tests
tests/roofline-model.test.mjs
docs/PORTFOLIO_NOTES.md
```

## License

MIT
