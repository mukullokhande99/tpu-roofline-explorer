# TPU Roofline Explorer

An interactive, SRAM-aware roofline model for TPU-style systolic accelerators.
The project combines a browser-based architecture workbench with a reusable
Python CLI model.

**Live site:** https://mukullokhande99.github.io/tpu-roofline-explorer/

## Assignments

- **Assignment 1 · TPU Roofline Explorer:** the full-system analytical roofline workbench.
- **Assignment 2 · Flex-PE Research Explorer:** an interactive Flex-PE-to-TPU design-space study at [\`/assignment-2/\`](https://mukullokhande99.github.io/tpu-roofline-explorer/assignment-2/), grounded in the [IEEE TVLSI paper](https://doi.org/10.1109/TVLSI.2025.3553069).

## What it models

- GEMM operation count for `C[M,N] = A[M,K] @ B[K,N]`
- Tiled HBM traffic for activations, weights, and outputs
- SRAM-constrained weight and activation reuse
- Systolic MXU padding plus fill/drain utilization
- Compute-bound, HBM-bound, and estimated latency
- Arithmetic intensity and the roofline operating point
- Posit-(4,1), FP2, INT4, INT8, BF16, and FP32 storage precision
- Posit-8, Posit-16, MXFP4, MXINT8, and NVFP4 storage formats
- Model/layer presets for Llama 3 8B, Qwen2.5 1.5B, ViT-B/16, and custom GEMMs
- Structured-pruning sweep from 0% to 100% in 5% increments

## Development branches

- `master` is the stable, tested explorer.
- `full-system-modeling` is the integration branch for the expanded model.
- The integration branch returns to `master` only after the web UI, TypeScript
  model, Python model, lint checks, and both test suites pass.

The full-system branch adds independent activation, weight,
output, and accumulator precision; a four-stage Posit/FP MAC pipeline
(decode, multiply, accumulate, encode) with quire finalization cost; 64 KiB
SRAM banking and explicit A/B/C allocation; loop-order and dataflow controls;
HBM, SRAM, NoC, and compute overlap; multicast and multiple TPU cores/MXUs;
tile pipeline overlap; compiler transformations and fusion; host and launch
overhead; and energy/power estimates. These are analytical approximations, not
cycle-accurate TPU claims, and each coefficient will remain visible in the UI.

## Interactive UI

The interface exposes architecture and workload presets plus selectable model
layers and independent
activation/weight/output/accumulator precision; MXU/core counts; 64 KiB SRAM
banking and A/B/C allocation; HBM/SRAM/NoC efficiency; multicast; loop order,
dataflow, fusion, compiler, tile-overlap, and host controls; plus energy
coefficients. The roofline graph and latency, traffic, pipeline, utilization,
and energy breakdowns update live.

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

The Python implementation lives in `python/roofline.py`. Its `GemmWorkload`
accepts `pruning_percent` at 0, 5, 10, …, 100; pruning reduces useful MACs and
weight traffic while leaving dense MXU slot capacity visible in utilization.

```bash
cd python
python roofline.py \
  --precision mxfp4 \
  --pruning-percent 25 \
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
Compute latency = operations / (MXU peak * workers * utilization * compiler efficiency)
Resource latency = compute, HBM, SRAM, or NoC bytes / effective bandwidth
Device latency = max(resources) + (1 - overlap) * (sum(resources) - max(resources))
End-to-end latency = device latency + host overhead + kernels * launch overhead
Energy = compute + HBM + SRAM + NoC dynamic energy + static power * latency
```

At 100% overlap the device equation becomes the normal roofline maximum. Posit
and floating-point MACs use a four-stage Decode → Multiply → Accumulate → Encode
pipeline whose stages affect tile fill/drain, not steady-state issue rate.
Quire accumulation adds finalization cycles. SRAM residency, reuse/dataflow,
spill, compiler, and energy terms are transparent heuristics rather than a
cycle-accurate TPU simulator. See `docs/PORTFOLIO_NOTES.md` for interpretation
and limitations.

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
