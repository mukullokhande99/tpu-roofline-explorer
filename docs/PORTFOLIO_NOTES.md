# Portfolio Notes: SRAM-Aware TPU Roofline

## Design question

How do MXU width, HBM bandwidth, SRAM capacity, operand reuse, and precision
interact for large prefill GEMMs versus token-limited decode GEMMs?

## Experimental setup

- Architecture A: one 128x128 MXU, 32.768 TOPS
- Architecture B: one 256x256 MXU, 131.072 TOPS
- Both: 900 GB/s HBM and 64 MiB SRAM
- Precision: BF16
- Requested weight reuse: 8x
- Requested activation reuse: 4x
- Performance: `max(compute time, HBM time)`
- Compute efficiency: padding plus systolic fill/drain
- Memory efficiency: tiled reloads reduced by SRAM-achievable reuse

## Default results

| Workload | A latency | B latency | B speedup | A/B arithmetic intensity |
|---|---:|---:|---:|---:|
| 4096x4096x4096 | 4.45 ms | 1.18 ms | 3.77x | 315 / 585 op/B |
| Up-projection prefill | 11.97 ms | 3.17 ms | 3.78x | 315 / 585 op/B |
| Down-projection prefill | 11.53 ms | 2.95 ms | 3.91x | 331 / 643 op/B |
| Up-projection, M=32 | 374.10 us | 198.06 us | 1.89x | 29.9 / 30.8 op/B |
| Up-projection, M=1 | 374.10 us | 198.06 us | 1.89x | 1.0 / 1.0 op/B |

## Architectural interpretation

The larger MXU is compelling for prefill-sized matrices: its theoretical 4x
compute increase becomes approximately 3.8-3.9x lower compute time after
systolic overhead. It also needs fewer output tiles, so the same reuse factors
produce less HBM traffic and higher arithmetic intensity.

The result changes sharply for decode and small-token work. With only one or 32
output rows, most physical rows are inactive. Weight reuse is clipped to 1x
because there is only one M tile, so the large 4096x11008 weight matrix must be
streamed from HBM. Architecture B still wins, but its incremental hardware
produces much less incremental performance.

The roofline graph separates three effects:

1. Roof position: peak TOPS and HBM bandwidth define theoretical ceilings.
2. Horizontal position: precision, tiling, SRAM, and reuse determine arithmetic
   intensity.
3. Vertical gap below the roof: MXU shape utilization lowers achieved compute.

The design implication is stronger than “use a larger MXU”: a production TPU
benefits from partitionable arrays, multi-tenant packing, adequate SRAM, and
different prefill/decode execution modes.

## Caveat

These are analytic estimates rather than cycle-accurate simulation or silicon
measurements. SRAM residency is a capacity heuristic, and real traffic depends
on loop order, double buffering, bank conflicts, dataflow, fusion, and compiler
scheduling. Calibration against traces or counters is the next validation step.

