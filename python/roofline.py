"""SRAM-aware roofline model for TPU-style GEMM accelerators.

The modeled operation is C[M, N] = A[M, K] @ B[K, N]. This first-order model
captures tiled HBM traffic, SRAM-constrained reuse, arithmetic intensity,
systolic MXU utilization, roofline latency, and a presentation-ready plot.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


PRECISION_BITS = {
    "posit-(4,1)": 4,
    "posit-8": 8,
    "posit-16": 16,
    "fp2": 2,
    "int4": 4,
    "int8": 8,
    "mxfp4": 4,
    "mxint8": 8,
    "nvfp4": 4,
    "bf16": 16,
    "fp32": 32,
}


@dataclass(frozen=True)
class Architecture:
    name: str
    peak_compute_tops: float
    hbm_bandwidth_gbs: float
    sram_capacity_mib: float
    mxu_rows: int
    mxu_cols: int

    def __post_init__(self) -> None:
        if self.peak_compute_tops <= 0 or self.hbm_bandwidth_gbs <= 0:
            raise ValueError("Peak compute and HBM bandwidth must be positive")
        if self.sram_capacity_mib < 0:
            raise ValueError("SRAM capacity cannot be negative")
        if self.mxu_rows <= 0 or self.mxu_cols <= 0:
            raise ValueError("MXU dimensions must be positive integers")


@dataclass(frozen=True)
class GemmWorkload:
    name: str
    m: int
    n: int
    k: int
    precision: str = "bf16"
    weight_reuse_factor: float = 8.0
    activation_reuse_factor: float = 4.0
    pruning_percent: int = 0

    def __post_init__(self) -> None:
        normalized = self.precision.lower()
        if min(self.m, self.n, self.k) <= 0:
            raise ValueError("M, N, and K must be positive integers")
        if normalized not in PRECISION_BITS:
            raise ValueError(
                f"Unsupported precision {self.precision!r}; choose from "
                + ", ".join(PRECISION_BITS)
            )
        if self.weight_reuse_factor < 1 or self.activation_reuse_factor < 1:
            raise ValueError("Reuse factors must be at least 1")
        if not 0 <= self.pruning_percent <= 100 or self.pruning_percent % 5:
            raise ValueError("Pruning must be an integer from 0 to 100 in 5% steps")


@dataclass(frozen=True)
class MemoryTraffic:
    activation_read_bytes: float
    weight_read_bytes: float
    output_bytes: float
    total_bytes: float
    tile_working_set_bytes: float
    sram_residency: float
    effective_weight_reuse: float
    effective_activation_reuse: float


@dataclass(frozen=True)
class RooflineResult:
    architecture: str
    workload: str
    m: int
    n: int
    k: int
    precision: str
    peak_compute_tops: float
    hbm_bandwidth_gbs: float
    sram_capacity_mib: float
    requested_weight_reuse: float
    requested_activation_reuse: float
    effective_weight_reuse: float
    effective_activation_reuse: float
    sram_residency: float
    tile_working_set_bytes: float
    total_operations: int
    dense_operations: int
    pruning_percent: int
    bytes_transferred: float
    arithmetic_intensity_ops_per_byte: float
    ridge_point_ops_per_byte: float
    mxu_utilization: float
    compute_bound_latency_s: float
    memory_bound_latency_s: float
    estimated_latency_s: float
    estimated_performance_tops: float
    bottleneck: str


def precision_bytes(precision: str) -> float:
    return PRECISION_BITS[precision.lower()] / 8.0


def total_operations(workload: GemmWorkload) -> int:
    """Return 2*M*N*K operations: one multiply plus one add per MAC."""
    return 2 * workload.m * workload.n * workload.k


def _tile_counts(workload: GemmWorkload, architecture: Architecture) -> tuple[int, int]:
    return (
        math.ceil(workload.m / architecture.mxu_rows),
        math.ceil(workload.n / architecture.mxu_cols),
    )


def estimate_hbm_traffic(
    workload: GemmWorkload,
    architecture: Architecture,
    *,
    read_output: bool = False,
) -> MemoryTraffic:
    """Estimate HBM bytes from tiled reloads, reuse, and SRAM residency.

    With no reuse, an A tile is reloaded for every N tile and a B tile for every
    M tile. Requested reuse is clipped to the available tile count. SRAM limits
    reuse using a simple residency ratio for one output tile's A, B, and C
    working set. Traffic never falls below one compulsory read of A/B.
    """
    value_bytes = precision_bytes(workload.precision)
    density = 1.0 - workload.pruning_percent / 100.0
    tiles_m, tiles_n = _tile_counts(workload, architecture)
    tile_m = min(workload.m, architecture.mxu_rows)
    tile_n = min(workload.n, architecture.mxu_cols)

    a_compulsory = workload.m * workload.k * value_bytes
    b_compulsory = workload.k * workload.n * value_bytes * density
    c_write = workload.m * workload.n * value_bytes
    if read_output:
        c_write *= 2

    tile_working_set = (
        tile_m * workload.k + workload.k * tile_n + tile_m * tile_n
    ) * value_bytes
    sram_bytes = architecture.sram_capacity_mib * 2**20
    residency = min(1.0, sram_bytes / tile_working_set) if tile_working_set else 1.0

    requested_weight_reuse = min(workload.weight_reuse_factor, float(tiles_m))
    requested_activation_reuse = min(workload.activation_reuse_factor, float(tiles_n))
    effective_weight_reuse = 1.0 + (requested_weight_reuse - 1.0) * residency
    effective_activation_reuse = 1.0 + (requested_activation_reuse - 1.0) * residency

    activation_reads = max(
        a_compulsory, a_compulsory * tiles_n / effective_activation_reuse
    )
    weight_reads = max(b_compulsory, b_compulsory * tiles_m / effective_weight_reuse)
    total = activation_reads + weight_reads + c_write
    return MemoryTraffic(
        activation_read_bytes=activation_reads,
        weight_read_bytes=weight_reads,
        output_bytes=c_write,
        total_bytes=total,
        tile_working_set_bytes=tile_working_set,
        sram_residency=residency,
        effective_weight_reuse=effective_weight_reuse,
        effective_activation_reuse=effective_activation_reuse,
    )


def approximate_bytes_transferred(
    workload: GemmWorkload,
    architecture: Architecture,
    *,
    read_output: bool = False,
) -> float:
    """Compatibility helper returning total estimated HBM traffic."""
    return estimate_hbm_traffic(
        workload, architecture, read_output=read_output
    ).total_bytes


def mxu_utilization(workload: GemmWorkload, architecture: Architecture) -> float:
    """Estimate useful/scheduled work including padding and fill/drain."""
    rows, cols = architecture.mxu_rows, architecture.mxu_cols
    tiles_m, tiles_n = _tile_counts(workload, architecture)
    scheduled_mac_slots = tiles_m * tiles_n * rows * cols * (
        workload.k + rows + cols - 2
    )
    useful_macs = workload.m * workload.n * workload.k * (1.0 - workload.pruning_percent / 100.0)
    return useful_macs / scheduled_mac_slots


def evaluate(
    workload: GemmWorkload,
    architecture: Architecture,
    *,
    read_output: bool = False,
) -> RooflineResult:
    dense_ops = total_operations(workload)
    ops = int(dense_ops * (1.0 - workload.pruning_percent / 100.0))
    traffic = estimate_hbm_traffic(workload, architecture, read_output=read_output)
    intensity = ops / traffic.total_bytes
    utilization = mxu_utilization(workload, architecture)

    effective_compute_ops_s = architecture.peak_compute_tops * 1e12 * utilization
    compute_latency = ops / effective_compute_ops_s
    memory_latency = traffic.total_bytes / (architecture.hbm_bandwidth_gbs * 1e9)
    estimated_latency = max(compute_latency, memory_latency)
    ridge_point = architecture.peak_compute_tops * 1000 / architecture.hbm_bandwidth_gbs

    return RooflineResult(
        architecture=architecture.name,
        workload=workload.name,
        m=workload.m,
        n=workload.n,
        k=workload.k,
        precision=workload.precision.lower(),
        peak_compute_tops=architecture.peak_compute_tops,
        hbm_bandwidth_gbs=architecture.hbm_bandwidth_gbs,
        sram_capacity_mib=architecture.sram_capacity_mib,
        requested_weight_reuse=workload.weight_reuse_factor,
        requested_activation_reuse=workload.activation_reuse_factor,
        effective_weight_reuse=traffic.effective_weight_reuse,
        effective_activation_reuse=traffic.effective_activation_reuse,
        sram_residency=traffic.sram_residency,
        tile_working_set_bytes=traffic.tile_working_set_bytes,
        total_operations=ops,
        dense_operations=dense_ops,
        pruning_percent=workload.pruning_percent,
        bytes_transferred=traffic.total_bytes,
        arithmetic_intensity_ops_per_byte=intensity,
        ridge_point_ops_per_byte=ridge_point,
        mxu_utilization=utilization,
        compute_bound_latency_s=compute_latency,
        memory_bound_latency_s=memory_latency,
        estimated_latency_s=estimated_latency,
        estimated_performance_tops=ops / estimated_latency / 1e12,
        bottleneck="compute" if compute_latency >= memory_latency else "memory",
    )


def default_architectures() -> list[Architecture]:
    """Single-MXU designs at 1 GHz, with equal HBM bandwidth and SRAM."""
    return [
        Architecture("A-128x128", 32.768, 900.0, 64.0, 128, 128),
        Architecture("B-256x256", 131.072, 900.0, 64.0, 256, 256),
    ]


def default_workloads(
    precision: str = "bf16",
    weight_reuse_factor: float = 8.0,
    activation_reuse_factor: float = 4.0,
    pruning_percent: int = 0,
) -> list[GemmWorkload]:
    common = (precision, weight_reuse_factor, activation_reuse_factor, pruning_percent)
    return [
        GemmWorkload("square-4096", 4096, 4096, 4096, *common),
        GemmWorkload("up-projection-prefill", 4096, 11008, 4096, *common),
        GemmWorkload("down-projection-prefill", 4096, 4096, 11008, *common),
        GemmWorkload("up-projection-small-batch", 32, 11008, 4096, *common),
        GemmWorkload("up-projection-single-token", 1, 11008, 4096, *common),
    ]


def evaluate_many(
    architectures: Iterable[Architecture],
    workloads: Iterable[GemmWorkload],
    *,
    read_output: bool = False,
) -> list[RooflineResult]:
    return [
        evaluate(workload, architecture, read_output=read_output)
        for workload in workloads
        for architecture in architectures
    ]


def _latency_text(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    return f"{seconds * 1e3:.2f} ms"


def markdown_table(results: Sequence[RooflineResult]) -> str:
    header = (
        "| Workload | Arch | Precision | AI (op/B) | HBM bytes | SRAM fit | "
        "W/A reuse | MXU util. | Tcompute | Tmemory | Estimated | Bound |\n"
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    rows = []
    for result in results:
        rows.append(
            "| {workload} | {architecture} | {precision} | {ai:.2f} | "
            "{hbm:.2f} MB | {sram:.1%} | {wr:.2f}x/{ar:.2f}x | {util:.2%} | "
            "{tc} | {tm} | {te} | {bound} |".format(
                workload=result.workload,
                architecture=result.architecture,
                precision=result.precision,
                ai=result.arithmetic_intensity_ops_per_byte,
                hbm=result.bytes_transferred / 1e6,
                sram=result.sram_residency,
                wr=result.effective_weight_reuse,
                ar=result.effective_activation_reuse,
                util=result.mxu_utilization,
                tc=_latency_text(result.compute_bound_latency_s),
                tm=_latency_text(result.memory_bound_latency_s),
                te=_latency_text(result.estimated_latency_s),
                bound=result.bottleneck,
            )
        )
    return "\n".join([header, *rows])


def write_csv(results: Sequence[RooflineResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def plot_roofline(results: Sequence[RooflineResult], path: Path) -> None:
    """Generate a log-log roofline plot with workload operating points."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.lines import Line2D
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError("Plotting requires matplotlib and numpy") from exc

    architectures = {result.architecture: result for result in results}
    max_ai = max(result.arithmetic_intensity_ops_per_byte for result in results)
    x_values = np.logspace(-1, math.log10(max(1e4, max_ai * 2)), 500)

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    colors = ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd"]
    workload_markers = {
        "square-4096": ("o", "4096 cube"),
        "up-projection-prefill": ("s", "Up-projection prefill"),
        "down-projection-prefill": ("^", "Down-projection prefill"),
        "up-projection-small-batch": ("D", "Up-projection, M=32"),
        "up-projection-single-token": ("X", "Up-projection, M=1"),
    }

    for color, (architecture_name, representative) in zip(colors, architectures.items()):
        bandwidth_ceiling = representative.hbm_bandwidth_gbs * x_values / 1000.0
        roof = np.minimum(bandwidth_ceiling, representative.peak_compute_tops)
        ax.plot(
            x_values,
            roof,
            color=color,
            linewidth=2.5,
            label=(
                f"{architecture_name}: {representative.peak_compute_tops:g} TOPS, "
                f"{representative.hbm_bandwidth_gbs:g} GB/s"
            ),
        )
        arch_results = [r for r in results if r.architecture == architecture_name]
        for result in arch_results:
            marker, _ = workload_markers.get(result.workload, ("o", result.workload))
            is_square = result.workload == "square-4096"
            ax.scatter(
                result.arithmetic_intensity_ops_per_byte,
                result.estimated_performance_tops,
                facecolor="none" if is_square else color,
                marker=marker,
                edgecolor=color if is_square else "white",
                linewidth=2.0 if is_square else 0.8,
                s=130 if is_square else 80,
                zorder=4 if is_square else 3,
            )

    precision = results[0].precision.upper()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, max(1e4, max_ai * 2))
    ax.set_ylim(0.05, max(r.peak_compute_tops for r in results) * 1.7)
    ax.grid(True, which="both", linewidth=0.5, alpha=0.25)
    ax.set_xlabel("Arithmetic intensity (operations / HBM byte)")
    ax.set_ylabel("Estimated performance (TOPS)")
    ax.set_title(f"TPU-style roofline: {precision}, SRAM-aware HBM traffic")
    architecture_legend = ax.legend(loc="upper left", frameon=False, title="Architecture")
    ax.add_artist(architecture_legend)
    marker_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            color="none",
            markerfacecolor="#555555",
            markeredgecolor="white",
            markersize=8,
            label=label,
        )
        for marker, label in workload_markers.values()
    ]
    ax.legend(
        handles=marker_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.79),
        frameon=False,
        title="Workload",
    )
    ax.text(
        0.99,
        0.02,
        "Points include MXU utilization; roofs show theoretical ceilings.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#555555",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, help="Rows/tokens in A and C")
    parser.add_argument("--n", type=int, help="Columns/features in B and C")
    parser.add_argument("--k", type=int, help="Reduction dimension")
    parser.add_argument(
        "--precision",
        "--datatype",
        dest="precision",
        default="bf16",
        choices=list(PRECISION_BITS),
    )
    parser.add_argument("--peak-tops", type=float, help="Peak compute in TOPS")
    parser.add_argument(
        "--hbm-bandwidth-gbs",
        "--bandwidth-gbs",
        dest="hbm_bandwidth_gbs",
        type=float,
        help="HBM bandwidth in GB/s",
    )
    parser.add_argument("--sram-mib", type=float, help="On-chip SRAM capacity in MiB")
    parser.add_argument("--weight-reuse", type=float, default=8.0)
    parser.add_argument("--activation-reuse", type=float, default=4.0)
    parser.add_argument("--pruning-percent", type=int, default=0, choices=range(0, 101, 5), help="Structured pruning percentage (0..100 in 5% steps)")
    parser.add_argument("--mxu-rows", type=int, help="MXU row count")
    parser.add_argument("--mxu-cols", type=int, help="MXU column count")
    parser.add_argument("--name", default="custom", help="Custom architecture name")
    parser.add_argument(
        "--read-output",
        action="store_true",
        help="Include a C read for beta*C accumulation",
    )
    parser.add_argument("--csv", type=Path, help="Write machine-readable results")
    parser.add_argument("--plot", type=Path, help="Write a roofline graph (PNG/PDF/SVG)")
    return parser


def _custom_mode(args: argparse.Namespace) -> bool:
    custom_fields = (
        args.m,
        args.n,
        args.k,
        args.peak_tops,
        args.hbm_bandwidth_gbs,
        args.sram_mib,
        args.mxu_rows,
        args.mxu_cols,
    )
    if not any(value is not None for value in custom_fields):
        return False
    if not all(value is not None for value in custom_fields):
        raise SystemExit(
            "Custom mode requires --m --n --k --peak-tops --hbm-bandwidth-gbs "
            "--sram-mib --mxu-rows and --mxu-cols"
        )
    return True


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if _custom_mode(args):
        architectures = [
            Architecture(
                args.name,
                args.peak_tops,
                args.hbm_bandwidth_gbs,
                args.sram_mib,
                args.mxu_rows,
                args.mxu_cols,
            )
        ]
        workloads = [
            GemmWorkload(
                "custom-gemm",
                args.m,
                args.n,
                args.k,
                args.precision,
                args.weight_reuse,
                args.activation_reuse,
                args.pruning_percent,
            )
        ]
    else:
        architectures = default_architectures()
        workloads = default_workloads(
            args.precision, args.weight_reuse, args.activation_reuse, args.pruning_percent
        )

    results = evaluate_many(architectures, workloads, read_output=args.read_output)
    print(markdown_table(results))
    if args.csv:
        write_csv(results, args.csv)
        print(f"\nWrote {args.csv}")
    if args.plot:
        plot_roofline(results, args.plot)
        print(f"Wrote {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
