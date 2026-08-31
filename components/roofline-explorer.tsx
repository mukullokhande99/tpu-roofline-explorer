"use client";

import { useMemo, useState } from "react";
import { Activity, Boxes, Cpu, Gauge, MemoryStick, Waves } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import {
  type Architecture,
  type Precision,
  type RooflineResult,
  type Workload,
  evaluateRoofline,
  formatBytes,
  formatLatency,
} from "@/lib/roofline";

const ARCHITECTURES: Record<string, Architecture> = {
  a: {
    name: "A · 128×128",
    peakComputeTops: 32.768,
    hbmBandwidthGbs: 900,
    sramCapacityMib: 64,
    mxuRows: 128,
    mxuCols: 128,
  },
  b: {
    name: "B · 256×256",
    peakComputeTops: 131.072,
    hbmBandwidthGbs: 900,
    sramCapacityMib: 64,
    mxuRows: 256,
    mxuCols: 256,
  },
};

const WORKLOADS = {
  square: { name: "4096 cube", m: 4096, n: 4096, k: 4096 },
  up: { name: "Up-projection prefill", m: 4096, n: 11008, k: 4096 },
  down: { name: "Down-projection prefill", m: 4096, n: 4096, k: 11008 },
  m32: { name: "Up-projection · M=32", m: 32, n: 11008, k: 4096 },
  m1: { name: "Up-projection · M=1", m: 1, n: 11008, k: 4096 },
};

const PRECISIONS: Array<{ value: Precision; label: string }> = [
  { value: "posit-(4,1)", label: "Posit-(4,1)" },
  { value: "fp2", label: "FP2" },
  { value: "int4", label: "INT4" },
  { value: "int8", label: "INT8" },
  { value: "bf16", label: "BF16" },
  { value: "fp32", label: "FP32" },
];

function NumberField({
  label,
  value,
  onChange,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: number;
}) {
  return (
    <label className="control-field">
      <span>{label}</span>
      <Input
        type="number"
        min={step}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function MetricCard({
  icon: Icon,
  label,
  value,
  detail,
}: {
  icon: typeof Cpu;
  label: string;
  value: string;
  detail: string;
}) {
  return (
    <Card className="metric-card">
      <CardContent className="p-5">
        <div className="metric-label">
          <Icon aria-hidden="true" />
          <span>{label}</span>
        </div>
        <div className="metric-value">{value}</div>
        <div className="metric-detail">{detail}</div>
      </CardContent>
    </Card>
  );
}

function RooflineGraph({
  architecture,
  workload,
  result,
}: {
  architecture: Architecture;
  workload: Workload;
  result: RooflineResult;
}) {
  const width = 900;
  const height = 430;
  const margin = { left: 72, right: 28, top: 26, bottom: 62 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const xMin = 0.1;
  const xMax = Math.max(10000, result.arithmeticIntensity * 2);
  const yMin = 0.05;
  const yMax = architecture.peakComputeTops * 1.75;
  const logX = (value: number) =>
    margin.left +
    ((Math.log10(value) - Math.log10(xMin)) /
      (Math.log10(xMax) - Math.log10(xMin))) *
      plotWidth;
  const logY = (value: number) =>
    margin.top +
    (1 -
      (Math.log10(value) - Math.log10(yMin)) /
        (Math.log10(yMax) - Math.log10(yMin))) *
      plotHeight;
  const samples = Array.from({ length: 100 }, (_, index) => {
    const ratio = index / 99;
    const intensity = 10 ** (Math.log10(xMin) + ratio * (Math.log10(xMax) - Math.log10(xMin)));
    const memoryCeiling = (architecture.hbmBandwidthGbs * intensity) / 1000;
    return {
      intensity,
      memoryCeiling,
      roof: Math.min(memoryCeiling, architecture.peakComputeTops),
    };
  });
  const roofPath = samples
    .map((point, index) => `${index ? "L" : "M"}${logX(point.intensity)},${logY(point.roof)}`)
    .join(" ");
  const memoryPath = samples
    .filter((point) => point.memoryCeiling <= yMax)
    .map(
      (point, index) =>
        `${index ? "L" : "M"}${logX(point.intensity)},${logY(point.memoryCeiling)}`,
    )
    .join(" ");
  const xTicks = [0.1, 1, 10, 100, 1000, 10000].filter((tick) => tick <= xMax);
  const yTicks = [0.1, 1, 10, 100, 1000].filter((tick) => tick <= yMax);
  const pointX = logX(result.arithmeticIntensity);
  const pointY = logY(result.estimatedPerformanceTops);
  const labelAnchor = pointX > width * 0.74 ? "end" : "start";
  const labelX = pointX + (labelAnchor === "end" ? -12 : 12);

  return (
    <div className="roofline-wrap">
      <div className="graph-heading">
        <div>
          <p className="eyebrow">OPERATING POINT</p>
          <h2>{architecture.name} roofline</h2>
        </div>
        <div className="graph-legend" aria-label="Graph legend">
          <span><i className="legend-roof" />Effective roof</span>
          <span><i className="legend-memory" />HBM ceiling</span>
          <span><i className="legend-point" />{workload.name}</span>
        </div>
      </div>
      <svg
        className="roofline-svg"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-labelledby="roofline-title roofline-description"
      >
        <title id="roofline-title">Roofline for {architecture.name}</title>
        <desc id="roofline-description">
          Logarithmic arithmetic intensity versus estimated performance with HBM and compute ceilings.
        </desc>
        <defs>
          <linearGradient id="roofline-glow" x1="0" x2="1">
            <stop offset="0" stopColor="var(--cyan)" />
            <stop offset="1" stopColor="var(--violet)" />
          </linearGradient>
        </defs>
        <rect
          x={margin.left}
          y={margin.top}
          width={plotWidth}
          height={plotHeight}
          className="plot-frame"
        />
        {xTicks.map((tick) => (
          <g key={`x-${tick}`}>
            <line
              x1={logX(tick)}
              x2={logX(tick)}
              y1={margin.top}
              y2={height - margin.bottom}
              className="grid-line"
            />
            <text x={logX(tick)} y={height - margin.bottom + 23} textAnchor="middle" className="axis-tick">
              {tick}
            </text>
          </g>
        ))}
        {yTicks.map((tick) => (
          <g key={`y-${tick}`}>
            <line
              x1={margin.left}
              x2={width - margin.right}
              y1={logY(tick)}
              y2={logY(tick)}
              className="grid-line"
            />
            <text x={margin.left - 14} y={logY(tick) + 4} textAnchor="end" className="axis-tick">
              {tick}
            </text>
          </g>
        ))}
        <path d={memoryPath} className="memory-line" />
        <path d={roofPath} className="roof-line" />
        <line x1={pointX} x2={pointX} y1={pointY} y2={height - margin.bottom} className="point-guide" />
        <line x1={margin.left} x2={pointX} y1={pointY} y2={pointY} className="point-guide" />
        <circle cx={pointX} cy={pointY} r="8" className="workload-point" />
        <circle cx={pointX} cy={pointY} r="14" className="workload-halo" />
        <text x={labelX} y={pointY - 14} textAnchor={labelAnchor} className="point-label">
          {result.arithmeticIntensity.toFixed(1)} op/B · {result.estimatedPerformanceTops.toFixed(2)} TOPS
        </text>
        <text x={margin.left + plotWidth / 2} y={height - 14} textAnchor="middle" className="axis-title">
          Arithmetic intensity · operations per HBM byte
        </text>
        <text
          x="18"
          y={margin.top + plotHeight / 2}
          textAnchor="middle"
          transform={`rotate(-90 18 ${margin.top + plotHeight / 2})`}
          className="axis-title"
        >
          Performance · TOPS
        </text>
      </svg>
    </div>
  );
}

export function RooflineExplorer() {
  const [architectureKey, setArchitectureKey] = useState("a");
  const [workloadKey, setWorkloadKey] = useState<keyof typeof WORKLOADS>("square");
  const [architecture, setArchitecture] = useState<Architecture>(ARCHITECTURES.a);
  const [workload, setWorkload] = useState<Workload>({
    ...WORKLOADS.square,
    precision: "bf16",
    weightReuseFactor: 8,
    activationReuseFactor: 4,
  });
  const result = useMemo(
    () => evaluateRoofline(architecture, workload),
    [architecture, workload],
  );

  const selectArchitecture = (key: string) => {
    setArchitectureKey(key);
    setArchitecture(ARCHITECTURES[key]);
  };
  const selectWorkload = (key: string) => {
    const typedKey = key as keyof typeof WORKLOADS;
    setWorkloadKey(typedKey);
    setWorkload((current) => ({ ...current, ...WORKLOADS[typedKey] }));
  };
  const updateArchitecture = (field: keyof Architecture, value: number) =>
    setArchitecture((current) => ({ ...current, [field]: value }));
  const updateWorkload = (field: keyof Workload, value: number | Precision) =>
    setWorkload((current) => ({ ...current, [field]: value }));

  return (
    <main className="explorer-shell">
      <header className="site-header">
        <div className="brand-mark" aria-hidden="true"><span /><span /><span /></div>
        <div>
          <p className="eyebrow">ARCHITECTURE LAB · 01</p>
          <h1>TPU Roofline Explorer</h1>
          <p className="header-copy">
            Shape-aware GEMM performance with systolic utilization, SRAM-constrained reuse, and HBM traffic.
          </p>
        </div>
        <Badge variant="outline" className="model-badge">ANALYTIC MODEL</Badge>
      </header>

      <section className="explorer-grid">
        <Card className="controls-panel">
          <CardHeader className="controls-header">
            <CardTitle>Model inputs</CardTitle>
            <span>All values update live</span>
          </CardHeader>
          <CardContent className="control-stack">
            <div className="control-section">
              <p className="section-label">Presets</p>
              <label className="control-field">
                <span>Architecture</span>
                <Select value={architectureKey} onValueChange={selectArchitecture}>
                  <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="a">A · 128×128 MXU</SelectItem>
                    <SelectItem value="b">B · 256×256 MXU</SelectItem>
                  </SelectContent>
                </Select>
              </label>
              <label className="control-field">
                <span>Workload</span>
                <Select value={workloadKey} onValueChange={selectWorkload}>
                  <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {Object.entries(WORKLOADS).map(([key, value]) => (
                      <SelectItem key={key} value={key}>{value.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </label>
              <label className="control-field">
                <span>Precision</span>
                <Select
                  value={workload.precision}
                  onValueChange={(value) => updateWorkload("precision", value as Precision)}
                >
                  <SelectTrigger className="w-full"><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {PRECISIONS.map((precision) => (
                      <SelectItem key={precision.value} value={precision.value}>{precision.label}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </label>
            </div>

            <div className="control-section">
              <p className="section-label">Matrix · C[M,N] = A[M,K] × B[K,N]</p>
              <div className="number-grid">
                <NumberField label="M" value={workload.m} onChange={(value) => updateWorkload("m", value)} />
                <NumberField label="N" value={workload.n} onChange={(value) => updateWorkload("n", value)} />
                <NumberField label="K" value={workload.k} onChange={(value) => updateWorkload("k", value)} />
              </div>
            </div>

            <div className="control-section">
              <p className="section-label">Compute fabric</p>
              <NumberField
                label="Peak compute · TOPS"
                value={architecture.peakComputeTops}
                step={0.001}
                onChange={(value) => updateArchitecture("peakComputeTops", value)}
              />
              <div className="number-grid two">
                <NumberField label="MXU rows" value={architecture.mxuRows} onChange={(value) => updateArchitecture("mxuRows", value)} />
                <NumberField label="MXU cols" value={architecture.mxuCols} onChange={(value) => updateArchitecture("mxuCols", value)} />
              </div>
            </div>

            <div className="control-section slider-stack">
              <p className="section-label">Memory system</p>
              <label className="slider-field">
                <span>HBM bandwidth <strong>{architecture.hbmBandwidthGbs} GB/s</strong></span>
                <Slider
                  min={100}
                  max={2000}
                  step={50}
                  value={[architecture.hbmBandwidthGbs]}
                  onValueChange={([value]) => updateArchitecture("hbmBandwidthGbs", value)}
                />
              </label>
              <label className="slider-field">
                <span>SRAM capacity <strong>{architecture.sramCapacityMib} MiB</strong></span>
                <Slider
                  min={1}
                  max={256}
                  step={1}
                  value={[architecture.sramCapacityMib]}
                  onValueChange={([value]) => updateArchitecture("sramCapacityMib", value)}
                />
              </label>
              <label className="slider-field">
                <span>Weight reuse <strong>{workload.weightReuseFactor}×</strong></span>
                <Slider
                  min={1}
                  max={32}
                  step={1}
                  value={[workload.weightReuseFactor]}
                  onValueChange={([value]) => updateWorkload("weightReuseFactor", value)}
                />
              </label>
              <label className="slider-field">
                <span>Activation reuse <strong>{workload.activationReuseFactor}×</strong></span>
                <Slider
                  min={1}
                  max={32}
                  step={1}
                  value={[workload.activationReuseFactor]}
                  onValueChange={([value]) => updateWorkload("activationReuseFactor", value)}
                />
              </label>
            </div>
          </CardContent>
        </Card>

        <div className="results-panel" aria-live="polite">
          <div className="metric-grid">
            <MetricCard
              icon={Activity}
              label="Estimated latency"
              value={formatLatency(result.estimatedLatencySeconds)}
              detail={`${result.bottleneck}-bound`}
            />
            <MetricCard
              icon={Gauge}
              label="Arithmetic intensity"
              value={result.arithmeticIntensity.toFixed(2)}
              detail={`ridge · ${result.ridgePoint.toFixed(2)} op/B`}
            />
            <MetricCard
              icon={Cpu}
              label="MXU utilization"
              value={`${(result.mxuUtilization * 100).toFixed(2)}%`}
              detail={`${result.estimatedPerformanceTops.toFixed(2)} TOPS estimated`}
            />
            <MetricCard
              icon={Waves}
              label="HBM traffic"
              value={formatBytes(result.bytesTransferred)}
              detail={`${result.tilesM}×${result.tilesN} output tiles`}
            />
          </div>

          <Card className="graph-card">
            <CardContent className="p-0">
              <RooflineGraph architecture={architecture} workload={workload} result={result} />
            </CardContent>
          </Card>

          <div className="detail-grid">
            <Card className="detail-card">
              <CardHeader><CardTitle><MemoryStick aria-hidden="true" /> Memory path</CardTitle></CardHeader>
              <CardContent>
                <div className="detail-row"><span>Activation reads</span><strong>{formatBytes(result.activationReadBytes)}</strong></div>
                <div className="detail-row"><span>Weight reads</span><strong>{formatBytes(result.weightReadBytes)}</strong></div>
                <div className="detail-row"><span>Output write</span><strong>{formatBytes(result.outputBytes)}</strong></div>
                <div className="detail-row"><span>SRAM tile set</span><strong>{formatBytes(result.tileWorkingSetBytes)}</strong></div>
                <div className="detail-row"><span>SRAM residency</span><strong>{(result.sramResidency * 100).toFixed(1)}%</strong></div>
              </CardContent>
            </Card>
            <Card className="detail-card">
              <CardHeader><CardTitle><Boxes aria-hidden="true" /> Execution path</CardTitle></CardHeader>
              <CardContent>
                <div className="detail-row"><span>Total operations</span><strong>{(result.operations / 1e9).toFixed(3)} GOP</strong></div>
                <div className="detail-row"><span>Compute latency</span><strong>{formatLatency(result.computeLatencySeconds)}</strong></div>
                <div className="detail-row"><span>HBM latency</span><strong>{formatLatency(result.memoryLatencySeconds)}</strong></div>
                <div className="detail-row"><span>Effective weight reuse</span><strong>{result.effectiveWeightReuse.toFixed(2)}×</strong></div>
                <div className="detail-row"><span>Effective activation reuse</span><strong>{result.effectiveActivationReuse.toFixed(2)}×</strong></div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      <footer className="model-note">
        <span>MODEL ASSUMPTION</span>
        Estimated latency is max(compute, HBM), assuming perfect overlap. SRAM residency is a capacity heuristic; peak TOPS must match the selected precision.
      </footer>
    </main>
  );
}
