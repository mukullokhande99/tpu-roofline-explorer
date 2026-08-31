"use client";

import { useMemo, useState } from "react";
import { Activity, Cpu, Gauge, MemoryStick, Network, Zap } from "lucide-react";
import Link from "next/link";

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
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  type AccumulatorPrecision,
  type Architecture,
  type CompilerLevel,
  type ComputeFabric,
  type Dataflow,
  type FusionLevel,
  type ExecutionPhase,
  type LoopOrder,
  type StoragePrecision,
  type Workload,
  buildAccuracyEnergyPareto,
  evaluateRoofline,
  formatBytes,
  formatEnergy,
  formatLatency,
} from "@/lib/roofline";

const ARCHITECTURES: Record<string, Architecture> = {
  mxu32: {
    name: "32×32 MXU",
    computeFabric: "mxu",
    vectorLanes: 32,
    peakComputeTopsPerMxu: 2.048,
    hbmBandwidthGbs: 900,
    hbmEfficiency: 0.8,
    sramBankCount: 1024,
    sramBankBandwidthGbs: 32,
    sramEfficiency: 0.8,
    sramAllocationA: 35,
    sramAllocationB: 45,
    sramAllocationC: 20,
    mxuRows: 32,
    mxuCols: 32,
    coreCount: 1,
    mxusPerCore: 1,
    nocBandwidthGbs: 800,
    multicastFactor: 1,
    overlapEfficiency: 0.85,
    pipelineOverlap: 0.8,
    hostOverheadUs: 8,
    launchOverheadUs: 4,
    computeEnergyPjPerOp: 0.9,
    hbmEnergyPjPerByte: 12,
    sramEnergyPjPerByte: 1.2,
    nocEnergyPjPerByte: 2,
    staticPowerW: 20,
  },
  mxu64: {
    name: "64×64 MXU",
    computeFabric: "mxu",
    vectorLanes: 64,
    peakComputeTopsPerMxu: 8.192,
    hbmBandwidthGbs: 900,
    hbmEfficiency: 0.8,
    sramBankCount: 1024,
    sramBankBandwidthGbs: 32,
    sramEfficiency: 0.8,
    sramAllocationA: 35,
    sramAllocationB: 45,
    sramAllocationC: 20,
    mxuRows: 64,
    mxuCols: 64,
    coreCount: 1,
    mxusPerCore: 1,
    nocBandwidthGbs: 800,
    multicastFactor: 1,
    overlapEfficiency: 0.85,
    pipelineOverlap: 0.8,
    hostOverheadUs: 8,
    launchOverheadUs: 4,
    computeEnergyPjPerOp: 0.8,
    hbmEnergyPjPerByte: 12,
    sramEnergyPjPerByte: 1.2,
    nocEnergyPjPerByte: 2,
    staticPowerW: 30,
  },
  a: {
    name: "A · 128×128 MXU",
    computeFabric: "mxu",
    vectorLanes: 128,
    peakComputeTopsPerMxu: 32.768,
    hbmBandwidthGbs: 900,
    hbmEfficiency: 0.8,
    sramBankCount: 1024,
    sramBankBandwidthGbs: 32,
    sramEfficiency: 0.8,
    sramAllocationA: 35,
    sramAllocationB: 45,
    sramAllocationC: 20,
    mxuRows: 128,
    mxuCols: 128,
    coreCount: 1,
    mxusPerCore: 1,
    nocBandwidthGbs: 800,
    multicastFactor: 1,
    overlapEfficiency: 0.85,
    pipelineOverlap: 0.8,
    hostOverheadUs: 8,
    launchOverheadUs: 4,
    computeEnergyPjPerOp: 0.7,
    hbmEnergyPjPerByte: 12,
    sramEnergyPjPerByte: 1.2,
    nocEnergyPjPerByte: 2,
    staticPowerW: 45,
  },
  b: {
    name: "B · 256×256 MXU",
    computeFabric: "mxu",
    vectorLanes: 256,
    peakComputeTopsPerMxu: 131.072,
    hbmBandwidthGbs: 900,
    hbmEfficiency: 0.8,
    sramBankCount: 1024,
    sramBankBandwidthGbs: 32,
    sramEfficiency: 0.8,
    sramAllocationA: 35,
    sramAllocationB: 45,
    sramAllocationC: 20,
    mxuRows: 256,
    mxuCols: 256,
    coreCount: 1,
    mxusPerCore: 1,
    nocBandwidthGbs: 800,
    multicastFactor: 1,
    overlapEfficiency: 0.85,
    pipelineOverlap: 0.8,
    hostOverheadUs: 8,
    launchOverheadUs: 4,
    computeEnergyPjPerOp: 0.55,
    hbmEnergyPjPerByte: 12,
    sramEnergyPjPerByte: 1.2,
    nocEnergyPjPerByte: 2,
    staticPowerW: 70,
  },
  vxu256: {
    name: "256-lane VXU",
    computeFabric: "vxu",
    vectorLanes: 256,
    peakComputeTopsPerMxu: 0.512,
    hbmBandwidthGbs: 900,
    hbmEfficiency: 0.8,
    sramBankCount: 1024,
    sramBankBandwidthGbs: 32,
    sramEfficiency: 0.8,
    sramAllocationA: 35,
    sramAllocationB: 45,
    sramAllocationC: 20,
    mxuRows: 1,
    mxuCols: 256,
    coreCount: 1,
    mxusPerCore: 1,
    nocBandwidthGbs: 800,
    multicastFactor: 1,
    overlapEfficiency: 0.85,
    pipelineOverlap: 0.8,
    hostOverheadUs: 8,
    launchOverheadUs: 4,
    computeEnergyPjPerOp: 0.85,
    hbmEnergyPjPerByte: 12,
    sramEnergyPjPerByte: 1.2,
    nocEnergyPjPerByte: 2,
    staticPowerW: 25,
  },
  vxu1024: {
    name: "1024-lane VXU",
    computeFabric: "vxu",
    vectorLanes: 1024,
    peakComputeTopsPerMxu: 2.048,
    hbmBandwidthGbs: 900,
    hbmEfficiency: 0.8,
    sramBankCount: 1024,
    sramBankBandwidthGbs: 32,
    sramEfficiency: 0.8,
    sramAllocationA: 35,
    sramAllocationB: 45,
    sramAllocationC: 20,
    mxuRows: 1,
    mxuCols: 1024,
    coreCount: 1,
    mxusPerCore: 1,
    nocBandwidthGbs: 800,
    multicastFactor: 1,
    overlapEfficiency: 0.85,
    pipelineOverlap: 0.8,
    hostOverheadUs: 8,
    launchOverheadUs: 4,
    computeEnergyPjPerOp: 0.75,
    hbmEnergyPjPerByte: 12,
    sramEnergyPjPerByte: 1.2,
    nocEnergyPjPerByte: 2,
    staticPowerW: 35,
  },
  vxu4096: {
    name: "4096-lane VXU",
    computeFabric: "vxu",
    vectorLanes: 4096,
    peakComputeTopsPerMxu: 8.192,
    hbmBandwidthGbs: 900,
    hbmEfficiency: 0.8,
    sramBankCount: 1024,
    sramBankBandwidthGbs: 32,
    sramEfficiency: 0.8,
    sramAllocationA: 35,
    sramAllocationB: 45,
    sramAllocationC: 20,
    mxuRows: 1,
    mxuCols: 4096,
    coreCount: 1,
    mxusPerCore: 1,
    nocBandwidthGbs: 800,
    multicastFactor: 1,
    overlapEfficiency: 0.85,
    pipelineOverlap: 0.8,
    hostOverheadUs: 8,
    launchOverheadUs: 4,
    computeEnergyPjPerOp: 0.65,
    hbmEnergyPjPerByte: 12,
    sramEnergyPjPerByte: 1.2,
    nocEnergyPjPerByte: 2,
    staticPowerW: 55,
  },
};

const WORKLOADS = {
  square: { name: "4096 × 4096 GEMM", m: 4096, n: 4096, k: 4096 },
  up: { name: "4096 × 11008 GEMM", m: 4096, n: 11008, k: 4096 },
  down: { name: "11008 × 4096 GEMM", m: 11008, n: 4096, k: 4096 },
  token: { name: "Small-token / batch GEMM", m: 1, n: 11008, k: 4096 },
};

const MODEL_LAYERS = {
  llama3_8b: {
    name: "Llama 3 · 8B",
    layers: {
      qkv: [4096, 4096, 4096],
      mlp_up: [4096, 11008, 4096],
      mlp_down: [4096, 4096, 11008],
      lm_head: [1, 128256, 4096],
    },
  },
  qwen25_15b: {
    name: "Qwen2.5 · 1.5B",
    layers: {
      qkv: [1, 896, 896],
      mlp_up: [1, 4864, 896],
      mlp_down: [1, 896, 4864],
      lm_head: [1, 151936, 896],
    },
  },
  vit_b16: {
    name: "ViT-B/16",
    layers: {
      patch_embed: [197, 768, 768],
      qkv: [197, 2304, 768],
      mlp_up: [197, 3072, 768],
      mlp_down: [197, 768, 3072],
    },
  },
  custom: { name: "Custom GEMM", layers: { custom: [4096, 4096, 4096] } },
} as const;
type ModelKey = keyof typeof MODEL_LAYERS;
type LayerKey = string;
const STORAGE: Array<[StoragePrecision, string]> = [
  ["posit-(4,1)", "Posit-(4,1)"],
  ["posit-8", "Posit-8"],
  ["posit-16", "Posit-16"],
  ["fp2", "FP2"],
  ["int4", "INT4"],
  ["int8", "INT8"],
  ["mxfp4", "MXFP4 · microscaled"],
  ["mxint8", "MXINT8 · microscaled"],
  ["nvfp4", "NVFP4"],
  ["bf16", "BF16"],
  ["fp32", "FP32"],
];
const ACCUMULATORS: Array<[AccumulatorPrecision, string]> = [
  ["int32", "INT32"],
  ["bf16", "BF16"],
  ["fp32", "FP32"],
  ["quire128", "Quire-128"],
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
        min={0}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function SliderField({
  label,
  value,
  min,
  max,
  step,
  onChange,
  suffix = "",
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (value: number) => void;
  suffix?: string;
}) {
  return (
    <label className="slider-field">
      <span>
        {label}
        <strong>
          {value}
          {suffix}
        </strong>
      </span>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={([next]) => onChange(next)}
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

function DetailCard({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <Card className="detail-card">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="detail-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function RooflineGraph({
  result,
  architecture,
  workload,
}: {
  result: ReturnType<typeof evaluateRoofline>;
  architecture: Architecture;
  workload: Workload;
}) {
  const width = 860,
    height = 390,
    left = 72,
    bottom = 54,
    right = 24,
    top = 22;
  const plotW = width - left - right,
    plotH = height - top - bottom;
  const xMin = 0.1,
    xMax = Math.max(10000, result.arithmeticIntensity * 2),
    yMin = 0.05,
    yMax = Math.max(10, result.effectiveComputeCeilingTops * 1.5);
  const x = (v: number) =>
    left +
    ((Math.log10(v) - Math.log10(xMin)) /
      (Math.log10(xMax) - Math.log10(xMin))) *
      plotW;
  const y = (v: number) =>
    top +
    (1 -
      (Math.log10(Math.max(v, yMin)) - Math.log10(yMin)) /
        (Math.log10(yMax) - Math.log10(yMin))) *
      plotH;
  const points = Array.from({ length: 72 }, (_, i) => {
    const ai =
      10 **
      (Math.log10(xMin) + (i / 71) * (Math.log10(xMax) - Math.log10(xMin)));
    return [
      ai,
      Math.min(
        (result.effectiveHbmBandwidthGbs * ai) / 1000,
        result.effectiveComputeCeilingTops,
      ),
    ] as const;
  });
  const roof = points
    .map(([ai, tops], i) => `${i ? "L" : "M"}${x(ai)},${y(tops)}`)
    .join(" ");
  const memory = points
    .map(
      ([ai]) => `${x(ai)},${y((result.effectiveHbmBandwidthGbs * ai) / 1000)}`,
    )
    .join(" L");
  const pointX = x(result.arithmeticIntensity),
    pointY = y(Math.min(yMax, result.estimatedPerformanceTops));
  return (
    <div className="roofline-wrap">
      <div className="graph-heading">
        <div>
          <p className="eyebrow">OPERATING POINT</p>
          <h2>{architecture.name} · full-system roofline</h2>
        </div>
        <div className="graph-legend">
          <span>
            <i className="legend-roof" />
            compute/HBM roof
          </span>
          <span>
            <i className="legend-memory" />
            effective HBM
          </span>
          <span>
            <i className="legend-point" />
            {workload.name}
          </span>
        </div>
      </div>
      <svg
        className="roofline-svg"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label="Roofline graph"
      >
        <defs>
          <linearGradient id="roofline-glow" x1="0" x2="1">
            <stop offset="0" stopColor="var(--cyan)" />
            <stop offset="1" stopColor="var(--violet)" />
          </linearGradient>
        </defs>
        <rect
          x={left}
          y={top}
          width={plotW}
          height={plotH}
          className="plot-frame"
        />
        {[0.1, 1, 10, 100, 1000, 10000]
          .filter((v) => v <= xMax)
          .map((v) => (
            <g key={v}>
              <line
                x1={x(v)}
                x2={x(v)}
                y1={top}
                y2={height - bottom}
                className="grid-line"
              />
              <text
                x={x(v)}
                y={height - bottom + 21}
                textAnchor="middle"
                className="axis-tick"
              >
                {v}
              </text>
            </g>
          ))}
        {[0.1, 1, 10, 100, 1000]
          .filter((v) => v <= yMax)
          .map((v) => (
            <g key={v}>
              <line
                x1={left}
                x2={width - right}
                y1={y(v)}
                y2={y(v)}
                className="grid-line"
              />
              <text
                x={left - 12}
                y={y(v) + 4}
                textAnchor="end"
                className="axis-tick"
              >
                {v}
              </text>
            </g>
          ))}
        <path d={`M${memory}`} className="memory-line" />
        <path d={roof} className="roof-line" />
        <line
          x1={pointX}
          x2={pointX}
          y1={pointY}
          y2={height - bottom}
          className="point-guide"
        />
        <circle cx={pointX} cy={pointY} r="8" className="workload-point" />
        <circle cx={pointX} cy={pointY} r="14" className="workload-halo" />
        <text
          x={left + plotW / 2}
          y={height - 12}
          textAnchor="middle"
          className="axis-title"
        >
          Arithmetic intensity · operations / HBM byte
        </text>
        <text
          x="17"
          y={top + plotH / 2}
          textAnchor="middle"
          transform={`rotate(-90 17 ${top + plotH / 2})`}
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
  const [workloadKey, setWorkloadKey] =
    useState<keyof typeof WORKLOADS>("square");
  const [modelKey, setModelKey] = useState<ModelKey>("llama3_8b");
  const [layerKey, setLayerKey] = useState<LayerKey>("mlp_up");
  const [architecture, setArchitecture] = useState<Architecture>(
    ARCHITECTURES.a,
  );
  const [workload, setWorkload] = useState<Workload>({
    ...WORKLOADS.square,
    activationPrecision: "bf16",
    weightPrecision: "bf16",
    outputPrecision: "bf16",
    accumulatorPrecision: "fp32",
    weightReuseFactor: 8,
    activationReuseFactor: 4,
    loopOrder: "m-n-k",
    dataflow: "output-stationary",
    fusionLevel: "epilogue",
    compilerLevel: "tiled",
    pruningPercent: 0,
    executionPhase: "custom",
    batchSize: 1,
    sequenceLength: 4096,
    decodeTokens: 1,
    layerCount: 1,
    baselineAccuracyPercent: 75,
    precisionSensitivity: 2,
    pruningSensitivity: 12,
    pruningExponent: 1.5,
  });
  const result = useMemo(
    () => evaluateRoofline(architecture, workload),
    [architecture, workload],
  );
  const pareto = useMemo(
    () => buildAccuracyEnergyPareto(architecture, workload),
    [architecture, workload],
  );
  const updateArch = (
    field: keyof Architecture,
    value: Architecture[keyof Architecture],
  ) =>
    setArchitecture(
      (current) => ({ ...current, [field]: value }) as Architecture,
    );
  const updateFabric = (fabric: ComputeFabric) =>
    setArchitecture((current) =>
      fabric === "vxu"
        ? {
            ...current,
            computeFabric: fabric,
            vectorLanes: current.vectorLanes ?? current.mxuCols,
          }
        : {
            ...current,
            computeFabric: fabric,
            mxuRows: current.mxuRows === 1 ? 128 : current.mxuRows,
            mxuCols: current.mxuRows === 1 ? 128 : current.mxuCols,
          },
    );
  const updateVectorLanes = (lanes: number) =>
    setArchitecture((current) => ({
      ...current,
      vectorLanes: lanes,
      mxuCols: lanes,
    }));
  const updateWorkload = (
    field: keyof Workload,
    value: Workload[keyof Workload],
  ) => setWorkload((current) => ({ ...current, [field]: value }) as Workload);
  const selectArchitecture = (key: string) => {
    setArchitectureKey(key);
    setArchitecture(ARCHITECTURES[key]);
  };
  const selectWorkload = (key: string) => {
    const typed = key as keyof typeof WORKLOADS;
    setWorkloadKey(typed);
    setWorkload((current) => ({ ...current, ...WORKLOADS[typed] }));
  };
  const selectModel = (key: string) => {
    const typed = key as ModelKey;
    const firstLayer = Object.keys(MODEL_LAYERS[typed].layers)[0];
    const dims = MODEL_LAYERS[typed].layers[
      firstLayer as never
    ] as readonly number[];
    setModelKey(typed);
    setLayerKey(firstLayer);
    setWorkload((current) => ({
      ...current,
      name: `${MODEL_LAYERS[typed].name} · ${firstLayer}`,
      m: dims[0],
      n: dims[1],
      k: dims[2],
    }));
  };
  const selectLayer = (key: string) => {
    const dims = MODEL_LAYERS[modelKey].layers[
      key as never
    ] as readonly number[];
    setLayerKey(key);
    setWorkload((current) => ({
      ...current,
      name: `${MODEL_LAYERS[modelKey].name} · ${key}`,
      m: dims[0],
      n: dims[1],
      k: dims[2],
    }));
  };

  return (
    <main className="explorer-shell">
      <header className="site-header">
        <div className="brand-mark" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        <div>
          <p className="eyebrow">ARCHITECTURE LAB · 01</p>
          <h1>TPU Roofline Explorer</h1>
          <p className="header-copy">
            A transparent tiled-GEMM model spanning arithmetic format, on-chip
            memory, interconnect, compiler choices, and energy.
          </p>
        </div>
        <nav
          className="assignment-nav"
          aria-label="Architecture lab assignments"
        >
          <Link className="assignment-tab assignment-tab-active" href="/">
            Assignment 1<span>Roofline</span>
          </Link>
          <Link className="assignment-tab" href="/assignment-2">
            Assignment 2<span>Flex-TPU</span>
          </Link>
        </nav>
        <Badge variant="outline" className="model-badge">
          ANALYTIC MODEL
        </Badge>
      </header>
      <section className="explorer-grid">
        <Card className="controls-panel">
          <CardHeader className="controls-header">
            <CardTitle>System inputs</CardTitle>
            <span>Updates live</span>
          </CardHeader>
          <CardContent className="control-stack">
            <Tabs defaultValue="model" className="w-full">
              <TabsList className="model-tabs">
                <TabsTrigger value="model">Model</TabsTrigger>
                <TabsTrigger value="memory">Memory</TabsTrigger>
                <TabsTrigger value="system">System</TabsTrigger>
                <TabsTrigger value="energy">Energy</TabsTrigger>
              </TabsList>
              <TabsContent value="model" className="tab-content">
                <div className="control-section">
                  <p className="section-label">Presets</p>
                  <label className="control-field">
                    <span>Architecture</span>
                    <Select
                      value={architectureKey}
                      onValueChange={selectArchitecture}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {Object.entries(ARCHITECTURES).map(([key, preset]) => (
                          <SelectItem key={key} value={key}>
                            {preset.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </label>
                  <label className="control-field">
                    <span>Workload</span>
                    <Select value={workloadKey} onValueChange={selectWorkload}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {Object.entries(WORKLOADS).map(([key, value]) => (
                          <SelectItem key={key} value={key}>
                            {value.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </label>
                  <label className="control-field">
                    <span>Model</span>
                    <Select value={modelKey} onValueChange={selectModel}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {Object.entries(MODEL_LAYERS).map(([key, model]) => (
                          <SelectItem key={key} value={key}>
                            {model.name}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </label>
                  <label className="control-field">
                    <span>Layer / projection</span>
                    <Select value={layerKey} onValueChange={selectLayer}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {Object.keys(MODEL_LAYERS[modelKey].layers).map(
                          (key) => (
                            <SelectItem key={key} value={key}>
                              {key.replaceAll("_", " ")}
                            </SelectItem>
                          ),
                        )}
                      </SelectContent>
                    </Select>
                  </label>
                </div>
                <div className="control-section">
                  <p className="section-label">C[M,N] = A[M,K] × B[K,N]</p>
                  <div className="number-grid">
                    <NumberField
                      label="M"
                      value={workload.m}
                      onChange={(v) => updateWorkload("m", v)}
                    />
                    <NumberField
                      label="N"
                      value={workload.n}
                      onChange={(v) => updateWorkload("n", v)}
                    />
                    <NumberField
                      label="K"
                      value={workload.k}
                      onChange={(v) => updateWorkload("k", v)}
                    />
                  </div>
                  <label className="control-field">
                    <span>Execution phase</span>
                    <Select
                      value={workload.executionPhase ?? "custom"}
                      onValueChange={(v) => updateWorkload("executionPhase", v as ExecutionPhase)}
                    >
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="custom">Custom GEMM · use M</SelectItem>
                        <SelectItem value="prefill">Prefill · M = batch × sequence</SelectItem>
                        <SelectItem value="decode">Decode · M = batch</SelectItem>
                      </SelectContent>
                    </Select>
                  </label>
                  <div className="number-grid two">
                    <NumberField label="Batch" value={workload.batchSize ?? 1} onChange={(v) => updateWorkload("batchSize", v)} />
                    <NumberField label="Sequence length" value={workload.sequenceLength ?? workload.m} onChange={(v) => updateWorkload("sequenceLength", v)} />
                    <NumberField label="Layer count" value={workload.layerCount ?? 1} onChange={(v) => updateWorkload("layerCount", v)} />
                    <NumberField label="Decode tokens" value={workload.decodeTokens ?? 1} onChange={(v) => updateWorkload("decodeTokens", v)} />
                  </div>
                  <p className="input-note">
                    Custom preserves the original M knob. Prefill and decode derive M and scale totals across layers and generated tokens.
                  </p>
                </div>
                <div className="control-section">
                  <p className="section-label">Independent precisions</p>
                  {(
                    [
                      ["Activation · A", "activationPrecision"],
                      ["Weight · B", "weightPrecision"],
                      ["Output · C", "outputPrecision"],
                    ] as const
                  ).map(([label, field]) => (
                    <label className="control-field" key={field}>
                      <span>{label}</span>
                      <Select
                        value={workload[field]}
                        onValueChange={(v) =>
                          updateWorkload(field, v as StoragePrecision)
                        }
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {STORAGE.map(([v, text]) => (
                            <SelectItem key={v} value={v}>
                              {text}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </label>
                  ))}
                  <label className="control-field">
                    <span>Accumulator</span>
                    <Select
                      value={workload.accumulatorPrecision}
                      onValueChange={(v) =>
                        updateWorkload(
                          "accumulatorPrecision",
                          v as AccumulatorPrecision,
                        )
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        {ACCUMULATORS.map(([v, text]) => (
                          <SelectItem key={v} value={v}>
                            {text}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </label>
                </div>
                <div className="control-section slider-stack">
                  <p className="section-label">Structured pruning</p>
                  <SliderField
                    label="Pruned weights"
                    value={workload.pruningPercent}
                    min={0}
                    max={100}
                    step={5}
                    suffix="%"
                    onChange={(v) => updateWorkload("pruningPercent", v)}
                  />
                  <p className="input-note">
                    Evaluates 0–100% pruning in 5% steps; useful MACs and weight
                    traffic scale with the remaining density.
                  </p>
                </div>
                <div className="control-section">
                  <p className="section-label">Compiler and dataflow</p>
                  <label className="control-field">
                    <span>Loop order</span>
                    <Select
                      value={workload.loopOrder}
                      onValueChange={(v) =>
                        updateWorkload("loopOrder", v as LoopOrder)
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="m-n-k">
                          M → N → K · favor A reuse
                        </SelectItem>
                        <SelectItem value="n-m-k">
                          N → M → K · favor B reuse
                        </SelectItem>
                        <SelectItem value="k-m-n">
                          K → M → N · partial-sum pressure
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </label>
                  <label className="control-field">
                    <span>Dataflow</span>
                    <Select
                      value={workload.dataflow}
                      onValueChange={(v) =>
                        updateWorkload("dataflow", v as Dataflow)
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="output-stationary">
                          Output stationary
                        </SelectItem>
                        <SelectItem value="weight-stationary">
                          Weight stationary
                        </SelectItem>
                        <SelectItem value="activation-stationary">
                          Activation stationary
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </label>
                  <label className="control-field">
                    <span>Fusion</span>
                    <Select
                      value={workload.fusionLevel}
                      onValueChange={(v) =>
                        updateWorkload("fusionLevel", v as FusionLevel)
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="none">No fusion</SelectItem>
                        <SelectItem value="epilogue">Fuse epilogue</SelectItem>
                        <SelectItem value="aggressive">
                          Aggressive fusion
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </label>
                  <label className="control-field">
                    <span>Compiler schedule</span>
                    <Select
                      value={workload.compilerLevel}
                      onValueChange={(v) =>
                        updateWorkload("compilerLevel", v as CompilerLevel)
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="basic">Basic · 75%</SelectItem>
                        <SelectItem value="tiled">Tiled · 90%</SelectItem>
                        <SelectItem value="aggressive">
                          Aggressive · 97%
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </label>
                </div>
              </TabsContent>
              <TabsContent value="memory" className="tab-content">
                <div className="control-section slider-stack">
                  <p className="section-label">HBM</p>
                  <SliderField
                    label="Peak bandwidth"
                    value={architecture.hbmBandwidthGbs}
                    min={100}
                    max={3000}
                    step={50}
                    suffix=" GB/s"
                    onChange={(v) => updateArch("hbmBandwidthGbs", v)}
                  />
                  <SliderField
                    label="HBM efficiency"
                    value={Math.round(architecture.hbmEfficiency * 100)}
                    min={10}
                    max={100}
                    step={1}
                    suffix="%"
                    onChange={(v) => updateArch("hbmEfficiency", v / 100)}
                  />
                </div>
                <div className="control-section">
                  <p className="section-label">
                    Banked SRAM · capacity = bank size × M
                  </p>
                  <div className="number-grid two">
                    <NumberField
                      label="Banks"
                      value={architecture.sramBankCount}
                      onChange={(v) => updateArch("sramBankCount", v)}
                    />
                    <NumberField
                      label="Bank size · KiB"
                      value={architecture.sramBankSizeKib ?? 64}
                      onChange={(v) => updateArch("sramBankSizeKib", v)}
                    />
                    <NumberField
                      label="GB/s per bank"
                      value={architecture.sramBankBandwidthGbs}
                      onChange={(v) => updateArch("sramBankBandwidthGbs", v)}
                    />
                  </div>
                  <label className="control-field">
                    <span>Buffering</span>
                    <Select
                      value={(architecture.doubleBuffering ?? false) ? "double" : "single"}
                      onValueChange={(v) => updateArch("doubleBuffering", v === "double")}
                    >
                      <SelectTrigger><SelectValue /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="single">Single buffer</SelectItem>
                        <SelectItem value="double">Double buffer · ping-pong</SelectItem>
                      </SelectContent>
                    </Select>
                  </label>
                  <SliderField
                    label="SRAM efficiency"
                    value={Math.round(architecture.sramEfficiency * 100)}
                    min={10}
                    max={100}
                    step={1}
                    suffix="%"
                    onChange={(v) => updateArch("sramEfficiency", v / 100)}
                  />
                  <p className="input-note">
                    Bank size is independently configurable. Double buffering reserves half the capacity for the next tile.
                  </p>
                </div>
                <div className="control-section slider-stack">
                  <p className="section-label">Explicit SRAM allocation</p>
                  <SliderField
                    label="A allocation"
                    value={architecture.sramAllocationA}
                    min={0}
                    max={100}
                    step={1}
                    suffix="%"
                    onChange={(v) => updateArch("sramAllocationA", v)}
                  />
                  <SliderField
                    label="B allocation"
                    value={architecture.sramAllocationB}
                    min={0}
                    max={100}
                    step={1}
                    suffix="%"
                    onChange={(v) => updateArch("sramAllocationB", v)}
                  />
                  <SliderField
                    label="C / accumulator allocation"
                    value={architecture.sramAllocationC}
                    min={0}
                    max={100}
                    step={1}
                    suffix="%"
                    onChange={(v) => updateArch("sramAllocationC", v)}
                  />
                  <p className="input-note">
                    Allocations are normalized if they do not total 100%.
                  </p>
                </div>
                <div className="control-section slider-stack">
                  <p className="section-label">Reuse intent</p>
                  <SliderField
                    label="Weight reuse"
                    value={workload.weightReuseFactor}
                    min={1}
                    max={32}
                    step={1}
                    suffix="×"
                    onChange={(v) => updateWorkload("weightReuseFactor", v)}
                  />
                  <SliderField
                    label="Activation reuse"
                    value={workload.activationReuseFactor}
                    min={1}
                    max={32}
                    step={1}
                    suffix="×"
                    onChange={(v) => updateWorkload("activationReuseFactor", v)}
                  />
                </div>
              </TabsContent>
              <TabsContent value="system" className="tab-content">
                <div className="control-section">
                  <p className="section-label">Compute fabric</p>
                  <label className="control-field">
                    <span>Fabric type</span>
                    <Select
                      value={architecture.computeFabric ?? "mxu"}
                      onValueChange={(v) => updateFabric(v as ComputeFabric)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="mxu">2-D systolic MXU</SelectItem>
                        <SelectItem value="vxu">Vector VXU</SelectItem>
                      </SelectContent>
                    </Select>
                  </label>
                  <NumberField
                    label={`Peak compute per ${architecture.computeFabric === "vxu" ? "VXU" : "MXU"} · TOPS`}
                    value={architecture.peakComputeTopsPerMxu}
                    step={0.001}
                    onChange={(v) => updateArch("peakComputeTopsPerMxu", v)}
                  />
                  {architecture.computeFabric === "vxu" ? (
                    <>
                      <NumberField label="Vector lanes" value={architecture.vectorLanes ?? architecture.mxuCols} onChange={updateVectorLanes} />
                      <div className="number-grid two">
                        <NumberField label="RF capacity · KiB" value={architecture.vectorRegisterFileKib ?? 256} onChange={(v) => updateArch("vectorRegisterFileKib", v)} />
                        <NumberField label="RF bandwidth · GB/s" value={architecture.vectorRegisterBandwidthGbs ?? 2048} onChange={(v) => updateArch("vectorRegisterBandwidthGbs", v)} />
                        <NumberField label="Vector issue width" value={architecture.vectorIssueWidth ?? 1} step={0.25} onChange={(v) => updateArch("vectorIssueWidth", v)} />
                      </div>
                    </>
                  ) : (
                    <div className="number-grid two">
                      <NumberField
                        label="MXU rows"
                        value={architecture.mxuRows}
                        onChange={(v) => updateArch("mxuRows", v)}
                      />
                      <NumberField
                        label="MXU cols"
                        value={architecture.mxuCols}
                        onChange={(v) => updateArch("mxuCols", v)}
                      />
                    </div>
                  )}
                  <div className="number-grid two">
                    <NumberField
                      label="TPU cores"
                      value={architecture.coreCount}
                      onChange={(v) => updateArch("coreCount", v)}
                    />
                    <NumberField
                      label={`${architecture.computeFabric === "vxu" ? "VXUs" : "MXUs"} per core`}
                      value={architecture.mxusPerCore}
                      onChange={(v) => updateArch("mxusPerCore", v)}
                    />
                  </div>
                </div>
                <div className="control-section">
                  <p className="section-label">DVFS operating point</p>
                  <div className="number-grid two">
                    <NumberField label="Frequency · GHz" value={architecture.frequencyGhz ?? 1} step={0.05} onChange={(v) => updateArch("frequencyGhz", v)} />
                    <NumberField label="Nominal frequency · GHz" value={architecture.nominalFrequencyGhz ?? 1} step={0.05} onChange={(v) => updateArch("nominalFrequencyGhz", v)} />
                    <NumberField label="Voltage · V" value={architecture.voltageV ?? 0.8} step={0.01} onChange={(v) => updateArch("voltageV", v)} />
                    <NumberField label="Nominal voltage · V" value={architecture.nominalVoltageV ?? 0.8} step={0.01} onChange={(v) => updateArch("nominalVoltageV", v)} />
                  </div>
                  <p className="input-note">Throughput scales with frequency; dynamic compute energy scales with voltage squared.</p>
                </div>
                <div className="control-section slider-stack">
                  <p className="section-label">NoC and overlap</p>
                  <SliderField
                    label="NoC bandwidth"
                    value={architecture.nocBandwidthGbs}
                    min={50}
                    max={3000}
                    step={50}
                    suffix=" GB/s"
                    onChange={(v) => updateArch("nocBandwidthGbs", v)}
                  />
                  <SliderField
                    label="Multicast fan-out"
                    value={architecture.multicastFactor}
                    min={1}
                    max={32}
                    step={1}
                    suffix="×"
                    onChange={(v) => updateArch("multicastFactor", v)}
                  />
                  <SliderField
                    label="Tile pipeline overlap"
                    value={Math.round(architecture.pipelineOverlap * 100)}
                    min={0}
                    max={100}
                    step={1}
                    suffix="%"
                    onChange={(v) => updateArch("pipelineOverlap", v / 100)}
                  />
                  <SliderField
                    label="Resource overlap efficiency"
                    value={Math.round(architecture.overlapEfficiency * 100)}
                    min={0}
                    max={100}
                    step={1}
                    suffix="%"
                    onChange={(v) => updateArch("overlapEfficiency", v / 100)}
                  />
                </div>
                <div className="control-section">
                  <p className="section-label">Host path</p>
                  <div className="number-grid two">
                    <NumberField
                      label="Host overhead · µs"
                      value={architecture.hostOverheadUs}
                      step={0.1}
                      onChange={(v) => updateArch("hostOverheadUs", v)}
                    />
                    <NumberField
                      label="Launch / kernel · µs"
                      value={architecture.launchOverheadUs}
                      step={0.1}
                      onChange={(v) => updateArch("launchOverheadUs", v)}
                    />
                  </div>
                </div>
              </TabsContent>
              <TabsContent value="energy" className="tab-content">
                <div className="control-section">
                  <p className="section-label">Energy coefficients · pJ</p>
                  <NumberField
                    label="Compute / operation"
                    value={architecture.computeEnergyPjPerOp}
                    step={0.01}
                    onChange={(v) => updateArch("computeEnergyPjPerOp", v)}
                  />
                  <NumberField
                    label="HBM / byte"
                    value={architecture.hbmEnergyPjPerByte}
                    step={0.1}
                    onChange={(v) => updateArch("hbmEnergyPjPerByte", v)}
                  />
                  <NumberField
                    label="SRAM / byte"
                    value={architecture.sramEnergyPjPerByte}
                    step={0.1}
                    onChange={(v) => updateArch("sramEnergyPjPerByte", v)}
                  />
                  <NumberField
                    label="NoC / byte"
                    value={architecture.nocEnergyPjPerByte}
                    step={0.1}
                    onChange={(v) => updateArch("nocEnergyPjPerByte", v)}
                  />
                  <NumberField
                    label="Static power · W"
                    value={architecture.staticPowerW}
                    step={1}
                    onChange={(v) => updateArch("staticPowerW", v)}
                  />
                  <p className="input-note">
                    Dynamic energy is traffic × pJ/byte plus operations × pJ/op;
                    static energy is power × total latency.
                  </p>
                </div>
                <div className="control-section">
                  <p className="section-label">Accuracy proxy calibration</p>
                  <div className="number-grid two">
                    <NumberField label="Baseline score · %" value={workload.baselineAccuracyPercent ?? 75} step={0.1} onChange={(v) => updateWorkload("baselineAccuracyPercent", v)} />
                    <NumberField label="Precision sensitivity" value={workload.precisionSensitivity ?? 2} step={0.1} onChange={(v) => updateWorkload("precisionSensitivity", v)} />
                    <NumberField label="Pruning sensitivity" value={workload.pruningSensitivity ?? 12} step={0.1} onChange={(v) => updateWorkload("pruningSensitivity", v)} />
                    <NumberField label="Pruning exponent" value={workload.pruningExponent ?? 1.5} step={0.1} onChange={(v) => updateWorkload("pruningExponent", v)} />
                  </div>
                  <p className="input-note">This is a user-calibrated analytical score, not measured benchmark accuracy.</p>
                </div>
              </TabsContent>
            </Tabs>
          </CardContent>
        </Card>
        <div className="results-panel" aria-live="polite">
          <div className="metric-grid">
            <MetricCard
              icon={Activity}
              label="End-to-end latency"
              value={formatLatency(result.estimatedLatencySeconds)}
              detail={`${result.bottleneck} dominates device time`}
            />
            <MetricCard
              icon={Gauge}
              label="Arithmetic intensity"
              value={result.arithmeticIntensity.toFixed(1)}
              detail={`ridge · ${result.ridgePoint.toFixed(1)} op/B`}
            />
            <MetricCard
              icon={Cpu}
              label={`${result.computeFabric === "vxu" ? "VXU lane" : "MXU"} utilization`}
              value={`${(result.mxuUtilization * 100).toFixed(1)}%`}
              detail={`${result.workers} worker${result.workers === 1 ? "" : "s"} · ${(result.parallelUtilization * 100).toFixed(0)}% occupied`}
            />
            <MetricCard
              icon={MemoryStick}
              label="HBM traffic"
              value={formatBytes(result.bytesTransferred)}
              detail={`${result.effectiveHbmBandwidthGbs.toFixed(0)} GB/s effective`}
            />
            <MetricCard
              icon={Zap}
              label="Total energy"
              value={formatEnergy(result.totalEnergyJ)}
              detail={`${result.averagePowerW.toFixed(1)} W average`}
            />
            <MetricCard
              icon={Network}
              label="NoC traffic"
              value={formatBytes(result.nocTrafficBytes)}
              detail={`${architecture.multicastFactor}× multicast fan-out`}
            />
          </div>
          <Card className="graph-card">
            <CardContent className="p-0">
              <RooflineGraph
                result={result}
                architecture={architecture}
                workload={workload}
              />
            </CardContent>
          </Card>
          <div className="detail-grid">
            <DetailCard title="Memory & SRAM">
              <Row
                label="A / B / C tile footprints"
                value={`${formatBytes(result.tileABytes)} / ${formatBytes(result.tileBBytes)} / ${formatBytes(result.tileCBytes)}`}
              />
              <Row
                label="SRAM capacity / bandwidth"
                value={`${formatBytes(result.usableSramCapacityBytes)} usable / ${formatBytes(result.sramCapacityBytes)} raw · ${result.sramBandwidthGbs.toFixed(0)} GB/s`}
              />
              <Row
                label="A / B / C residency"
                value={`${(result.residencyA * 100).toFixed(0)}% / ${(result.residencyB * 100).toFixed(0)}% / ${(result.residencyC * 100).toFixed(0)}%`}
              />
              <Row
                label="Effective A / B reuse"
                value={`${result.effectiveActivationReuse.toFixed(2)}× / ${result.effectiveWeightReuse.toFixed(2)}×`}
              />
              <Row
                label="C spill / SRAM traffic"
                value={`${formatBytes(result.cSpillBytes)} / ${formatBytes(result.sramTrafficBytes)}`}
              />
            </DetailCard>
            <DetailCard title="Pipeline & latency">
              <Row
                label="Numeric MAC pipeline"
                value={`${result.pipelineDepth} stages · ${result.pipelineDescription}`}
              />
              <Row
                label="Quire finalization"
                value={`${result.quireFinalizeCycles} cycles / tile`}
              />
              <Row
                label="Cycles per output tile"
                value={result.cyclesPerTile.toFixed(1)}
              />
              <Row
                label="Compute / HBM / SRAM / NoC / RF"
                value={`${formatLatency(result.computeLatencySeconds)} / ${formatLatency(result.hbmLatencySeconds)} / ${formatLatency(result.sramLatencySeconds)} / ${formatLatency(result.nocLatencySeconds)} / ${formatLatency(result.registerFileLatencySeconds)}`}
              />
              <Row
                label="Device + host / launch"
                value={`${formatLatency(result.deviceLatencySeconds)} + ${formatLatency(result.hostLatencySeconds)}`}
              />
            </DetailCard>
            <DetailCard title="System schedule">
              <Row
                label={result.computeFabric === "vxu" ? "Vector waves" : "Output tiles"}
                value={
                  result.computeFabric === "vxu"
                    ? `${result.vectorWaves} × ${result.vectorLanes} lanes`
                    : `${result.tilesM} × ${result.tilesN}`
                }
              />
              <Row
                label="Peak / compiler ceiling"
                value={`${result.totalPeakTops.toFixed(2)} / ${result.effectiveComputeCeilingTops.toFixed(2)} TOPS`}
              />
              <Row
                label="Fusion / kernel count"
                value={`${workload.fusionLevel} · ${result.kernelCount}`}
              />
              <Row
                label="Loop / dataflow"
                value={`${workload.loopOrder} · ${workload.dataflow}`}
              />
              <Row label="Runtime shape" value={`${result.executionPhase} · M=${result.effectiveM} · ${result.invocationCount} invocation${result.invocationCount === 1 ? "" : "s"}`} />
              <Row label="DVFS" value={`${result.frequencyGhz.toFixed(2)} GHz · ${result.voltageV.toFixed(2)} V`} />
              {result.computeFabric === "vxu" && (
                <Row label="VXU RF / issue" value={`${(result.registerFileResidency * 100).toFixed(0)}% resident · ${result.vectorIssueEfficiency.toFixed(2)}× issue`} />
              )}
              <Row
                label="NoC multicast traffic"
                value={formatBytes(result.nocTrafficBytes)}
              />
            </DetailCard>
            <DetailCard title="Accuracy–energy Pareto">
              <Row label="Current estimated score" value={`${result.estimatedAccuracyPercent.toFixed(2)}%`} />
              {pareto.filter((_, index) => index === 0 || index === pareto.length - 1 || index % Math.max(1, Math.floor(pareto.length / 4)) === 0).slice(0, 6).map((point) => (
                <Row key={`${point.precision}-${point.pruningPercent}`} label={`${point.precision} · ${point.pruningPercent}% pruned`} value={`${point.estimatedAccuracyPercent.toFixed(1)}% · ${formatEnergy(point.totalEnergyJ)}`} />
              ))}
              <p className="input-note">Non-dominated sweep: maximize calibrated score while minimizing modeled energy.</p>
            </DetailCard>
            <DetailCard title="Energy accounting">
              <Row
                label="Compute energy"
                value={formatEnergy(result.computeEnergyJ)}
              />
              <Row
                label="HBM / SRAM energy"
                value={`${formatEnergy(result.hbmEnergyJ)} / ${formatEnergy(result.sramEnergyJ)}`}
              />
              <Row
                label="NoC / static energy"
                value={`${formatEnergy(result.nocEnergyJ)} / ${formatEnergy(result.staticEnergyJ)}`}
              />
              <Row
                label="Total / average power"
                value={`${formatEnergy(result.totalEnergyJ)} / ${result.averagePowerW.toFixed(2)} W`}
              />
              <Row
                label="Output materialization"
                value={`${(result.fusionOutputFactor * 100).toFixed(0)}% after fusion`}
              />
            </DetailCard>
          </div>
        </div>
      </section>
      <footer className="model-note">
        <span>MODEL ASSUMPTION</span>Posit/FP uses a fully pipelined Decode →
        Multiply → Accumulate → Encode path: four stages affect fill/drain
        latency, not steady-state MAC issue rate. MXUs use 2-D systolic
        fill/drain and padding; VXUs use vector-lane occupancy and wave
        scheduling. Dataflow, reuse, spill, compiler, and energy coefficients
        are visible heuristics rather than cycle-accurate TPU measurements.
      </footer>
    </main>
  );
}
