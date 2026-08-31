export const STORAGE_BITS = {
  "posit-(4,1)": 4,
  fp2: 2,
  int4: 4,
  int8: 8,
  bf16: 16,
  fp32: 32,
} as const;

export const ACCUMULATOR_BITS = {
  int32: 32,
  bf16: 16,
  fp32: 32,
  quire128: 128,
} as const;

export type StoragePrecision = keyof typeof STORAGE_BITS;
export type AccumulatorPrecision = keyof typeof ACCUMULATOR_BITS;
export type LoopOrder = "m-n-k" | "n-m-k" | "k-m-n";
export type Dataflow = "output-stationary" | "weight-stationary" | "activation-stationary";
export type FusionLevel = "none" | "epilogue" | "aggressive";
export type CompilerLevel = "basic" | "tiled" | "aggressive";

export type Architecture = {
  name: string;
  peakComputeTopsPerMxu: number;
  hbmBandwidthGbs: number;
  hbmEfficiency: number;
  sramBankCount: number;
  sramBankBandwidthGbs: number;
  sramEfficiency: number;
  sramAllocationA: number;
  sramAllocationB: number;
  sramAllocationC: number;
  mxuRows: number;
  mxuCols: number;
  coreCount: number;
  mxusPerCore: number;
  nocBandwidthGbs: number;
  multicastFactor: number;
  overlapEfficiency: number;
  pipelineOverlap: number;
  hostOverheadUs: number;
  launchOverheadUs: number;
  computeEnergyPjPerOp: number;
  hbmEnergyPjPerByte: number;
  sramEnergyPjPerByte: number;
  nocEnergyPjPerByte: number;
  staticPowerW: number;
};

export type Workload = {
  name: string;
  m: number;
  n: number;
  k: number;
  activationPrecision: StoragePrecision;
  weightPrecision: StoragePrecision;
  outputPrecision: StoragePrecision;
  accumulatorPrecision: AccumulatorPrecision;
  weightReuseFactor: number;
  activationReuseFactor: number;
  loopOrder: LoopOrder;
  dataflow: Dataflow;
  fusionLevel: FusionLevel;
  compilerLevel: CompilerLevel;
};

export type RooflineResult = {
  operations: number;
  bytesTransferred: number;
  arithmeticIntensity: number;
  mxuUtilization: number;
  parallelUtilization: number;
  computeLatencySeconds: number;
  hbmLatencySeconds: number;
  sramLatencySeconds: number;
  nocLatencySeconds: number;
  deviceLatencySeconds: number;
  hostLatencySeconds: number;
  estimatedLatencySeconds: number;
  estimatedPerformanceTops: number;
  bottleneck: "MXU" | "HBM" | "SRAM" | "NoC";
  tilesM: number;
  tilesN: number;
  workers: number;
  pipelineDepth: number;
  pipelineDescription: string;
  quireFinalizeCycles: number;
  cyclesPerTile: number;
  sramCapacityBytes: number;
  sramBandwidthGbs: number;
  allocationABytes: number;
  allocationBBytes: number;
  allocationCBytes: number;
  tileABytes: number;
  tileBBytes: number;
  tileCBytes: number;
  residencyA: number;
  residencyB: number;
  residencyC: number;
  effectiveWeightReuse: number;
  effectiveActivationReuse: number;
  activationReadBytes: number;
  weightReadBytes: number;
  outputBytes: number;
  cSpillBytes: number;
  sramTrafficBytes: number;
  nocTrafficBytes: number;
  effectiveHbmBandwidthGbs: number;
  totalPeakTops: number;
  effectiveComputeCeilingTops: number;
  ridgePoint: number;
  kernelCount: number;
  compilerEfficiency: number;
  fusionOutputFactor: number;
  computeEnergyJ: number;
  hbmEnergyJ: number;
  sramEnergyJ: number;
  nocEnergyJ: number;
  staticEnergyJ: number;
  totalEnergyJ: number;
  averagePowerW: number;
};

const positive = (value: number, fallback = 1) =>
  Math.max(Number.isFinite(value) ? value : fallback, Number.EPSILON);
const clamp01 = (value: number) => Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));

const COMPILER_EFFICIENCY: Record<CompilerLevel, number> = {
  basic: 0.75,
  tiled: 0.9,
  aggressive: 0.97,
};

const FUSION: Record<FusionLevel, { outputFactor: number; kernels: number }> = {
  none: { outputFactor: 1, kernels: 3 },
  epilogue: { outputFactor: 0.75, kernels: 2 },
  aggressive: { outputFactor: 0.5, kernels: 1 },
};

function normalizeAllocations(a: number, b: number, c: number) {
  const safe = [Math.max(0, a), Math.max(0, b), Math.max(0, c)];
  const sum = safe.reduce((total, value) => total + value, 0) || 1;
  return safe.map((value) => value / sum);
}

function isDecodedNumber(precision: StoragePrecision) {
  return precision === "posit-(4,1)" || precision === "fp2" || precision === "bf16" || precision === "fp32";
}

export function evaluateRoofline(
  architectureInput: Architecture,
  workloadInput: Workload,
): RooflineResult {
  const a = {
    ...architectureInput,
    peakComputeTopsPerMxu: positive(architectureInput.peakComputeTopsPerMxu),
    hbmBandwidthGbs: positive(architectureInput.hbmBandwidthGbs),
    hbmEfficiency: clamp01(architectureInput.hbmEfficiency),
    sramBankCount: Math.max(1, Math.round(positive(architectureInput.sramBankCount))),
    sramBankBandwidthGbs: positive(architectureInput.sramBankBandwidthGbs),
    sramEfficiency: clamp01(architectureInput.sramEfficiency),
    mxuRows: Math.max(1, Math.round(positive(architectureInput.mxuRows))),
    mxuCols: Math.max(1, Math.round(positive(architectureInput.mxuCols))),
    coreCount: Math.max(1, Math.round(positive(architectureInput.coreCount))),
    mxusPerCore: Math.max(1, Math.round(positive(architectureInput.mxusPerCore))),
    nocBandwidthGbs: positive(architectureInput.nocBandwidthGbs),
    multicastFactor: positive(architectureInput.multicastFactor),
    overlapEfficiency: clamp01(architectureInput.overlapEfficiency),
    pipelineOverlap: clamp01(architectureInput.pipelineOverlap),
  };
  const w = {
    ...workloadInput,
    m: Math.max(1, Math.round(positive(workloadInput.m))),
    n: Math.max(1, Math.round(positive(workloadInput.n))),
    k: Math.max(1, Math.round(positive(workloadInput.k))),
    weightReuseFactor: positive(workloadInput.weightReuseFactor),
    activationReuseFactor: positive(workloadInput.activationReuseFactor),
  };

  const activationBytes = STORAGE_BITS[w.activationPrecision] / 8;
  const weightBytes = STORAGE_BITS[w.weightPrecision] / 8;
  const outputValueBytes = STORAGE_BITS[w.outputPrecision] / 8;
  const accumulatorBytes = ACCUMULATOR_BITS[w.accumulatorPrecision] / 8;
  const tilesM = Math.ceil(w.m / a.mxuRows);
  const tilesN = Math.ceil(w.n / a.mxuCols);
  const tileM = Math.min(w.m, a.mxuRows);
  const tileN = Math.min(w.n, a.mxuCols);
  const workers = a.coreCount * a.mxusPerCore;
  const outputTileCount = tilesM * tilesN;

  const tileABytes = tileM * w.k * activationBytes;
  const tileBBytes = w.k * tileN * weightBytes;
  const tileCBytes = tileM * tileN * accumulatorBytes;
  const sramCapacityBytes = a.sramBankCount * 64 * 1024;
  const [fractionA, fractionB, fractionC] = normalizeAllocations(
    architectureInput.sramAllocationA,
    architectureInput.sramAllocationB,
    architectureInput.sramAllocationC,
  );
  const allocationABytes = sramCapacityBytes * fractionA;
  const allocationBBytes = sramCapacityBytes * fractionB;
  const allocationCBytes = sramCapacityBytes * fractionC;
  const residencyA = Math.min(1, allocationABytes / positive(tileABytes));
  const residencyB = Math.min(1, allocationBBytes / positive(tileBBytes));
  const residencyC = Math.min(1, allocationCBytes / positive(tileCBytes));

  const activationDataflowBonus = w.dataflow === "activation-stationary" ? 1.5 : w.dataflow === "output-stationary" ? 1.1 : 1;
  const weightDataflowBonus = w.dataflow === "weight-stationary" ? 1.5 : w.dataflow === "output-stationary" ? 1.1 : 1;
  const activationLoopBonus = w.loopOrder === "m-n-k" ? 1.25 : w.loopOrder === "k-m-n" ? 0.8 : 1;
  const weightLoopBonus = w.loopOrder === "n-m-k" ? 1.25 : w.loopOrder === "k-m-n" ? 0.8 : 1;
  const requestedActivationReuse = Math.min(tilesN, w.activationReuseFactor * activationDataflowBonus * activationLoopBonus);
  const requestedWeightReuse = Math.min(tilesM, w.weightReuseFactor * weightDataflowBonus * weightLoopBonus);
  const effectiveActivationReuse = 1 + (Math.max(1, requestedActivationReuse) - 1) * residencyA;
  const effectiveWeightReuse = 1 + (Math.max(1, requestedWeightReuse) - 1) * residencyB;

  const activationCompulsory = w.m * w.k * activationBytes;
  const weightCompulsory = w.k * w.n * weightBytes;
  const activationReadBytes = Math.max(activationCompulsory, activationCompulsory * tilesN / effectiveActivationReuse);
  const weightReadBytes = Math.max(weightCompulsory, weightCompulsory * tilesM / effectiveWeightReuse);
  const fusion = FUSION[w.fusionLevel];
  const outputBytes = w.m * w.n * outputValueBytes * fusion.outputFactor;
  const cSpillMultiplier =
    (w.dataflow === "output-stationary" ? 0.1 : 0.5) +
    (w.loopOrder === "k-m-n" ? 1 : 0);
  const cSpillBytes = 2 * w.m * w.n * accumulatorBytes * (1 - residencyC) * cSpillMultiplier;
  const bytesTransferred = activationReadBytes + weightReadBytes + outputBytes + cSpillBytes;
  const operations = 2 * w.m * w.n * w.k;
  const arithmeticIntensity = operations / bytesTransferred;

  const decodedPipeline = isDecodedNumber(w.activationPrecision) || isDecodedNumber(w.weightPrecision);
  const pipelineDepth = decodedPipeline ? 4 : 2;
  const pipelineDescription = decodedPipeline
    ? "Decode → Multiply → Accumulate → Encode"
    : "Multiply → Accumulate";
  const quireFinalizeCycles = w.accumulatorPrecision === "quire128" ? 2 : 0;
  const fillDrainCycles = a.mxuRows + a.mxuCols - 2 + pipelineDepth - 1 + quireFinalizeCycles;
  const cyclesPerTile = w.k + (1 - a.pipelineOverlap) * fillDrainCycles;
  const scheduledMacSlots = outputTileCount * a.mxuRows * a.mxuCols * cyclesPerTile;
  const usefulMacs = w.m * w.n * w.k;
  const mxuUtilization = Math.min(1, usefulMacs / scheduledMacSlots);
  const parallelUtilization = Math.min(1, outputTileCount / workers);
  const compilerEfficiency = COMPILER_EFFICIENCY[w.compilerLevel];
  const totalPeakTops = a.peakComputeTopsPerMxu * workers;
  const effectiveComputeCeilingTops = totalPeakTops * compilerEfficiency;
  const effectiveComputeOpsPerSecond = effectiveComputeCeilingTops * 1e12 * mxuUtilization * parallelUtilization;
  const computeLatencySeconds = operations / positive(effectiveComputeOpsPerSecond);

  const effectiveHbmBandwidthGbs = a.hbmBandwidthGbs * a.hbmEfficiency;
  const hbmLatencySeconds = bytesTransferred / positive(effectiveHbmBandwidthGbs * 1e9);
  const cAccumulatorTraffic = 2 * w.m * w.n * accumulatorBytes * (w.dataflow === "output-stationary" ? 0.2 : 1);
  const sramTrafficBytes =
    activationCompulsory * tilesN / activationDataflowBonus +
    weightCompulsory * tilesM / weightDataflowBonus +
    cAccumulatorTraffic;
  const sramBandwidthGbs = a.sramBankCount * a.sramBankBandwidthGbs * a.sramEfficiency;
  const sramLatencySeconds = sramTrafficBytes / positive(sramBandwidthGbs * 1e9);
  const multicastFactor = Math.min(workers, Math.max(1, a.multicastFactor));
  const nocTrafficBytes = workers <= 1
    ? 0
    : (activationReadBytes + weightReadBytes) * (workers - 1) / multicastFactor + cSpillBytes;
  const nocLatencySeconds = nocTrafficBytes / positive(a.nocBandwidthGbs * 1e9);

  const resourceTimes = [computeLatencySeconds, hbmLatencySeconds, sramLatencySeconds, nocLatencySeconds];
  const dominantTime = Math.max(...resourceTimes);
  const resourceSum = resourceTimes.reduce((sum, value) => sum + value, 0);
  const deviceLatencySeconds = dominantTime + (1 - a.overlapEfficiency) * (resourceSum - dominantTime);
  const hostLatencySeconds = Math.max(0, a.hostOverheadUs + fusion.kernels * a.launchOverheadUs) * 1e-6;
  const estimatedLatencySeconds = deviceLatencySeconds + hostLatencySeconds;
  const bottleneckNames: RooflineResult["bottleneck"][] = ["MXU", "HBM", "SRAM", "NoC"];
  const bottleneck = bottleneckNames[resourceTimes.indexOf(dominantTime)];

  const computeEnergyJ = operations * Math.max(0, a.computeEnergyPjPerOp) * 1e-12;
  const hbmEnergyJ = bytesTransferred * Math.max(0, a.hbmEnergyPjPerByte) * 1e-12;
  const sramEnergyJ = sramTrafficBytes * Math.max(0, a.sramEnergyPjPerByte) * 1e-12;
  const nocEnergyJ = nocTrafficBytes * Math.max(0, a.nocEnergyPjPerByte) * 1e-12;
  const staticEnergyJ = Math.max(0, a.staticPowerW) * estimatedLatencySeconds;
  const totalEnergyJ = computeEnergyJ + hbmEnergyJ + sramEnergyJ + nocEnergyJ + staticEnergyJ;

  return {
    operations,
    bytesTransferred,
    arithmeticIntensity,
    mxuUtilization,
    parallelUtilization,
    computeLatencySeconds,
    hbmLatencySeconds,
    sramLatencySeconds,
    nocLatencySeconds,
    deviceLatencySeconds,
    hostLatencySeconds,
    estimatedLatencySeconds,
    estimatedPerformanceTops: operations / estimatedLatencySeconds / 1e12,
    bottleneck,
    tilesM,
    tilesN,
    workers,
    pipelineDepth,
    pipelineDescription,
    quireFinalizeCycles,
    cyclesPerTile,
    sramCapacityBytes,
    sramBandwidthGbs,
    allocationABytes,
    allocationBBytes,
    allocationCBytes,
    tileABytes,
    tileBBytes,
    tileCBytes,
    residencyA,
    residencyB,
    residencyC,
    effectiveWeightReuse,
    effectiveActivationReuse,
    activationReadBytes,
    weightReadBytes,
    outputBytes,
    cSpillBytes,
    sramTrafficBytes,
    nocTrafficBytes,
    effectiveHbmBandwidthGbs,
    totalPeakTops,
    effectiveComputeCeilingTops,
    ridgePoint: effectiveComputeCeilingTops * 1000 / positive(effectiveHbmBandwidthGbs),
    kernelCount: fusion.kernels,
    compilerEfficiency,
    fusionOutputFactor: fusion.outputFactor,
    computeEnergyJ,
    hbmEnergyJ,
    sramEnergyJ,
    nocEnergyJ,
    staticEnergyJ,
    totalEnergyJ,
    averagePowerW: totalEnergyJ / estimatedLatencySeconds,
  };
}

export function formatLatency(seconds: number) {
  if (seconds < 1e-6) return `${(seconds * 1e9).toFixed(2)} ns`;
  if (seconds < 1e-3) return `${(seconds * 1e6).toFixed(2)} µs`;
  return `${(seconds * 1e3).toFixed(2)} ms`;
}

export function formatBytes(bytes: number) {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(2)} MB`;
  return `${(bytes / 1e3).toFixed(2)} KB`;
}

export function formatEnergy(joules: number) {
  if (joules < 1e-6) return `${(joules * 1e9).toFixed(2)} nJ`;
  if (joules < 1e-3) return `${(joules * 1e6).toFixed(2)} µJ`;
  return `${(joules * 1e3).toFixed(2)} mJ`;
}
