export const STORAGE_BITS = {
  "posit-(4,1)": 4,
  "posit-8": 8,
  "posit-16": 16,
  fp2: 2,
  int4: 4,
  int8: 8,
  mxfp4: 4,
  mxint8: 8,
  nvfp4: 4,
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
export type ComputeFabric = "mxu" | "vxu";
export type ExecutionPhase = "custom" | "prefill" | "decode";
export type ProcessCorner = "ss" | "tt" | "ff";
export type TechnologyNodeNm = 7 | 16 | 28 | 65;

export type Architecture = {
  name: string;
  computeFabric?: ComputeFabric;
  vectorLanes?: number;
  peakComputeTopsPerMxu: number;
  hbmBandwidthGbs: number;
  hbmEfficiency: number;
  sramBankCount: number;
  sramBankSizeKib?: number;
  doubleBuffering?: boolean;
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
  frequencyGhz?: number;
  nominalFrequencyGhz?: number;
  voltageV?: number;
  nominalVoltageV?: number;
  processCorner?: ProcessCorner;
  temperatureC?: number;
  technologyNodeNm?: TechnologyNodeNm;
  vectorRegisterFileKib?: number;
  vectorRegisterBandwidthGbs?: number;
  vectorIssueWidth?: number;
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
  pruningPercent: number;
  executionPhase?: ExecutionPhase;
  batchSize?: number;
  sequenceLength?: number;
  decodeTokens?: number;
  layerCount?: number;
  baselineAccuracyPercent?: number;
  precisionSensitivity?: number;
  pruningSensitivity?: number;
  pruningExponent?: number;
};

export type RooflineResult = {
  operations: number;
  denseOperations: number;
  pruningPercent: number;
  bytesTransferred: number;
  arithmeticIntensity: number;
  mxuUtilization: number;
  parallelUtilization: number;
  computeLatencySeconds: number;
  hbmLatencySeconds: number;
  sramLatencySeconds: number;
  nocLatencySeconds: number;
  registerFileLatencySeconds: number;
  deviceLatencySeconds: number;
  hostLatencySeconds: number;
  estimatedLatencySeconds: number;
  estimatedPerformanceTops: number;
  bottleneck: "MXU" | "VXU" | "HBM" | "SRAM" | "NoC" | "RF";
  computeFabric: ComputeFabric;
  vectorLanes: number;
  vectorWaves: number;
  tilesM: number;
  tilesN: number;
  workers: number;
  pipelineDepth: number;
  pipelineDescription: string;
  quireFinalizeCycles: number;
  cyclesPerTile: number;
  sramCapacityBytes: number;
  usableSramCapacityBytes: number;
  sramBankSizeKib: number;
  doubleBuffering: boolean;
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
  registerFileTrafficBytes: number;
  registerFileResidency: number;
  vectorIssueEfficiency: number;
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
  executionPhase: ExecutionPhase;
  effectiveM: number;
  invocationCount: number;
  layerCount: number;
  frequencyGhz: number;
  voltageV: number;
  estimatedAccuracyPercent: number;
  processCorner: ProcessCorner;
  temperatureC: number;
  technologyNodeNm: TechnologyNodeNm;
  cornerFrequencyFactor: number;
  temperatureFrequencyFactor: number;
  nodeFrequencyFactor: number;
  leakageScale: number;
  nodeDynamicEnergyFactor: number;
};

export type AccuracyEnergyPoint = {
  precision: StoragePrecision;
  pruningPercent: number;
  estimatedAccuracyPercent: number;
  totalEnergyJ: number;
  estimatedLatencySeconds: number;
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

const CORNER_FREQUENCY_FACTOR: Record<ProcessCorner, number> = {
  ss: 0.82,
  tt: 1,
  ff: 1.12,
};

const CORNER_LEAKAGE_FACTOR: Record<ProcessCorner, number> = {
  ss: 0.7,
  tt: 1,
  ff: 1.35,
};

const NODE_SCALING: Record<TechnologyNodeNm, { frequency: number; dynamicEnergy: number; leakage: number }> = {
  7: { frequency: 1.35, dynamicEnergy: 0.55, leakage: 1.6 },
  16: { frequency: 1, dynamicEnergy: 1, leakage: 1 },
  28: { frequency: 0.75, dynamicEnergy: 1.65, leakage: 0.75 },
  65: { frequency: 0.45, dynamicEnergy: 3.2, leakage: 0.45 },
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
    sramBankSizeKib: positive(architectureInput.sramBankSizeKib ?? 64),
    doubleBuffering: architectureInput.doubleBuffering ?? false,
    sramBankBandwidthGbs: positive(architectureInput.sramBankBandwidthGbs),
    sramEfficiency: clamp01(architectureInput.sramEfficiency),
    mxuRows: Math.max(1, Math.round(positive(architectureInput.mxuRows))),
    mxuCols: Math.max(1, Math.round(positive(architectureInput.mxuCols))),
    computeFabric: architectureInput.computeFabric ?? "mxu",
    vectorLanes: Math.max(
      1,
      Math.round(positive(architectureInput.vectorLanes ?? architectureInput.mxuCols)),
    ),
    coreCount: Math.max(1, Math.round(positive(architectureInput.coreCount))),
    mxusPerCore: Math.max(1, Math.round(positive(architectureInput.mxusPerCore))),
    nocBandwidthGbs: positive(architectureInput.nocBandwidthGbs),
    multicastFactor: positive(architectureInput.multicastFactor),
    overlapEfficiency: clamp01(architectureInput.overlapEfficiency),
    pipelineOverlap: clamp01(architectureInput.pipelineOverlap),
    frequencyGhz: positive(architectureInput.frequencyGhz ?? 1),
    nominalFrequencyGhz: positive(architectureInput.nominalFrequencyGhz ?? 1),
    voltageV: positive(architectureInput.voltageV ?? 0.8),
    nominalVoltageV: positive(architectureInput.nominalVoltageV ?? 0.8),
    processCorner: architectureInput.processCorner ?? "tt",
    temperatureC: Math.min(150, Math.max(-55, architectureInput.temperatureC ?? 25)),
    technologyNodeNm: architectureInput.technologyNodeNm ?? 16,
    vectorRegisterFileKib: positive(architectureInput.vectorRegisterFileKib ?? 256),
    vectorRegisterBandwidthGbs: positive(architectureInput.vectorRegisterBandwidthGbs ?? 2048),
    vectorIssueWidth: positive(architectureInput.vectorIssueWidth ?? 1),
  };
  const executionPhase = workloadInput.executionPhase ?? "custom";
  const batchSize = Math.max(1, Math.round(positive(workloadInput.batchSize ?? 1)));
  const sequenceLength = Math.max(1, Math.round(positive(workloadInput.sequenceLength ?? workloadInput.m)));
  const decodeTokens = Math.max(1, Math.round(positive(workloadInput.decodeTokens ?? 1)));
  const layerCount = Math.max(1, Math.round(positive(workloadInput.layerCount ?? 1)));
  const effectiveM = executionPhase === "prefill"
    ? batchSize * sequenceLength
    : executionPhase === "decode"
      ? batchSize
      : Math.max(1, Math.round(positive(workloadInput.m)));
  const invocationCount = layerCount * (executionPhase === "decode" ? decodeTokens : 1);
  const w = {
    ...workloadInput,
    m: effectiveM,
    n: Math.max(1, Math.round(positive(workloadInput.n))),
    k: Math.max(1, Math.round(positive(workloadInput.k))),
    weightReuseFactor: positive(workloadInput.weightReuseFactor),
    activationReuseFactor: positive(workloadInput.activationReuseFactor),
    pruningPercent: Math.min(100, Math.max(0, Math.round(workloadInput.pruningPercent ?? 0))),
  };

  const activationBytes = STORAGE_BITS[w.activationPrecision] / 8;
  const weightBytes = STORAGE_BITS[w.weightPrecision] / 8;
  const outputValueBytes = STORAGE_BITS[w.outputPrecision] / 8;
  const accumulatorBytes = ACCUMULATOR_BITS[w.accumulatorPrecision] / 8;
  const tileRows = a.computeFabric === "vxu" ? 1 : a.mxuRows;
  const tileCols = a.computeFabric === "vxu" ? a.vectorLanes : a.mxuCols;
  const tilesM = Math.ceil(w.m / tileRows);
  const tilesN = Math.ceil(w.n / tileCols);
  const tileM = Math.min(w.m, tileRows);
  const tileN = Math.min(w.n, tileCols);
  const workers = a.coreCount * a.mxusPerCore;
  const outputTileCount = tilesM * tilesN;

  const tileABytes = tileM * w.k * activationBytes;
  const tileBBytes = w.k * tileN * weightBytes;
  const tileCBytes = tileM * tileN * accumulatorBytes;
  const sramCapacityBytes = a.sramBankCount * a.sramBankSizeKib * 1024;
  const usableSramCapacityBytes = sramCapacityBytes / (a.doubleBuffering ? 2 : 1);
  const [fractionA, fractionB, fractionC] = normalizeAllocations(
    architectureInput.sramAllocationA,
    architectureInput.sramAllocationB,
    architectureInput.sramAllocationC,
  );
  const allocationABytes = usableSramCapacityBytes * fractionA;
  const allocationBBytes = usableSramCapacityBytes * fractionB;
  const allocationCBytes = usableSramCapacityBytes * fractionC;
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
  const density = 1 - w.pruningPercent / 100;
  const weightCompulsory = w.k * w.n * weightBytes * density;
  const activationReadBytes = Math.max(activationCompulsory, activationCompulsory * tilesN / effectiveActivationReuse) * invocationCount;
  const weightReadBytes = Math.max(weightCompulsory, weightCompulsory * tilesM / effectiveWeightReuse) * invocationCount;
  const fusion = FUSION[w.fusionLevel];
  const outputBytes = w.m * w.n * outputValueBytes * fusion.outputFactor * invocationCount;
  const cSpillMultiplier =
    (w.dataflow === "output-stationary" ? 0.1 : 0.5) +
    (w.loopOrder === "k-m-n" ? 1 : 0);
  const cSpillBytes = 2 * w.m * w.n * accumulatorBytes * (1 - residencyC) * cSpillMultiplier * invocationCount;
  const bytesTransferred = activationReadBytes + weightReadBytes + outputBytes + cSpillBytes;
  const denseOperations = 2 * w.m * w.n * w.k * invocationCount;
  const operations = denseOperations * density;
  const arithmeticIntensity = operations / bytesTransferred;

  const decodedPipeline = isDecodedNumber(w.activationPrecision) || isDecodedNumber(w.weightPrecision);
  const pipelineDepth = decodedPipeline ? 4 : 2;
  const pipelineDescription = decodedPipeline
    ? "Decode → Multiply → Accumulate → Encode"
    : "Multiply → Accumulate";
  const quireFinalizeCycles = w.accumulatorPrecision === "quire128" ? 2 : 0;
  const fillDrainCycles =
    (a.computeFabric === "mxu" ? tileRows + tileCols - 2 : 0) +
    pipelineDepth -
    1 +
    quireFinalizeCycles;
  const cyclesPerTile = w.k + (1 - a.pipelineOverlap) * fillDrainCycles;
  const scheduledMacSlots =
    outputTileCount * tileRows * tileCols * cyclesPerTile;
  const usefulMacs = w.m * w.n * w.k * density;
  const mxuUtilization = Math.min(1, usefulMacs / scheduledMacSlots);
  const parallelUtilization = Math.min(1, outputTileCount / workers);
  const compilerEfficiency = COMPILER_EFFICIENCY[w.compilerLevel];
  const cornerFrequencyFactor = CORNER_FREQUENCY_FACTOR[a.processCorner];
  const temperatureFrequencyFactor = Math.min(1.15, Math.max(0.7,
    1 - 0.0015 * (a.temperatureC - 25)));
  const nodeScaling = NODE_SCALING[a.technologyNodeNm];
  const nodeFrequencyFactor = nodeScaling.frequency;
  const frequencyScale = (a.frequencyGhz / a.nominalFrequencyGhz) *
    cornerFrequencyFactor * temperatureFrequencyFactor * nodeFrequencyFactor;
  const vectorIssueEfficiency = a.computeFabric === "vxu" ? a.vectorIssueWidth : 1;
  const totalPeakTops = a.peakComputeTopsPerMxu * workers * frequencyScale * vectorIssueEfficiency;
  const effectiveComputeCeilingTops = totalPeakTops * compilerEfficiency;
  const effectiveComputeOpsPerSecond = effectiveComputeCeilingTops * 1e12 * mxuUtilization * parallelUtilization;
  const computeLatencySeconds = operations / positive(effectiveComputeOpsPerSecond);

  const effectiveHbmBandwidthGbs = a.hbmBandwidthGbs * a.hbmEfficiency;
  const hbmLatencySeconds = bytesTransferred / positive(effectiveHbmBandwidthGbs * 1e9);
  const cAccumulatorTraffic = 2 * w.m * w.n * accumulatorBytes * (w.dataflow === "output-stationary" ? 0.2 : 1);
  const sramTrafficBytes = (
    activationCompulsory * tilesN / activationDataflowBonus +
    weightCompulsory * tilesM / weightDataflowBonus +
    cAccumulatorTraffic) * invocationCount;
  const sramBandwidthGbs = a.sramBankCount * a.sramBankBandwidthGbs * a.sramEfficiency;
  const sramLatencySeconds = sramTrafficBytes / positive(sramBandwidthGbs * 1e9);
  const multicastFactor = Math.min(workers, Math.max(1, a.multicastFactor));
  const nocTrafficBytes = workers <= 1
    ? 0
    : (activationReadBytes + weightReadBytes) * (workers - 1) / multicastFactor + cSpillBytes;
  const nocLatencySeconds = nocTrafficBytes / positive(a.nocBandwidthGbs * 1e9);

  const registerFileCapacityBytes = a.vectorRegisterFileKib * 1024;
  const registerFileWorkingSetBytes = a.vectorLanes * (activationBytes + weightBytes + accumulatorBytes);
  const registerFileResidency = a.computeFabric === "vxu"
    ? Math.min(1, registerFileCapacityBytes / positive(registerFileWorkingSetBytes))
    : 1;
  const registerFileTrafficBytes = a.computeFabric === "vxu"
    ? ((operations / 2) * (activationBytes + weightBytes) +
      2 * w.m * w.n * accumulatorBytes * invocationCount) / registerFileResidency
    : 0;
  const registerFileLatencySeconds = registerFileTrafficBytes /
    positive(a.vectorRegisterBandwidthGbs * 1e9 * vectorIssueEfficiency);

  const resourceTimes = [computeLatencySeconds, hbmLatencySeconds, sramLatencySeconds, nocLatencySeconds, registerFileLatencySeconds];
  const dominantTime = Math.max(...resourceTimes);
  const resourceSum = resourceTimes.reduce((sum, value) => sum + value, 0);
  const deviceLatencySeconds = dominantTime + (1 - a.overlapEfficiency) * (resourceSum - dominantTime);
  const hostLatencySeconds = Math.max(0, a.hostOverheadUs + fusion.kernels * a.launchOverheadUs) * 1e-6 * invocationCount;
  const estimatedLatencySeconds = deviceLatencySeconds + hostLatencySeconds;
  const bottleneckNames: RooflineResult["bottleneck"][] = [
    a.computeFabric === "vxu" ? "VXU" : "MXU",
    "HBM",
    "SRAM",
    "NoC",
    "RF",
  ];
  const bottleneck = bottleneckNames[resourceTimes.indexOf(dominantTime)];

  const voltageScaleSquared = (a.voltageV / a.nominalVoltageV) ** 2;
  const nodeDynamicEnergyFactor = nodeScaling.dynamicEnergy;
  const computeEnergyJ = operations * Math.max(0, a.computeEnergyPjPerOp) *
    voltageScaleSquared * nodeDynamicEnergyFactor * 1e-12;
  const hbmEnergyJ = bytesTransferred * Math.max(0, a.hbmEnergyPjPerByte) * 1e-12;
  const sramEnergyJ = sramTrafficBytes * Math.max(0, a.sramEnergyPjPerByte) * 1e-12;
  const nocEnergyJ = nocTrafficBytes * Math.max(0, a.nocEnergyPjPerByte) * 1e-12;
  const temperatureLeakageFactor = 2 ** ((a.temperatureC - 25) / 35);
  const leakageScale = CORNER_LEAKAGE_FACTOR[a.processCorner] *
    temperatureLeakageFactor * nodeScaling.leakage;
  const staticEnergyJ = Math.max(0, a.staticPowerW) *
    (a.voltageV / a.nominalVoltageV) * leakageScale * estimatedLatencySeconds;
  const totalEnergyJ = computeEnergyJ + hbmEnergyJ + sramEnergyJ + nocEnergyJ + staticEnergyJ;
  const precisionBits = Math.min(STORAGE_BITS[w.activationPrecision], STORAGE_BITS[w.weightPrecision]);
  const precisionLoss = Math.max(0, workloadInput.precisionSensitivity ?? 2) *
    (Math.max(0, 16 - precisionBits) / 12) ** 2;
  const pruningLoss = Math.max(0, workloadInput.pruningSensitivity ?? 12) *
    (w.pruningPercent / 100) ** positive(workloadInput.pruningExponent ?? 1.5);
  const estimatedAccuracyPercent = Math.max(0, Math.min(100,
    (workloadInput.baselineAccuracyPercent ?? 75) - precisionLoss - pruningLoss));

  return {
    operations,
    denseOperations,
    pruningPercent: w.pruningPercent,
    bytesTransferred,
    arithmeticIntensity,
    mxuUtilization,
    parallelUtilization,
    computeLatencySeconds,
    hbmLatencySeconds,
    sramLatencySeconds,
    nocLatencySeconds,
    registerFileLatencySeconds,
    deviceLatencySeconds,
    hostLatencySeconds,
    estimatedLatencySeconds,
    estimatedPerformanceTops: operations / estimatedLatencySeconds / 1e12,
    bottleneck,
    computeFabric: a.computeFabric,
    vectorLanes: a.vectorLanes,
    vectorWaves: outputTileCount,
    tilesM,
    tilesN,
    workers,
    pipelineDepth,
    pipelineDescription,
    quireFinalizeCycles,
    cyclesPerTile,
    sramCapacityBytes,
    usableSramCapacityBytes,
    sramBankSizeKib: a.sramBankSizeKib,
    doubleBuffering: a.doubleBuffering,
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
    registerFileTrafficBytes,
    registerFileResidency,
    vectorIssueEfficiency,
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
    executionPhase,
    effectiveM,
    invocationCount,
    layerCount,
    frequencyGhz: a.frequencyGhz,
    voltageV: a.voltageV,
    estimatedAccuracyPercent,
    processCorner: a.processCorner,
    temperatureC: a.temperatureC,
    technologyNodeNm: a.technologyNodeNm,
    cornerFrequencyFactor,
    temperatureFrequencyFactor,
    nodeFrequencyFactor,
    leakageScale,
    nodeDynamicEnergyFactor,
  };
}

export function buildAccuracyEnergyPareto(
  architecture: Architecture,
  workload: Workload,
): AccuracyEnergyPoint[] {
  const precisions: StoragePrecision[] = ["bf16", "posit-16", "int8", "posit-8", "mxfp4", "nvfp4", "int4", "posit-(4,1)", "fp2"];
  const candidates = precisions.flatMap((precision) =>
    Array.from({ length: 21 }, (_, index) => index * 5).map((pruningPercent) => {
      const result = evaluateRoofline(architecture, {
        ...workload,
        activationPrecision: precision,
        weightPrecision: precision,
        pruningPercent,
      });
      return {
        precision,
        pruningPercent,
        estimatedAccuracyPercent: result.estimatedAccuracyPercent,
        totalEnergyJ: result.totalEnergyJ,
        estimatedLatencySeconds: result.estimatedLatencySeconds,
      };
    }),
  ).sort((left, right) => left.totalEnergyJ - right.totalEnergyJ);

  let bestAccuracy = -Infinity;
  return candidates.filter((point) => {
    if (point.estimatedAccuracyPercent <= bestAccuracy + 1e-9) return false;
    bestAccuracy = point.estimatedAccuracyPercent;
    return true;
  });
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
