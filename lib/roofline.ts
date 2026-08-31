export const PRECISION_BITS = {
  "posit-(4,1)": 4,
  fp2: 2,
  int4: 4,
  int8: 8,
  bf16: 16,
  fp32: 32,
} as const;

export type Precision = keyof typeof PRECISION_BITS;

export type Architecture = {
  name: string;
  peakComputeTops: number;
  hbmBandwidthGbs: number;
  sramCapacityMib: number;
  mxuRows: number;
  mxuCols: number;
};

export type Workload = {
  name: string;
  m: number;
  n: number;
  k: number;
  precision: Precision;
  weightReuseFactor: number;
  activationReuseFactor: number;
};

export type RooflineResult = {
  operations: number;
  bytesTransferred: number;
  arithmeticIntensity: number;
  mxuUtilization: number;
  computeLatencySeconds: number;
  memoryLatencySeconds: number;
  estimatedLatencySeconds: number;
  estimatedPerformanceTops: number;
  bottleneck: "MXU / compute" | "HBM";
  tilesM: number;
  tilesN: number;
  tileWorkingSetBytes: number;
  sramResidency: number;
  effectiveWeightReuse: number;
  effectiveActivationReuse: number;
  activationReadBytes: number;
  weightReadBytes: number;
  outputBytes: number;
  ridgePoint: number;
};

const positive = (value: number) => Math.max(Number.isFinite(value) ? value : 1, 1);

export function evaluateRoofline(
  architectureInput: Architecture,
  workloadInput: Workload,
): RooflineResult {
  const architecture = {
    ...architectureInput,
    peakComputeTops: positive(architectureInput.peakComputeTops),
    hbmBandwidthGbs: positive(architectureInput.hbmBandwidthGbs),
    sramCapacityMib: Math.max(architectureInput.sramCapacityMib, 0),
    mxuRows: Math.round(positive(architectureInput.mxuRows)),
    mxuCols: Math.round(positive(architectureInput.mxuCols)),
  };
  const workload = {
    ...workloadInput,
    m: Math.round(positive(workloadInput.m)),
    n: Math.round(positive(workloadInput.n)),
    k: Math.round(positive(workloadInput.k)),
    weightReuseFactor: positive(workloadInput.weightReuseFactor),
    activationReuseFactor: positive(workloadInput.activationReuseFactor),
  };

  const bytesPerValue = PRECISION_BITS[workload.precision] / 8;
  const tilesM = Math.ceil(workload.m / architecture.mxuRows);
  const tilesN = Math.ceil(workload.n / architecture.mxuCols);
  const tileM = Math.min(workload.m, architecture.mxuRows);
  const tileN = Math.min(workload.n, architecture.mxuCols);

  const activationCompulsory = workload.m * workload.k * bytesPerValue;
  const weightCompulsory = workload.k * workload.n * bytesPerValue;
  const outputBytes = workload.m * workload.n * bytesPerValue;
  const tileWorkingSetBytes =
    (tileM * workload.k + workload.k * tileN + tileM * tileN) * bytesPerValue;
  const sramBytes = architecture.sramCapacityMib * 2 ** 20;
  const sramResidency = Math.min(1, sramBytes / tileWorkingSetBytes);

  const clippedWeightReuse = Math.min(workload.weightReuseFactor, tilesM);
  const clippedActivationReuse = Math.min(workload.activationReuseFactor, tilesN);
  const effectiveWeightReuse =
    1 + (clippedWeightReuse - 1) * sramResidency;
  const effectiveActivationReuse =
    1 + (clippedActivationReuse - 1) * sramResidency;

  const activationReadBytes = Math.max(
    activationCompulsory,
    (activationCompulsory * tilesN) / effectiveActivationReuse,
  );
  const weightReadBytes = Math.max(
    weightCompulsory,
    (weightCompulsory * tilesM) / effectiveWeightReuse,
  );
  const bytesTransferred = activationReadBytes + weightReadBytes + outputBytes;
  const operations = 2 * workload.m * workload.n * workload.k;
  const arithmeticIntensity = operations / bytesTransferred;

  const scheduledMacSlots =
    tilesM *
    tilesN *
    architecture.mxuRows *
    architecture.mxuCols *
    (workload.k + architecture.mxuRows + architecture.mxuCols - 2);
  const mxuUtilization = (workload.m * workload.n * workload.k) / scheduledMacSlots;
  const computeLatencySeconds =
    operations / (architecture.peakComputeTops * 1e12 * mxuUtilization);
  const memoryLatencySeconds =
    bytesTransferred / (architecture.hbmBandwidthGbs * 1e9);
  const estimatedLatencySeconds = Math.max(
    computeLatencySeconds,
    memoryLatencySeconds,
  );

  return {
    operations,
    bytesTransferred,
    arithmeticIntensity,
    mxuUtilization,
    computeLatencySeconds,
    memoryLatencySeconds,
    estimatedLatencySeconds,
    estimatedPerformanceTops: operations / estimatedLatencySeconds / 1e12,
    bottleneck: computeLatencySeconds >= memoryLatencySeconds ? "MXU / compute" : "HBM",
    tilesM,
    tilesN,
    tileWorkingSetBytes,
    sramResidency,
    effectiveWeightReuse,
    effectiveActivationReuse,
    activationReadBytes,
    weightReadBytes,
    outputBytes,
    ridgePoint:
      (architecture.peakComputeTops * 1000) / architecture.hbmBandwidthGbs,
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
