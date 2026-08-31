export const GPU_PRECISION_BITS = {
  fp32: 32,
  tf32: 19,
  bf16: 16,
  fp16: 16,
  "fp8-e4m3": 8,
  "fp8-e5m2": 8,
  fp4: 4,
  int8: 8,
  int4: 4,
} as const;

export type GpuPrecision = keyof typeof GPU_PRECISION_BITS;
export type GpuExecutionPhase = "custom" | "prefill" | "decode";
export type GpuInterconnect = "pcie" | "nvlink";

export type GpuArchitecture = {
  name: string;
  generation: string;
  technologyNodeNm: number;
  smCount: number;
  cudaCoresPerSm: number;
  tensorCoresPerSm: number;
  gpuClockGhz: number;
  peakTensorTflops: Partial<Record<GpuPrecision, number>>;
  hbmCapacityGb: number;
  hbmBandwidthGbs: number;
  hbmEfficiency: number;
  l2CapacityMb: number;
  l2BandwidthGbs: number;
  l2Efficiency: number;
  sharedMemoryKibPerSm: number;
  sharedMemoryBandwidthGbs: number;
  sharedMemoryEfficiency: number;
  registerFileKibPerSm: number;
  registerFileBandwidthGbs: number;
  registerFileEfficiency: number;
  warpSize: number;
  maxWarpsPerSm: number;
  warpSchedulersPerSm: number;
  launchOverheadUs: number;
  pcieBandwidthGbs: number;
  nvlinkBandwidthGbs: number;
  boardPowerW: number;
};

export type GpuWorkload = {
  name: string;
  m: number;
  n: number;
  k: number;
  precision: GpuPrecision;
  accumulatorBits: number;
  executionPhase: GpuExecutionPhase;
  batchSize: number;
  sequenceLength: number;
  decodeTokens: number;
  layerCount: number;
  pruningPercent: number;
  structuredTwoToFour: boolean;
  requestedOccupancy: number;
  activeWarpsPerSm: number;
  issueEfficiency: number;
  tensorTileM: number;
  tensorTileN: number;
  tensorTileK: number;
  l2HitRate: number;
  sharedReuseFactor: number;
  registerReuseFactor: number;
  multiGpuCount: number;
  interconnect: GpuInterconnect;
  communicationFraction: number;
};

export type GpuRooflineResult = {
  effectiveM: number;
  invocationCount: number;
  denseOperations: number;
  operations: number;
  density: number;
  peakTensorTflops: number;
  totalPeakTflops: number;
  occupancy: number;
  warpOccupancy: number;
  tensorShapeEfficiency: number;
  smParallelEfficiency: number;
  issueEfficiency: number;
  effectiveComputeTflops: number;
  compulsoryBytes: number;
  hbmBytes: number;
  l2Bytes: number;
  sharedMemoryBytes: number;
  registerFileBytes: number;
  hbmArithmeticIntensity: number;
  l2ArithmeticIntensity: number;
  sharedArithmeticIntensity: number;
  registerArithmeticIntensity: number;
  computeLatencySeconds: number;
  hbmLatencySeconds: number;
  l2LatencySeconds: number;
  sharedMemoryLatencySeconds: number;
  registerFileLatencySeconds: number;
  communicationLatencySeconds: number;
  launchLatencySeconds: number;
  estimatedLatencySeconds: number;
  estimatedPerformanceTflops: number;
  bottleneck: "Tensor Core" | "HBM" | "L2" | "Shared memory" | "Register file";
  tileCount: number;
  wavesPerSm: number;
};

const clamp01 = (value: number) => Math.min(1, Math.max(0, Number.isFinite(value) ? value : 0));
const positive = (value: number, fallback = 1) =>
  Math.max(Number.isFinite(value) ? value : fallback, Number.EPSILON);

export const NVIDIA_GPU_PRESETS: Record<string, GpuArchitecture> = {
  a100Sxm: {
    name: "NVIDIA A100 SXM4 80 GB",
    generation: "Ampere",
    technologyNodeNm: 7,
    smCount: 108,
    cudaCoresPerSm: 64,
    tensorCoresPerSm: 4,
    gpuClockGhz: 1.41,
    peakTensorTflops: { fp32: 19.5, tf32: 156, bf16: 312, fp16: 312, int8: 624 },
    hbmCapacityGb: 80,
    hbmBandwidthGbs: 2039,
    hbmEfficiency: 0.8,
    l2CapacityMb: 40,
    l2BandwidthGbs: 5000,
    l2Efficiency: 0.8,
    sharedMemoryKibPerSm: 164,
    sharedMemoryBandwidthGbs: 19000,
    sharedMemoryEfficiency: 0.75,
    registerFileKibPerSm: 256,
    registerFileBandwidthGbs: 25000,
    registerFileEfficiency: 0.75,
    warpSize: 32,
    maxWarpsPerSm: 64,
    warpSchedulersPerSm: 4,
    launchOverheadUs: 5,
    pcieBandwidthGbs: 32,
    nvlinkBandwidthGbs: 600,
    boardPowerW: 400,
  },
  h100Sxm: {
    name: "NVIDIA H100 SXM 80 GB",
    generation: "Hopper",
    technologyNodeNm: 4,
    smCount: 132,
    cudaCoresPerSm: 128,
    tensorCoresPerSm: 4,
    gpuClockGhz: 1.98,
    peakTensorTflops: { fp32: 66.9, tf32: 494.7, bf16: 989.4, fp16: 989.4, "fp8-e4m3": 1978.9, "fp8-e5m2": 1978.9, int8: 1978.9 },
    hbmCapacityGb: 80,
    hbmBandwidthGbs: 3350,
    hbmEfficiency: 0.82,
    l2CapacityMb: 50,
    l2BandwidthGbs: 12000,
    l2Efficiency: 0.82,
    sharedMemoryKibPerSm: 228,
    sharedMemoryBandwidthGbs: 30000,
    sharedMemoryEfficiency: 0.78,
    registerFileKibPerSm: 256,
    registerFileBandwidthGbs: 42000,
    registerFileEfficiency: 0.78,
    warpSize: 32,
    maxWarpsPerSm: 64,
    warpSchedulersPerSm: 4,
    launchOverheadUs: 4,
    pcieBandwidthGbs: 64,
    nvlinkBandwidthGbs: 900,
    boardPowerW: 700,
  },
  h200Sxm: {
    name: "NVIDIA H200 SXM 141 GB",
    generation: "Hopper",
    technologyNodeNm: 4,
    smCount: 132,
    cudaCoresPerSm: 128,
    tensorCoresPerSm: 4,
    gpuClockGhz: 1.98,
    peakTensorTflops: { fp32: 66.9, tf32: 494.7, bf16: 989.4, fp16: 989.4, "fp8-e4m3": 1978.9, "fp8-e5m2": 1978.9, int8: 1978.9 },
    hbmCapacityGb: 141,
    hbmBandwidthGbs: 4800,
    hbmEfficiency: 0.82,
    l2CapacityMb: 50,
    l2BandwidthGbs: 12000,
    l2Efficiency: 0.82,
    sharedMemoryKibPerSm: 228,
    sharedMemoryBandwidthGbs: 30000,
    sharedMemoryEfficiency: 0.78,
    registerFileKibPerSm: 256,
    registerFileBandwidthGbs: 42000,
    registerFileEfficiency: 0.78,
    warpSize: 32,
    maxWarpsPerSm: 64,
    warpSchedulersPerSm: 4,
    launchOverheadUs: 4,
    pcieBandwidthGbs: 64,
    nvlinkBandwidthGbs: 900,
    boardPowerW: 700,
  },
  b200Sxm: {
    name: "NVIDIA B200 SXM 180 GB",
    generation: "Blackwell",
    technologyNodeNm: 4,
    smCount: 160,
    cudaCoresPerSm: 128,
    tensorCoresPerSm: 4,
    gpuClockGhz: 2.1,
    peakTensorTflops: { bf16: 2250, fp16: 2250, "fp8-e4m3": 4500, "fp8-e5m2": 4500, fp4: 9000, int8: 4500, int4: 9000 },
    hbmCapacityGb: 180,
    hbmBandwidthGbs: 8000,
    hbmEfficiency: 0.82,
    l2CapacityMb: 96,
    l2BandwidthGbs: 18000,
    l2Efficiency: 0.82,
    sharedMemoryKibPerSm: 256,
    sharedMemoryBandwidthGbs: 50000,
    sharedMemoryEfficiency: 0.78,
    registerFileKibPerSm: 256,
    registerFileBandwidthGbs: 65000,
    registerFileEfficiency: 0.78,
    warpSize: 32,
    maxWarpsPerSm: 64,
    warpSchedulersPerSm: 4,
    launchOverheadUs: 4,
    pcieBandwidthGbs: 128,
    nvlinkBandwidthGbs: 1800,
    boardPowerW: 1000,
  },
};

export const DEFAULT_GPU_WORKLOAD: GpuWorkload = {
  name: "Llama-style MLP up projection",
  m: 4096,
  n: 11008,
  k: 4096,
  precision: "bf16",
  accumulatorBits: 32,
  executionPhase: "custom",
  batchSize: 1,
  sequenceLength: 4096,
  decodeTokens: 1,
  layerCount: 1,
  pruningPercent: 0,
  structuredTwoToFour: false,
  requestedOccupancy: 0.75,
  activeWarpsPerSm: 48,
  issueEfficiency: 0.85,
  tensorTileM: 16,
  tensorTileN: 16,
  tensorTileK: 16,
  l2HitRate: 0.55,
  sharedReuseFactor: 8,
  registerReuseFactor: 4,
  multiGpuCount: 1,
  interconnect: "nvlink",
  communicationFraction: 0,
};

export function evaluateGpuRoofline(
  architecture: GpuArchitecture,
  workload: GpuWorkload,
): GpuRooflineResult {
  const batch = Math.max(1, Math.round(positive(workload.batchSize)));
  const sequence = Math.max(1, Math.round(positive(workload.sequenceLength)));
  const layers = Math.max(1, Math.round(positive(workload.layerCount)));
  const decodeTokens = Math.max(1, Math.round(positive(workload.decodeTokens)));
  const effectiveM = workload.executionPhase === "prefill"
    ? batch * sequence
    : workload.executionPhase === "decode"
      ? batch
      : Math.max(1, Math.round(positive(workload.m)));
  const invocationCount = layers * (workload.executionPhase === "decode" ? decodeTokens : 1);
  const n = Math.max(1, Math.round(positive(workload.n)));
  const k = Math.max(1, Math.round(positive(workload.k)));
  const pruningDensity = 1 - Math.min(100, Math.max(0, workload.pruningPercent)) / 100;
  const density = Math.min(pruningDensity, workload.structuredTwoToFour ? 0.5 : 1);
  const denseOperations = 2 * effectiveM * n * k * invocationCount;
  const operations = denseOperations * density;

  const tileM = Math.max(1, Math.round(positive(workload.tensorTileM)));
  const tileN = Math.max(1, Math.round(positive(workload.tensorTileN)));
  const tilesM = Math.ceil(effectiveM / tileM);
  const tilesN = Math.ceil(n / tileN);
  const tileCount = tilesM * tilesN;
  const usefulOutputs = effectiveM * n;
  const scheduledOutputs = tilesM * tilesN * tileM * tileN;
  const tensorShapeEfficiency = usefulOutputs / scheduledOutputs;
  const warpOccupancy = Math.min(1,
    positive(workload.activeWarpsPerSm) / positive(architecture.maxWarpsPerSm));
  const occupancy = Math.min(clamp01(workload.requestedOccupancy), warpOccupancy);
  const smParallelEfficiency = Math.min(1, tileCount / positive(architecture.smCount));
  const issueEfficiency = clamp01(workload.issueEfficiency);
  const peakTensorTflops = positive(architecture.peakTensorTflops[workload.precision] ?? 0.001);
  const gpuCount = Math.max(1, Math.round(positive(workload.multiGpuCount)));
  const totalPeakTflops = peakTensorTflops * gpuCount;
  const effectiveComputeTflops = totalPeakTflops * occupancy *
    tensorShapeEfficiency * smParallelEfficiency * issueEfficiency;
  const computeLatencySeconds = operations / positive(effectiveComputeTflops * 1e12);

  const valueBytes = GPU_PRECISION_BITS[workload.precision] / 8;
  const accumulatorBytes = positive(workload.accumulatorBits) / 8;
  const aBytes = effectiveM * k * valueBytes;
  const bBytes = k * n * valueBytes * density;
  const cBytes = effectiveM * n * valueBytes;
  const compulsoryPerInvocation = aBytes + bBytes + cBytes;
  const compulsoryBytes = compulsoryPerInvocation * invocationCount;
  const tiledDemandPerInvocation =
    aBytes * tilesN + bBytes * tilesM + cBytes +
    2 * effectiveM * n * accumulatorBytes;
  const l2Bytes = tiledDemandPerInvocation * invocationCount;
  const l2MissBytes = l2Bytes * (1 - clamp01(workload.l2HitRate));
  const hbmBytes = Math.max(compulsoryBytes, l2MissBytes);
  const sharedMemoryBytes = Math.max(compulsoryBytes,
    (operations / 2) * (2 * valueBytes + accumulatorBytes) /
      positive(workload.sharedReuseFactor));
  const registerFileBytes = Math.max(sharedMemoryBytes,
    (operations / 2) * (valueBytes + accumulatorBytes) /
      positive(workload.registerReuseFactor));

  const hbmLatencySeconds = hbmBytes /
    positive(architecture.hbmBandwidthGbs * architecture.hbmEfficiency * 1e9 * gpuCount);
  const l2LatencySeconds = l2Bytes /
    positive(architecture.l2BandwidthGbs * architecture.l2Efficiency * 1e9 * gpuCount);
  const sharedMemoryLatencySeconds = sharedMemoryBytes /
    positive(architecture.sharedMemoryBandwidthGbs * architecture.sharedMemoryEfficiency * 1e9 * gpuCount);
  const registerFileLatencySeconds = registerFileBytes /
    positive(architecture.registerFileBandwidthGbs * architecture.registerFileEfficiency * 1e9 * gpuCount);

  const resourceTimes = [
    computeLatencySeconds,
    hbmLatencySeconds,
    l2LatencySeconds,
    sharedMemoryLatencySeconds,
    registerFileLatencySeconds,
  ];
  const dominantTime = Math.max(...resourceTimes);
  const bottleneckNames: GpuRooflineResult["bottleneck"][] = [
    "Tensor Core",
    "HBM",
    "L2",
    "Shared memory",
    "Register file",
  ];
  const bottleneck = bottleneckNames[resourceTimes.indexOf(dominantTime)];

  const communicationBytes = compulsoryBytes * clamp01(workload.communicationFraction) *
    Math.max(0, gpuCount - 1) / gpuCount;
  const interconnectBandwidth = workload.interconnect === "nvlink"
    ? architecture.nvlinkBandwidthGbs
    : architecture.pcieBandwidthGbs;
  const communicationLatencySeconds = communicationBytes /
    positive(interconnectBandwidth * 1e9);
  const launchLatencySeconds = architecture.launchOverheadUs * invocationCount * 1e-6;
  const estimatedLatencySeconds = dominantTime + communicationLatencySeconds + launchLatencySeconds;

  return {
    effectiveM,
    invocationCount,
    denseOperations,
    operations,
    density,
    peakTensorTflops,
    totalPeakTflops,
    occupancy,
    warpOccupancy,
    tensorShapeEfficiency,
    smParallelEfficiency,
    issueEfficiency,
    effectiveComputeTflops,
    compulsoryBytes,
    hbmBytes,
    l2Bytes,
    sharedMemoryBytes,
    registerFileBytes,
    hbmArithmeticIntensity: operations / hbmBytes,
    l2ArithmeticIntensity: operations / l2Bytes,
    sharedArithmeticIntensity: operations / sharedMemoryBytes,
    registerArithmeticIntensity: operations / registerFileBytes,
    computeLatencySeconds,
    hbmLatencySeconds,
    l2LatencySeconds,
    sharedMemoryLatencySeconds,
    registerFileLatencySeconds,
    communicationLatencySeconds,
    launchLatencySeconds,
    estimatedLatencySeconds,
    estimatedPerformanceTflops: operations / estimatedLatencySeconds / 1e12,
    bottleneck,
    tileCount,
    wavesPerSm: tileCount / positive(architecture.smCount * gpuCount),
  };
}
