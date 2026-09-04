import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const vite = await createServer({ appType: "custom", configFile: false, root, resolve: { alias: { "@": root } }, server: { middlewareMode: true } });
after(async () => vite.close());

test("provides Ampere, Hopper, and Blackwell architecture presets", async () => {
  const { NVIDIA_GPU_PRESETS } = await vite.ssrLoadModule("/lib/gpu-roofline.ts");
  assert.equal(NVIDIA_GPU_PRESETS.a100Sxm.generation, "Ampere");
  assert.equal(NVIDIA_GPU_PRESETS.h100Sxm.generation, "Hopper");
  assert.equal(NVIDIA_GPU_PRESETS.b200Sxm.generation, "Blackwell");
  assert.ok(NVIDIA_GPU_PRESETS.h200Sxm.hbmBandwidthGbs > NVIDIA_GPU_PRESETS.a100Sxm.hbmBandwidthGbs);
});

test("derives prefill and decode shapes and layer invocation counts", async () => {
  const { DEFAULT_GPU_WORKLOAD, NVIDIA_GPU_PRESETS, evaluateGpuRoofline } = await vite.ssrLoadModule("/lib/gpu-roofline.ts");
  const architecture = NVIDIA_GPU_PRESETS.h100Sxm;
  const prefill = evaluateGpuRoofline(architecture, { ...DEFAULT_GPU_WORKLOAD, executionPhase: "prefill", batchSize: 2, sequenceLength: 32, layerCount: 4 });
  const decode = evaluateGpuRoofline(architecture, { ...DEFAULT_GPU_WORKLOAD, executionPhase: "decode", batchSize: 2, decodeTokens: 8, layerCount: 4 });
  assert.equal(prefill.effectiveM, 64);
  assert.equal(prefill.invocationCount, 4);
  assert.equal(decode.effectiveM, 2);
  assert.equal(decode.invocationCount, 32);
});

test("models hierarchical bandwidth ceilings and tensor-shape utilization", async () => {
  const { DEFAULT_GPU_WORKLOAD, NVIDIA_GPU_PRESETS, evaluateGpuRoofline } = await vite.ssrLoadModule("/lib/gpu-roofline.ts");
  const result = evaluateGpuRoofline(NVIDIA_GPU_PRESETS.a100Sxm, { ...DEFAULT_GPU_WORKLOAD, m: 33, n: 65, k: 64 });
  assert.ok(result.tensorShapeEfficiency < 1);
  assert.ok(result.hbmBytes >= result.compulsoryBytes);
  assert.ok(result.l2Bytes >= result.hbmBytes);
  assert.ok(result.hbmArithmeticIntensity > 0);
  assert.ok(result.estimatedLatencySeconds >= result.computeLatencySeconds);
});

test("models occupancy, issue efficiency, 2:4 sparsity, and multi-GPU communication", async () => {
  const { DEFAULT_GPU_WORKLOAD, NVIDIA_GPU_PRESETS, evaluateGpuRoofline } = await vite.ssrLoadModule("/lib/gpu-roofline.ts");
  const architecture = NVIDIA_GPU_PRESETS.h100Sxm;
  const dense = evaluateGpuRoofline(architecture, DEFAULT_GPU_WORKLOAD);
  const sparse = evaluateGpuRoofline(architecture, { ...DEFAULT_GPU_WORKLOAD, structuredTwoToFour: true });
  const distributed = evaluateGpuRoofline(architecture, { ...DEFAULT_GPU_WORKLOAD, multiGpuCount: 8, communicationFraction: 0.2 });
  assert.equal(sparse.density, 0.5);
  assert.equal(sparse.operations, dense.operations / 2);
  assert.ok(sparse.hbmBytes < dense.hbmBytes);
  assert.ok(distributed.communicationLatencySeconds > 0);
  assert.ok(distributed.totalPeakTflops > dense.totalPeakTflops);
});

test("exposes the Assignment 3 interactive hierarchy controls and graph", async () => {
  const component = await readFile(new URL("../components/nvidia-gpu-explorer.tsx", import.meta.url), "utf8");
  const page = await readFile(new URL("../app/assignment-3/page.tsx", import.meta.url), "utf8");
  assert.match(page, /NvidiaGpuExplorer/);
  for (const control of ["Execution phase", "Batch", "Sequence", "Layers", "Precision", "Structured 2:4", "Requested occupancy", "L2 hit rate", "GPU count", "Interconnect", "GPU clock", "Power limit", "Measured latency", "Baseline accuracy"]) {
    assert.match(component, new RegExp(control));
  }
  for (const ceiling of ["HBM", "L2", "Shared", "Register"]) {
    assert.match(component, new RegExp(`label: \"${ceiling}\"`));
  }
});

test("calibrates DVFS, overlap, measured latency, energy, and accuracy", async () => {
  const { DEFAULT_GPU_SYSTEM_CONTROLS, DEFAULT_GPU_WORKLOAD, NVIDIA_GPU_PRESETS, evaluateGpuSystem } = await vite.ssrLoadModule("/lib/gpu-roofline.ts");
  const architecture = NVIDIA_GPU_PRESETS.h100Sxm;
  const baseline = evaluateGpuSystem(architecture, DEFAULT_GPU_WORKLOAD, DEFAULT_GPU_SYSTEM_CONTROLS);
  const constrained = evaluateGpuSystem(architecture, { ...DEFAULT_GPU_WORKLOAD, pruningPercent: 50 }, {
    ...DEFAULT_GPU_SYSTEM_CONTROLS,
    clockPercent: 70,
    powerLimitPercent: 60,
    overlapEfficiency: 0,
    measuredLatencyUs: baseline.calibratedLatencySeconds * 1e6,
  });
  assert.ok(constrained.calibratedLatencySeconds > baseline.calibratedLatencySeconds);
  assert.ok(constrained.predictedAccuracyPercent < baseline.predictedAccuracyPercent);
  assert.ok(constrained.energyJoules > 0);
  assert.ok(Number.isFinite(constrained.averagePowerW));
  assert.notEqual(constrained.measuredDeltaPercent, null);
});

test("generates a non-dominated accuracy-energy frontier", async () => {
  const { DEFAULT_GPU_SYSTEM_CONTROLS, DEFAULT_GPU_WORKLOAD, NVIDIA_GPU_PRESETS, generateGpuPareto } = await vite.ssrLoadModule("/lib/gpu-roofline.ts");
  const points = generateGpuPareto(NVIDIA_GPU_PRESETS.b200Sxm, DEFAULT_GPU_WORKLOAD, DEFAULT_GPU_SYSTEM_CONTROLS);
  const frontier = points.filter((point) => point.isPareto);
  assert.ok(points.length >= 64);
  assert.ok(frontier.length > 1);
  for (const point of frontier) {
    assert.ok(!points.some((other) => other.energyJoules < point.energyJoules && other.accuracyPercent > point.accuracyPercent));
  }
});
