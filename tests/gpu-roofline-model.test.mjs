import assert from "node:assert/strict";
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
