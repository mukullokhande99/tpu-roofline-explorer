import assert from "node:assert/strict";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const vite = await createServer({ appType: "custom", configFile: false, root, resolve: { alias: { "@": root } }, server: { middlewareMode: true } });
after(async () => vite.close());

const architecture = { name: "test", peakComputeTopsPerMxu: 1, hbmBandwidthGbs: 100, hbmEfficiency: 0.8, sramBankCount: 4, sramBankBandwidthGbs: 10, sramEfficiency: 0.8, sramAllocationA: 40, sramAllocationB: 40, sramAllocationC: 20, mxuRows: 4, mxuCols: 4, coreCount: 1, mxusPerCore: 1, nocBandwidthGbs: 100, multicastFactor: 1, overlapEfficiency: 1, pipelineOverlap: 0, hostOverheadUs: 0, launchOverheadUs: 0, computeEnergyPjPerOp: 1, hbmEnergyPjPerByte: 1, sramEnergyPjPerByte: 1, nocEnergyPjPerByte: 1, staticPowerW: 0 };
const workload = { name: "gemm", m: 8, n: 8, k: 4, activationPrecision: "bf16", weightPrecision: "int4", outputPrecision: "int8", accumulatorPrecision: "fp32", weightReuseFactor: 2, activationReuseFactor: 2, loopOrder: "m-n-k", dataflow: "output-stationary", fusionLevel: "none", compilerLevel: "tiled" };

test("models independent precision, bank capacity, and Posit pipeline", async () => {
  const { evaluateRoofline } = await vite.ssrLoadModule("/lib/roofline.ts");
  const result = evaluateRoofline(architecture, { ...workload, activationPrecision: "posit-(4,1)", accumulatorPrecision: "quire128" });
  assert.equal(result.sramCapacityBytes, 4 * 64 * 1024);
  assert.equal(result.pipelineDepth, 4);
  assert.equal(result.quireFinalizeCycles, 2);
  assert.ok(result.tileABytes < evaluateRoofline(architecture, workload).tileABytes);
});

test("overlap, HBM efficiency, multicore NoC, and energy are explicit", async () => {
  const { evaluateRoofline } = await vite.ssrLoadModule("/lib/roofline.ts");
  const serial = evaluateRoofline({ ...architecture, overlapEfficiency: 0, coreCount: 4, mxusPerCore: 2, multicastFactor: 8 }, workload);
  const overlapped = evaluateRoofline({ ...architecture, overlapEfficiency: 1, coreCount: 4, mxusPerCore: 2, multicastFactor: 8 }, workload);
  assert.ok(serial.deviceLatencySeconds >= overlapped.deviceLatencySeconds);
  assert.equal(overlapped.effectiveHbmBandwidthGbs, 80);
  assert.ok(overlapped.nocTrafficBytes > 0);
  assert.ok(overlapped.totalEnergyJ > 0);
  assert.equal(overlapped.averagePowerW, overlapped.totalEnergyJ / overlapped.estimatedLatencySeconds);
});

test("supports new formats, model-layer style workloads, and 5% pruning", async () => {
  const { evaluateRoofline, STORAGE_BITS } = await vite.ssrLoadModule("/lib/roofline.ts");
  for (const precision of ["posit-8", "posit-16", "mxfp4", "mxint8", "nvfp4"]) {
    assert.ok(STORAGE_BITS[precision] > 0);
  }
  const dense = evaluateRoofline(architecture, { ...workload, name: "Qwen2.5 · mlp_up", pruningPercent: 0, weightPrecision: "mxfp4" });
  const pruned = evaluateRoofline(architecture, { ...workload, pruningPercent: 50, weightPrecision: "mxfp4" });
  assert.equal(pruned.pruningPercent, 50);
  assert.equal(pruned.operations, dense.operations / 2);
  assert.ok(pruned.weightReadBytes < dense.weightReadBytes);
});

test("models VXU lane waves separately from systolic MXU tiles", async () => {
  const { evaluateRoofline } = await vite.ssrLoadModule("/lib/roofline.ts");
  const vxu = evaluateRoofline(
    { ...architecture, computeFabric: "vxu", vectorLanes: 256, mxuRows: 1, mxuCols: 256 },
    { ...workload, m: 1, n: 1025, k: 4, pruningPercent: 0 },
  );
  const mxu = evaluateRoofline(
    { ...architecture, computeFabric: "mxu", mxuRows: 32, mxuCols: 32 },
    { ...workload, m: 1, n: 1025, k: 4, pruningPercent: 0 },
  );
  assert.equal(vxu.computeFabric, "vxu");
  assert.equal(vxu.vectorLanes, 256);
  assert.equal(vxu.vectorWaves, 5);
  assert.equal(vxu.bottleneck === "VXU" || vxu.bottleneck !== "MXU", true);
  assert.notEqual(vxu.cyclesPerTile, mxu.cyclesPerTile);
});

test("derives prefill and decode M and scales across layers and tokens", async () => {
  const { evaluateRoofline } = await vite.ssrLoadModule("/lib/roofline.ts");
  const prefill = evaluateRoofline(architecture, { ...workload, executionPhase: "prefill", batchSize: 2, sequenceLength: 16, layerCount: 4 });
  const decode = evaluateRoofline(architecture, { ...workload, executionPhase: "decode", batchSize: 2, decodeTokens: 8, layerCount: 4 });
  assert.equal(prefill.effectiveM, 32);
  assert.equal(prefill.invocationCount, 4);
  assert.equal(decode.effectiveM, 2);
  assert.equal(decode.invocationCount, 32);
  assert.equal(prefill.denseOperations, 2 * 32 * workload.n * workload.k * 4);
});

test("models bank size, double buffering, and DVFS", async () => {
  const { evaluateRoofline } = await vite.ssrLoadModule("/lib/roofline.ts");
  const base = evaluateRoofline({ ...architecture, sramBankSizeKib: 128, frequencyGhz: 1, nominalFrequencyGhz: 1, voltageV: 0.8, nominalVoltageV: 0.8 }, workload);
  const buffered = evaluateRoofline({ ...architecture, sramBankSizeKib: 128, doubleBuffering: true }, workload);
  const turbo = evaluateRoofline({ ...architecture, frequencyGhz: 2, nominalFrequencyGhz: 1, voltageV: 1.0, nominalVoltageV: 0.8 }, workload);
  assert.equal(base.sramCapacityBytes, 4 * 128 * 1024);
  assert.equal(buffered.usableSramCapacityBytes, buffered.sramCapacityBytes / 2);
  assert.equal(turbo.totalPeakTops, 2 * base.totalPeakTops);
  assert.ok(turbo.computeEnergyJ > base.computeEnergyJ);
});

test("models process corner, temperature, and technology-node PVT scaling", async () => {
  const { evaluateRoofline } = await vite.ssrLoadModule("/lib/roofline.ts");
  const tt16 = evaluateRoofline({ ...architecture, processCorner: "tt", temperatureC: 25, technologyNodeNm: 16 }, workload);
  const ssHot = evaluateRoofline({ ...architecture, processCorner: "ss", temperatureC: 125, technologyNodeNm: 16 }, workload);
  const ffWarm = evaluateRoofline({ ...architecture, processCorner: "ff", temperatureC: 85, technologyNodeNm: 16 }, workload);
  const tt7 = evaluateRoofline({ ...architecture, processCorner: "tt", temperatureC: 25, technologyNodeNm: 7 }, workload);
  assert.ok(ssHot.totalPeakTops < tt16.totalPeakTops);
  assert.ok(ssHot.leakageScale > tt16.leakageScale);
  assert.ok(ffWarm.cornerFrequencyFactor > tt16.cornerFrequencyFactor);
  assert.ok(tt7.totalPeakTops > tt16.totalPeakTops);
  assert.ok(tt7.computeEnergyJ < tt16.computeEnergyJ);
  assert.equal(ssHot.temperatureC, 125);
  assert.equal(tt7.technologyNodeNm, 7);
});

test("applies VXU RF and issue constraints and emits a non-dominated Pareto frontier", async () => {
  const { evaluateRoofline, buildAccuracyEnergyPareto } = await vite.ssrLoadModule("/lib/roofline.ts");
  const constrained = evaluateRoofline({ ...architecture, computeFabric: "vxu", vectorLanes: 4096, vectorRegisterFileKib: 1, vectorRegisterBandwidthGbs: 0.001, vectorIssueWidth: 0.5 }, workload);
  assert.ok(constrained.registerFileResidency < 1);
  assert.equal(constrained.vectorIssueEfficiency, 0.5);
  assert.equal(constrained.bottleneck, "RF");
  const frontier = buildAccuracyEnergyPareto(architecture, { ...workload, baselineAccuracyPercent: 80, precisionSensitivity: 2, pruningSensitivity: 10, pruningExponent: 1.5 });
  assert.ok(frontier.length > 1);
  for (let index = 1; index < frontier.length; index += 1) {
    assert.ok(frontier[index].totalEnergyJ >= frontier[index - 1].totalEnergyJ);
    assert.ok(frontier[index].estimatedAccuracyPercent > frontier[index - 1].estimatedAccuracyPercent);
  }
});
