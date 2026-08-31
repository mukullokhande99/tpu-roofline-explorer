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
