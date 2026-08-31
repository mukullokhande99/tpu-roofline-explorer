import assert from "node:assert/strict";
import test, { after } from "node:test";
import { fileURLToPath } from "node:url";
import { createServer } from "vite";

const root = fileURLToPath(new URL("..", import.meta.url));
const vite = await createServer({
  appType: "custom",
  configFile: false,
  root,
  resolve: { alias: { "@": root } },
  server: { middlewareMode: true },
});

after(async () => vite.close());

test("matches the 256x256 single-token reference", async () => {
  const { evaluateRoofline } = await vite.ssrLoadModule("/lib/roofline.ts");
  const result = evaluateRoofline(
    {
      name: "B",
      peakComputeTops: 131.072,
      hbmBandwidthGbs: 900,
      sramCapacityMib: 64,
      mxuRows: 256,
      mxuCols: 256,
    },
    {
      name: "decode",
      m: 1,
      n: 11008,
      k: 4096,
      precision: "bf16",
      weightReuseFactor: 8,
      activationReuseFactor: 4,
    },
  );

  assert.equal(result.tilesM, 1);
  assert.equal(result.tilesN, 43);
  assert.ok(Math.abs(result.mxuUtilization - 0.0034737299) < 1e-9);
  assert.ok(Math.abs(result.estimatedLatencySeconds * 1e6 - 198.058) < 0.001);
  assert.equal(result.effectiveWeightReuse, 1);
});

test("more SRAM cannot reduce reuse", async () => {
  const { evaluateRoofline } = await vite.ssrLoadModule("/lib/roofline.ts");
  const workload = {
    name: "square",
    m: 4096,
    n: 4096,
    k: 4096,
    precision: "bf16",
    weightReuseFactor: 8,
    activationReuseFactor: 4,
  };
  const base = {
    name: "A",
    peakComputeTops: 32.768,
    hbmBandwidthGbs: 900,
    mxuRows: 128,
    mxuCols: 128,
  };
  const small = evaluateRoofline({ ...base, sramCapacityMib: 0.25 }, workload);
  const large = evaluateRoofline({ ...base, sramCapacityMib: 64 }, workload);

  assert.ok(large.effectiveWeightReuse >= small.effectiveWeightReuse);
  assert.ok(large.effectiveActivationReuse >= small.effectiveActivationReuse);
  assert.ok(large.bytesTransferred <= small.bytesTransferred);
});
