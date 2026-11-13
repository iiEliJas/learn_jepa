const test = require("node:test");
const assert = require("node:assert/strict");
const Core = require("../assets/projector_core.js");

test("normalCdf is accurate at the usual anchors", () => {
  assert.ok(Math.abs(Core.normalCdf(0) - 0.5) < 1e-9);
  assert.ok(Math.abs(Core.normalCdf(1.96) - 0.975) < 5e-4);
  assert.ok(Math.abs(Core.normalCdf(-1.96) - 0.025) < 5e-4);
  assert.ok(Core.normalCdf(-8) < 1e-12);
  assert.ok(Core.normalCdf(8) > 1 - 1e-12);
});

test("columnStats reports moments for a symmetric three-point sample", () => {
  const points = new Float32Array([-1, 0, 1]);
  const stats = Core.columnStats(points, 3, 1);
  assert.equal(stats.mean[0], 0);
  assert.ok(Math.abs(stats.std[0] - Math.sqrt(2 / 3)) < 1e-12);
  assert.ok(Math.abs(stats.skew[0]) < 1e-12);
  assert.ok(Math.abs(stats.kurtosis[0] + 1.5) < 1e-12);
  assert.equal(stats.min[0], -1);
  assert.equal(stats.max[0], 1);
  assert.ok(stats.ks[0] > 0 && stats.ks[0] < 0.6);
});

test("columnStats handles a constant column without dividing by zero", () => {
  const stats = Core.columnStats(new Float32Array([2, 2, 2]), 3, 1);
  assert.equal(stats.std[0], 0);
  assert.equal(stats.skew[0], 0);
  assert.equal(stats.kurtosis[0], 0);
});

test("correlationMatrix is symmetric with unit diagonal", () => {
  const points = new Float32Array([
    1, -2,
    2, -4,
    3, -6,
    4, -8,
  ]);
  const matrix = Core.correlationMatrix(points, 4, 2);
  assert.equal(matrix[0][0], 1);
  assert.equal(matrix[1][1], 1);
  assert.ok(Math.abs(matrix[0][1] + 1) < 1e-12);
  assert.equal(matrix[0][1], matrix[1][0]);
});

test("correlationSummary reports the strongest off-diagonal pair", () => {
  const matrix = [
    [1, 0.1, -0.6],
    [0.1, 1, 0.3],
    [-0.6, 0.3, 1],
  ];
  const summary = Core.correlationSummary(matrix);
  assert.deepEqual(summary.maxPair, [0, 2]);
  assert.ok(Math.abs(summary.maxAbs - 0.6) < 1e-12);
  assert.ok(Math.abs(summary.maxValue + 0.6) < 1e-12);
  assert.ok(summary.meanAbs > 0.3 && summary.meanAbs < 0.35);
});

test("rescaleTs keeps the null floor fixed and scales the signal", () => {
  const nullExpected = 1.0525;
  const reference = Core.rescaleTs(20.664, 2000, 256, nullExpected);
  assert.ok(
    Math.abs(reference - (nullExpected + (20.664 - nullExpected) * 0.128)) < 1e-12
  );
  assert.ok(reference > 20.664 * 0.128);
  assert.ok(reference < 4);
  assert.equal(Core.rescaleTs(3.5, 256, 256, nullExpected), 3.5);
  assert.equal(Core.rescaleTs(3.5, 0, 256, nullExpected), nullExpected);
  assert.equal(Core.rescaleTs(nullExpected, 2000, 256, nullExpected), nullExpected);
});

test("sampleIndices is deterministic, unique, sorted and bounded", () => {
  const rng = () => 0.25;
  const first = Core.sampleIndices(10, 4, rng);
  const second = Core.sampleIndices(10, 4, rng);
  assert.deepEqual(first, second);
  assert.equal(first.length, 4);
  assert.equal(new Set(first).size, 4);
  assert.deepEqual(first.slice().sort((a, b) => a - b), first);
  for (const value of first) assert.ok(value >= 0 && value < 10);
  assert.deepEqual(Core.sampleIndices(4, 9, rng), [0, 1, 2, 3]);
});
