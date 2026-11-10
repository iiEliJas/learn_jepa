const test = require("node:test");
const assert = require("node:assert/strict");
const Core = require("../assets/sigreg_core.js");

const POINTS = [
  [0.1, 0.2],
  [-0.3, 0.4],
  [0.5, -0.1],
  [-0.2, -0.5],
  [0.7, 0.3],
  [-0.6, 0.2],
  [0.0, 0.8],
  [0.4, -0.7],
];
const DIRECTIONS = [
  [1, 0],
  [0, 1],
  [Math.SQRT1_2, Math.SQRT1_2],
];
const OPTIONS = { tMax: 3, nKnots: 17 };

test("quadrature uses the official one-sided folded weights", () => {
  const quad = Core.quadrature(3, 17);
  assert.equal(quad.t.length, 17);
  assert.equal(quad.t[0], 0);
  assert.equal(quad.t[16], 3);
  const dt = 3 / 16;
  assert.ok(Math.abs(quad.raw[0] - dt) < 1e-15);
  assert.ok(Math.abs(quad.raw[16] - dt) < 1e-15);
  for (let i = 1; i < 16; i += 1) {
    assert.ok(Math.abs(quad.raw[i] - 2 * dt) < 1e-15);
  }
  for (let i = 0; i < 17; i += 1) {
    assert.ok(Math.abs(quad.weights[i] - quad.raw[i] * quad.phi0[i]) < 1e-15);
  }
});

test("sigmaReg matches the PyTorch reference value", () => {
  const result = Core.sigmaReg(POINTS, DIRECTIONS, OPTIONS);
  // Computed with pysrc/lejepa/sigreg.py in FP64 on the same points and
  // directions. This pins both the quadrature and the 1/S averaging.
  assert.ok(Math.abs(result.value - 1.6191886765419181) < 1e-12);
  const expected = [1.6479710760113573, 1.417823927128129, 1.7917710264862683];
  result.perSlice.forEach((value, index) => {
    assert.ok(Math.abs(value - expected[index]) < 1e-12);
  });
});

test("gaussianNullExpected is the finite-grid expectation, not zero", () => {
  const expected = Core.gaussianNullExpected(OPTIONS);
  // sum_t (1 - phi0^2) * weights for [0, 3], 17 knots.
  assert.ok(Math.abs(expected - 1.0524641547991607) < 1e-12);
  assert.ok(expected > 1.0);
});

test("analytic gradient matches central finite differences", () => {
  const result = Core.sigmaReg(POINTS, DIRECTIONS, OPTIONS);
  const epsilon = 1e-6;
  for (const [index, dim] of [
    [0, 0],
    [0, 1],
    [3, 1],
    [7, 0],
  ]) {
    const plus = POINTS.map((point) => point.slice());
    const minus = POINTS.map((point) => point.slice());
    plus[index][dim] += epsilon;
    minus[index][dim] -= epsilon;
    const up = Core.sigmaRegValue(plus, DIRECTIONS, OPTIONS);
    const down = Core.sigmaRegValue(minus, DIRECTIONS, OPTIONS);
    const numerical = (up - down) / (2 * epsilon);
    const analytic = result.gradient[index][dim];
    const scale = Math.max(1, Math.abs(numerical), Math.abs(analytic));
    assert.ok(
      Math.abs(numerical - analytic) / scale < 2e-5,
      `gradient mismatch at point ${index} dim ${dim}: ${analytic} vs ${numerical}`
    );
  }
});

test("anisotropic and shifted clouds cost more than a Gaussian cloud", () => {
  const rng = Core.mulberry32(2026);
  const gaussian = Core.standardNormalPoints(1200, 2, rng);
  const anisotropic = gaussian.map(([x, y]) => [2.6 * x, 0.22 * y]);
  const shifted = gaussian.map(([x, y]) => [x + 2.2, y]);
  const directions = Core.sampleDirections(2, 64, Core.mulberry32(7));
  const g = Core.sigmaRegValue(gaussian, directions, OPTIONS);
  const a = Core.sigmaRegValue(anisotropic, directions, OPTIONS);
  const s = Core.sigmaRegValue(shifted, directions, OPTIONS);
  assert.ok(a > g * 2, `expected anisotropic ${a} >> gaussian ${g}`);
  assert.ok(s > g * 2, `expected shifted ${s} >> gaussian ${g}`);
});

test("correlatedGaussianPoints has the requested shape and rough scale", () => {
  const rng = Core.mulberry32(3);
  const points = Core.correlatedGaussianPoints(
    500,
    { stdX: 2, stdY: 0.5, corr: 0.8, offsetX: 1, offsetY: -1 },
    rng
  );
  assert.equal(points.length, 500);
  const xs = points.map((point) => point[0]);
  const ys = points.map((point) => point[1]);
  const mx = Core.meanAndStd(xs);
  const my = Core.meanAndStd(ys);
  assert.ok(Math.abs(mx.mean - 1) < 0.3);
  assert.ok(Math.abs(my.mean + 1) < 0.15);
  assert.ok(Math.abs(mx.std - 2) < 0.3);
  assert.ok(Math.abs(my.std - 0.5) < 0.15);
});
