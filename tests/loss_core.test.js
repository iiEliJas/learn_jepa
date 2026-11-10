const test = require("node:test");
const assert = require("node:assert/strict");
const Core = require("../assets/sigreg_core.js");
const Loss = require("../loss_calculator/loss_core.js");

function makeInvariantBatch() {
  const base = [
    [-1.2, 0.3],
    [0.7, -1.1],
    [1.4, 0.9],
    [0.1, 1.3],
    [-0.4, -0.8],
    [1.0, 0.2],
  ];
  return Array.from({ length: 4 }, () => base.map((row) => row.slice()));
}

test("prediction term is zero for identical views and mu is the view", () => {
  const z = makeInvariantBatch();
  const result = Loss.predictionTerms(z, { Vg: 4, reduction: "mean_over_k" });
  assert.equal(result.lPred, 0);
  assert.deepEqual(result.mu[0], z[0][0]);
});

test("sum_over_k is K times mean_over_k", () => {
  const z = makeInvariantBatch();
  z[2][3][1] += 0.8;
  const mean = Loss.predictionTerms(z, { Vg: 4, reduction: "mean_over_k" });
  const sum = Loss.predictionTerms(z, { Vg: 4, reduction: "sum_over_k" });
  assert.ok(Math.abs(sum.lPred - mean.lPred * 2) < 1e-12);
});

test("V_g changes the target mean but not the view set", () => {
  const z = makeInvariantBatch();
  z[3][0][0] += 2.0;
  const all = Loss.predictionTerms(z, { Vg: 4 });
  const global = Loss.predictionTerms(z, { Vg: 2 });
  assert.ok(global.mu[0][0] < all.mu[0][0]);
  assert.notEqual(global.lPred, all.lPred);
});

test("SIGReg is large for a collapsed batch", () => {
  const z = makeInvariantBatch().map((view) =>
    view.map(() => [0.1, 0.1])
  );
  const directions = Loss.makeDirections(2, 64, 987);
  const result = Loss.lejepa(z, {
    lambda: 0.02,
    Vg: 4,
    directions,
    sigregMode: "per_view",
  });
  assert.equal(result.lPred, 0);
  assert.ok(result.lSig > Core.gaussianNullExpected({ tMax: 3, nKnots: 17 }));
});

test("lambda endpoints isolate the two terms", () => {
  const z = makeInvariantBatch();
  z[0][0][0] += 0.5;
  const directions = Loss.makeDirections(2, 32, 5);
  const onlyPred = Loss.lejepa(z, { lambda: 0, directions });
  const onlySig = Loss.lejepa(z, { lambda: 1, directions });
  assert.ok(Math.abs(onlyPred.loss - onlyPred.lPred) < 1e-12);
  assert.ok(Math.abs(onlySig.loss - onlySig.lSig) < 1e-12);
});

test("per_view and joint modes are different estimators", () => {
  const z = makeInvariantBatch();
  z[0][0][0] += 3.0;
  const directions = Loss.makeDirections(2, 64, 11);
  const perView = Loss.lejepa(z, { directions, sigregMode: "per_view" });
  const joint = Loss.lejepa(z, { directions, sigregMode: "joint" });
  assert.notEqual(perView.lSig, joint.lSig);
  assert.equal(perView.perView.length, 4);
  assert.equal(joint.perView.length, 1);
});

test("typicalSigreg returns finite draws with a spread", () => {
  const z = makeInvariantBatch();
  const values = Loss.typicalSigreg(z, { sigregMode: "per_view" }, 12);
  assert.equal(values.length, 12);
  assert.ok(values.every((value) => Number.isFinite(value) && value >= 0));
  assert.ok(Math.max(...values) > Math.min(...values));
});

test("validation rejects inconsistent tensors", () => {
  assert.throws(() => Loss.validate([]), /at least one view/);
  assert.throws(() => Loss.validate([[[1, 2]], [[1]]]), /same number of samples|same dimension/);
});
