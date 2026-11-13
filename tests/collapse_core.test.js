const test = require("node:test");
const assert = require("node:assert/strict");
const SigregCore = require("../assets/sigreg_core.js");
const Core = require("../collapse_sim/collapse_core.js");

test("makeSyntheticData is deterministic and keeps views near the base point", () => {
  const first = Core.makeSyntheticData({ count: 24, seed: 3, noise: 0.2 });
  const second = Core.makeSyntheticData({ count: 24, seed: 3, noise: 0.2 });
  assert.deepEqual(first.base, second.base);
  assert.deepEqual(first.views, second.views);
  assert.equal(first.base.length, 24);
  assert.equal(first.views.length, 2);
  assert.equal(first.views[0].length, 24);
  assert.equal(first.clusters[0], 0);
  assert.equal(first.clusters[3], 0);

  const other = Core.makeSyntheticData({ count: 24, seed: 4, noise: 0.2 });
  assert.notDeepEqual(first.base, other.base);

  let total = 0;
  for (let index = 0; index < 24; index += 1) {
    const view = first.views[0][index];
    total += Math.hypot(view[0] - first.base[index][0], view[1] - first.base[index][1]);
  }
  assert.ok(total / 24 > 0.05 && total / 24 < 0.6);
});

test("encoder and predictor have the expected shapes", () => {
  const data = Core.makeSyntheticData({ count: 5, seed: 3 });
  const params = Core.initParams({ hidden: 6, seed: 4 });
  const encoded = Core.encode(params, data.views[0]);
  assert.equal(encoded.hidden.length, 5);
  assert.equal(encoded.hidden[0].length, 6);
  assert.equal(encoded.embeddings[0].length, 2);
  const predicted = Core.predict(params, encoded.embeddings);
  assert.equal(predicted.predicted.length, 5);
  assert.equal(predicted.predicted[0].length, 2);
  for (const value of encoded.embeddings[0]) assert.ok(Number.isFinite(value));
});

test("analytic gradients match central finite differences", () => {
  const data = Core.makeSyntheticData({ count: 8, seed: 3, noise: 0.15 });
  const params = Core.initParams({ hidden: 5, seed: 4 });
  const directions = SigregCore.sampleDirections(2, 4, SigregCore.mulberry32(9));
  const options = { lambda: 0.4, useSig: true, directions };
  const { grads } = Core.lossAndGradients(params, data, options);
  const total = () => Core.evaluate(params, data, options).total;
  const epsilon = 1e-6;

  const checks = [
    ["W1", 0, 0],
    ["W1", 3, 1],
    ["b1", 2],
    ["W2", 1, 0],
    ["b2", 0],
    ["P1", 0, 1],
    ["p1", 1],
    ["P2", 0, 2],
    ["p2", 1],
  ];

  for (const [name, row, column] of checks) {
    const target = params[name];
    const original = column === undefined ? target[row] : target[row][column];
    const analytic = column === undefined ? grads[name][row] : grads[name][row][column];
    const bump = (delta) => {
      if (column === undefined) target[row] = original + delta;
      else target[row][column] = original + delta;
    };
    bump(epsilon);
    const up = total();
    bump(-epsilon);
    const down = total();
    bump(0);
    const numeric = (up - down) / (2 * epsilon);
    assert.ok(
      Math.abs(numeric - analytic) < 2e-4,
      `${name}[${row}${column === undefined ? "" : `,${column}`}] analytic ${analytic} numeric ${numeric}`
    );
  }
});

test("participationRatio separates a line from an isotropic cloud", () => {
  const line = [
    [0, 0],
    [1, 0],
    [2, 0],
    [3, 0],
  ];
  assert.ok(Math.abs(Core.participationRatio(line, 4, 2) - 1) < 1e-12);

  const disk = [
    [1, 0],
    [0, 1],
    [-1, 0],
    [0, -1],
  ];
  assert.ok(Math.abs(Core.participationRatio(disk, 4, 2) - 2) < 1e-12);

  const point = [
    [2, -1],
    [2, -1],
  ];
  assert.equal(Core.participationRatio(point, 2, 2), 0);
  assert.throws(() => Core.participationRatio(line, 4, 3), /two dimensions/);
});

test("perDimensionStd reports both axes", () => {
  const points = [
    [-1, 0],
    [1, 2],
  ];
  const std = Core.perDimensionStd(points, 2, 2);
  assert.ok(Math.abs(std[0] - 1) < 1e-12);
  assert.ok(Math.abs(std[1] - 1) < 1e-12);
});

test("training is deterministic for a fixed seed", () => {
  const options = { count: 24, steps: 20, dirs: 4, seed: 5, noise: 0.2 };
  const first = Core.runTraining(options);
  const second = Core.runTraining(options);
  const a = first.snapshots[first.snapshots.length - 1];
  const b = second.snapshots[second.snapshots.length - 1];
  assert.equal(a.lPred, b.lPred);
  assert.equal(a.lSig, b.lSig);
  assert.equal(a.rank, b.rank);
});

test("without SIGReg the embedding collapses to a line; with it the rank stays high", () => {
  const common = { count: 48, steps: 160, dirs: 8, seed: 7, noise: 0.2, hidden: 8, lr: 0.02 };
  const off = Core.runTraining({ ...common, useSig: false });
  const on = Core.runTraining({ ...common, useSig: true, lambda: 0.5 });
  const offLast = off.snapshots[off.snapshots.length - 1];
  const onLast = on.snapshots[on.snapshots.length - 1];
  assert.ok(offLast.rank < 1.15, `rank without SIGReg ${offLast.rank}`);
  assert.ok(onLast.rank > 1.7, `rank with SIGReg ${onLast.rank}`);
  assert.ok(offLast.lSig > 5, `L_sig without SIGReg ${offLast.lSig}`);
  assert.ok(onLast.lSig < 4, `L_sig with SIGReg ${onLast.lSig}`);
  assert.ok(offLast.lPred < onLast.lPred);
});
