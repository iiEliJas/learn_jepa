const test = require("node:test");
const assert = require("node:assert/strict");
const Core = require("../assets/atlas_core.js");

function base64(bytes) {
  return Buffer.from(bytes).toString("base64");
}

test("decodeMatrix round-trips int8 and float32 payloads", () => {
  const quantized = new Int8Array([1, -2, 3, 4, -5, 6]);
  const intSpace = {
    encoding: "int8",
    rows: 3,
    dim: 2,
    scale: [0.5, 2],
    data: base64(quantized),
  };
  const intRows = Core.decodeMatrix(intSpace);
  assert.deepEqual(Array.from(intRows), [0.5, -4, 1.5, 8, -2.5, 12]);

  const floats = new Float32Array([1.5, -2.25, 0, 3.75]);
  const floatSpace = {
    encoding: "float32",
    rows: 2,
    dim: 2,
    data: base64(new Uint8Array(floats.buffer)),
  };
  assert.deepEqual(Array.from(Core.decodeMatrix(floatSpace)), [1.5, -2.25, 0, 3.75]);
});

test("decodeFloat32 handles the per-view and local-layout payloads", () => {
  const values = new Float32Array([0, 1, 2, -3, 4.5, -6]);
  const decoded = Core.decodeFloat32(base64(new Uint8Array(values.buffer)));
  assert.equal(decoded.length, values.length);
  for (let index = 0; index < values.length; index += 1) {
    assert.ok(Math.abs(decoded[index] - values[index]) < 1e-6);
  }
});

test("centerRows removes the per-dimension mean", () => {
  const rows = new Float32Array([1, 2, 3, 4, 5, 6]);
  const centered = Core.centerRows(rows, 3, 2);
  assert.ok(Math.abs(centered[0] + 2) < 1e-6);
  assert.ok(Math.abs(centered[2]) < 1e-6);
  assert.ok(Math.abs(centered[4] - 2) < 1e-6);
});

test("predictionFromFlat matches the LeJEPA prediction formula", () => {
  const result = Core.predictionFromFlat(new Float32Array([1, 0, 3, 0]), 2, 2);
  assert.deepEqual(result.mu, [2, 0]);
  assert.deepEqual(result.contributions, [0.5, 0.5]);
  assert.equal(result.lPred, 0.5);
});

test("per-view contributions average to L_pred", () => {
  const flat = new Float32Array([0.2, -0.4, 1.1, 0.7, -0.6, 0.3, 0.9, -1.2, 0.1, 0.4]);
  const result = Core.predictionFromFlat(flat, 5, 2);
  const average =
    result.contributions.reduce((sum, value) => sum + value, 0) /
    result.contributions.length;
  assert.ok(Math.abs(average - result.lPred) < 1e-12);
});

test("shuffledPredictionForImage breaks the image correspondence", () => {
  const views = new Float32Array([
    0, 0,
    0.1, 0,
    10, 10,
    10.1, 10,
    20, 20,
    20.1, 20,
  ]);
  const paired = Core.predictionForImage(views, 3, 2, 2, 0);
  const shuffled = Core.shuffledPredictionForImage(views, 3, 2, 2, 0, [1, 2]);
  assert.deepEqual(shuffled.picked, [1, 2]);
  assert.ok(paired.lPred < 0.01);
  assert.ok(shuffled.lPred > 20);
  assert.ok(shuffled.lPred > paired.lPred);
});

test("pairwiseCosine is symmetric with a unit diagonal", () => {
  const flat = new Float32Array([1, 0, 0, 1, -1, 0]);
  const matrix = Core.pairwiseCosine(flat, 3, 2);
  for (let row = 0; row < 3; row += 1) {
    assert.equal(matrix[row][row], 1);
    for (let column = 0; column < 3; column += 1) {
      assert.ok(Math.abs(matrix[row][column] - matrix[column][row]) < 1e-12);
    }
  }
  assert.equal(matrix[0][1], 0);
  assert.equal(matrix[0][2], -1);
});

test("confusionMatrix counts rows as truth and columns as predictions", () => {
  const matrix = Core.confusionMatrix([0, 0, 1, 1, 1], [0, 1, 1, 1, 0], 3);
  assert.equal(matrix[0][0], 1);
  assert.equal(matrix[0][1], 1);
  assert.equal(matrix[1][2], 0);
  assert.equal(matrix[1][0], 1);
  assert.equal(matrix[1][1], 2);
});

test("nearestNeighbors sorts by cosine and excludes the query", () => {
  const rows = new Float32Array([1, 0, 0.9, 0.1, 0, 1]);
  const result = Core.nearestNeighbors(rows, 3, 2, 0, 2);
  assert.equal(result.length, 2);
  assert.equal(result[0].index, 1);
  assert.equal(result[1].index, 2);
  assert.ok(result[0].score > result[1].score);
});

test("classPurity and accuracy are simple fractions", () => {
  const labels = [0, 1, 1, 0];
  assert.equal(Core.classPurity([0, 2, 3], labels, 0), 2 / 3);
  assert.equal(Core.classPurity([1, 0], labels, 1), 1 / 2);
  assert.equal(Core.accuracy([0, 1, 1, 0], [0, 0, 1, 1]), 0.5);
});

test("meanDirectionRatio is one for a shared non-zero direction", () => {
  const rows = new Float32Array([1, 0, 2, 0, 3, 0]);
  assert.ok(Math.abs(Core.meanDirectionRatio(rows, 3, 2) - 1) < 1e-12);
  const centered = Core.centerRows(rows, 3, 2);
  assert.ok(Core.meanDirectionRatio(centered, 3, 2) < 1e-6);
});
