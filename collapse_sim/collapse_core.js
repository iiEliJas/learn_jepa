/* Tiny LeJEPA trainer for the Collapse Simulator.
 *
 * A 2 -> hidden -> 2 tanh encoder, a 2 -> hidden -> 2 tanh predictor, and the
 * LeJEPA objective on two views of synthetic 2-D data:
 *
 *   L = L_pred + lambda * L_SIGReg
 *
 * Everything is plain arrays and manual backprop so the page can train a few
 * hundred steps in the browser without a framework.  The SIGReg value and its
 * gradient come from `assets/sigreg_core.js`, the same code the lab and the
 * loss calculator use.
 *
 * UMD-wrapped: classic script in the browser, require() in the Node tests.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory(require("../assets/sigreg_core.js"));
  } else {
    root.CollapseCore = factory(root.SigregCore);
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function (SigregCore) {
  "use strict";

  if (!SigregCore) {
    throw new Error("CollapseCore needs SigregCore loaded first");
  }

  const K = 2;

  function zeros(count) {
    return new Array(count).fill(0);
  }

  function matrix(rows, columns, rng, scale) {
    const out = new Array(rows);
    for (let row = 0; row < rows; row += 1) {
      out[row] = new Array(columns);
      for (let column = 0; column < columns; column += 1) {
        out[row][column] = (rng() * 2 - 1) * scale;
      }
    }
    return out;
  }

  function gaussian(rng) {
    let u = 0;
    let v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /* A triangle of three Gaussian clusters.  The triangle makes a collapsed
   * embedding easy to see: three blobs become one line, then one point. */
  function makeSyntheticData(options) {
    const opts = Object.assign({ count: 96, seed: 7, noise: 0.2, spread: 0.16 }, options || {});
    const rng = SigregCore.mulberry32(opts.seed >>> 0);
    const centers = [
      [1, 0],
      [-0.5, 0.866],
      [-0.5, -0.866],
    ];
    const base = new Array(opts.count);
    const views = [new Array(opts.count), new Array(opts.count)];
    const clusters = new Array(opts.count);
    for (let index = 0; index < opts.count; index += 1) {
      const cluster = index % centers.length;
      const center = centers[cluster];
      const point = [
        center[0] + opts.spread * gaussian(rng),
        center[1] + opts.spread * gaussian(rng),
      ];
      base[index] = point;
      clusters[index] = cluster;
      for (let view = 0; view < 2; view += 1) {
        views[view][index] = [
          point[0] + opts.noise * gaussian(rng),
          point[1] + opts.noise * gaussian(rng),
        ];
      }
    }
    return { count: opts.count, base, views, clusters, noise: opts.noise, seed: opts.seed };
  }

  function initParams(options) {
    const opts = Object.assign({ hidden: 8, seed: 11 }, options || {});
    const rng = SigregCore.mulberry32(opts.seed >>> 0);
    const inputScale = Math.sqrt(2 / 2);
    const hiddenScale = Math.sqrt(2 / K);
    return {
      hidden: opts.hidden,
      W1: matrix(opts.hidden, 2, rng, inputScale),
      b1: zeros(opts.hidden),
      W2: matrix(K, opts.hidden, rng, hiddenScale),
      b2: zeros(K),
      P1: matrix(opts.hidden, K, rng, inputScale),
      p1: zeros(opts.hidden),
      P2: matrix(K, opts.hidden, rng, hiddenScale),
      p2: zeros(K),
    };
  }

  function encode(params, points) {
    const count = points.length;
    const hidden = new Array(count);
    const embeddings = new Array(count);
    for (let row = 0; row < count; row += 1) {
      const h = new Array(params.W1.length);
      for (let j = 0; j < params.W1.length; j += 1) {
        let sum = params.b1[j];
        sum += params.W1[j][0] * points[row][0];
        sum += params.W1[j][1] * points[row][1];
        h[j] = Math.tanh(sum);
      }
      const z = new Array(K);
      for (let k = 0; k < K; k += 1) {
        let sum = params.b2[k];
        for (let j = 0; j < h.length; j += 1) sum += params.W2[k][j] * h[j];
        z[k] = sum;
      }
      hidden[row] = h;
      embeddings[row] = z;
    }
    return { hidden, embeddings };
  }

  function predict(params, embeddings) {
    const count = embeddings.length;
    const hidden = new Array(count);
    const predicted = new Array(count);
    for (let row = 0; row < count; row += 1) {
      const h = new Array(params.P1.length);
      for (let j = 0; j < params.P1.length; j += 1) {
        let sum = params.p1[j];
        sum += params.P1[j][0] * embeddings[row][0];
        sum += params.P1[j][1] * embeddings[row][1];
        h[j] = Math.tanh(sum);
      }
      const out = new Array(K);
      for (let k = 0; k < K; k += 1) {
        let sum = params.p2[k];
        for (let j = 0; j < h.length; j += 1) sum += params.P2[k][j] * h[j];
        out[k] = sum;
      }
      hidden[row] = h;
      predicted[row] = out;
    }
    return { hidden, predicted };
  }

  function sigregOptions(options) {
    return { tMax: options.tMax || 3, nKnots: options.nKnots || 17 };
  }

  function fixedDirections(options) {
    if (options.directions) return options.directions;
    const count = options.dirs || 12;
    if (options.rng) return SigregCore.sampleDirections(K, count, options.rng);
    return SigregCore.sampleDirections(K, count, SigregCore.mulberry32(1));
  }

  function predictionLoss(predicted, target) {
    const count = predicted.length;
    let total = 0;
    for (let row = 0; row < count; row += 1) {
      for (let k = 0; k < K; k += 1) {
        const error = predicted[row][k] - target[row][k];
        total += error * error;
      }
    }
    return total / (count * K);
  }

  /* Forward pass + objective, without gradients (used for the initial snapshot). */
  function evaluate(params, data, options) {
    const opts = Object.assign({ lambda: 0.5, useSig: true }, options || {});
    const f1 = encode(params, data.views[0]);
    const f2 = encode(params, data.views[1]);
    const out = predict(params, f1.embeddings);
    const lPred = predictionLoss(out.predicted, f2.embeddings);
    const directions = fixedDirections(opts);
    const sigOpts = sigregOptions(opts);
    const s1 = SigregCore.sigmaReg(f1.embeddings, directions, sigOpts);
    const s2 = SigregCore.sigmaReg(f2.embeddings, directions, sigOpts);
    const lSig = (s1.value + s2.value) / 2;
    return {
      lPred,
      lSig,
      total: lPred + (opts.useSig ? opts.lambda * lSig : 0),
      z1: f1.embeddings,
      z2: f2.embeddings,
      predicted: out.predicted,
      directions,
    };
  }

  /* Forward + backward pass.  The returned gradients are averaged over the
   * batch; callers apply the SGD update. */
  function lossAndGradients(params, data, options) {
    const opts = Object.assign({ lambda: 0.5, useSig: true }, options || {});
    const count = data.views[0].length;
    const hidden = params.W1.length;
    const f1 = encode(params, data.views[0]);
    const f2 = encode(params, data.views[1]);
    const out = predict(params, f1.embeddings);
    const lPred = predictionLoss(out.predicted, f2.embeddings);

    const dp = new Array(count);
    for (let row = 0; row < count; row += 1) {
      dp[row] = new Array(K);
      for (let k = 0; k < K; k += 1) {
        const error = out.predicted[row][k] - f2.embeddings[row][k];
        dp[row][k] = (2 * error) / (count * K);
      }
    }

    const directions = fixedDirections(opts);
    const sigOpts = sigregOptions(opts);
    const s1 = SigregCore.sigmaReg(f1.embeddings, directions, sigOpts);
    const s2 = SigregCore.sigmaReg(f2.embeddings, directions, sigOpts);
    const lSig = (s1.value + s2.value) / 2;
    const sigWeight = opts.useSig ? opts.lambda / 2 : 0;

    /* Predictor backward. */
    const gP2 = Array.from({ length: K }, () => zeros(hidden));
    const gp2 = zeros(K);
    const dhP = Array.from({ length: count }, () => zeros(hidden));
    for (let row = 0; row < count; row += 1) {
      for (let k = 0; k < K; k += 1) {
        gp2[k] += dp[row][k];
        for (let j = 0; j < hidden; j += 1) {
          gP2[k][j] += dp[row][k] * out.hidden[row][j];
        }
      }
    }
    for (let row = 0; row < count; row += 1) {
      for (let j = 0; j < hidden; j += 1) {
        let sum = 0;
        for (let k = 0; k < K; k += 1) sum += dp[row][k] * params.P2[k][j];
        const h = out.hidden[row][j];
        dhP[row][j] = sum * (1 - h * h);
      }
    }
    const gP1 = Array.from({ length: hidden }, () => zeros(K));
    const gp1 = zeros(hidden);
    const dz1 = Array.from({ length: count }, () => zeros(K));
    for (let row = 0; row < count; row += 1) {
      for (let j = 0; j < hidden; j += 1) {
        gp1[j] += dhP[row][j];
        for (let k = 0; k < K; k += 1) {
          gP1[j][k] += dhP[row][j] * f1.embeddings[row][k];
          dz1[row][k] += dhP[row][j] * params.P1[j][k];
        }
      }
    }

    /* SIGReg gradients on both views.  z2 is also the prediction target, so the
     * prediction loss contributes -dp to its gradient. */
    const dz2 = new Array(count);
    for (let row = 0; row < count; row += 1) {
      dz2[row] = new Array(K);
      for (let k = 0; k < K; k += 1) {
        dz2[row][k] = -dp[row][k] + (sigWeight !== 0 ? sigWeight * s2.gradient[row][k] : 0);
      }
    }
    if (sigWeight !== 0) {
      for (let row = 0; row < count; row += 1) {
        for (let k = 0; k < K; k += 1) dz1[row][k] += sigWeight * s1.gradient[row][k];
      }
    }

    /* Encoder backward: z = W2 * tanh(W1 x + b1) + b2. */
    const gW2 = Array.from({ length: K }, () => zeros(hidden));
    const gb2 = zeros(K);
    const dh1 = Array.from({ length: count }, () => zeros(hidden));
    const dh2 = Array.from({ length: count }, () => zeros(hidden));
    for (let row = 0; row < count; row += 1) {
      for (let k = 0; k < K; k += 1) {
        gb2[k] += dz1[row][k] + dz2[row][k];
        for (let j = 0; j < hidden; j += 1) {
          gW2[k][j] +=
            dz1[row][k] * f1.hidden[row][j] + dz2[row][k] * f2.hidden[row][j];
        }
      }
    }
    for (let row = 0; row < count; row += 1) {
      for (let j = 0; j < hidden; j += 1) {
        let sum1 = 0;
        let sum2 = 0;
        for (let k = 0; k < K; k += 1) {
          sum1 += dz1[row][k] * params.W2[k][j];
          sum2 += dz2[row][k] * params.W2[k][j];
        }
        const h1 = f1.hidden[row][j];
        const h2 = f2.hidden[row][j];
        dh1[row][j] = sum1 * (1 - h1 * h1);
        dh2[row][j] = sum2 * (1 - h2 * h2);
      }
    }
    const gW1 = Array.from({ length: hidden }, () => zeros(2));
    const gb1 = zeros(hidden);
    for (let row = 0; row < count; row += 1) {
      for (let j = 0; j < hidden; j += 1) {
        gb1[j] += dh1[row][j] + dh2[row][j];
        for (let d = 0; d < 2; d += 1) {
          gW1[j][d] +=
            dh1[row][j] * data.views[0][row][d] + dh2[row][j] * data.views[1][row][d];
        }
      }
    }

    return {
      lPred,
      lSig,
      total: lPred + (opts.useSig ? opts.lambda * lSig : 0),
      z1: f1.embeddings,
      z2: f2.embeddings,
      predicted: out.predicted,
      directions,
      grads: { W1: gW1, b1: gb1, W2: gW2, b2: gb2, P1: gP1, p1: gp1, P2: gP2, p2: gp2 },
    };
  }

  function applyUpdate(params, grads, learningRate) {
    const stepMatrix = (weights, gradient) => {
      for (let row = 0; row < weights.length; row += 1) {
        for (let column = 0; column < weights[row].length; column += 1) {
          weights[row][column] -= learningRate * gradient[row][column];
        }
      }
    };
    const stepVector = (values, gradient) => {
      for (let index = 0; index < values.length; index += 1) {
        values[index] -= learningRate * gradient[index];
      }
    };
    stepMatrix(params.W1, grads.W1);
    stepVector(params.b1, grads.b1);
    stepMatrix(params.W2, grads.W2);
    stepVector(params.b2, grads.b2);
    stepMatrix(params.P1, grads.P1);
    stepVector(params.p1, grads.p1);
    stepMatrix(params.P2, grads.P2);
    stepVector(params.p2, grads.p2);
  }

  function trainStep(params, data, options) {
    const opts = Object.assign({ lambda: 0.5, useSig: true, lr: 0.02, dirs: 8 }, options || {});
    const result = lossAndGradients(params, data, opts);
    applyUpdate(params, result.grads, opts.lr);
    return result;
  }

  function participationRatio(points, count, dim) {
    if (dim !== 2) throw new Error("participationRatio currently supports two dimensions");
    const mean = [0, 0];
    for (let row = 0; row < count; row += 1) {
      mean[0] += points[row][0];
      mean[1] += points[row][1];
    }
    mean[0] /= count;
    mean[1] /= count;
    let xx = 0;
    let yy = 0;
    let xy = 0;
    for (let row = 0; row < count; row += 1) {
      const dx = points[row][0] - mean[0];
      const dy = points[row][1] - mean[1];
      xx += dx * dx;
      yy += dy * dy;
      xy += dx * dy;
    }
    xx /= count;
    yy /= count;
    xy /= count;
    const trace = xx + yy;
    const gap = Math.sqrt((xx - yy) * (xx - yy) + 4 * xy * xy);
    const first = Math.max(0, (trace + gap) / 2);
    const second = Math.max(0, (trace - gap) / 2);
    const total = first + second;
    if (total <= 1e-12) return 0;
    return (total * total) / (first * first + second * second);
  }

  function perDimensionStd(points, count, dim) {
    if (dim !== 2) throw new Error("perDimensionStd currently supports two dimensions");
    const mean = [0, 0];
    for (let row = 0; row < count; row += 1) {
      mean[0] += points[row][0];
      mean[1] += points[row][1];
    }
    mean[0] /= count;
    mean[1] /= count;
    const variance = [0, 0];
    for (let row = 0; row < count; row += 1) {
      variance[0] += (points[row][0] - mean[0]) ** 2;
      variance[1] += (points[row][1] - mean[1]) ** 2;
    }
    return [Math.sqrt(variance[0] / count), Math.sqrt(variance[1] / count)];
  }

  function snapshotFor(step, z1, z2, result) {
    const joint = z1.concat(z2);
    return {
      step,
      lPred: result.lPred,
      lSig: result.lSig,
      total: result.total,
      rank: participationRatio(joint, joint.length, 2),
      std: perDimensionStd(joint, joint.length, 2),
      z1,
      z2,
    };
  }

  /* Train from scratch and record one snapshot per step. */
  function runTraining(options) {
    const opts = Object.assign(
      {
        count: 96,
        seed: 7,
        noise: 0.2,
        hidden: 8,
        steps: 240,
        lambda: 0.5,
        useSig: true,
        lr: 0.02,
        dirs: 8,
        tMax: 3,
        nKnots: 17,
      },
      options || {}
    );
    const data = opts.data || makeSyntheticData(opts);
    const params = initParams({ hidden: opts.hidden, seed: opts.seed + 1 });
    const rng = SigregCore.mulberry32((opts.seed + 2) >>> 0);
    const sigOpts = { tMax: opts.tMax, nKnots: opts.nKnots };

    const initial = evaluate(params, data, {
      lambda: opts.lambda,
      useSig: opts.useSig,
      dirs: opts.dirs,
      rng,
      tMax: opts.tMax,
      nKnots: opts.nKnots,
    });
    const snapshots = [snapshotFor(0, initial.z1, initial.z2, initial)];

    for (let step = 1; step <= opts.steps; step += 1) {
      const directions = SigregCore.sampleDirections(K, opts.dirs, rng);
      const result = lossAndGradients(params, data, {
        lambda: opts.lambda,
        useSig: opts.useSig,
        lr: opts.lr,
        directions,
        ...sigOpts,
      });
      applyUpdate(params, result.grads, opts.lr);
      snapshots.push(snapshotFor(step, result.z1, result.z2, result));
    }

    return { data, params, snapshots, config: opts };
  }

  return {
    K,
    makeSyntheticData,
    initParams,
    encode,
    predict,
    evaluate,
    lossAndGradients,
    applyUpdate,
    trainStep,
    participationRatio,
    perDimensionStd,
    runTraining,
  };
});
