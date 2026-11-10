/* LeJEPA objective helpers for the interactive loss calculator.
 *
 * z is a nested array: z[view][sample][dimension].
 * The conventions match `pysrc/lejepa/loss.py`:
 *   mu      = mean of the first V_g views
 *   L_pred  = mean over (view, sample, dimension) of (mu - z)^2
 *   L_sig   = mean over views of SIGReg(z[view])
 *   L       = (1 - lambda) * L_pred + lambda * L_sig
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory(require("../assets/sigreg_core.js"));
  } else {
    root.LossCore = factory(root.SigregCore);
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function (SigregCore) {
  "use strict";

  if (!SigregCore) {
    throw new Error("LossCore needs SigregCore loaded first");
  }

  function zeros(shape) {
    return Array.from({ length: shape[0] }, () =>
      Array.from({ length: shape[1] }, () => 0)
    );
  }

  function validate(z) {
    if (!Array.isArray(z) || z.length === 0) throw new Error("z must have at least one view");
    const N = z[0].length;
    const K = z[0][0].length;
    if (N === 0 || K === 0) throw new Error("z must have at least one sample and one dimension");
    for (const view of z) {
      if (view.length !== N) throw new Error("all views must have the same number of samples");
      for (const sample of view) {
        if (sample.length !== K) throw new Error("all samples must have the same dimension");
      }
    }
    return { V: z.length, N, K };
  }

  function flatten(z) {
    const { V, N, K } = validate(z);
    const points = [];
    for (let v = 0; v < V; v += 1) {
      for (let n = 0; n < N; n += 1) {
        points.push(z[v][n].slice());
      }
    }
    return { points, V, N, K };
  }

  function predictionTerms(z, options) {
    const opts = Object.assign({ Vg: null, reduction: "mean_over_k" }, options || {});
    const { V, N, K } = validate(z);
    const Vg = opts.Vg === null || opts.Vg === undefined ? V : Math.max(1, Math.min(V, opts.Vg));

    const mu = Array.from({ length: N }, () => new Array(K).fill(0));
    for (let n = 0; n < N; n += 1) {
      for (let k = 0; k < K; k += 1) {
        let sum = 0;
        for (let v = 0; v < Vg; v += 1) sum += z[v][n][k];
        mu[n][k] = sum / Vg;
      }
    }

    const residuals = z.map((view) =>
      view.map((sample, n) => sample.map((value, k) => mu[n][k] - value))
    );

    let lPred;
    let elementCount;
    if (opts.reduction === "sum_over_k") {
      let total = 0;
      for (let v = 0; v < V; v += 1) {
        for (let n = 0; n < N; n += 1) {
          let row = 0;
          for (let k = 0; k < K; k += 1) row += residuals[v][n][k] ** 2;
          total += row;
        }
      }
      lPred = total / (V * N);
      elementCount = V * N * K;
    } else if (opts.reduction === "mean_over_k") {
      let total = 0;
      for (let v = 0; v < V; v += 1) {
        for (let n = 0; n < N; n += 1) {
          for (let k = 0; k < K; k += 1) total += residuals[v][n][k] ** 2;
        }
      }
      lPred = total / (V * N * K);
      elementCount = V * N * K;
    } else {
      throw new Error("reduction must be mean_over_k or sum_over_k");
    }

    return { mu, residuals, lPred, V, N, K, Vg, reduction: opts.reduction, elementCount };
  }

  function sigregTerms(z, options) {
    const opts = Object.assign(
      {
        directions: null,
        sigregMode: "per_view",
        tMax: 3,
        nKnots: 17,
      },
      options || {}
    );
    const { V, N, K } = validate(z);
    let directions = opts.directions;
    if (!directions) {
      directions = SigregCore.sampleDirections(K, 64, SigregCore.mulberry32(1234));
    }
    const sigOpts = { tMax: opts.tMax, nKnots: opts.nKnots };
    if (opts.sigregMode === "joint") {
      const { points } = flatten(z);
      const result = SigregCore.sigmaReg(points, directions, sigOpts);
      return {
        lSig: result.value,
        perView: [result.value],
        mode: "joint",
        directions,
        results: [result],
        N,
        K,
      };
    }
    if (opts.sigregMode !== "per_view") {
      throw new Error("sigregMode must be per_view or joint");
    }
    const results = z.map((view) => SigregCore.sigmaReg(view, directions, sigOpts));
    const perView = results.map((result) => result.value);
    const lSig = perView.reduce((sum, value) => sum + value, 0) / perView.length;
    return { lSig, perView, mode: "per_view", directions, results, N, K };
  }

  function lejepa(z, options) {
    const opts = Object.assign(
      {
        lambda: 0.02,
        Vg: null,
        reduction: "mean_over_k",
        sigregMode: "per_view",
        directions: null,
        tMax: 3,
        nKnots: 17,
      },
      options || {}
    );
    if (!(opts.lambda >= 0 && opts.lambda <= 1)) {
      throw new Error("lambda must be in [0, 1]");
    }
    const pred = predictionTerms(z, opts);
    const sig = sigregTerms(z, opts);
    const weightedPred = (1 - opts.lambda) * pred.lPred;
    const weightedSig = opts.lambda * sig.lSig;
    return {
      ...pred,
      ...sig,
      lambda: opts.lambda,
      loss: weightedPred + weightedSig,
      weightedPred,
      weightedSig,
      sigFraction: pred.lPred + sig.lSig > 0
        ? weightedSig / (weightedPred + weightedSig)
        : 0,
    };
  }

  function typicalSigreg(z, options, draws) {
    const opts = Object.assign(
      { lambda: 0.02, Vg: null, reduction: "mean_over_k", sigregMode: "per_view", tMax: 3, nKnots: 17, seed: 7 },
      options || {}
    );
    const count = draws || 24;
    const { K } = validate(z);
    const values = [];
    for (let d = 0; d < count; d += 1) {
      const directions = SigregCore.sampleDirections(
        K,
        64,
        SigregCore.mulberry32((opts.seed + d * 7919) >>> 0)
      );
      const sig = sigregTerms(z, { ...opts, directions });
      values.push(sig.lSig);
    }
    return values;
  }

  function makeDirections(dim, count, seed) {
    return SigregCore.sampleDirections(dim, count, SigregCore.mulberry32(seed >>> 0));
  }

  return {
    validate,
    flatten,
    predictionTerms,
    sigregTerms,
    lejepa,
    typicalSigreg,
    makeDirections,
    zeros,
  };
});
