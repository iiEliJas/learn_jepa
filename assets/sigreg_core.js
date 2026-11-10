/* Exact SIGReg math used by the interactive learning demos.
 *
 * This file is intentionally dependency-free and UMD-wrapped:
 *   - the browser pages load it as a classic script and use `SigregCore`;
 *   - the Node tests `require()` it.
 *
 * Conventions match `pysrc/lejepa/sigreg.py` and protocol section 5.2:
 * one-sided quadrature on [0, t_max], n_knots odd, endpoint weights dt and
 * interior weights 2*dt, all multiplied by phi_0(t) = exp(-t^2 / 2). The
 * statistic is T_s = N * sum_t err(t) * weights(t), averaged over slices.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.SigregCore = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function linspace(start, end, count) {
    if (count <= 0) return [];
    if (count === 1) return [start];
    const step = (end - start) / (count - 1);
    return Array.from({ length: count }, (_, index) => start + index * step);
  }

  function mulberry32(seed) {
    let state = seed >>> 0;
    return function random() {
      state = (state + 0x6d2b79f5) >>> 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function randn(rng) {
    const random = rng || Math.random;
    let u = 0;
    let v = 0;
    while (u === 0) u = random();
    while (v === 0) v = random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  function quadrature(tMax, nKnots) {
    if (!(tMax > 0)) throw new Error("tMax must be positive");
    if (!Number.isInteger(nKnots) || nKnots < 3 || nKnots % 2 === 0) {
      throw new Error("nKnots must be an odd integer >= 3");
    }
    const t = linspace(0, tMax, nKnots);
    const dt = tMax / (nKnots - 1);
    const raw = t.map((_, index) => (index === 0 || index === nKnots - 1 ? dt : 2 * dt));
    const phi0 = t.map((value) => Math.exp(-0.5 * value * value));
    const weights = raw.map((value, index) => value * phi0[index]);
    return { t, dt, raw, phi0, weights, tMax, nKnots };
  }

  function normalize(vector) {
    let norm = 0;
    for (const value of vector) norm += value * value;
    norm = Math.sqrt(norm);
    if (norm === 0) return vector.map(() => 0);
    return vector.map((value) => value / norm);
  }

  function sampleDirections(dim, count, rng) {
    const random = rng || Math.random;
    const directions = [];
    for (let s = 0; s < count; s += 1) {
      const vector = Array.from({ length: dim }, () => randn(random));
      directions.push(normalize(vector));
    }
    return directions;
  }

  function dot(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i += 1) sum += a[i] * b[i];
    return sum;
  }

  function project(points, direction) {
    return points.map((point) => dot(point, direction));
  }

  /* Compute SIGReg for one batch of points.
   *
   * points:     [[x0, x1, ...], ...] (N x K)
   * directions: [[a0, a1, ...], ...] (S x K), unit norm
   * options:    { tMax, nKnots }
   *
   * Returns:
   *   value      mean T_s over slices
   *   perSlice   T_s for every direction
   *   slices     per-slice projections, C(t), S(t), err(t), integrand, T_s
   *   gradient   dL/dz for every point, shape N x K
   *   quadrature t, phi0, raw weights and effective weights
   */
  function sigmaReg(points, directions, options) {
    const opts = Object.assign({ tMax: 3, nKnots: 17 }, options || {});
    if (!points || points.length === 0) throw new Error("points must not be empty");
    if (!directions || directions.length === 0) throw new Error("directions must not be empty");
    const N = points.length;
    const K = points[0].length;
    const S = directions.length;
    for (const point of points) {
      if (point.length !== K) throw new Error("inconsistent point dimension");
    }
    for (const direction of directions) {
      if (direction.length !== K) throw new Error("direction dimension does not match points");
    }

    const quad = quadrature(opts.tMax, opts.nKnots);
    const slices = [];
    const gradient = points.map(() => new Array(K).fill(0));
    let sumT = 0;

    for (let s = 0; s < S; s += 1) {
      const direction = directions[s];
      const projections = project(points, direction);
      const C = new Array(quad.nKnots);
      const Sn = new Array(quad.nKnots);
      for (let ti = 0; ti < quad.nKnots; ti += 1) {
        const t = quad.t[ti];
        let cosine = 0;
        let sine = 0;
        for (let j = 0; j < N; j += 1) {
          const angle = t * projections[j];
          cosine += Math.cos(angle);
          sine += Math.sin(angle);
        }
        C[ti] = cosine / N;
        Sn[ti] = sine / N;
      }

      const err = new Array(quad.nKnots);
      const integrand = new Array(quad.nKnots);
      let integral = 0;
      for (let ti = 0; ti < quad.nKnots; ti += 1) {
        const dc = C[ti] - quad.phi0[ti];
        err[ti] = dc * dc + Sn[ti] * Sn[ti];
        integrand[ti] = err[ti] * quad.phi0[ti];
        integral += err[ti] * quad.weights[ti];
      }
      const T = N * integral;
      sumT += T;

      // dT/dx_j = 2 * integral w(t) t [-(C-phi0) sin(t x_j) + S cos(t x_j)] dt
      // dL/dz_j = (1/S) sum_s dT/dx_j * a_s
      for (let j = 0; j < N; j += 1) {
        let g = 0;
        for (let ti = 0; ti < quad.nKnots; ti += 1) {
          const t = quad.t[ti];
          const angle = t * projections[j];
          g +=
            quad.weights[ti] *
            t *
            (-(C[ti] - quad.phi0[ti]) * Math.sin(angle) + Sn[ti] * Math.cos(angle));
        }
        g *= (2 / S);
        for (let k = 0; k < K; k += 1) gradient[j][k] += g * direction[k];
      }

      slices.push({
        index: s,
        direction,
        projections,
        C,
        S: Sn,
        err,
        integrand,
        integral,
        T,
      });
    }

    return {
      value: sumT / S,
      perSlice: slices.map((slice) => slice.T),
      slices,
      gradient,
      quadrature: quad,
      points,
      directions,
      N,
      K,
      S,
    };
  }

  function sigmaRegValue(points, directions, options) {
    return sigmaReg(points, directions, options).value;
  }

  /* Expected value of one slice's statistic for iid standard-normal data.
   *
   * E[err(t)] = (1 - phi0(t)^2) / N, so
   * E[T] = sum_t (1 - phi0(t)^2) * weights(t).
   * The finite-sample value fluctuates around this; it does not go to zero.
   */
  function gaussianNullExpected(options) {
    const opts = Object.assign({ tMax: 3, nKnots: 17 }, options || {});
    const quad = quadrature(opts.tMax, opts.nKnots);
    let expected = 0;
    for (let ti = 0; ti < quad.nKnots; ti += 1) {
      expected += (1 - quad.phi0[ti] * quad.phi0[ti]) * quad.weights[ti];
    }
    return expected;
  }

  function quantile(values, q) {
    if (!values || values.length === 0) return NaN;
    const sorted = values.slice().sort((a, b) => a - b);
    const position = (sorted.length - 1) * Math.min(1, Math.max(0, q));
    const lower = Math.floor(position);
    const upper = Math.ceil(position);
    if (lower === upper) return sorted[lower];
    const weight = position - lower;
    return sorted[lower] * (1 - weight) + sorted[upper] * weight;
  }

  function standardNormalPoints(count, dim, rng) {
    return Array.from({ length: count }, () =>
      Array.from({ length: dim }, () => randn(rng))
    );
  }

  function correlatedGaussianPoints(count, options, rng) {
    const opts = Object.assign(
      { stdX: 1, stdY: 1, corr: 0, offsetX: 0, offsetY: 0 },
      options || {}
    );
    const corr = Math.max(-0.999, Math.min(0.999, opts.corr));
    const points = [];
    for (let i = 0; i < count; i += 1) {
      const x = randn(rng);
      const y = randn(rng);
      points.push([
        opts.offsetX + opts.stdX * x,
        opts.offsetY + opts.stdY * (corr * x + Math.sqrt(1 - corr * corr) * y),
      ]);
    }
    return points;
  }

  function meanAndStd(values) {
    if (!values || values.length === 0) return { mean: 0, std: 0 };
    let mean = 0;
    for (const value of values) mean += value;
    mean /= values.length;
    let variance = 0;
    for (const value of values) variance += (value - mean) ** 2;
    variance /= values.length;
    return { mean, std: Math.sqrt(variance) };
  }

  return {
    linspace,
    mulberry32,
    randn,
    quadrature,
    sampleDirections,
    project,
    sigmaReg,
    sigmaRegValue,
    gaussianNullExpected,
    quantile,
    standardNormalPoints,
    correlatedGaussianPoints,
    meanAndStd,
    normalize,
    dot,
  };
});
