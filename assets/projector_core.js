/* Distribution and correlation helpers for the Projector Diagnostics page.
 *
 * Points are row-major: count rows of `dim` values.  The module is UMD-wrapped
 * so the browser pages can load it as a classic script and the Node tests can
 * require() it.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.ProjectorCore = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  /* Abramowitz-Stegun 26.2.17, |error| < 7.5e-8. */
  function normalCdf(x) {
    const z = Math.abs(x);
    const t = 1 / (1 + 0.2316419 * z);
    const density = 0.3989422804014327 * Math.exp(-0.5 * z * z);
    const tail =
      density *
      t *
      (0.319381530 +
        t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    return x >= 0 ? 1 - tail : tail;
  }

  function sortedColumn(points, count, dim, column) {
    const values = new Array(count);
    for (let row = 0; row < count; row += 1) values[row] = points[row * dim + column];
    values.sort((a, b) => a - b);
    return values;
  }

  function ksDistance(sorted, cdf) {
    const count = sorted.length;
    let distance = 0;
    for (let index = 0; index < count; index += 1) {
      const empirical = (index + 1) / count;
      const lower = index / count;
      const theoretical = cdf(sorted[index]);
      distance = Math.max(distance, empirical - theoretical, theoretical - lower);
    }
    return distance;
  }

  /* Per-dimension moments and a Kolmogorov-Smirnov distance to N(0,1). */
  function columnStats(points, count, dim) {
    const mean = new Float64Array(dim);
    const std = new Float64Array(dim);
    const skew = new Float64Array(dim);
    const kurtosis = new Float64Array(dim);
    const ks = new Float64Array(dim);
    const min = new Float64Array(dim);
    const max = new Float64Array(dim);

    for (let column = 0; column < dim; column += 1) {
      let sum = 0;
      let low = Infinity;
      let high = -Infinity;
      for (let row = 0; row < count; row += 1) {
        const value = points[row * dim + column];
        sum += value;
        if (value < low) low = value;
        if (value > high) high = value;
      }
      const mu = count > 0 ? sum / count : 0;
      let m2 = 0;
      let m3 = 0;
      let m4 = 0;
      for (let row = 0; row < count; row += 1) {
        const delta = points[row * dim + column] - mu;
        const squared = delta * delta;
        m2 += squared;
        m3 += squared * delta;
        m4 += squared * squared;
      }
      m2 /= Math.max(1, count);
      m3 /= Math.max(1, count);
      m4 /= Math.max(1, count);
      const sigma = Math.sqrt(m2);
      mean[column] = mu;
      std[column] = sigma;
      skew[column] = sigma > 0 ? m3 / (sigma * sigma * sigma) : 0;
      kurtosis[column] = sigma > 0 ? m4 / (sigma * sigma * sigma * sigma) - 3 : 0;
      min[column] = low;
      max[column] = high;
      ks[column] = count > 0 ? ksDistance(sortedColumn(points, count, dim, column), normalCdf) : 0;
    }

    return { count, dim, mean, std, skew, kurtosis, ks, min, max };
  }

  /* Pearson correlation; the diagonal is 1 when the column has variance. */
  function correlationMatrix(points, count, dim, stats) {
    const summary = stats || columnStats(points, count, dim);
    const matrix = Array.from({ length: dim }, () => new Float64Array(dim));
    for (let a = 0; a < dim; a += 1) {
      for (let b = a; b < dim; b += 1) {
        let covariance = 0;
        for (let row = 0; row < count; row += 1) {
          covariance +=
            (points[row * dim + a] - summary.mean[a]) *
            (points[row * dim + b] - summary.mean[b]);
        }
        covariance /= Math.max(1, count);
        const denominator = summary.std[a] * summary.std[b];
        const value =
          a === b ? (summary.std[a] > 0 ? 1 : 0) : denominator > 0 ? covariance / denominator : 0;
        matrix[a][b] = value;
        matrix[b][a] = value;
      }
    }
    return matrix;
  }

  function correlationSummary(matrix) {
    let maxAbs = 0;
    let maxValue = 0;
    let maxPair = null;
    let total = 0;
    let pairs = 0;
    for (let a = 0; a < matrix.length; a += 1) {
      for (let b = a + 1; b < matrix.length; b += 1) {
        const value = matrix[a][b];
        const magnitude = Math.abs(value);
        total += magnitude;
        pairs += 1;
        if (magnitude > maxAbs) {
          maxAbs = magnitude;
          maxValue = value;
          maxPair = [a, b];
        }
      }
    }
    return { maxAbs, maxValue, maxPair, meanAbs: pairs > 0 ? total / pairs : 0 };
  }

  /* Map T_s from one point count to another.
   *
   * T_s = null + N * signal: the finite-sample Gaussian floor does not scale with
   * N, so only the part above the null is rescaled.
   */
  function rescaleTs(value, pointCount, referenceCount, nullExpected) {
    if (pointCount <= 0) return nullExpected;
    return nullExpected + (value - nullExpected) * (referenceCount / pointCount);
  }

  /* Partial Fisher-Yates sample without replacement, returned in ascending order. */
  function sampleIndices(total, count, rng) {
    const wanted = Math.max(0, Math.min(total, Math.floor(count)));
    if (wanted >= total) return Array.from({ length: total }, (_, index) => index);
    const pool = Array.from({ length: total }, (_, index) => index);
    for (let index = 0; index < wanted; index += 1) {
      const pick = index + Math.floor(rng() * (total - index));
      const swap = pool[index];
      pool[index] = pool[pick];
      pool[pick] = swap;
    }
    return pool.slice(0, wanted).sort((a, b) => a - b);
  }

  return {
    normalCdf,
    ksDistance,
    columnStats,
    correlationMatrix,
    correlationSummary,
    rescaleTs,
    sampleIndices,
  };
});
