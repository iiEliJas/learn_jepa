/* Pure helpers for the CIFAR atlas and per-image card.
 *
 * The data comes from cifar_atlas/generate.py as base64-encoded matrices.  This
 * file is UMD-wrapped so the Node tests can require() it and the browser pages
 * can load it as a classic script.
 */
(function (root, factory) {
  if (typeof module === "object" && module.exports) {
    module.exports = factory();
  } else {
    root.AtlasCore = factory();
  }
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function decodeBase64(text) {
    if (typeof Buffer !== "undefined" && typeof Buffer.from === "function") {
      const buffer = Buffer.from(text, "base64");
      return new Uint8Array(buffer.buffer, buffer.byteOffset, buffer.byteLength);
    }
    const binary = atob(text);
    const bytes = new Uint8Array(binary.length);
    for (let index = 0; index < binary.length; index += 1) {
      bytes[index] = binary.charCodeAt(index);
    }
    return bytes;
  }

  /* Decode a space payload from the exporter.
   *
   * int8 spaces carry one scale per feature; float32 spaces are little-endian.
   * The returned array is row-major: rows * dim floats.
   */
  function decodeMatrix(space) {
    const bytes = decodeBase64(space.data);
    const count = space.rows * space.dim;
    const out = new Float32Array(count);
    if (space.encoding === "float32") {
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      for (let index = 0; index < count; index += 1) {
        out[index] = view.getFloat32(index * 4, true);
      }
      return out;
    }
    if (space.encoding === "int8") {
      const quantized = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const scale = space.scale;
      for (let row = 0; row < space.rows; row += 1) {
        for (let dim = 0; dim < space.dim; dim += 1) {
          out[row * space.dim + dim] = quantized[row * space.dim + dim] * scale[dim];
        }
      }
      return out;
    }
    throw new Error(`unknown matrix encoding: ${space.encoding}`);
  }

  function decodeFloat32(text) {
    const bytes = decodeBase64(text);
    const count = Math.floor(bytes.byteLength / 4);
    const out = new Float32Array(count);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (let index = 0; index < count; index += 1) {
      out[index] = view.getFloat32(index * 4, true);
    }
    return out;
  }

  function centerRows(rows, count, dim) {
    const mean = new Float64Array(dim);
    for (let row = 0; row < count; row += 1) {
      const offset = row * dim;
      for (let d = 0; d < dim; d += 1) mean[d] += rows[offset + d];
    }
    for (let d = 0; d < dim; d += 1) mean[d] /= Math.max(1, count);
    const out = new Float32Array(count * dim);
    for (let row = 0; row < count; row += 1) {
      const offset = row * dim;
      for (let d = 0; d < dim; d += 1) out[offset + d] = rows[offset + d] - mean[d];
    }
    return out;
  }

  function rowNorms(rows, count, dim) {
    const norms = new Float32Array(count);
    for (let row = 0; row < count; row += 1) {
      let total = 0;
      const offset = row * dim;
      for (let d = 0; d < dim; d += 1) total += rows[offset + d] * rows[offset + d];
      norms[row] = Math.sqrt(total);
    }
    return norms;
  }

  function meanDirectionRatio(rows, count, dim) {
    const mean = new Float64Array(dim);
    let meanNorm = 0;
    for (let row = 0; row < count; row += 1) {
      const offset = row * dim;
      let rowSquared = 0;
      for (let d = 0; d < dim; d += 1) {
        const value = rows[offset + d];
        mean[d] += value;
        rowSquared += value * value;
      }
      meanNorm += Math.sqrt(rowSquared);
    }
    let norm = 0;
    for (let d = 0; d < dim; d += 1) {
      mean[d] /= Math.max(1, count);
      norm += mean[d] * mean[d];
    }
    norm = Math.sqrt(norm);
    meanNorm /= Math.max(1, count);
    return meanNorm > 0 ? norm / meanNorm : 0;
  }

  function dotRows(rows, a, b, dim) {
    let total = 0;
    const offsetA = a * dim;
    const offsetB = b * dim;
    for (let d = 0; d < dim; d += 1) total += rows[offsetA + d] * rows[offsetB + d];
    return total;
  }

  function cosineRows(rows, a, b, dim, norms) {
    const denominator = (norms[a] || 0) * (norms[b] || 0);
    return denominator > 0 ? dotRows(rows, a, b, dim) / denominator : 0;
  }

  function nearestNeighbors(rows, count, dim, index, k, norms) {
    const rowNormsLocal = norms || rowNorms(rows, count, dim);
    const base = index * dim;
    const results = [];
    for (let row = 0; row < count; row += 1) {
      if (row === index) continue;
      const score = cosineRows(rows, index, row, dim, rowNormsLocal);
      results.push({ index: row, score });
    }
    results.sort((a, b) => b.score - a.score);
    return results.slice(0, Math.max(0, k));
  }

  /* Prediction terms for four views of one image, flat layout [v * K + k]. */
  function predictionFromFlat(flat, V, K) {
    const mu = new Float32Array(K);
    for (let view = 0; view < V; view += 1) {
      const offset = view * K;
      for (let d = 0; d < K; d += 1) mu[d] += flat[offset + d];
    }
    for (let d = 0; d < K; d += 1) mu[d] /= Math.max(1, V);
    const contributions = new Array(V).fill(0);
    let total = 0;
    for (let view = 0; view < V; view += 1) {
      const offset = view * K;
      let sum = 0;
      for (let d = 0; d < K; d += 1) {
        const residual = mu[d] - flat[offset + d];
        sum += residual * residual;
      }
      contributions[view] = sum / Math.max(1, K);
      total += sum;
    }
    return {
      mu: Array.from(mu),
      contributions,
      lPred: total / Math.max(1, V * K),
    };
  }

  function viewFlat(views, N, V, K, imageIndex) {
    const out = new Float32Array(V * K);
    for (let view = 0; view < V; view += 1) {
      const source = (imageIndex * V + view) * K;
      out.set(views.subarray(source, source + K), view * K);
    }
    return out;
  }

  function predictionForImage(views, N, V, K, imageIndex) {
    return predictionFromFlat(viewFlat(views, N, V, K, imageIndex), V, K);
  }

  /* Break the image correspondence: view v of image i is replaced by view v of
   * image i + offsets[v].  With one sample this is the local shuffled control.
   */
  function shuffledPredictionForImage(views, N, V, K, imageIndex, offsets) {
    const picked = new Array(V);
    const flat = new Float32Array(V * K);
    for (let view = 0; view < V; view += 1) {
      const offset = offsets && offsets[view] !== undefined ? offsets[view] : 0;
      const sourceImage = ((imageIndex + offset) % N + N) % N;
      picked[view] = sourceImage;
      const source = (sourceImage * V + view) * K;
      flat.set(views.subarray(source, source + K), view * K);
    }
    return Object.assign(predictionFromFlat(flat, V, K), { picked });
  }

  function pairwiseCosine(flat, V, K) {
    const norms = new Array(V).fill(0);
    for (let view = 0; view < V; view += 1) {
      let total = 0;
      const offset = view * K;
      for (let d = 0; d < K; d += 1) total += flat[offset + d] * flat[offset + d];
      norms[view] = Math.sqrt(total);
    }
    const matrix = Array.from({ length: V }, () => new Array(V).fill(0));
    for (let a = 0; a < V; a += 1) {
      for (let b = 0; b < V; b += 1) {
        let total = 0;
        const offsetA = a * K;
        const offsetB = b * K;
        for (let d = 0; d < K; d += 1) total += flat[offsetA + d] * flat[offsetB + d];
        const denominator = norms[a] * norms[b];
        matrix[a][b] = denominator > 0 ? total / denominator : 0;
      }
    }
    return matrix;
  }

  function confusionMatrix(labels, preds, numClasses) {
    const classes = numClasses || 10;
    const matrix = Array.from({ length: classes }, () => new Array(classes).fill(0));
    const count = Math.min(labels.length, preds.length);
    for (let index = 0; index < count; index += 1) {
      const actual = labels[index];
      const predicted = preds[index];
      if (actual >= 0 && actual < classes && predicted >= 0 && predicted < classes) {
        matrix[actual][predicted] += 1;
      }
    }
    return matrix;
  }

  function classPurity(indices, labels, targetClass) {
    if (!indices || indices.length === 0) return 0;
    const wanted = targetClass === undefined || targetClass === null ? labels[indices[0]] : targetClass;
    let matches = 0;
    for (const index of indices) if (labels[index] === wanted) matches += 1;
    return matches / indices.length;
  }

  function meanNeighborScore(neighbors) {
    if (!neighbors || neighbors.length === 0) return 0;
    let total = 0;
    for (const neighbor of neighbors) total += neighbor.score;
    return total / neighbors.length;
  }

  /* Jaccard similarity for two neighbour-id lists. */
  function jaccard(a, b) {
    const setA = new Set(a);
    const setB = new Set(b);
    if (setA.size === 0 && setB.size === 0) return 1;
    let shared = 0;
    for (const value of setA) if (setB.has(value)) shared += 1;
    const union = setA.size + setB.size - shared;
    return union > 0 ? shared / union : 1;
  }

  /* Majority label among neighbours; ties go to the lower class index. */
  function neighborVote(neighbors, labels) {
    const counts = new Map();
    for (const neighbor of neighbors) {
      const label = labels[neighbor.index];
      counts.set(label, (counts.get(label) || 0) + 1);
    }
    let label = null;
    let count = 0;
    for (const [candidate, tally] of counts) {
      if (tally > count || (tally === count && candidate < label)) {
        label = candidate;
        count = tally;
      }
    }
    const total = neighbors ? neighbors.length : 0;
    return { label, count, fraction: total > 0 ? count / total : 0 };
  }

  function accuracy(labels, preds) {
    const count = Math.min(labels.length, preds.length);
    if (count === 0) return 0;
    let correct = 0;
    for (let index = 0; index < count; index += 1) {
      if (labels[index] === preds[index]) correct += 1;
    }
    return correct / count;
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

  return {
    decodeBase64,
    decodeMatrix,
    decodeFloat32,
    centerRows,
    rowNorms,
    meanDirectionRatio,
    dotRows,
    cosineRows,
    nearestNeighbors,
    predictionFromFlat,
    predictionForImage,
    shuffledPredictionForImage,
    viewFlat,
    pairwiseCosine,
    confusionMatrix,
    classPurity,
    meanNeighborScore,
    jaccard,
    neighborVote,
    accuracy,
    quantile,
  };
});
