/* Projector Diagnostics: marginals, correlation and real-data SIGReg. */
(function () {
  "use strict";

  const DATA = window.AtlasData;
  const Core = window.AtlasCore;
  const Sigreg = window.SigregCore;
  const Loss = window.LossCore;
  const Projector = window.ProjectorCore;
  const Kit = window.CanvasKit;
  const $ = (id) => document.getElementById(id);

  const N = DATA.indices.length;
  const K = DATA.spaces.proj.dim;
  const V = DATA.meta.views;
  const REFERENCE_N = 256;
  const VIEW_COLORS = ["#2458d6", "#0f766e", "#c2610a", "#b42318"];
  const CENTER = Core.decodeMatrix(DATA.spaces.proj);
  const VIEWS = Core.decodeFloat32(DATA.views.data);

  const state = {
    dataset: "center",
    mode: "per_view",
    batchSize: 256,
    batchSeed: 11,
    directionSeed: 1234,
    directions: 64,
    tMax: 3,
    knots: 17,
    selectedPair: null,
  };

  function fmt(value, digits) {
    const places = digits === undefined ? 3 : digits;
    if (!isFinite(value)) return "–";
    const abs = Math.abs(value);
    if (abs >= 10000 || (abs > 0 && abs < 0.0005)) return value.toExponential(2);
    return value.toFixed(places);
  }

  function pad(index) {
    return String(index).padStart(2, "0");
  }

  function mean(values) {
    if (!values.length) return 0;
    let total = 0;
    for (const value of values) total += value;
    return total / values.length;
  }

  function sampleBatch() {
    const rng = Sigreg.mulberry32(state.batchSeed >>> 0);
    return Projector.sampleIndices(N, state.batchSize, rng);
  }

  function distributionPoints(indices) {
    if (state.dataset === "center") {
      const points = new Float32Array(indices.length * K);
      indices.forEach((image, row) => {
        points.set(CENTER.subarray(image * K, (image + 1) * K), row * K);
      });
      return points;
    }
    const points = new Float32Array(indices.length * V * K);
    let cursor = 0;
    for (const image of indices) {
      for (let view = 0; view < V; view += 1) {
        const base = (image * V + view) * K;
        points.set(VIEWS.subarray(base, base + K), cursor);
        cursor += K;
      }
    }
    return points;
  }

  function centerPoints(indices) {
    return indices.map((image) => Array.from(CENTER.subarray(image * K, (image + 1) * K)));
  }

  function viewsNested(indices) {
    const z = new Array(V);
    for (let view = 0; view < V; view += 1) {
      const rows = new Array(indices.length);
      for (let row = 0; row < indices.length; row += 1) {
        const base = (indices[row] * V + view) * K;
        rows[row] = Array.from(VIEWS.subarray(base, base + K));
      }
      z[view] = rows;
    }
    return z;
  }

  function currentDirections() {
    return Sigreg.sampleDirections(
      K,
      state.directions,
      Sigreg.mulberry32(state.directionSeed >>> 0)
    );
  }

  function compute() {
    const indices = sampleBatch();
    const points = distributionPoints(indices);
    const pointCount = points.length / K;
    const stats = Projector.columnStats(points, pointCount, K);
    const matrix = Projector.correlationMatrix(points, pointCount, K, stats);
    const corrSummary = Projector.correlationSummary(matrix);
    const directions = currentDirections();
    const opts = { tMax: state.tMax, nKnots: state.knots };

    let sig;
    if (state.dataset === "center") {
      const result = Sigreg.sigmaReg(centerPoints(indices), directions, opts);
      sig = {
        mode: "center",
        lSig: result.value,
        results: [result],
        perView: [result.value],
      };
    } else {
      const result = Loss.sigregTerms(viewsNested(indices), {
        directions,
        sigregMode: state.mode,
        tMax: state.tMax,
        nKnots: state.knots,
      });
      sig = {
        mode: result.mode,
        lSig: result.lSig,
        results: result.results,
        perView: result.perView,
      };
    }

    const nullExpected = Sigreg.gaussianNullExpected(opts);
    /* T_s = null + N * signal, so scale only the part above the null floor.  Scaling
     * the whole value would understate a small batch relative to a large one. */
    const scaled = mean(
      sig.results.map((result) =>
        Projector.rescaleTs(result.value, result.N, REFERENCE_N, nullExpected)
      )
    );
    return {
      indices,
      points,
      pointCount,
      stats,
      matrix,
      corrSummary,
      directions,
      sig,
      scaled,
      nullExpected,
    };
  }

  function updateStats(comp) {
    const sig = comp.sig;
    $("statSig").textContent = fmt(comp.scaled, 3);
    $("statSigSub").textContent = `null + (T_s − null) × 256/N`;
    $("statRaw").textContent = fmt(sig.lSig, 3);
    $("statRawSub").textContent =
      sig.mode === "per_view"
        ? `mean over ${V} views · N = ${comp.indices.length} each`
        : sig.mode === "joint"
          ? `joint · N = ${comp.pointCount}`
          : `center crop · N = ${comp.pointCount}`;
    $("statNull").textContent = fmt(comp.nullExpected, 3);
    $("statNullSub").textContent = `t_max ${state.tMax} · ${state.knots} knots`;
    $("statCorr").textContent = fmt(comp.corrSummary.maxAbs, 3);
    $("statCorrSub").textContent = comp.corrSummary.maxPair
      ? `d${pad(comp.corrSummary.maxPair[0])} ↔ d${pad(comp.corrSummary.maxPair[1])}`
      : "off-diagonal";
    $("statBatch").textContent = comp.pointCount.toLocaleString();
    $("statBatchSub").textContent =
      state.dataset === "center"
        ? `${comp.indices.length} center-crop vectors`
        : `${comp.indices.length} images × ${V} views`;
  }

  function rescale(value, pointCount, nullExpected) {
    return Projector.rescaleTs(value, pointCount, REFERENCE_N, nullExpected);
  }

  function chartSeries(comp) {
    if (comp.sig.mode === "per_view") {
      return comp.sig.results.map((result, view) => ({
        label: `view ${view}`,
        color: VIEW_COLORS[view % VIEW_COLORS.length],
        values: result.perSlice.map((value) =>
          rescale(value, result.N, comp.nullExpected)
        ),
      }));
    }
    const result = comp.sig.results[0];
    return [
      {
        label: comp.sig.mode === "joint" ? "joint" : "center crop",
        color: VIEW_COLORS[0],
        values: result.perSlice.map((value) =>
          rescale(value, result.N, comp.nullExpected)
        ),
      },
    ];
  }

  function drawSliceChart(comp) {
    const series = chartSeries(comp);
    const length = series[0].values.length;
    const average = Array.from({ length }, (_, index) =>
      mean(series.map((entry) => entry.values[index]))
    );
    let ymax = Math.max(comp.nullExpected * 1.3, comp.scaled * 1.25);
    for (const entry of series) {
      for (const value of entry.values) ymax = Math.max(ymax, value * 1.12);
    }
    if (!isFinite(ymax) || ymax <= 0) ymax = 1;

    const plot = new Kit.Plot($("sliceCanvas"), {
      xmin: 0,
      xmax: Math.max(1, length - 1),
      ymin: 0,
      ymax,
      xlabel: "direction index",
      ylabel: "T_s (N = 256 equivalent)",
      title: `mean ${fmt(comp.scaled, 3)}   ·   Gaussian null ${fmt(comp.nullExpected, 3)}`,
      grid: true,
      padding: { l: 66, r: 18, t: 28, b: 38 },
    }).draw();

    plot.clip(() => {
      plot.hline(comp.nullExpected, {
        stroke: Kit.COLORS.orange,
        width: 1.4,
        dash: [5, 4],
      });
      plot.hline(comp.scaled, { stroke: Kit.COLORS.green, width: 1.5 });
      series.forEach((entry) => {
        plot.line(
          entry.values.map((value, index) => [index, value]),
          {
            stroke: entry.color,
            width: 1.0,
            alpha: series.length > 1 ? 0.55 : 0.9,
          }
        );
      });
      if (series.length > 1) {
        plot.line(
          average.map((value, index) => [index, value]),
          { stroke: Kit.COLORS.ink, width: 1.8, alpha: 0.9 }
        );
      }
      const dots = series.length > 1 ? average : series[0].values;
      plot.scatter(
        dots.map((value, index) => ({ 0: index, 1: value, index })),
        { r: length > 70 ? 1.4 : 2.1, fill: Kit.COLORS.ink, alpha: 0.7 }
      );
    });

    const entries = series.map(
      (entry) =>
        `<span><span class="swatch line" style="background:${entry.color}"></span>${entry.label}</span>`
    );
    if (series.length > 1) {
      entries.push(
        '<span><span class="swatch line" style="background:#18202a"></span>view mean</span>'
      );
    }
    entries.push(
      '<span><span class="swatch line" style="background:#16794a"></span>T_s mean</span>',
      '<span><span class="swatch line" style="background:#c2610a"></span>Gaussian null</span>'
    );
    $("sliceLegend").innerHTML = entries.join("");
    $("sigNote").innerHTML =
      (comp.sig.mode === "per_view"
        ? `Per-view mode scores each of the ${V} views separately, then averages the ${V} statistics. `
        : comp.sig.mode === "joint"
          ? `Joint mode pools all ${comp.pointCount} view embeddings into one statistic. `
          : `Center-crop mode has one embedding per image, so per-view and joint are identical. `) +
      `Raw <span class="mono">T_s</span> is ${fmt(comp.sig.lSig, 3)}; the chart maps each slice to the same ` +
      `number of points as a <span class="mono">${REFERENCE_N}</span>-point batch, keeping the ` +
      `Gaussian-null floor fixed. The model is <strong>${fmt(
        comp.scaled / comp.nullExpected,
        1
      )}×</strong> the Gaussian null at the same point count.`;
  }

  function columnValues(points, count, dim, column) {
    const values = new Float32Array(count);
    for (let row = 0; row < count; row += 1) values[row] = points[row * dim + column];
    return values;
  }

  function drawHistograms(comp) {
    const grid = $("histGrid");
    grid.innerHTML = "";
    const bins = 28;
    const xmin = -4;
    const xmax = 4;
    const nodes = [];

    for (let dimension = 0; dimension < K; dimension += 1) {
      const figure = document.createElement("figure");
      figure.className = "hist-cell";
      figure.dataset.dim = String(dimension);
      if (state.selectedPair && state.selectedPair.includes(dimension)) {
        figure.classList.add("selected");
      }
      const caption = document.createElement("figcaption");
      caption.innerHTML =
        `<span>d${pad(dimension)}</span>` +
        `<span class="hist-sigma">σ ${fmt(comp.stats.std[dimension], 2)}</span>`;
      const canvas = document.createElement("canvas");
      canvas.title =
        `dimension ${dimension} · mean ${fmt(comp.stats.mean[dimension], 3)} · ` +
        `std ${fmt(comp.stats.std[dimension], 3)} · skew ${fmt(
          comp.stats.skew[dimension],
          3
        )} · excess kurtosis ${fmt(comp.stats.kurtosis[dimension], 3)} · KS ${fmt(
          comp.stats.ks[dimension],
          3
        )}`;
      figure.appendChild(caption);
      figure.appendChild(canvas);
      grid.appendChild(figure);
      nodes.push({
        dimension,
        canvas,
        values: columnValues(comp.points, comp.pointCount, K, dimension),
      });
    }

    for (const node of nodes) {
      const hist = Kit.histogram(node.values, bins, xmin, xmax);
      let ymax = 0.55;
      for (const density of hist.density) ymax = Math.max(ymax, density);
      ymax *= 1.16;
      const plot = new Kit.Plot(node.canvas, {
        xmin,
        xmax,
        ymin: 0,
        ymax,
        xTicks: [-4, -2, 0, 2, 4],
        yTicks: [],
        grid: false,
        zeroLines: false,
        padding: { l: 22, r: 6, t: 6, b: 16 },
      }).draw();
      plot.clip(() => {
        const steps = Kit.stepPoints(hist);
        plot.area(steps, 0, { fill: "rgba(36, 88, 214, 0.08)" });
        plot.line(steps, { stroke: Kit.COLORS.accent, width: 1.15 });
        const curve = [];
        for (let index = 0; index <= 80; index += 1) {
          const x = xmin + ((xmax - xmin) * index) / 80;
          curve.push([x, Kit.normalPdf(x)]);
        }
        plot.line(curve, { stroke: Kit.COLORS.ink, width: 1, dash: [4, 3], alpha: 0.8 });
      });
    }

    const lowStd = [];
    for (let dimension = 0; dimension < K; dimension += 1) {
      const std = comp.stats.std[dimension];
      if (std < 0.8 || std > 1.2) lowStd.push(`d${pad(dimension)} (${fmt(std, 2)})`);
    }
    $("histNote").textContent =
      `${comp.pointCount.toLocaleString()} points · mean KS ${fmt(mean(Array.from(comp.stats.ks)), 3)} · ` +
      (lowStd.length
        ? `std outside 0.8–1.2: ${lowStd.join(", ")}.`
        : "every std is inside 0.8–1.2.");
  }

  function renderCorrelation(comp) {
    const container = $("corrMatrix");
    container.innerHTML = "";
    const corner = document.createElement("div");
    corner.className = "corr-corner";
    corner.textContent = "d";
    container.appendChild(corner);
    for (let column = 0; column < K; column += 1) {
      const head = document.createElement("div");
      head.className = "corr-head";
      head.textContent = String(column);
      container.appendChild(head);
    }

    for (let row = 0; row < K; row += 1) {
      const head = document.createElement("div");
      head.className = "corr-row-head";
      head.textContent = String(row);
      container.appendChild(head);
      for (let column = 0; column < K; column += 1) {
        const value = comp.matrix[row][column];
        const cell = document.createElement("div");
        cell.className = "corr-cell";
        const pinned =
          state.selectedPair &&
          ((row === state.selectedPair[0] && column === state.selectedPair[1]) ||
            (row === state.selectedPair[1] && column === state.selectedPair[0]));
        if (pinned) cell.classList.add("selected");
        const strength = Math.min(1, Math.abs(value));
        const rgb = value >= 0 ? "36, 88, 214" : "180, 35, 24";
        cell.style.background = `rgba(${rgb}, ${0.06 + 0.68 * strength})`;
        if (strength > 0.62) cell.style.color = "#ffffff";
        cell.title = `r(d${pad(row)}, d${pad(column)}) = ${fmt(value, 3)}`;
        cell.setAttribute(
          "aria-label",
          `correlation between dimension ${row} and ${column} is ${fmt(value, 3)}`
        );
        cell.addEventListener("click", () => {
          if (
            state.selectedPair &&
            state.selectedPair[0] === row &&
            state.selectedPair[1] === column
          ) {
            state.selectedPair = null;
          } else {
            state.selectedPair = [row, column];
          }
          renderCorrelation(comp);
          highlightPair();
          renderTable(comp);
        });
        container.appendChild(cell);
      }
    }
    updateCorrNote(comp);
  }

  function updateCorrNote(comp) {
    const pair =
      state.selectedPair ||
      comp.corrSummary.maxPair;
    if (!pair) {
      $("corrNote").textContent = "No off-diagonal entries.";
      return;
    }
    const value = comp.matrix[pair[0]][pair[1]];
    const pinned = Boolean(state.selectedPair);
    const strongest =
      comp.corrSummary.maxPair &&
      ` Strongest off-diagonal: <span class="mono">d${pad(
        comp.corrSummary.maxPair[0]
      )} ↔ d${pad(comp.corrSummary.maxPair[1])}</span> at ${fmt(
        comp.corrSummary.maxValue,
        3
      )}.`;
    $("corrNote").innerHTML =
      `${pinned ? "Pinned" : "Largest"} pair <span class="mono">d${pad(pair[0])} ↔ d${pad(
        pair[1]
      )}</span>: <span class="mono">r = ${fmt(value, 3)}</span>. ` +
      `Mean |r| = <span class="mono">${fmt(comp.corrSummary.meanAbs, 3)}</span>.` +
      (pinned ? strongest || "" : "");
  }

  function highlightPair() {
    const pair = state.selectedPair;
    for (const cell of document.querySelectorAll(".hist-cell")) {
      const dimension = Number(cell.dataset.dim);
      cell.classList.toggle("selected", Boolean(pair && pair.includes(dimension)));
    }
  }

  function renderTable(comp) {
    const rows = [];
    for (let dimension = 0; dimension < K; dimension += 1) {
      const selected = state.selectedPair && state.selectedPair.includes(dimension);
      rows.push(
        `<tr${selected ? ' class="selected"' : ""}>` +
          `<td class="mono">d${pad(dimension)}</td>` +
          `<td>${fmt(comp.stats.mean[dimension], 3)}</td>` +
          `<td>${fmt(comp.stats.std[dimension], 3)}</td>` +
          `<td>${fmt(comp.stats.skew[dimension], 3)}</td>` +
          `<td>${fmt(comp.stats.kurtosis[dimension], 3)}</td>` +
          `<td>${fmt(comp.stats.ks[dimension], 3)}</td>` +
          `<td class="mono">${fmt(comp.stats.min[dimension], 2)} … ${fmt(
            comp.stats.max[dimension],
            2
          )}</td>` +
          "</tr>"
      );
    }
    $("dimBody").innerHTML = rows.join("");
  }

  function updateControls() {
    for (const button of $("dataset").querySelectorAll("button")) {
      button.setAttribute("aria-pressed", String(button.dataset.dataset === state.dataset));
    }
    for (const button of $("sigMode").querySelectorAll("button")) {
      button.setAttribute("aria-pressed", String(button.dataset.mode === state.mode));
    }
    $("modeControl").hidden = state.dataset !== "views";
    $("batchSize").value = String(state.batchSize);
    $("dirs").value = String(state.directions);
    $("tMax").value = String(state.tMax);
    $("knots").value = String(state.knots);
    $("batchNote").textContent =
      state.dataset === "center"
        ? `${state.batchSize === N ? "All 2,000" : state.batchSize} center-crop embeddings, sampled with seed ${state.batchSeed}.`
        : `${state.batchSize === N ? "All 2,000" : state.batchSize} images × ${V} views = ${
            state.batchSize * V
          } projector vectors, seed ${state.batchSeed}.`;
  }

  function updateUrl() {
    try {
      const url = new URL(window.location.href);
      url.searchParams.set("dataset", state.dataset);
      url.searchParams.set("batch", String(state.batchSize));
      url.searchParams.set("mode", state.mode);
      url.searchParams.set("seed", String(state.batchSeed));
      url.searchParams.set("dseed", String(state.directionSeed));
      url.searchParams.set("dirs", String(state.directions));
      url.searchParams.set("tmax", String(state.tMax));
      url.searchParams.set("knots", String(state.knots));
      window.history.replaceState(null, "", url.toString());
    } catch (error) {
      /* file:// in some browsers refuses replaceState; the page still works. */
    }
  }

  function readUrl() {
    const params = new URLSearchParams(window.location.search);
    if (params.get("dataset") === "views") state.dataset = "views";
    if (params.get("mode") === "joint") state.mode = "joint";
    const batch = Number(params.get("batch"));
    if ([128, 256, 512, 1000, 2000].includes(batch)) state.batchSize = batch;
    const seedText = params.get("seed");
    const seed = seedText === null ? NaN : Number(seedText);
    if (Number.isInteger(seed) && seed >= 0) state.batchSeed = seed;
    const dseedText = params.get("dseed");
    const dseed = dseedText === null ? NaN : Number(dseedText);
    if (Number.isInteger(dseed) && dseed >= 0) state.directionSeed = dseed;
    const dirs = Number(params.get("dirs"));
    if ([16, 32, 64, 96].includes(dirs)) state.directions = dirs;
    const tMax = Number(params.get("tmax"));
    if ([3, 4].includes(tMax)) state.tMax = tMax;
    const knots = Number(params.get("knots"));
    if ([17, 33].includes(knots)) state.knots = knots;
  }

  function render() {
    const comp = compute();
    updateControls();
    updateStats(comp);
    drawSliceChart(comp);
    drawHistograms(comp);
    renderCorrelation(comp);
    renderTable(comp);
    updateUrl();
  }

  function bind() {
    for (const button of $("dataset").querySelectorAll("button")) {
      button.addEventListener("click", () => {
        state.dataset = button.dataset.dataset;
        state.selectedPair = null;
        render();
      });
    }
    for (const button of $("sigMode").querySelectorAll("button")) {
      button.addEventListener("click", () => {
        state.mode = button.dataset.mode;
        render();
      });
    }
    $("batchSize").addEventListener("change", (event) => {
      state.batchSize = Number(event.target.value);
      state.selectedPair = null;
      render();
    });
    $("resampleBatch").addEventListener("click", () => {
      state.batchSeed += 1;
      state.selectedPair = null;
      render();
    });
    $("newDirections").addEventListener("click", () => {
      state.directionSeed += 1;
      render();
    });
    $("dirs").addEventListener("change", (event) => {
      state.directions = Number(event.target.value);
      render();
    });
    $("tMax").addEventListener("change", (event) => {
      state.tMax = Number(event.target.value);
      render();
    });
    $("knots").addEventListener("change", (event) => {
      state.knots = Number(event.target.value);
      render();
    });
  }

  function init() {
    $("footerCommit").textContent = DATA.meta.git_commit || "unknown";
    const sha = DATA.meta.checkpoint_sha256 || "";
    $("footerSha").textContent = sha ? `${sha.slice(0, 12)}…` : "unknown";
    $("footerSha").title = sha;
    readUrl();
    bind();
    render();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
