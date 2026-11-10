/* SIGReg Lab: DOM, controls and canvas rendering. */
(function () {
  "use strict";

  const Core = window.SigregCore;
  const Kit = window.CanvasKit;
  const $ = (id) => document.getElementById(id);

  const DEFAULTS = {
    preset: "gaussian",
    n: 256,
    stdX: 1,
    stdY: 1,
    corr: 0,
    offsetX: 0,
    offsetY: 0,
    seed: 7,
    dirSeed: 11,
    slices: 64,
    knots: 17,
    tMax: 3,
    selected: 0,
    showGrad: true,
    showDirections: true,
  };

  const PRESETS = {
    gaussian: { stdX: 1, stdY: 1, corr: 0, offsetX: 0, offsetY: 0 },
    collapsed: { stdX: 0.03, stdY: 0.03, corr: 0, offsetX: 0, offsetY: 0 },
    anisotropic: { stdX: 2.5, stdY: 0.25, corr: 0, offsetX: 0, offsetY: 0 },
    shifted: { stdX: 1, stdY: 1, corr: 0, offsetX: 2, offsetY: 0 },
    two_clusters: { stdX: 0.7, stdY: 0.7, corr: 0, offsetX: 0, offsetY: 0 },
    heavy_tails: { stdX: 1, stdY: 1, corr: 0, offsetX: 0, offsetY: 0 },
    uniform: { stdX: 2, stdY: 2, corr: 0, offsetX: 0, offsetY: 0 },
    custom: {},
  };

  const state = Object.assign({}, DEFAULTS);
  let points = [];
  let directions = [];
  let result = null;
  let nullExpected = 0;
  let selectedSlice = 0;
  let framePending = false;

  function fmt(value, digits) {
    const d = digits === undefined ? 3 : digits;
    if (!isFinite(value)) return "–";
    const abs = Math.abs(value);
    if (abs >= 10000 || (abs > 0 && abs < 0.0005)) return value.toExponential(2);
    return value.toFixed(d);
  }

  function clamp(value, low, high) {
    return Math.max(low, Math.min(high, value));
  }

  function hashSeed(seed) {
    return (Math.imul(seed >>> 0, 2654435761) ^ 0x9e3779b9) >>> 0;
  }

  function readControls() {
    state.preset = $("preset").value;
    state.n = Number($("nPoints").value);
    state.stdX = Number($("stdX").value);
    state.stdY = Number($("stdY").value);
    state.corr = Number($("corr").value);
    state.offsetX = Number($("offsetX").value);
    state.offsetY = Number($("offsetY").value);
    state.seed = Math.max(0, Math.floor(Number($("seed").value) || 0));
    state.slices = Number($("slices").value);
    state.knots = Number($("knots").value);
    state.tMax = Number($("tMax").value);
    state.selected = Number($("selectedSlice").value);
    state.showGrad = $("showGrad").checked;
    state.showDirections = $("showDirections").checked;
  }

  function syncControls() {
    $("preset").value = state.preset;
    $("nPoints").value = state.n;
    $("stdX").value = state.stdX;
    $("stdY").value = state.stdY;
    $("corr").value = state.corr;
    $("offsetX").value = state.offsetX;
    $("offsetY").value = state.offsetY;
    $("seed").value = state.seed;
    $("slices").value = state.slices;
    $("knots").value = String(state.knots);
    $("tMax").value = state.tMax;
    $("showGrad").checked = state.showGrad;
    $("showDirections").checked = state.showDirections;
    $("selectedSlice").max = String(Math.max(0, state.slices - 1));
    $("selectedSlice").value = String(state.selected);
    syncOutputs();
  }

  function syncOutputs() {
    $("nPointsOut").textContent = String(state.n);
    $("stdXOut").textContent = Number(state.stdX).toFixed(2);
    $("stdYOut").textContent = Number(state.stdY).toFixed(2);
    $("corrOut").textContent = Number(state.corr).toFixed(2);
    $("offsetXOut").textContent = Number(state.offsetX).toFixed(2);
    $("offsetYOut").textContent = Number(state.offsetY).toFixed(2);
    $("slicesOut").textContent = String(state.slices);
    $("tMaxOut").textContent = Number(state.tMax).toFixed(1);
    $("selectedSliceOut").textContent = String(selectedSlice);
  }

  function applyPreset(name) {
    const preset = PRESETS[name] || {};
    Object.assign(state, preset);
    state.preset = name;
    if (name === "collapsed") {
      state.stdX = 0.03;
      state.stdY = 0.03;
    }
    state.selected = 0;
    selectedSlice = 0;
    syncControls();
  }

  function generatePoints() {
    const rng = Core.mulberry32(hashSeed(state.seed + 1));
    const { n, stdX, stdY, corr, offsetX, offsetY, preset } = state;
    const c = clamp(corr, -0.999, 0.999);
    const pointsOut = [];

    if (preset === "two_clusters") {
      for (let i = 0; i < n; i += 1) {
        const center = i % 2 === 0 ? -2.2 : 2.2;
        const x = Core.randn(rng);
        const y = Core.randn(rng);
        pointsOut.push([
          offsetX + center + stdX * x,
          offsetY + stdY * (c * x + Math.sqrt(1 - c * c) * y),
        ]);
      }
      return pointsOut;
    }

    if (preset === "heavy_tails") {
      for (let i = 0; i < n; i += 1) {
        const scale = rng() < 0.16 ? 2.8 : 1;
        const x = Core.randn(rng);
        const y = Core.randn(rng);
        pointsOut.push([
          offsetX + scale * stdX * x,
          offsetY + scale * stdY * (c * x + Math.sqrt(1 - c * c) * y),
        ]);
      }
      return pointsOut;
    }

    if (preset === "uniform") {
      const halfX = Math.sqrt(3) * stdX;
      const halfY = Math.sqrt(3) * stdY;
      for (let i = 0; i < n; i += 1) {
        pointsOut.push([
          offsetX + (rng() * 2 - 1) * halfX,
          offsetY + (rng() * 2 - 1) * halfY,
        ]);
      }
      return pointsOut;
    }

    return Core.correlatedGaussianPoints(
      n,
      { stdX, stdY, corr: c, offsetX, offsetY },
      rng
    );
  }

  function generateDirections() {
    return Core.sampleDirections(2, state.slices, Core.mulberry32(hashSeed(state.dirSeed + 17)));
  }

  function recompute() {
    points = generatePoints();
    directions = generateDirections();
    const sig = Core.sigmaReg(points, directions, {
      tMax: state.tMax,
      nKnots: state.knots,
    });
    result = sig;
    nullExpected = Core.gaussianNullExpected({ tMax: state.tMax, nKnots: state.knots });
    state.selected = clamp(state.selected, 0, state.slices - 1);
    selectedSlice = state.selected;
    $("selectedSlice").max = String(Math.max(0, state.slices - 1));
  }

  function scheduleRender(recomputeFirst) {
    if (framePending) return;
    framePending = true;
    window.requestAnimationFrame(() => {
      framePending = false;
      if (recomputeFirst) recompute();
      render();
    });
  }

  function updateStats() {
    const slice = result.slices[selectedSlice];
    const ratio = nullExpected > 0 ? result.value / nullExpected : 0;
    $("statSig").textContent = fmt(result.value, 3);
    $("statSigSub").textContent = `mean over ${result.S} slices`;
    $("statNull").textContent = fmt(nullExpected, 3);
    $("statRatio").textContent = fmt(ratio, 2) + "×";
    $("statSlice").textContent = fmt(slice.T, 3);
    $("statSliceSub").textContent = `slice ${slice.index}`;
    $("statBatch").textContent = `${result.N} × ${result.S}`;
    $("statBatchSub").textContent = `${result.N} points · ${result.S} directions`;

    const card = $("statRatioCard");
    card.className = "stat";
    card.style.borderColor = "";
    card.style.background = "";
    if (ratio < 2.5) card.classList.add("good");
    else if (ratio < 8.0) card.classList.add("warn");
    else {
      card.style.borderColor = "#efc7c3";
      card.style.background = "linear-gradient(180deg, #fffdfc, #fdf0ee)";
    }
  }

  function cloudExtent() {
    let maxAbs = 0;
    for (const [x, y] of points) maxAbs = Math.max(maxAbs, Math.abs(x), Math.abs(y));
    return Math.max(0.18, maxAbs * 1.22 + 0.12);
  }

  function drawCloud() {
    const canvas = $("cloudCanvas");
    const extent = cloudExtent();
    const plot = new Kit.Plot(canvas, {
      xmin: -extent,
      xmax: extent,
      ymin: -extent,
      ymax: extent,
      xlabel: "z₀",
      ylabel: "z₁",
      zeroLines: true,
      grid: true,
      padding: { l: 44, r: 20, t: 24, b: 40 },
    }).draw();

    plot.clip(() => {
      if (state.showDirections) {
        const count = Math.min(directions.length, 40);
        for (let i = 0; i < count; i += 1) {
          const dir = directions[i];
          const scale = (i / Math.max(1, count - 1)) * 0.92;
          plot.line(
            [
              [0, 0],
              [dir[0] * extent * scale, dir[1] * extent * scale],
            ],
            { stroke: Kit.COLORS.accentSoft, width: 0.8, alpha: 0.22 }
          );
        }
      }

      const selected = result.slices[selectedSlice];
      plot.line(
        [
          [0, 0],
          [selected.direction[0] * extent * 0.92, selected.direction[1] * extent * 0.92],
        ],
        { stroke: Kit.COLORS.accent, width: 1.8, alpha: 0.75 }
      );

      if (state.showGrad) {
        let maxGrad = 0;
        for (const g of result.gradient) {
          maxGrad = Math.max(maxGrad, Math.hypot(g[0], g[1]));
        }
        const stride = result.N > 140 ? Math.ceil(result.N / 140) : 1;
        const scale = (0.22 * extent) / Math.max(1e-9, maxGrad);
        for (let i = 0; i < result.N; i += stride) {
          const [x, y] = points[i];
          const [gx, gy] = result.gradient[i];
          plot.arrow(x, y, x - gx * scale, y - gy * scale, {
            stroke: Kit.COLORS.red,
            width: 0.9,
            head: 3.2,
            alpha: 0.42,
          });
        }
      }

      const radius = result.N > 300 ? 1.7 : result.N > 120 ? 2.1 : 2.8;
      plot.scatter(points, {
        r: radius,
        fill: "rgba(36,88,214,0.62)",
        stroke: "rgba(255,255,255,0.55)",
        lineWidth: 0.4,
      });

      plot.text(
        -extent * 0.97,
        extent * 0.92,
        `N = ${result.N}   S = ${result.S}`,
        { fill: Kit.COLORS.muted, font: "11px ui-monospace, monospace" }
      );
    });
  }

  function drawHistograms() {
    const canvas = $("histCanvas");
    const S = result.S;
    const step = Math.max(1, Math.floor(S / 3));
    const indices = [selectedSlice, (selectedSlice + step) % S, (selectedSlice + 2 * step) % S];
    const colors = [Kit.COLORS.accent, Kit.COLORS.teal, Kit.COLORS.orange];
    let maxStd = 1;
    const projections = indices.map((index) => {
      const values = result.slices[index].projections;
      const stats = Core.meanAndStd(values);
      maxStd = Math.max(maxStd, stats.std);
      return values;
    });
    const xmax = Math.min(10, Math.max(4.2, 3.1 * maxStd));
    const bins = 44;
    const histograms = projections.map((values) => Kit.histogram(values, bins, -xmax, xmax));
    let ymax = 0.5;
    for (const hist of histograms) {
      for (const density of hist.density) ymax = Math.max(ymax, density);
    }
    ymax *= 1.18;

    const plot = new Kit.Plot(canvas, {
      xmin: -xmax,
      xmax: xmax,
      ymin: 0,
      ymax: ymax,
      xlabel: "a_s · z",
      ylabel: "density",
      title: `selected slice ${selectedSlice}`,
      grid: true,
    }).draw();

    plot.clip(() => {
      const grid = [];
      for (let i = 0; i <= 220; i += 1) {
        const x = -xmax + (2 * xmax * i) / 220;
        grid.push([x, Kit.normalPdf(x)]);
      }
      plot.line(grid, { stroke: Kit.COLORS.ink, width: 1.8, dash: [5, 4], alpha: 0.8 });
      histograms.forEach((hist, index) => {
        plot.line(Kit.stepPoints(hist), { stroke: colors[index], width: 1.7, alpha: 0.9 });
      });
    });

    $("histLegend").innerHTML = indices
      .map(
        (index, i) =>
          `<span><span class="swatch" style="background:${colors[i]}"></span>slice ${index}</span>`
      )
      .concat(
        '<span><span class="swatch line" style="background:#18202a"></span>N(0,1)</span>'
      )
      .join("");
  }

  function drawCharacteristic() {
    const slice = result.slices[selectedSlice];
    const t = result.quadrature.t;
    const plot = new Kit.Plot($("cfCanvas"), {
      xmin: 0,
      xmax: state.tMax,
      ymin: -0.18,
      ymax: 1.06,
      xlabel: "t",
      ylabel: "value",
      title: `slice ${selectedSlice}`,
      zeroLines: true,
      grid: true,
    }).draw();

    plot.clip(() => {
      plot.line(
        t.map((value, index) => [value, result.quadrature.phi0[index]]),
        { stroke: Kit.COLORS.ink, width: 1.9, dash: [5, 4] }
      );
      plot.line(
        t.map((value, index) => [value, slice.C[index]]),
        { stroke: Kit.COLORS.accent, width: 2.1 }
      );
      plot.line(
        t.map((value, index) => [value, slice.S[index]]),
        { stroke: Kit.COLORS.orange, width: 1.8 }
      );
    });
  }

  function drawIntegrand() {
    const slice = result.slices[selectedSlice];
    const t = result.quadrature.t;
    const pointsOut = t.map((value, index) => [value, slice.integrand[index]]);
    let ymax = 0;
    for (const point of pointsOut) ymax = Math.max(ymax, point[1]);
    ymax = Math.max(0.05, ymax * 1.22);

    const plot = new Kit.Plot($("integrandCanvas"), {
      xmin: 0,
      xmax: state.tMax,
      ymin: 0,
      ymax,
      xlabel: "t",
      ylabel: "err(t) · φ₀(t)",
      title: `T_s = ${fmt(slice.T, 3)}`,
      grid: true,
    }).draw();

    plot.clip(() => {
      for (const value of t) {
        plot.line(
          [
            [value, 0],
            [value, ymax],
          ],
          { stroke: Kit.COLORS.faint, width: 0.7, alpha: 0.38 }
        );
      }
      plot.area(pointsOut, 0, { fill: "rgba(36,88,214,0.16)" });
      plot.line(pointsOut, { stroke: Kit.COLORS.accent, width: 2.0 });
      plot.scatter(
        pointsOut.map((point) => ({ 0: point[0], 1: point[1] })),
        { r: 2.2, fill: Kit.COLORS.accent, alpha: 0.85 }
      );
    });
  }

  function drawSlices() {
    const values = result.perSlice;
    let ymax = nullExpected * 1.35;
    for (const value of values) ymax = Math.max(ymax, value * 1.15);
    if (!isFinite(ymax) || ymax <= 0) ymax = 1;
    const plot = new Kit.Plot($("sliceCanvas"), {
      xmin: 0,
      xmax: Math.max(1, values.length - 1),
      ymin: 0,
      ymax,
      xlabel: "slice index",
      ylabel: "T_s",
      title: `mean ${fmt(result.value, 3)}   ·   Gaussian null ${fmt(nullExpected, 3)}`,
      grid: true,
    }).draw();

    plot.clip(() => {
      plot.hline(nullExpected, { stroke: Kit.COLORS.orange, width: 1.4, dash: [5, 4] });
      plot.hline(result.value, { stroke: Kit.COLORS.green, width: 1.6 });
      plot.line(
        values.map((value, index) => [index, value]),
        { stroke: Kit.COLORS.accent, width: 1.1, alpha: 0.75 }
      );
      plot.scatter(
        values.map((value, index) => ({ 0: index, 1: value, index })),
        { r: values.length > 220 ? 1.3 : 2.3, fill: Kit.COLORS.accent, alpha: 0.8 }
      );
      plot.vline(selectedSlice, { stroke: Kit.COLORS.red, width: 1.2, dash: [2, 3] });
    });
  }

  function renderTable() {
    const slice = result.slices[selectedSlice];
    const quad = result.quadrature;
    const rows = [];
    for (let i = 0; i < quad.t.length; i += 1) {
      rows.push(
        `<tr>` +
          `<td>${fmt(quad.t[i], 3)}</td>` +
          `<td>${fmt(slice.C[i], 4)}</td>` +
          `<td>${fmt(slice.S[i], 4)}</td>` +
          `<td>${fmt(quad.phi0[i], 4)}</td>` +
          `<td>${fmt(slice.err[i], 5)}</td>` +
          `<td>${fmt(quad.weights[i], 5)}</td>` +
          `<td>${fmt(result.N * slice.err[i] * quad.weights[i], 5)}</td>` +
          `</tr>`
      );
    }
    $("sliceTableBody").innerHTML = rows.join("");
  }

  function render() {
    selectedSlice = clamp(selectedSlice, 0, result.S - 1);
    state.selected = selectedSlice;
    $("selectedSlice").value = String(selectedSlice);
    $("selectedSlice").max = String(Math.max(0, result.S - 1));
    syncOutputs();
    updateStats();
    drawCloud();
    drawHistograms();
    drawCharacteristic();
    drawIntegrand();
    drawSlices();
    renderTable();
  }

  function bindControls() {
    $("preset").addEventListener("change", (event) => {
      applyPreset(event.target.value);
      scheduleRender(true);
    });

    const pointInputs = ["nPoints", "stdX", "stdY", "corr", "offsetX", "offsetY", "seed"];
    const geometryInputs = new Set(["stdX", "stdY", "corr", "offsetX", "offsetY"]);
    const distributionPresets = new Set(["two_clusters", "heavy_tails", "uniform"]);
    for (const id of pointInputs) {
      $(id).addEventListener("input", () => {
        readControls();
        if (geometryInputs.has(id) && !distributionPresets.has(state.preset)) {
          state.preset = "custom";
          $("preset").value = "custom";
        }
        syncOutputs();
        scheduleRender(true);
      });
    }

    const sigInputs = ["slices", "knots", "tMax"];
    for (const id of sigInputs) {
      $(id).addEventListener("input", () => {
        readControls();
        if (state.preset === "gaussian" && (id === "stdX" || id === "stdY" || id === "corr")) {
          // no-op: the preset name is only a starting point
        }
        state.dirSeed += id === "slices" ? 1 : 0;
        syncOutputs();
        scheduleRender(true);
      });
    }

    $("selectedSlice").addEventListener("input", () => {
      selectedSlice = Number($("selectedSlice").value);
      state.selected = selectedSlice;
      syncOutputs();
      scheduleRender(false);
    });

    $("showGrad").addEventListener("change", () => {
      state.showGrad = $("showGrad").checked;
      scheduleRender(false);
    });
    $("showDirections").addEventListener("change", () => {
      state.showDirections = $("showDirections").checked;
      scheduleRender(false);
    });

    $("newSample").addEventListener("click", () => {
      state.seed = (state.seed + 1) % 1000000;
      $("seed").value = String(state.seed);
      scheduleRender(true);
    });

    $("resampleDirs").addEventListener("click", () => {
      state.dirSeed = (state.dirSeed + 1) % 2147483647;
      scheduleRender(true);
    });

    $("reset").addEventListener("click", () => {
      Object.assign(state, DEFAULTS);
      selectedSlice = 0;
      syncControls();
      scheduleRender(true);
    });
  }

  function init() {
    syncControls();
    bindControls();
    recompute();
    render();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
