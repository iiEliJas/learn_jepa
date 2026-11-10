/* LeJEPA Loss Calculator: editable tiny tensor, loss arithmetic and plots. */
(function () {
  "use strict";

  const Core = window.SigregCore;
  const Loss = window.LossCore;
  const Kit = window.CanvasKit;
  const $ = (id) => document.getElementById(id);

  const DEFAULTS = {
    preset: "near_invariant",
    V: 4,
    N: 6,
    K: 2,
    Vg: 4,
    lambda: 0.02,
    slices: 64,
    mode: "per_view",
    reduction: "mean_over_k",
    seed: 1234,
    dirSeed: 987,
  };

  const VIEW_COLORS = [
    Kit.COLORS.accent,
    Kit.COLORS.teal,
    Kit.COLORS.orange,
    Kit.COLORS.red,
    Kit.COLORS.purple,
    Kit.COLORS.green,
  ];

  const state = Object.assign({}, DEFAULTS);
  let z = [];
  let result = null;
  let typical = [];
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

  function deepCopy(matrix) {
    return matrix.map((row) => row.map((cell) => cell.slice()));
  }

  function readControls() {
    state.preset = $("preset").value;
    state.V = Number($("views").value);
    state.N = Number($("samples").value);
    state.K = Number($("dim").value);
    state.Vg = clamp(Number($("globalViews").value), 1, state.V);
    state.lambda = Number($("lambda").value);
    state.slices = Number($("slices").value);
    state.mode = $("mode").value;
    state.reduction = $("reduction").value;
    state.seed = Math.max(0, Math.floor(Number($("seed").value) || 0));
  }

  function syncControls() {
    $("preset").value = state.preset;
    $("views").value = state.V;
    $("samples").value = state.N;
    $("dim").value = String(state.K);
    $("globalViews").max = String(state.V);
    $("globalViews").value = state.Vg;
    $("lambda").value = state.lambda;
    $("slices").value = String(state.slices);
    $("mode").value = state.mode;
    $("reduction").value = state.reduction;
    $("seed").value = state.seed;
    syncOutputs();
  }

  function syncOutputs() {
    $("viewsOut").textContent = String(state.V);
    $("samplesOut").textContent = String(state.N);
    $("globalViewsOut").textContent = String(state.Vg);
    $("lambdaOut").textContent = Number(state.lambda).toFixed(3);
  }

  function makeBase(N, K, rng, scale) {
    return Array.from({ length: N }, () =>
      Array.from({ length: K }, () => Core.randn(rng) * (scale || 1.4))
    );
  }

  function makePreset(name) {
    const shapeMatches =
      z.length === state.V &&
      z[0] &&
      z[0].length === state.N &&
      z[0][0] &&
      z[0][0].length === state.K;
    if (name === "custom" && shapeMatches) return deepCopy(z);
    const rng = Core.mulberry32(hashSeed(state.seed + name.length * 131 + state.V * 17 + state.N));
    const base = makeBase(state.N, state.K, rng, 1.5);
    const out = [];

    if (name === "collapsed") {
      for (let v = 0; v < state.V; v += 1) {
        out.push(
          Array.from({ length: state.N }, () => new Array(state.K).fill(0.1))
        );
      }
      return out;
    }

    if (name === "near_invariant") {
      for (let v = 0; v < state.V; v += 1) {
        out.push(
          base.map((row) => row.map((value) => value + 0.16 * Core.randn(rng)))
        );
      }
      return out;
    }

    if (name === "invariant_diverse") {
      for (let v = 0; v < state.V; v += 1) out.push(base.map((row) => row.slice()));
      return out;
    }

    if (name === "noisy") {
      for (let v = 0; v < state.V; v += 1) {
        out.push(
          base.map((row) =>
            row.map((value) => value + 0.55 * Core.randn(rng))
          )
        );
      }
      return out;
    }

    if (name === "outlier") {
      const copies = Array.from({ length: state.V }, () => base.map((row) => row.slice()));
      copies[0][0][0] += 4.0;
      return copies;
    }

    if (name === "non_gaussian") {
      const copies = Array.from({ length: state.V }, () => base.map((row) => row.slice()));
      copies[state.V - 1][state.N - 1] = new Array(state.K).fill(4.5);
      if (state.K > 1) copies[state.V - 1][state.N - 1][1] = -3.5;
      return copies;
    }

    return Array.from({ length: state.V }, () => base.map((row) => row.slice()));
  }

  function randomize() {
    const rng = Core.mulberry32(hashSeed(state.seed + 4242));
    z = Array.from({ length: state.V }, () =>
      Array.from({ length: state.N }, () =>
        Array.from({ length: state.K }, () => Core.randn(rng) * 1.2)
      )
    );
    state.preset = "custom";
    $("preset").value = "custom";
  }

  function buildMatrix() {
    const container = $("matrix");
    container.innerHTML = "";
    for (let v = 0; v < state.V; v += 1) {
      const row = document.createElement("div");
      row.className = "matrix-row";
      const label = document.createElement("div");
      label.className = "matrix-label";
      label.textContent = `view ${v}`;
      row.appendChild(label);
      for (let n = 0; n < state.N; n += 1) {
        const cell = document.createElement("div");
        cell.className = "cell";
        for (let k = 0; k < state.K; k += 1) {
          const input = document.createElement("input");
          input.type = "number";
          input.step = "0.1";
          input.value = Number(z[v][n][k]).toFixed(2);
          input.dataset.v = String(v);
          input.dataset.n = String(n);
          input.dataset.k = String(k);
          input.setAttribute("aria-label", `z view ${v} sample ${n} dim ${k}`);
          input.title = `view ${v}, sample ${n}, dimension ${k}`;
          cell.appendChild(input);
        }
        row.appendChild(cell);
      }
      container.appendChild(row);
    }
  }

  function compute() {
    const directions = Loss.makeDirections(state.K, state.slices, hashSeed(state.dirSeed));
    result = Loss.lejepa(z, {
      lambda: state.lambda,
      Vg: state.Vg,
      reduction: state.reduction,
      sigregMode: state.mode,
      directions,
      tMax: 3,
      nKnots: 17,
    });
    typical = Loss.typicalSigreg(
      z,
      {
        lambda: state.lambda,
        Vg: state.Vg,
        reduction: state.reduction,
        sigregMode: state.mode,
        tMax: 3,
        nKnots: 17,
        seed: state.dirSeed,
      },
      16
    );
  }

  function schedule(recomputeFirst) {
    if (framePending) return;
    framePending = true;
    window.requestAnimationFrame(() => {
      framePending = false;
      if (recomputeFirst) compute();
      render();
    });
  }

  function updateStats() {
    $("statPred").textContent = fmt(result.lPred, 4);
    $("statPredSub").textContent =
      state.reduction === "mean_over_k"
        ? `mean over ${result.V}·${result.N}·${result.K} = ${result.elementCount}`
        : `sum over K, mean over ${result.V}·${result.N}`;
    $("statSig").textContent = fmt(result.lSig, 4);
    $("statSigSub").textContent = `${result.mode.replace("_", " ")} · ${state.slices} slices`;
    $("statLoss").textContent = fmt(result.loss, 4);
    $("statLossSub").textContent = `λ = ${Number(state.lambda).toFixed(3)}`;
    $("statShare").textContent = `${fmt(100 * result.sigFraction, 1)}%`;
    const values = result.perView;
    $("statRange").textContent = `${fmt(Math.min(...values), 2)} … ${fmt(Math.max(...values), 2)}`;
    $("statRangeSub").textContent =
      result.mode === "joint" ? "joint test (one value)" : `${values.length} views`;
  }

  function updateBars() {
    const total = Math.max(1e-12, result.weightedPred + result.weightedSig);
    const maxRaw = Math.max(1e-12, result.lPred, result.lSig);
    $("barPred").style.width = `${(100 * result.weightedPred) / total}%`;
    $("barSig").style.width = `${(100 * result.weightedSig) / total}%`;
    $("barPredRaw").style.width = `${(100 * result.lPred) / maxRaw}%`;
    $("barSigRaw").style.width = `${(100 * result.lSig) / maxRaw}%`;
    $("barPredNum").textContent = fmt(result.weightedPred, 4);
    $("barSigNum").textContent = fmt(result.weightedSig, 4);
    $("barPredRawNum").textContent = fmt(result.lPred, 4);
    $("barSigRawNum").textContent = fmt(result.lSig, 4);

    const note = $("balanceNote");
    if (result.lPred < 1e-9) {
      note.className = "note warn";
      note.innerHTML =
        "All views have the same mean, so <span class=\"mono\">L_pred</span> is zero. " +
        "Only SIGReg prevents this trivial solution.";
    } else if (result.sigFraction < 0.05) {
      note.className = "note";
      note.innerHTML =
        "The prediction term dominates. The regularizer is still active, but its weighted " +
        "share is small.";
    } else if (result.sigFraction > 0.8) {
      note.className = "note warn";
      note.innerHTML =
        "SIGReg dominates the total. At this <span class=\"mono\">lambda</span> the " +
        "representation is being pushed hard toward the Gaussian target.";
    } else if (result.sigFraction > 0.5) {
      note.className = "note";
      note.innerHTML =
        "SIGReg has the larger weighted share here, but both terms are active. " +
        "On a tiny batch with small prediction error this is normal.";
    } else {
      note.className = "note good";
      note.innerHTML =
        "Both terms matter. This is the regime the default " +
        "<span class=\"mono\">lambda = 0.02</span> is designed for on the real model.";
    }
  }

  function updateStepByStep() {
    const mu = result.mu;
    const v = 0;
    const n = 0;
    const z0 = z[v][n];
    const mu0 = mu[n];
    const residual = mu0.map((value, k) => value - z0[k]);
    const sq = residual.reduce((sum, value) => sum + value * value, 0);
    const denom =
      state.reduction === "mean_over_k" ? result.V * result.N * result.K : result.V * result.N;
    const lines = [];
    lines.push(
      `mu[${n}] = mean_{v < ${state.Vg}} z[v][${n}] = [${mu0.map((value) => fmt(value, 3)).join(", ")}]`
    );
    lines.push(
      `mu[${n}] - z[${v}][${n}] = [${residual.map((value) => fmt(value, 3)).join(", ")}]`
    );
    lines.push(`||mu[${n}] - z[${v}][${n}]||^2 = ${fmt(sq, 4)}`);
    lines.push(
      `L_pred = ${state.reduction === "mean_over_k" ? "mean" : "sum_K mean"} = ${fmt(result.lPred, 5)}  (denominator ${denom})`
    );
    lines.push(
      `L_sig  = mean_v SIGReg(z[v]) = ${fmt(result.lSig, 5)}  (per_view: ${fmt(Math.min(...result.perView), 3)} … ${fmt(Math.max(...result.perView), 3)})`
    );
    lines.push(
      `L      = (1 - ${Number(state.lambda).toFixed(3)})·${fmt(result.lPred, 4)} + ${Number(state.lambda).toFixed(3)}·${fmt(result.lSig, 4)} = ${fmt(result.loss, 5)}`
    );
    $("stepByStep").textContent = lines.join("\n");
  }

  function updatePerView() {
    const values = result.perView;
    const max = Math.max(1e-12, ...values);
    const labels =
      result.mode === "joint"
        ? ["joint"]
        : values.map((_, index) => `view ${index}`);
    $("perViewBars").innerHTML = values
      .map((value, index) => {
        const pct = (100 * value) / max;
        return (
          `<div class="bar-row">` +
          `<span>${labels[index]}</span>` +
          `<div class="bar-track"><div class="bar-fill teal" style="width:${pct}%"></div></div>` +
          `<span class="num">${fmt(value, 3)}</span>` +
          `</div>`
        );
      })
      .join("");

    const q05 = Core.quantile(typical, 0.05);
    const q95 = Core.quantile(typical, 0.95);
    const nullExpected = Core.gaussianNullExpected({ tMax: 3, nKnots: 17 });
    $("sigregNoise").innerHTML =
      `Gaussian reference for this quadrature: <span class="mono">${fmt(nullExpected, 3)}</span>. ` +
      `One direction draw is shown; over 16 fresh draws the same batch gives ` +
      `<span class="mono">[${fmt(q05, 3)}, ${fmt(q95, 3)}]</span> (5–95%). ` +
      `With N = ${state.N}, the estimator is noisy; the real recipe uses N = 128 per view.`;
  }

  function drawScatter() {
    const canvas = $("scatterCanvas");
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    const consider = (x, y) => {
      minX = Math.min(minX, x);
      maxX = Math.max(maxX, x);
      minY = Math.min(minY, y);
      maxY = Math.max(maxY, y);
    };
    for (let v = 0; v < result.V; v += 1) {
      for (let n = 0; n < result.N; n += 1) consider(z[v][n][0], z[v][n][1] || 0);
    }
    for (const point of result.mu) consider(point[0], point[1] || 0);

    const spanX = Math.max(0.4, maxX - minX);
    const spanY = Math.max(0.4, maxY - minY);
    const padX = 0.18 * spanX + 0.08;
    const padY = 0.18 * spanY + 0.08;
    const xmin = minX - padX;
    const xmax = maxX + padX;
    const ymin = minY - padY;
    const ymax = maxY + padY;

    const plot = new Kit.Plot(canvas, {
      xmin,
      xmax,
      ymin,
      ymax,
      xlabel: "z₀",
      ylabel: "z₁",
      zeroLines: true,
      grid: true,
      padding: { l: 44, r: 20, t: 24, b: 40 },
    }).draw();

    plot.clip(() => {
      for (let v = 0; v < result.V; v += 1) {
        const color = VIEW_COLORS[v % VIEW_COLORS.length];
        for (let n = 0; n < result.N; n += 1) {
          const point = z[v][n];
          const target = result.mu[n];
          if (
            Math.abs(point[0] - target[0]) < 1e-9 &&
            Math.abs((point[1] || 0) - (target[1] || 0)) < 1e-9
          ) {
            continue;
          }
          plot.arrow(point[0], point[1] || 0, target[0], target[1] || 0, {
            stroke: color,
            width: 0.9,
            head: 3.0,
            alpha: 0.38,
          });
        }
      }

      for (let v = 0; v < result.V; v += 1) {
        const color = VIEW_COLORS[v % VIEW_COLORS.length];
        for (let n = 0; n < result.N; n += 1) {
          const point = z[v][n];
          plot.scatter([[point[0], point[1] || 0]], {
            r: 5.2,
            fill: color,
            stroke: "#ffffff",
            lineWidth: 1.2,
            alpha: 0.9,
          });
        }
      }

      for (const point of result.mu) {
        plot.scatter([[point[0], point[1] || 0]], {
          r: 4.2,
          fill: "#18202a",
          stroke: "#ffffff",
          lineWidth: 1.1,
        });
      }
    });
  }

  function render() {
    updateStats();
    updateBars();
    updateStepByStep();
    updatePerView();
    drawScatter();
  }

  function rebuildFromPreset() {
    readControls();
    state.Vg = clamp(state.Vg, 1, state.V);
    z = makePreset(state.preset);
    syncControls();
    buildMatrix();
    schedule(true);
  }

  function bind() {
    $("preset").addEventListener("change", () => {
      readControls();
      if (state.preset !== "custom") z = makePreset(state.preset);
      syncControls();
      buildMatrix();
      schedule(true);
    });

    for (const id of ["views", "samples", "dim"]) {
      $(id).addEventListener("input", () => {
        rebuildFromPreset();
      });
      $(id).addEventListener("change", () => {
        rebuildFromPreset();
      });
    }

    $("globalViews").addEventListener("input", () => {
      readControls();
      syncOutputs();
      schedule(true);
    });

    for (const id of ["lambda", "slices", "mode", "reduction", "seed"]) {
      $(id).addEventListener("input", () => {
        readControls();
        syncOutputs();
        schedule(true);
      });
      $(id).addEventListener("change", () => {
        readControls();
        syncOutputs();
        schedule(true);
      });
    }

    $("matrix").addEventListener("input", (event) => {
      const input = event.target;
      if (!(input instanceof HTMLInputElement)) return;
      const v = Number(input.dataset.v);
      const n = Number(input.dataset.n);
      const k = Number(input.dataset.k);
      const value = Number(input.value);
      z[v][n][k] = Number.isFinite(value) ? value : 0;
      state.preset = "custom";
      $("preset").value = "custom";
      schedule(true);
    });

    $("randomize").addEventListener("click", () => {
      readControls();
      randomize();
      buildMatrix();
      schedule(true);
    });

    $("resample").addEventListener("click", () => {
      state.dirSeed = (state.dirSeed + 1) % 2147483647;
      schedule(true);
    });

    $("reset").addEventListener("click", () => {
      Object.assign(state, DEFAULTS);
      z = makePreset(state.preset);
      syncControls();
      buildMatrix();
      schedule(true);
    });
  }

  function init() {
    syncControls();
    z = makePreset(state.preset);
    buildMatrix();
    bind();
    compute();
    render();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
