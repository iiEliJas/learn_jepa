/* Collapse Simulator: train a tiny LeJEPA in the browser and scrub the run. */
(function () {
  "use strict";

  const Core = window.CollapseCore;
  const Sigreg = window.SigregCore;
  const Kit = window.CanvasKit;
  const $ = (id) => document.getElementById(id);

  const CLUSTER_COLORS = ["#2458d6", "#0f766e", "#c2610a"];
  const PAIR_STROKE = "rgba(92, 103, 117, 0.22)";
  const FRAME_MS = 30;

  const state = {
    useSig: true,
    lambda: 0.5,
    noise: 0.2,
    seed: 7,
    steps: 240,
    dirs: 8,
    lr: 0.02,
    hidden: 8,
    count: 96,
    step: 240,
    pendingStep: null,
    playing: false,
  };

  let run = null;
  let embedRange = 1;
  let animation = null;
  let lastFrame = 0;

  function fmt(value, digits) {
    const places = digits === undefined ? 3 : digits;
    if (!isFinite(value)) return "–";
    const abs = Math.abs(value);
    if (abs >= 10000 || (abs > 0 && abs < 0.0005)) return value.toExponential(2);
    return value.toFixed(places);
  }

  function pct(value) {
    return `${(100 * value).toFixed(0)}%`;
  }

  function meanOfRows(rows) {
    const mean = [0, 0];
    for (const row of rows) {
      mean[0] += row[0];
      mean[1] += row[1];
    }
    mean[0] /= rows.length;
    mean[1] /= rows.length;
    return mean;
  }

  function dataRange(points) {
    let maxAbs = 0.001;
    for (const point of points) {
      maxAbs = Math.max(maxAbs, Math.abs(point[0]), Math.abs(point[1]));
    }
    return maxAbs * 1.12;
  }

  function trainingOptions() {
    return {
      count: state.count,
      seed: state.seed,
      noise: state.noise,
      hidden: state.hidden,
      steps: state.steps,
      lambda: state.lambda,
      useSig: state.useSig,
      lr: state.lr,
      dirs: state.dirs,
    };
  }

  function computeEmbedRange() {
    let maxAbs = 0.05;
    for (const snapshot of run.snapshots) {
      for (const row of snapshot.z1) {
        maxAbs = Math.max(maxAbs, Math.abs(row[0]), Math.abs(row[1]));
      }
      for (const row of snapshot.z2) {
        maxAbs = Math.max(maxAbs, Math.abs(row[0]), Math.abs(row[1]));
      }
    }
    return maxAbs * 1.1;
  }

  function retrain() {
    stopPlayback();
    const started = performance.now();
    run = Core.runTraining(trainingOptions());
    state.step = run.snapshots.length - 1;
    embedRange = computeEmbedRange();
    const elapsed = performance.now() - started;
    $("trainNote").textContent =
      `${state.steps} steps · ${state.count} points · ` +
      `${run.snapshots.length} snapshots · ${elapsed.toFixed(0)} ms` +
      (state.useSig ? "" : " · SIGReg is diagnostic only in this run");
    resizeStepControl();
    render();
  }

  function resizeStepControl() {
    const slider = $("step");
    slider.max = String(run.snapshots.length - 1);
    state.step = Math.max(0, Math.min(state.step, run.snapshots.length - 1));
  }

  function snapshot() {
    return run.snapshots[state.step];
  }

  function updateControls() {
    for (const button of $("antiCollapse").querySelectorAll("button")) {
      button.setAttribute("aria-pressed", String((button.dataset.on === "true") === state.useSig));
    }
    $("lambda").value = String(state.lambda);
    $("lambda").disabled = !state.useSig;
    $("lambdaOut").textContent = state.useSig ? state.lambda.toFixed(2) : "off";
    $("noise").value = String(state.noise);
    $("noiseOut").textContent = state.noise.toFixed(2);
    $("step").value = String(state.step);
    $("stepOut").textContent = `${state.step} / ${run.snapshots.length - 1}`;
    $("playPause").textContent = state.playing ? "Pause" : "Replay training";
  }

  function updateStats(snap) {
    $("statStep").textContent = String(snap.step);
    $("statStepSub").textContent = `of ${run.snapshots.length - 1}`;
    $("statPred").textContent = fmt(snap.lPred, 4);
    $("statSig").textContent = fmt(snap.lSig, 3);
    $("statSigSub").textContent = `Gaussian null ≈ ${fmt(1.0525, 2)}`;
    $("statTotal").textContent = fmt(snap.total, 3);
    $("statTotalSub").textContent = state.useSig
      ? `L_pred + ${state.lambda.toFixed(2)} · L_sig`
      : "L_pred (SIGReg off)";
    $("statRank").textContent = fmt(snap.rank, 3);
    $("statStd").textContent = `${fmt(snap.std[0], 2)} / ${fmt(snap.std[1], 2)}`;

    const rankBox = $("statRank").parentElement;
    if (rankBox && rankBox.classList) {
      rankBox.classList.toggle("good", snap.rank > 1.8);
      rankBox.classList.toggle("warn", snap.rank < 1.2);
    }

    const note = $("collapseNote");
    let tone = "";
    let text;
    if (snap.rank < 0.15) {
      tone = "warn";
      text =
        `<strong>Collapsed to a point.</strong> Every embedding is nearly the same; ` +
        `the rank is ${fmt(snap.rank, 2)} and both σs are close to zero. ` +
        `The prediction loss can be tiny because there is nothing left to predict.`;
    } else if (snap.rank < 1.2) {
      tone = "warn";
      text =
        `<strong>Dimensional collapse.</strong> The cloud is essentially a line: ` +
        `effective rank ${fmt(snap.rank, 2)}, σ₁ = ${fmt(snap.std[0], 2)}, σ₂ = ${fmt(
          snap.std[1],
          2
        )}. The prediction term is satisfied without a 2-D representation.`;
    } else if (snap.rank > 1.85) {
      tone = "good";
      text =
        `<strong>Spread across both dimensions.</strong> Effective rank ${fmt(
          snap.rank,
          2
        )}, with per-dimension σs ${fmt(snap.std[0], 2)} and ${fmt(
          snap.std[1],
          2
        )}. L_SIGReg is ${fmt(snap.lSig, 2)} against a null of about 1.05.`;
    } else {
      text =
        `<strong>Partially collapsed.</strong> Effective rank ${fmt(
          snap.rank,
          2
        )} sits between a line (1) and a filled 2-D cloud (2). ` +
        `This is the boundary where the prediction term starts to win.`;
    }
    if (!state.useSig) {
      text +=
        ` SIGReg is switched off, so the reported value is a diagnostic, not part of the objective.`;
    }
    note.className = `note${tone ? ` ${tone}` : ""}`;
    note.innerHTML = text;
  }

  function drawFrame(canvas, range) {
    const { ctx, w, h } = Kit.fitCanvas(canvas);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);
    const pad = 16;
    const radius = Math.min(w, h) / 2 - pad;
    const scale = radius / range;
    const cx = w / 2;
    const cy = h / 2;
    ctx.save();
    ctx.strokeStyle = "#e8ecf2";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx - radius - 8, cy);
    ctx.lineTo(cx + radius + 8, cy);
    ctx.moveTo(cx, cy - radius - 8);
    ctx.lineTo(cx, cy + radius + 8);
    ctx.stroke();
    ctx.restore();
    return {
      ctx,
      w,
      h,
      scale,
      x: (value) => cx + value * scale,
      y: (value) => cy - value * scale,
      radius,
    };
  }

  function drawInput() {
    const data = run.data;
    const points = data.base.concat(data.views[0], data.views[1]);
    const frame = drawFrame($("inputCanvas"), dataRange(points));
    const { ctx, x, y } = frame;
    ctx.save();
    ctx.globalAlpha = 0.34;
    for (let view = 0; view < 2; view += 1) {
      for (let index = 0; index < data.count; index += 1) {
        const point = data.views[view][index];
        ctx.beginPath();
        ctx.fillStyle = CLUSTER_COLORS[data.clusters[index] % CLUSTER_COLORS.length];
        ctx.arc(x(point[0]), y(point[1]), 2.4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
    for (let index = 0; index < data.count; index += 1) {
      const point = data.base[index];
      ctx.beginPath();
      ctx.fillStyle = CLUSTER_COLORS[data.clusters[index] % CLUSTER_COLORS.length];
      ctx.arc(x(point[0]), y(point[1]), 3.6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.8)";
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }
    ctx.restore();
    $("inputLegend").innerHTML =
      CLUSTER_COLORS.map(
        (color, index) =>
          `<span><span class="swatch" style="background:${color}"></span>cluster ${index}</span>`
      ).join("") +
      '<span class="small muted">large = base · small = views</span>';
  }

  function drawEmbedding(snap) {
    const frame = drawFrame($("embedCanvas"), embedRange);
    const { ctx, x, y } = frame;
    ctx.save();
    ctx.strokeStyle = PAIR_STROKE;
    ctx.lineWidth = 1;
    for (let index = 0; index < snap.z1.length; index += 1) {
      const a = snap.z1[index];
      const b = snap.z2[index];
      ctx.beginPath();
      ctx.moveTo(x(a[0]), y(a[1]));
      ctx.lineTo(x(b[0]), y(b[1]));
      ctx.stroke();
    }
    for (let index = 0; index < snap.z1.length; index += 1) {
      const color = CLUSTER_COLORS[run.data.clusters[index] % CLUSTER_COLORS.length];
      const a = snap.z1[index];
      ctx.beginPath();
      ctx.fillStyle = color;
      ctx.globalAlpha = 0.88;
      ctx.arc(x(a[0]), y(a[1]), 3.1, 0, Math.PI * 2);
      ctx.fill();
      ctx.globalAlpha = 1;
      const b = snap.z2[index];
      ctx.beginPath();
      ctx.fillStyle = "#ffffff";
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.arc(x(b[0]), y(b[1]), 2.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    const mean = meanOfRows(snap.z1.concat(snap.z2));
    ctx.strokeStyle = "#18202a";
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(x(mean[0]) - 5, y(mean[1]));
    ctx.lineTo(x(mean[0]) + 5, y(mean[1]));
    ctx.moveTo(x(mean[0]), y(mean[1]) - 5);
    ctx.lineTo(x(mean[0]), y(mean[1]) + 5);
    ctx.stroke();
    ctx.restore();
    $("embedLegend").innerHTML =
      CLUSTER_COLORS.map(
        (color, index) =>
          `<span><span class="swatch" style="background:${color}"></span>cluster ${index}</span>`
      ).join("") +
      '<span class="small muted">● view 1 · ○ view 2 · ✚ mean</span>';
  }

  function series(key) {
    return run.snapshots.map((snap) => [snap.step, snap[key]]);
  }

  function drawPrediction() {
    const values = series("lPred");
    let ymax = 0.001;
    for (const [, value] of values) ymax = Math.max(ymax, value);
    ymax *= 1.15;
    const snap = snapshot();
    const plot = new Kit.Plot($("predCanvas"), {
      xmin: 0,
      xmax: run.snapshots.length - 1,
      ymin: 0,
      ymax,
      xlabel: "step",
      ylabel: "L_pred",
      title: `now ${fmt(snap.lPred, 4)}`,
      grid: true,
      padding: { l: 52, r: 12, t: 24, b: 34 },
    }).draw();
    plot.clip(() => {
      plot.area(values, 0, { fill: "rgba(36, 88, 214, 0.10)" });
      plot.line(values, { stroke: Kit.COLORS.accent, width: 1.8 });
      plot.vline(snap.step, { stroke: "rgba(24, 32, 42, 0.35)", width: 1, dash: [4, 3] });
      plot.scatter([{ 0: snap.step, 1: snap.lPred }], {
        r: 3,
        fill: Kit.COLORS.accent,
        stroke: "#ffffff",
        lineWidth: 1,
      });
    });
  }

  function drawSigreg() {
    const values = series("lSig");
    let ymax = 3;
    for (const [, value] of values) ymax = Math.max(ymax, value);
    ymax *= 1.12;
    const snap = snapshot();
    const nullValue = 1.0525;
    const plot = new Kit.Plot($("sigCanvas"), {
      xmin: 0,
      xmax: run.snapshots.length - 1,
      ymin: 0,
      ymax,
      xlabel: "step",
      ylabel: "L_SIGReg",
      title: `now ${fmt(snap.lSig, 3)}`,
      grid: true,
      padding: { l: 54, r: 12, t: 24, b: 34 },
    }).draw();
    plot.clip(() => {
      plot.hline(nullValue, { stroke: Kit.COLORS.orange, width: 1.2, dash: [5, 4] });
      plot.line(values, { stroke: Kit.COLORS.teal, width: 1.8 });
      plot.vline(snap.step, { stroke: "rgba(24, 32, 42, 0.35)", width: 1, dash: [4, 3] });
      plot.scatter([{ 0: snap.step, 1: snap.lSig }], {
        r: 3,
        fill: Kit.COLORS.teal,
        stroke: "#ffffff",
        lineWidth: 1,
      });
    });
    $("sigLegend").innerHTML =
      '<span><span class="swatch line" style="background:#0f766e"></span>L_SIGReg</span>' +
      '<span><span class="swatch line" style="background:#c2610a"></span>Gaussian null</span>';
  }

  function drawGeometry() {
    const ranks = run.snapshots.map((snap) => [snap.step, snap.rank]);
    const first = run.snapshots.map((snap) => [snap.step, snap.std[0]]);
    const second = run.snapshots.map((snap) => [snap.step, snap.std[1]]);
    const snap = snapshot();
    const plot = new Kit.Plot($("geomCanvas"), {
      xmin: 0,
      xmax: run.snapshots.length - 1,
      ymin: 0,
      ymax: 2.05,
      xlabel: "step",
      ylabel: "rank / σ",
      title: `now ${fmt(snap.rank, 2)} · ${fmt(snap.std[0], 2)} / ${fmt(snap.std[1], 2)}`,
      grid: true,
      padding: { l: 54, r: 12, t: 24, b: 34 },
      yTicks: [0, 0.5, 1, 1.5, 2],
    }).draw();
    plot.clip(() => {
      plot.hline(1, { stroke: "rgba(92,103,117,0.28)", width: 1, dash: [3, 4] });
      plot.hline(2, { stroke: "rgba(92,103,117,0.18)", width: 1, dash: [3, 4] });
      plot.line(first, { stroke: Kit.COLORS.teal, width: 1.2, dash: [4, 3], alpha: 0.85 });
      plot.line(second, { stroke: Kit.COLORS.orange, width: 1.2, dash: [4, 3], alpha: 0.85 });
      plot.line(ranks, { stroke: Kit.COLORS.accent, width: 2.0 });
      plot.vline(snap.step, { stroke: "rgba(24, 32, 42, 0.35)", width: 1, dash: [4, 3] });
      plot.scatter([{ 0: snap.step, 1: snap.rank }], {
        r: 3,
        fill: Kit.COLORS.accent,
        stroke: "#ffffff",
        lineWidth: 1,
      });
    });
    $("geomLegend").innerHTML =
      '<span><span class="swatch line" style="background:#2458d6"></span>effective rank</span>' +
      '<span><span class="swatch line" style="background:#0f766e"></span>σ₁</span>' +
      '<span><span class="swatch line" style="background:#c2610a"></span>σ₂</span>';
  }

  function render() {
    if (!run) return;
    const snap = snapshot();
    updateControls();
    updateStats(snap);
    drawInput();
    drawEmbedding(snap);
    drawPrediction();
    drawSigreg();
    drawGeometry();
    updateUrl();
  }

  function setStep(value, pause) {
    if (pause !== false) stopPlayback();
    state.step = Math.max(0, Math.min(run.snapshots.length - 1, Math.round(value)));
    render();
  }

  function stopPlayback() {
    state.playing = false;
    if (animation !== null) {
      cancelAnimationFrame(animation);
      animation = null;
    }
    lastFrame = 0;
    if (run) updateControls();
  }

  function frame(timestamp) {
    if (!state.playing) return;
    if (lastFrame === 0) lastFrame = timestamp;
    if (timestamp - lastFrame >= FRAME_MS) {
      lastFrame = timestamp;
      if (state.step >= run.snapshots.length - 1) {
        stopPlayback();
        render();
        return;
      }
      setStep(state.step + 1, false);
    }
    animation = requestAnimationFrame(frame);
  }

  function startPlayback() {
    stopPlayback();
    state.step = 0;
    state.playing = true;
    animation = requestAnimationFrame(frame);
    render();
  }

  function togglePlayback() {
    if (state.playing) {
      stopPlayback();
      render();
    } else {
      startPlayback();
    }
  }

  function readNumber(params, name, fallback) {
    const text = params.get(name);
    if (text === null) return fallback;
    const value = Number(text);
    return Number.isFinite(value) ? value : fallback;
  }

  function readUrl() {
    let params;
    try {
      params = new URLSearchParams(window.location.search);
    } catch (error) {
      return;
    }
    if (params.get("sig") === "0") state.useSig = false;
    else if (params.get("sig") === "1") state.useSig = true;
    const lambda = readNumber(params, "lambda", state.lambda);
    if (lambda > 0 && lambda <= 1) state.lambda = lambda;
    const noise = readNumber(params, "noise", state.noise);
    if (noise > 0 && noise <= 1) state.noise = noise;
    const seed = readNumber(params, "seed", state.seed);
    if (Number.isInteger(seed) && seed >= 0) state.seed = seed;
    const step = readNumber(params, "step", NaN);
    if (Number.isInteger(step) && step >= 0) state.pendingStep = step;
  }

  function updateUrl() {
    try {
      const url = new URL(window.location.href);
      url.searchParams.set("sig", state.useSig ? "1" : "0");
      url.searchParams.set("lambda", state.lambda.toFixed(2));
      url.searchParams.set("noise", state.noise.toFixed(2));
      url.searchParams.set("seed", String(state.seed));
      url.searchParams.set("step", String(state.step));
      window.history.replaceState(null, "", url.toString());
    } catch (error) {
      /* file:// in some browsers refuses replaceState; the page still works. */
    }
  }

  function bind() {
    for (const button of $("antiCollapse").querySelectorAll("button")) {
      button.addEventListener("click", () => {
        state.useSig = button.dataset.on === "true";
        retrain();
      });
    }

    $("lambda").addEventListener("input", (event) => {
      state.lambda = Number(event.target.value);
      $("lambdaOut").textContent = state.useSig ? state.lambda.toFixed(2) : "off";
    });
    $("lambda").addEventListener("change", (event) => {
      state.lambda = Number(event.target.value);
      retrain();
    });

    $("noise").addEventListener("input", (event) => {
      state.noise = Number(event.target.value);
      $("noiseOut").textContent = state.noise.toFixed(2);
    });
    $("noise").addEventListener("change", (event) => {
      state.noise = Number(event.target.value);
      retrain();
    });

    $("step").addEventListener("input", (event) => {
      setStep(Number(event.target.value));
    });

    $("playPause").addEventListener("click", togglePlayback);
    $("newData").addEventListener("click", () => {
      state.seed += 1;
      retrain();
    });

    document.addEventListener("keydown", (event) => {
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLSelectElement ||
        event.target instanceof HTMLButtonElement
      ) {
        return;
      }
      if (event.key === " ") {
        event.preventDefault();
        togglePlayback();
      } else if (event.key === "ArrowLeft") {
        setStep(state.step - 1);
      } else if (event.key === "ArrowRight") {
        setStep(state.step + 1);
      }
    });
  }

  function init() {
    readUrl();
    bind();
    retrain();
    if (state.pendingStep !== null) {
      state.step = Math.max(0, Math.min(run.snapshots.length - 1, state.pendingStep));
      state.pendingStep = null;
    } else {
      state.step = Math.min(state.step, run.snapshots.length - 1);
    }
    render();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
