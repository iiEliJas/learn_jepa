/* Per-image embedding card: four views, local layout, cosine matrix and L_pred. */
(function () {
  "use strict";

  const DATA = window.AtlasData;
  const Core = window.AtlasCore;
  const UI = window.AtlasUI;
  const Kit = window.CanvasKit;
  const $ = (id) => document.getElementById(id);

  const N = DATA.indices.length;
  const CLASS_NAMES = DATA.meta.class_names;
  const IMAGE_SPRITE = DATA.sprites.images;
  const VIEW_SPRITE = DATA.sprites.views;
  const V = DATA.meta.views;
  const K = DATA.spaces.proj.dim;
  const VIEWS = Core.decodeFloat32(DATA.views.data);
  const VIEW2D = Core.decodeFloat32(DATA.view2d.data);
  const sprites = {};
  const VIEW_COLORS = ["#2458d6", "#0f766e", "#c2610a", "#b42318", "#6d4bc3"];

  const state = {
    a: 0,
    b: 1,
    active: "a",
    classFilter: 0,
    offsets: {
      a: [3, 17, 41, 67],
      b: [5, 23, 59, 83],
    },
  };

  function fmt(value, digits) {
    return UI.fmt(value, digits);
  }

  function clamp(value, low, high) {
    return UI.clamp(value, low, high);
  }

  function chips(label, pred) {
    return label === pred
      ? '<span class="chip good">correct</span>'
      : '<span class="chip warn">misclassified</span>';
  }

  function imagesOfClass(classIndex) {
    const out = [];
    for (let index = 0; index < N; index += 1) {
      if (DATA.labels[index] === classIndex) out.push(index);
    }
    return out;
  }

  function view2d(index) {
    const out = [];
    for (let view = 0; view < V; view += 1) {
      const offset = (index * V + view) * 2;
      out.push([VIEW2D[offset], VIEW2D[offset + 1]]);
    }
    return out;
  }

  function meanCosine(flat) {
    const matrix = Core.pairwiseCosine(flat, V, K);
    let total = 0;
    let count = 0;
    for (let a = 0; a < V; a += 1) {
      for (let b = a + 1; b < V; b += 1) {
        total += matrix[a][b];
        count += 1;
      }
    }
    return { matrix, mean: count > 0 ? total / count : 0 };
  }

  function drawLocal(slot, index) {
    const canvas = $(`local${slot.toUpperCase()}`);
    const coords = view2d(index);
    const mu = [
      coords.reduce((sum, point) => sum + point[0], 0) / V,
      coords.reduce((sum, point) => sum + point[1], 0) / V,
    ];
    let minX = mu[0];
    let maxX = mu[0];
    let minY = mu[1];
    let maxY = mu[1];
    for (const point of coords) {
      minX = Math.min(minX, point[0]);
      maxX = Math.max(maxX, point[0]);
      minY = Math.min(minY, point[1]);
      maxY = Math.max(maxY, point[1]);
    }
    const spanX = Math.max(0.05, maxX - minX);
    const spanY = Math.max(0.05, maxY - minY);
    const padX = 0.18 * spanX + 0.05;
    const padY = 0.18 * spanY + 0.05;
    const plot = new Kit.Plot(canvas, {
      xmin: minX - padX,
      xmax: maxX + padX,
      ymin: minY - padY,
      ymax: maxY + padY,
      xlabel: "local PC1",
      ylabel: "local PC2",
      zeroLines: true,
      grid: true,
      padding: { l: 42, r: 16, t: 18, b: 36 },
    }).draw();

    plot.clip(() => {
      for (let view = 0; view < V; view += 1) {
        if (
          Math.abs(coords[view][0] - mu[0]) < 1e-9 &&
          Math.abs(coords[view][1] - mu[1]) < 1e-9
        ) {
          continue;
        }
        plot.arrow(coords[view][0], coords[view][1], mu[0], mu[1], {
          stroke: VIEW_COLORS[view % VIEW_COLORS.length],
          width: 1,
          head: 4,
          alpha: 0.55,
        });
      }
      for (let view = 0; view < V; view += 1) {
        plot.scatter([coords[view]], {
          r: 5.4,
          fill: VIEW_COLORS[view % VIEW_COLORS.length],
          stroke: "#ffffff",
          lineWidth: 1.2,
        });
        plot.text(coords[view][0], coords[view][1] + 0.055 * spanY, `v${view}`, {
          fill: VIEW_COLORS[view % VIEW_COLORS.length],
          font: "600 10px ui-sans-serif, system-ui, sans-serif",
          align: "center",
          background: "rgba(255,255,255,0.8)",
        });
      }
      plot.scatter([mu], {
        r: 4.4,
        fill: "#18202a",
        stroke: "#ffffff",
        lineWidth: 1.1,
      });
      plot.text(mu[0], mu[1] - 0.075 * spanY, "μ", {
        fill: "#18202a",
        font: "600 11px ui-sans-serif, system-ui, sans-serif",
        align: "center",
        background: "rgba(255,255,255,0.85)",
      });
    });
  }

  function cosineMatrixHtml(slot, matrix) {
    let html = '<div class="cos-matrix">';
    html += '<span class="cos-label"></span>';
    for (let view = 0; view < V; view += 1) {
      html += `<span class="cos-label">v${view}</span>`;
    }
    for (let row = 0; row < V; row += 1) {
      html += `<span class="cos-label">v${row}</span>`;
      for (let column = 0; column < V; column += 1) {
        const value = matrix[row][column];
        const strength = Math.min(1, Math.abs(value));
        const color =
          value >= 0
            ? `rgba(22, 121, 74, ${0.08 + 0.72 * strength})`
            : `rgba(180, 35, 24, ${0.08 + 0.72 * strength})`;
        const textColor = strength > 0.62 ? "#ffffff" : "#18202a";
        html +=
          `<span class="cos-cell${row === column ? " diagonal" : ""}"` +
          ` style="background:${color};color:${textColor}"` +
          ` title="cosine(v${row}, v${column}) = ${value.toFixed(3)}">${value.toFixed(
            2
          )}</span>`;
      }
    }
    html += "</div>";
    html += `<p class="small muted" style="margin:6px 0 0">Diagonal is 1 by construction. Cell (a,b) is the projector-space cosine between two views.</p>`;
    $(`cos${slot.toUpperCase()}`).innerHTML = html;
  }

  function barsHtml(slot, contributions) {
    const max = Math.max(1e-9, ...contributions);
    $(`bars${slot.toUpperCase()}`).innerHTML = contributions
      .map((value, view) => {
        const width = (100 * value) / max;
        return (
          `<div class="bar-row">` +
          `<span>v${view}</span>` +
          `<div class="bar-track"><div class="bar-fill" style="width:${width}%;background:${VIEW_COLORS[view % VIEW_COLORS.length]}"></div></div>` +
          `<span class="num">${fmt(value, 3)}</span>` +
          `</div>`
        );
      })
      .join("");
  }

  function renderViewStrip(slot, index, contributions) {
    const strip = $(`views${slot.toUpperCase()}`);
    strip.innerHTML = "";
    for (let view = 0; view < V; view += 1) {
      const figure = document.createElement("figure");
      figure.className = "view-tile";
      const canvas = UI.thumbNode(
        sprites.views,
        VIEW_SPRITE,
        index * V + view,
        64,
        "thumb view-thumb"
      );
      const caption = document.createElement("figcaption");
      caption.innerHTML = `<strong>v${view}</strong><span>${fmt(contributions[view], 3)}</span>`;
      figure.appendChild(canvas);
      figure.appendChild(caption);
      strip.appendChild(figure);
    }
  }

  function renderCard(slot, index) {
    const id = slot.toUpperCase();
    const label = DATA.labels[index];
    const pred = DATA.preds[index];
    const top = DATA.top3[index];
    const paired = Core.predictionForImage(VIEWS, N, V, K, index);
    const flat = Core.viewFlat(VIEWS, N, V, K, index);
    const cosines = meanCosine(flat);
    state[`paired${id}`] = paired.lPred;
    state[`cos${id}`] = cosines.mean;

    const card = $(`card${id}`);
    card.innerHTML =
      '<div class="image-card-head">' +
      `<canvas id="thumb${id}" class="thumb large"></canvas>` +
      "<div>" +
      `<h2>Image ${id} · ${CLASS_NAMES[label]}</h2>` +
      `<p class="small muted">test index ${DATA.indices[index]} · predicted <strong>${CLASS_NAMES[pred]}</strong> (${fmt(
        100 * top[0][1],
        0
      )}%)</p>` +
      `${chips(label, pred)}` +
      `<span class="small muted" style="margin-left:8px">top-3: ${top
        .map(([classIndex, probability]) => `${CLASS_NAMES[classIndex]} ${fmt(100 * probability, 0)}%`)
        .join(" · ")}</span>` +
      "</div></div>" +
      '<div class="view-strip" id="views' +
      id +
      '"></div>' +
      '<p class="small muted" style="margin:8px 0 2px">Four training-recipe views. The number under each tile is its per-view residual contribution to <span class="mono">L_pred</span>.</p>' +
      '<div class="plot-grid card-inner-grid">' +
      "<div>" +
      '<h3 class="inner-title">Local 2-D view</h3>' +
      '<p class="hint">PCA of the four 16-d points. Arrows point to <span class="mono">μ</span>; their squared length is the part of the prediction error visible in this 2-D shadow.</p>' +
      `<div class="canvas-wrap square"><canvas id="local${id}"></canvas></div>` +
      '<div class="legend">' +
      VIEW_COLORS.slice(0, V)
        .map(
          (color, view) =>
            `<span><span class="swatch" style="background:${color}"></span>v${view}</span>`
        )
        .join("") +
      '<span><span class="swatch" style="background:#18202a"></span>μ</span>' +
      "</div>" +
      "</div>" +
      "<div>" +
      '<h3 class="inner-title">Pairwise cosine</h3>' +
      '<p class="hint">Projector space, one 4×4 matrix. Values range from −1 to 1.</p>' +
      `<div id="cos${id}"></div>` +
      '<h3 class="inner-title" style="margin-top:14px">Per-view residual</h3>' +
      '<p class="hint">The mean of these four bars is <span class="mono">L_pred</span>.</p>' +
      `<div id="bars${id}"></div>` +
      "</div>" +
      "</div>" +
      '<div class="two-col card-stats">' +
      `<div class="stat accent"><span class="k">paired L_pred</span><span class="v">${fmt(
        paired.lPred,
        4
      )}</span><span class="s">same-image views</span></div>` +
      `<div class="stat warn"><span class="k">shuffled control</span><span class="v" id="shuffleValue${id}">–</span><span class="s" id="shuffleSub${id}">broken correspondence</span></div>` +
      "</div>" +
      '<div class="btn-row" style="margin-top:10px">' +
      `<button class="btn" id="reshuffle${id}" type="button">Reshuffle control</button>` +
      "</div>";

    UI.paintTile($(`thumb${id}`), sprites.images, IMAGE_SPRITE, index, 72);
    renderViewStrip(slot, index, paired.contributions);
    drawLocal(slot, index);
    cosineMatrixHtml(slot, cosines.matrix);
    barsHtml(slot, paired.contributions);
    updateShuffled(slot, index);
  }

  function updateShuffled(slot, index) {
    const id = slot.toUpperCase();
    const shuffled = Core.shuffledPredictionForImage(
      VIEWS,
      N,
      V,
      K,
      index,
      state.offsets[slot]
    );
    state[`shuffled${id}`] = shuffled.lPred;
    $(`shuffleValue${id}`).textContent = fmt(shuffled.lPred, 4);
    $(`shuffleSub${id}`).textContent = `views from images ${shuffled.picked
      .map((picked) => `#${DATA.indices[picked]}`)
      .join(", ")}`;
    $(`reshuffle${id}`).onclick = () => {
      state.offsets[slot] = Array.from({ length: V }, () => {
        let offset = 1 + Math.floor(Math.random() * (N - 1));
        if (offset === 0) offset = 1;
        return offset;
      });
      updateShuffled(slot, index);
      updateStats();
      renderCompare();
    };
  }

  function updateStats() {
    const a = state.pairedA || 0;
    const b = state.pairedB || 0;
    const sa = state.shuffledA || 0;
    const sb = state.shuffledB || 0;
    $("statPairedA").textContent = fmt(a, 4);
    $("statPairedB").textContent = fmt(b, 4);
    $("statShuffleA").textContent = fmt(sa, 4);
    $("statShuffleB").textContent = fmt(sb, 4);
    $("statPairedASub").textContent = `${CLASS_NAMES[DATA.labels[state.a]]} · test #${DATA.indices[state.a]}`;
    $("statPairedBSub").textContent = `${CLASS_NAMES[DATA.labels[state.b]]} · test #${DATA.indices[state.b]}`;
    $("statReference").textContent = `${fmt(DATA.meta.prediction_loss.paired, 3)} / ${fmt(
      DATA.meta.prediction_loss.shuffled,
      3
    )}`;
    $("statReferenceSub").textContent = `N = ${N}, paired / shuffled`;
    $("referenceNote").innerHTML =
      `Whole subset: paired <span class="mono">${fmt(
        DATA.meta.prediction_loss.paired,
        4
      )}</span>, shuffled <span class="mono">${fmt(
        DATA.meta.prediction_loss.shuffled,
        4
      )}</span>, ratio <span class="mono">${fmt(
        DATA.meta.prediction_loss.shuffled / DATA.meta.prediction_loss.paired,
        1
      )}×</span>. The selected-image controls use the same projection convention, so they are directly comparable.`;
  }

  function renderCompare() {
    const rows = ["a", "b"].map((slot) => {
      const id = slot.toUpperCase();
      const index = state[slot];
      const label = DATA.labels[index];
      const pred = DATA.preds[index];
      const paired = state[`paired${id}`] || 0;
      const shuffled = state[`shuffled${id}`] || 0;
      const ratio = paired > 0 ? shuffled / paired : Infinity;
      return (
        "<tr>" +
        `<td>${id}</td>` +
        `<td>${CLASS_NAMES[label]}</td>` +
        `<td>${CLASS_NAMES[pred]}</td>` +
        `<td>${fmt(paired, 4)}</td>` +
        `<td>${fmt(shuffled, 4)}</td>` +
        `<td>${isFinite(ratio) ? `${fmt(ratio, 2)}×` : "–"}</td>` +
        `<td>${fmt(state[`cos${id}`] || 0, 3)}</td>` +
        "</tr>"
      );
    });
    $("compareBody").innerHTML = rows.join("");
  }

  function syncControls() {
    for (const button of $("slot").querySelectorAll("button")) {
      button.setAttribute("aria-pressed", String(button.dataset.slot === state.active));
    }
    $("classPick").value = String(state.classFilter);
    $("openAtlas").href = `../cifar_atlas/index.html?image=${state.a}`;
  }

  function renderPicker() {
    const grid = $("pickGrid");
    const images = imagesOfClass(state.classFilter);
    grid.innerHTML = "";
    for (const index of images) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "thumb-button";
      button.title = `test index ${DATA.indices[index]} · predicted ${CLASS_NAMES[DATA.preds[index]]}`;
      button.appendChild(UI.thumbNode(sprites.images, IMAGE_SPRITE, index, 42, "thumb"));
      const caption = document.createElement("span");
      caption.textContent = `#${DATA.indices[index]}`;
      button.appendChild(caption);
      if (index === state.a) button.classList.add("active-a");
      if (index === state.b) button.classList.add("active-b");
      button.addEventListener("click", () => setSlot(state.active, index));
      grid.appendChild(button);
    }
  }

  function setSlot(slot, index) {
    state[slot] = clamp(index, 0, N - 1);
    state.classFilter = DATA.labels[state[slot]];
    syncControls();
    renderPicker();
    renderCard(slot, state[slot]);
    updateStats();
    renderCompare();
    updateUrl();
  }

  function updateUrl() {
    try {
      const url = new URL(window.location.href);
      url.searchParams.set("image", String(state.a));
      url.searchParams.set("b", String(state.b));
      window.history.replaceState(null, "", url.toString());
    } catch (error) {
      /* file:// may refuse replaceState; the page still works. */
    }
  }

  function step(direction) {
    const images = imagesOfClass(state.classFilter);
    if (!images.length) return;
    const current = state[state.active];
    let position = images.indexOf(current);
    if (position < 0) position = 0;
    position = (position + direction + images.length) % images.length;
    setSlot(state.active, images[position]);
  }

  function randomFromClass() {
    const images = imagesOfClass(state.classFilter);
    if (!images.length) return;
    setSlot(state.active, images[Math.floor(Math.random() * images.length)]);
  }

  function chooseInitialPair() {
    let firstError = null;
    for (let index = 0; index < N; index += 1) {
      if (DATA.labels[index] !== DATA.preds[index]) {
        firstError = index;
        break;
      }
    }
    let firstCorrect = null;
    const errorClass = firstError === null ? null : DATA.labels[firstError];
    for (let index = 0; index < N; index += 1) {
      if (DATA.labels[index] === DATA.preds[index] && DATA.labels[index] !== errorClass) {
        firstCorrect = index;
        break;
      }
    }
    state.a = firstError === null ? 0 : firstError;
    state.b = firstCorrect === null ? Math.min(1, N - 1) : firstCorrect;
    state.classFilter = DATA.labels[state.a];
  }

  function readUrl() {
    const params = new URLSearchParams(window.location.search);
    const a = Number(params.get("image"));
    const b = Number(params.get("b"));
    if (Number.isInteger(a) && a >= 0 && a < N) state.a = a;
    if (Number.isInteger(b) && b >= 0 && b < N) state.b = b;
    state.classFilter = DATA.labels[state.a];
  }

  function bind() {
    for (const button of $("slot").querySelectorAll("button")) {
      button.addEventListener("click", () => {
        state.active = button.dataset.slot;
        state.classFilter = DATA.labels[state[state.active]];
        syncControls();
        renderPicker();
      });
    }
    $("classPick").addEventListener("change", (event) => {
      state.classFilter = Number(event.target.value);
      renderPicker();
    });
    $("prevPick").addEventListener("click", () => step(-1));
    $("nextPick").addEventListener("click", () => step(1));
    $("randomPick").addEventListener("click", randomFromClass);
    $("swapCards").addEventListener("click", () => {
      const temporary = state.a;
      state.a = state.b;
      state.b = temporary;
      const offsets = state.offsets.a;
      state.offsets.a = state.offsets.b;
      state.offsets.b = offsets;
      state.classFilter = DATA.labels[state.a];
      syncControls();
      renderPicker();
      renderCard("a", state.a);
      renderCard("b", state.b);
      updateStats();
      renderCompare();
      updateUrl();
    });
    document.addEventListener("keydown", (event) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) {
        return;
      }
      if (event.key === "a" || event.key === "A") {
        state.active = "a";
        state.classFilter = DATA.labels[state.a];
        syncControls();
        renderPicker();
      } else if (event.key === "b" || event.key === "B") {
        state.active = "b";
        state.classFilter = DATA.labels[state.b];
        syncControls();
        renderPicker();
      }
    });
    $("openAtlas").addEventListener("click", () => {
      $("openAtlas").href = `../cifar_atlas/index.html?image=${state.a}`;
    });
  }

  function init() {
    $("footerCommit").textContent = DATA.meta.git_commit || "unknown";
    const sha = DATA.meta.checkpoint_sha256 || "";
    $("footerSha").textContent = sha ? `${sha.slice(0, 12)}…` : "unknown";
    $("footerSha").title = sha;
    $("classPick").innerHTML = CLASS_NAMES.map(
      (name, index) => `<option value="${index}">${name}</option>`
    ).join("");
    chooseInitialPair();
    readUrl();
    bind();
    UI.loadSprites(DATA, (error, loaded) => {
      if (error) {
        $("cardA").innerHTML = `<p class="small muted">Could not load the sprite sheets: ${error.message}</p>`;
        return;
      }
      Object.assign(sprites, loaded);
      syncControls();
      renderPicker();
      renderCard("a", state.a);
      renderCard("b", state.b);
      updateStats();
      renderCompare();
      updateUrl();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
