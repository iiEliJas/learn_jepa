/* CIFAR Embedding Atlas: canvas map, filters, confusion drill-down and cards. */
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
  const spaceCache = {};
  const sprites = {};

  const state = {
    space: "head",
    centered: true,
    projection: DATA.projections.head.centered.tsne ? "tsne" : "pca",
    filter: "all",
    classFilter: 0,
    cell: null,
    selected: null,
    hover: null,
    neighbors: [],
    screen: null,
    mask: null,
  };

  const SPACE_LABELS = {
    cls: "raw ViT CLS (192-d)",
    head: "proj_head (512-d)",
    proj: "projector (16-d)",
  };

  function fmt(value, digits) {
    return UI.fmt(value, digits);
  }

  function getSpace(name, centered) {
    const key = `${name}:${centered ? "centered" : "raw"}`;
    if (!spaceCache[key]) {
      const spec = DATA.spaces[name];
      let rows = Core.decodeMatrix(spec);
      if (centered) rows = Core.centerRows(rows, N, spec.dim);
      spaceCache[key] = {
        rows,
        dim: spec.dim,
        norms: Core.rowNorms(rows, N, spec.dim),
      };
    }
    return spaceCache[key];
  }

  function projectionEntry() {
    const entry = DATA.projections[state.space][state.centered ? "centered" : "raw"];
    const method = entry[state.projection] ? state.projection : "pca";
    if (method !== state.projection) state.projection = method;
    return { method, data: entry[method], all: entry };
  }

  function makeMask() {
    const mask = new Uint8Array(N);
    let count = 0;
    for (let index = 0; index < N; index += 1) {
      let visible = true;
      if (state.filter === "errors") visible = DATA.labels[index] !== DATA.preds[index];
      else if (state.filter === "class") visible = DATA.labels[index] === state.classFilter;
      else if (state.filter === "cell" && state.cell) {
        visible =
          DATA.labels[index] === state.cell[0] && DATA.preds[index] === state.cell[1];
      }
      mask[index] = visible ? 1 : 0;
      if (visible) count += 1;
    }
    return { mask, count };
  }

  function computeNeighbors() {
    if (state.selected === null) return [];
    const space = getSpace(state.space, state.centered);
    return Core.nearestNeighbors(
      space.rows,
      N,
      space.dim,
      state.selected,
      8,
      space.norms
    );
  }

  function buildLegend() {
    const legend = $("mapLegend");
    legend.innerHTML = "";
    const items = CLASS_NAMES.map((name, index) => {
      const item = document.createElement("span");
      item.className = "legend-marker";
      const canvas = document.createElement("canvas");
      canvas.width = 14;
      canvas.height = 14;
      canvas.style.width = "14px";
      canvas.style.height = "14px";
      const ctx = canvas.getContext("2d");
      UI.drawMarker(ctx, 7, 7, 4.2, UI.classShape(index), UI.classColor(index), null);
      const label = document.createElement("span");
      label.textContent = name;
      label.title = `marker shape and colour for ${name}`;
      item.appendChild(canvas);
      item.appendChild(label);
      return item;
    });
    for (const item of items) legend.appendChild(item);
    const selected = document.createElement("span");
    selected.className = "legend-marker";
    selected.innerHTML = '<span class="ring-swatch"></span>selected';
    legend.appendChild(selected);
    const neighbor = document.createElement("span");
    neighbor.className = "legend-marker";
    neighbor.innerHTML = '<span class="dash-swatch"></span>nearest neighbours';
    legend.appendChild(neighbor);
  }

  function draw() {
    const canvas = $("atlasCanvas");
    if (!canvas) return;
    const projection = projectionEntry();
    const coords = projection.data.xy;
    const { mask, count } = makeMask();
    state.mask = mask;

    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const point of coords) {
      minX = Math.min(minX, point[0]);
      maxX = Math.max(maxX, point[0]);
      minY = Math.min(minY, point[1]);
      maxY = Math.max(maxY, point[1]);
    }
    const spanX = Math.max(0.02, maxX - minX);
    const spanY = Math.max(0.02, maxY - minY);
    const padX = 0.06 * spanX + 0.01;
    const padY = 0.06 * spanY + 0.01;

    const isTsne = projection.method === "tsne";
    const plot = new Kit.Plot(canvas, {
      xmin: minX - padX,
      xmax: maxX + padX,
      ymin: minY - padY,
      ymax: maxY + padY,
      xlabel: isTsne ? "t-SNE 1" : "PC1",
      ylabel: isTsne ? "t-SNE 2" : "PC2",
      grid: !isTsne,
      zeroLines: !isTsne,
      xTicks: isTsne ? [] : null,
      yTicks: isTsne ? [] : null,
      padding: { l: isTsne ? 44 : 52, r: 18, t: 22, b: 38 },
    }).draw();

    const ctx = plot.ctx;
    const screen = new Float32Array(N * 2);
    for (let index = 0; index < N; index += 1) {
      screen[index * 2] = plot.x(coords[index][0]);
      screen[index * 2 + 1] = plot.y(coords[index][1]);
    }
    state.screen = screen;

    // Draw the dimmed context first, then the matching points on top.
    plot.clip(() => {
      for (let index = 0; index < N; index += 1) {
        if (mask[index]) continue;
        const x = screen[index * 2];
        const y = screen[index * 2 + 1];
        UI.drawMarker(ctx, x, y, 2.1, UI.classShape(DATA.labels[index]), "#b9c3cf", null);
      }
      ctx.globalAlpha = 0.9;
      for (let index = 0; index < N; index += 1) {
        if (!mask[index]) continue;
        const x = screen[index * 2];
        const y = screen[index * 2 + 1];
        const radius = index === state.selected ? 4.6 : 3.4;
        UI.drawMarker(
          ctx,
          x,
          y,
          radius,
          UI.classShape(DATA.labels[index]),
          UI.classColor(DATA.labels[index]),
          "rgba(255,255,255,0.75)",
          0.7
        );
      }
      ctx.globalAlpha = 1;

      if (state.neighbors.length) {
        ctx.save();
        ctx.setLineDash([3, 2]);
        for (const neighbor of state.neighbors) {
          const index = neighbor.index;
          if (!mask[index]) continue;
          UI.drawMarker(
            ctx,
            screen[index * 2],
            screen[index * 2 + 1],
            5.8,
            "circle",
            null,
            "#18202a",
            1.2
          );
        }
        ctx.restore();
      }

      if (state.selected !== null) {
        UI.drawMarker(
          ctx,
          screen[state.selected * 2],
          screen[state.selected * 2 + 1],
          7.2,
          "circle",
          null,
          "#18202a",
          2.2
        );
        UI.drawMarker(
          ctx,
          screen[state.selected * 2],
          screen[state.selected * 2 + 1],
          2.2,
          "circle",
          "#ffffff",
          null
        );
      }

      if (state.hover !== null) {
        UI.drawMarker(
          ctx,
          screen[state.hover * 2],
          screen[state.hover * 2 + 1],
          6.2,
          "circle",
          null,
          "#2458d6",
          1.8
        );
      }
    });

    $("mapTitle").textContent = `${SPACE_LABELS[state.space]} · ${
      state.centered ? "mean-centered" : "raw"
    } · ${projection.method.toUpperCase()}`;
    $("mapCount").textContent = `${count.toLocaleString()} shown`;
    return { count, projection };
  }

  function scheduleDraw() {
    if (scheduleDraw.pending) return;
    scheduleDraw.pending = true;
    window.requestAnimationFrame(() => {
      scheduleDraw.pending = false;
      draw();
    });
  }

  function updateStats() {
    const { mask, count } = makeMask();
    let correct = 0;
    for (let index = 0; index < N; index += 1) {
      if (mask[index] && DATA.labels[index] === DATA.preds[index]) correct += 1;
    }
    const accuracy = count > 0 ? correct / count : 0;
    $("statShown").textContent = count.toLocaleString();
    $("statShownSub").textContent = `of ${N.toLocaleString()} test images`;
    $("statProbe").textContent = `${(100 * DATA.meta.probe_accuracy).toFixed(1)}%`;
    $("statProbeSub").textContent = `shown accuracy ${(100 * accuracy).toFixed(1)}%`;
    const knn = DATA.meta.knn_accuracy[state.space][state.centered ? "centered" : "raw"];
    $("statKnn").textContent = `${(100 * knn).toFixed(1)}%`;
    $("statKnnSub").textContent = `k=20 · ${state.centered ? "centered" : "raw"}`;
    $("statAniso").textContent = fmt(DATA.meta.mean_direction_ratio[state.space], 3);
    $("statAnisoSub").textContent = "raw ||mean|| / mean ||x||";
    const projection = projectionEntry();
    if (projection.method === "pca") {
      const explained = projection.data.explained || [0, 0];
      $("statPca").textContent = `${(100 * (explained[0] + explained[1])).toFixed(1)}%`;
      $("statPcaSub").textContent = "variance in PC1 + PC2";
    } else {
      $("statPca").textContent = "t-SNE";
      $("statPcaSub").textContent = "precomputed, perplexity 30";
    }
    $("statPca").title =
      "PCA is linear and keeps the axes interpretable. t-SNE is non-linear and only preserves neighbourhoods.";
  }

  function updateControls() {
    $("space").value = state.space;
    $("projection").value = state.projection;
    $("filter").value = state.filter;
    $("classFilter").value = String(state.classFilter);
    $("classFilter").disabled = state.filter !== "class";
    for (const button of $("centering").querySelectorAll("button")) {
      button.setAttribute("aria-pressed", String(button.dataset.centered === String(state.centered)));
    }
  }

  function updateSelected() {
    const card = $("selectedCard");
    const link = $("openCard");
    if (state.selected === null) {
      card.innerHTML =
        '<p class="small muted">Click a point on the map to lock an image, or use the confusion cell thumbnails.</p>';
      link.classList.add("disabled");
      link.setAttribute("aria-disabled", "true");
      link.href = "#";
      return;
    }
    const index = state.selected;
    const label = DATA.labels[index];
    const pred = DATA.preds[index];
    const top = DATA.top3[index];
    card.innerHTML =
      '<div class="selected-inline">' +
      '<canvas id="selectedThumb" class="thumb large"></canvas>' +
      "<div>" +
      `<strong>${CLASS_NAMES[label]}</strong>` +
      `<div class="small muted">test index ${DATA.indices[index]}</div>` +
      `<div class="small ${label === pred ? "chip good" : "chip warn"}">pred ${CLASS_NAMES[pred]}</div>` +
      '<div class="small muted" id="selectedTop3"></div>' +
      "</div></div>";
    UI.paintTile($("selectedThumb"), sprites.images, IMAGE_SPRITE, index, 72);
    $("selectedTop3").textContent = top
      .map(([classIndex, probability]) => `${CLASS_NAMES[classIndex]} ${(100 * probability).toFixed(0)}%`)
      .join(" · ");
    link.classList.remove("disabled");
    link.removeAttribute("aria-disabled");
    link.href = `../per_image/index.html?image=${index}`;
  }

  function updateNeighbors() {
    const grid = $("neighborGrid");
    const stats = $("neighborStats");
    state.neighbors = computeNeighbors();
    grid.innerHTML = "";
    if (state.selected === null || state.neighbors.length === 0) {
      stats.textContent =
        "Neighbours are computed in the selected space with cosine similarity. Pick a point first.";
      $("neighborHint").textContent = "Pick a point to see the eight closest embeddings.";
      return;
    }
    const selectedLabel = DATA.labels[state.selected];
    const neighborIndices = state.neighbors.map((neighbor) => neighbor.index);
    const purity = Core.classPurity(neighborIndices, DATA.labels, selectedLabel);
    for (const neighbor of state.neighbors) {
      const index = neighbor.index;
      const button = document.createElement("button");
      button.className = "thumb-button";
      button.type = "button";
      button.title = `${CLASS_NAMES[DATA.labels[index]]} · cosine ${neighbor.score.toFixed(3)} · click to select`;
      const canvas = UI.thumbNode(sprites.images, IMAGE_SPRITE, index, 42, "thumb");
      button.appendChild(canvas);
      const caption = document.createElement("span");
      caption.textContent = `${CLASS_NAMES[DATA.labels[index]]} ${neighbor.score.toFixed(2)}`;
      button.appendChild(caption);
      if (DATA.labels[index] !== selectedLabel) button.classList.add("mismatch");
      button.addEventListener("click", () => selectImage(index));
      grid.appendChild(button);
    }
    $("neighborHint").textContent = `Eight nearest neighbours in ${
      SPACE_LABELS[state.space]
    } (${state.centered ? "mean-centered" : "raw"}), cosine similarity.`;
    stats.innerHTML =
      `Class purity <span class="mono">${(100 * purity).toFixed(0)}%</span> — ` +
      `${purity === 1 ? "every neighbour has the same label." : "neighbours include other labels."} ` +
      `The selected point is <span class="mono">${CLASS_NAMES[selectedLabel]}</span>.`;
  }

  function buildConfusion() {
    const matrix = Core.confusionMatrix(DATA.labels, DATA.preds, 10);
    const max = Math.max(...matrix.flat(), 1);
    const container = $("confusionMatrix");
    container.innerHTML = "";
    const corner = document.createElement("div");
    corner.className = "confusion-corner";
    corner.textContent = "true ↓ / pred →";
    container.appendChild(corner);
    for (let predicted = 0; predicted < 10; predicted += 1) {
      const head = document.createElement("div");
      head.className = "confusion-head";
      head.textContent = CLASS_NAMES[predicted].slice(0, 4);
      head.title = `predicted ${CLASS_NAMES[predicted]}`;
      container.appendChild(head);
    }
    for (let actual = 0; actual < 10; actual += 1) {
      const rowHead = document.createElement("div");
      rowHead.className = "confusion-row-head";
      rowHead.textContent = CLASS_NAMES[actual];
      container.appendChild(rowHead);
      for (let predicted = 0; predicted < 10; predicted += 1) {
        const count = matrix[actual][predicted];
        const button = document.createElement("button");
        button.type = "button";
        button.className = "confusion-cell";
        button.dataset.actual = String(actual);
        button.dataset.predicted = String(predicted);
        button.textContent = String(count);
        const strength = count / max;
        const color =
          actual === predicted
            ? `rgba(22, 121, 74, ${0.1 + 0.75 * strength})`
            : `rgba(180, 35, 24, ${0.05 + 0.75 * strength})`;
        button.style.background = color;
        button.style.color = strength > 0.52 ? "#ffffff" : "#18202a";
        button.setAttribute(
          "aria-label",
          `${count} images of ${CLASS_NAMES[actual]} predicted as ${CLASS_NAMES[predicted]}`
        );
        button.title = `${CLASS_NAMES[actual]} → ${CLASS_NAMES[predicted]}: ${count}`;
        button.addEventListener("click", () => setCell(actual, predicted));
        if (
          state.cell &&
          state.cell[0] === actual &&
          state.cell[1] === predicted &&
          state.filter === "cell"
        ) {
          button.classList.add("active");
        }
        container.appendChild(button);
      }
    }
  }

  function updateCellDetail() {
    const detail = $("cellDetail");
    if (!state.cell || state.filter !== "cell") {
      detail.innerHTML =
        '<p class="small muted">Click a cell to show the images behind that prediction.</p>';
      return;
    }
    const [actual, predicted] = state.cell;
    const matches = [];
    for (let index = 0; index < N; index += 1) {
      if (DATA.labels[index] === actual && DATA.preds[index] === predicted) matches.push(index);
    }
    if (matches.length === 0) {
      detail.innerHTML = `<p class="small muted">No ${CLASS_NAMES[actual]} → ${CLASS_NAMES[predicted]} errors in this subset.</p>`;
      return;
    }
    detail.innerHTML = `<p class="small"><strong>${matches.length}</strong> images of <strong>${
      CLASS_NAMES[actual]
    }</strong> predicted as <strong>${CLASS_NAMES[predicted]}</strong>. Click a thumbnail.</p>`;
    const grid = document.createElement("div");
    grid.className = "thumb-grid";
    for (const index of matches.slice(0, 12)) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "thumb-button";
      button.appendChild(UI.thumbNode(sprites.images, IMAGE_SPRITE, index, 42, "thumb"));
      const caption = document.createElement("span");
      caption.textContent = `#${DATA.indices[index]}`;
      button.appendChild(caption);
      button.title = `test index ${DATA.indices[index]} · click to select`;
      button.addEventListener("click", () => selectImage(index));
      grid.appendChild(button);
    }
    detail.appendChild(grid);
  }

  function setCell(actual, predicted) {
    state.cell = [actual, predicted];
    state.filter = "cell";
    let representative = null;
    for (let index = 0; index < N; index += 1) {
      if (DATA.labels[index] === actual && DATA.preds[index] === predicted) {
        representative = index;
        break;
      }
    }
    if (representative !== null) state.selected = representative;
    buildConfusion();
    updateControls();
    updateCellDetail();
    render();
  }

  function selectImage(index) {
    state.selected = UI.clamp(index, 0, N - 1);
    updateUrl();
    render();
  }

  function updateUrl() {
    try {
      const url = new URL(window.location.href);
      if (state.selected === null) url.searchParams.delete("image");
      else url.searchParams.set("image", String(state.selected));
      window.history.replaceState(null, "", url.toString());
    } catch (error) {
      /* file:// in some browsers refuses replaceState; the page still works. */
    }
  }

  function hideTooltip() {
    const tooltip = $("atlasTooltip");
    tooltip.hidden = true;
    if (state.hover !== null) {
      state.hover = null;
      scheduleDraw();
    }
  }

  function showTooltip(index, x, y) {
    const tooltip = $("atlasTooltip");
    const wrap = $("atlasWrap");
    const label = DATA.labels[index];
    const pred = DATA.preds[index];
    tooltip.hidden = false;
    tooltip.innerHTML =
      '<canvas id="tooltipThumb" class="thumb"></canvas>' +
      `<div><strong>${CLASS_NAMES[label]}</strong>` +
      `<div class="small muted">pred ${CLASS_NAMES[pred]} · test #${DATA.indices[index]}</div></div>`;
    UI.paintTile($("tooltipThumb"), sprites.images, IMAGE_SPRITE, index, 58);
    const width = 210;
    const left = UI.clamp(x + 16, 8, Math.max(8, wrap.clientWidth - width - 8));
    const top = UI.clamp(y + 16, 8, Math.max(8, wrap.clientHeight - 84));
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  }

  function findPoint(event) {
    if (!state.screen) return null;
    const canvas = $("atlasCanvas");
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    let best = null;
    let bestDistance = 13 * 13;
    for (let index = 0; index < N; index += 1) {
      if (state.mask && !state.mask[index]) continue;
      const dx = state.screen[index * 2] - x;
      const dy = state.screen[index * 2 + 1] - y;
      const distance = dx * dx + dy * dy;
      if (distance < bestDistance) {
        bestDistance = distance;
        best = { index, x: x, y: y };
      }
    }
    return best;
  }

  function bindCanvas() {
    const canvas = $("atlasCanvas");
    canvas.addEventListener("mousemove", (event) => {
      const point = findPoint(event);
      if (!point) {
        hideTooltip();
        canvas.style.cursor = "default";
        return;
      }
      canvas.style.cursor = "pointer";
      if (state.hover !== point.index) {
        state.hover = point.index;
        scheduleDraw();
      }
      showTooltip(point.index, point.x, point.y);
    });
    canvas.addEventListener("mouseleave", () => {
      hideTooltip();
      canvas.style.cursor = "default";
    });
    canvas.addEventListener("click", (event) => {
      const point = findPoint(event);
      if (point) selectImage(point.index);
    });
  }

  function render(rebuildConfusion) {
    if (rebuildConfusion) buildConfusion();
    state.neighbors = computeNeighbors();
    updateControls();
    updateStats();
    updateSelected();
    updateNeighbors();
    updateCellDetail();
    draw();
  }

  function init() {
    $("footerCommit").textContent = DATA.meta.git_commit || "unknown";
    const sha = DATA.meta.checkpoint_sha256 || "";
    $("footerSha").textContent = sha ? `${sha.slice(0, 12)}…` : "unknown";
    $("footerSha").title = sha;
    const params = new URLSearchParams(window.location.search);
    const requested = Number(params.get("image"));
    if (Number.isInteger(requested) && requested >= 0 && requested < N) {
      state.selected = requested;
    } else {
      let firstError = null;
      for (let index = 0; index < N; index += 1) {
        if (DATA.labels[index] !== DATA.preds[index]) {
          firstError = index;
          break;
        }
      }
      state.selected = firstError === null ? 0 : firstError;
    }

    // Build class options.
    $("classFilter").innerHTML = CLASS_NAMES.map(
      (name, index) => `<option value="${index}">${name}</option>`
    ).join("");
    $("classFilter").value = String(state.classFilter);

    // Hide the t-SNE option if this build did not precompute it.
    if (!DATA.projections.head.centered.tsne) {
      const option = $("projection").querySelector('option[value="tsne"]');
      if (option) option.disabled = true;
      state.projection = "pca";
    }

    $("space").addEventListener("change", (event) => {
      state.space = event.target.value;
      const entry = DATA.projections[state.space][state.centered ? "centered" : "raw"];
      if (!entry[state.projection]) state.projection = "pca";
      render();
    });
    $("projection").addEventListener("change", (event) => {
      state.projection = event.target.value;
      render();
    });
    for (const button of $("centering").querySelectorAll("button")) {
      button.addEventListener("click", () => {
        state.centered = button.dataset.centered === "true";
        const entry = DATA.projections[state.space][state.centered ? "centered" : "raw"];
        if (!entry[state.projection]) state.projection = "pca";
        render();
      });
    }
    $("filter").addEventListener("change", (event) => {
      state.filter = event.target.value;
      if (state.filter === "cell" && !state.cell) state.filter = "all";
      buildConfusion();
      render();
    });
    $("classFilter").addEventListener("change", (event) => {
      state.classFilter = Number(event.target.value);
      state.filter = "class";
      buildConfusion();
      render();
    });
    $("resetView").addEventListener("click", () => {
      state.space = "head";
      state.centered = true;
      state.projection = DATA.projections.head.centered.tsne ? "tsne" : "pca";
      state.filter = "all";
      state.classFilter = 0;
      state.cell = null;
      buildConfusion();
      render();
    });
    $("openCard").addEventListener("click", (event) => {
      if (state.selected === null) event.preventDefault();
    });
    window.addEventListener("resize", scheduleDraw);
    bindCanvas();

    buildLegend();
    buildConfusion();
    UI.loadSprites(DATA, (error, loaded) => {
      if (error) {
        $("mapTitle").textContent = "Sprite sheets could not be loaded";
        return;
      }
      Object.assign(sprites, loaded);
      render();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
