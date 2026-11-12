/* Three-Space Lens: one image and its neighbours in CLS, head and projector space. */
(function () {
  "use strict";

  const DATA = window.AtlasData;
  const Core = window.AtlasCore;
  const UI = window.AtlasUI;
  const $ = (id) => document.getElementById(id);

  const N = DATA.indices.length;
  const CLASS_NAMES = DATA.meta.class_names;
  const IMAGE_SPRITE = DATA.sprites.images;
  const spaceCache = {};
  const sprites = {};

  const SPACES = [
    {
      key: "cls",
      label: "raw ViT CLS",
      short: "CLS",
      cardId: "spaceCardCls",
      statId: "statCls",
      dim: DATA.spaces.cls.dim,
      role: "encoder token",
      color: "#6d4bc3",
      note:
        "The encoder's output before <span class=\"mono\">proj_head</span>. No label " +
        "training touches it directly.",
    },
    {
      key: "head",
      label: "proj_head",
      short: "head",
      cardId: "spaceCardHead",
      statId: "statHead",
      dim: DATA.spaces.head.dim,
      role: "probe feature",
      color: "#2458d6",
      note:
        "A linear map trained by the LeJEPA objective. The probe reads this space, " +
        "so it is the most semantic of the three.",
    },
    {
      key: "proj",
      label: "projector",
      short: "proj",
      cardId: "spaceCardProj",
      statId: "statProj",
      dim: DATA.spaces.proj.dim,
      role: "prediction + SIGReg",
      color: "#0f766e",
      note:
        "The 16-d output where <span class=\"mono\">L_pred</span> and SIGReg live. " +
        "It is diffuse by design, not trained to classify.",
    },
  ];

  const state = {
    image: 0,
    centered: true,
    k: 8,
    classFilter: 0,
  };

  function fmt(value, digits) {
    return UI.fmt(value, digits);
  }

  function pct(value, digits) {
    return `${(100 * value).toFixed(digits === undefined ? 0 : digits)}%`;
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

  function computeLens(space) {
    const data = getSpace(space.key, state.centered);
    const neighbors = Core.nearestNeighbors(
      data.rows,
      N,
      data.dim,
      state.image,
      state.k,
      data.norms
    );
    const indices = neighbors.map((neighbor) => neighbor.index);
    const trueLabel = DATA.labels[state.image];
    const purity = Core.classPurity(indices, DATA.labels, trueLabel);
    const meanCosine = Core.meanNeighborScore(neighbors);
    const vote = Core.neighborVote(neighbors, DATA.labels);
    return { space, neighbors, indices, purity, meanCosine, vote };
  }

  function renderStats(results) {
    const index = state.image;
    const label = DATA.labels[index];
    const pred = DATA.preds[index];
    const top = DATA.top3[index];
    $("statTrue").textContent = CLASS_NAMES[label];
    $("statTrueSub").textContent = `test index ${DATA.indices[index]}`;
    $("statPred").textContent = CLASS_NAMES[pred];
    $("statPredSub").textContent = `${pct(top[0][1], 1)} top-1${
      label === pred ? " · correct" : " · misclassified"
    }`;
    for (const space of SPACES) {
      const node = $(space.statId);
      if (!node) continue;
      node.textContent = pct(results[space.key].purity);
      const box = node.parentElement;
      if (box && box.classList) {
        box.classList.toggle("good", results[space.key].purity >= 0.75);
      }
    }
  }

  function renderBanner() {
    const index = state.image;
    const label = DATA.labels[index];
    const pred = DATA.preds[index];
    const top = DATA.top3[index];
    const banner = $("selectedBanner");
    banner.innerHTML =
      '<div class="selected-inline">' +
      '<canvas id="bannerThumb" class="thumb xlarge"></canvas>' +
      '<div class="selected-copy">' +
      `<h2>${CLASS_NAMES[label]}</h2>` +
      `<p class="small muted">test index ${DATA.indices[index]} · predicted ` +
      `<strong>${CLASS_NAMES[pred]}</strong> ${label === pred ? "correctly" : "incorrectly"}` +
      ` (${pct(top[0][1], 1)})</p>` +
      '<div class="banner-row">' +
      `<span class="chip ${label === pred ? "good" : "warn"}">${
        label === pred ? "probe correct" : "probe error"
      }</span>` +
      `<span class="small muted">top-3: ${top
        .map(([classIndex, probability]) => `${CLASS_NAMES[classIndex]} ${pct(probability)}`)
        .join(" · ")}</span>` +
      "</div></div>" +
      '<div class="banner-links">' +
      `<a class="btn" href="../cifar_atlas/index.html?image=${index}">Atlas map →</a>` +
      `<a class="btn" href="../per_image/index.html?image=${index}&b=${(index + 1) % N}">View card →</a>` +
      "</div>" +
      "</div>";
    UI.paintTile($("bannerThumb"), sprites.images, IMAGE_SPRITE, index, 84);
  }

  function renderSpaceCard(result) {
    const space = result.space;
    const card = $(result.space.cardId);
    const label = DATA.labels[state.image];
    const purityClass = result.purity >= 0.75 ? "good" : result.purity < 0.5 ? "warn" : "";
    const voteClass = result.vote.label === label ? "good" : "warn";
    const knn = DATA.meta.knn_accuracy[space.key][state.centered ? "centered" : "raw"];
    const ratio = DATA.meta.mean_direction_ratio[space.key];

    card.style.setProperty("--space-color", space.color);
    card.innerHTML =
      '<div class="space-head">' +
      '<span class="space-dot"></span>' +
      '<div class="space-head-copy">' +
      `<h3>${space.label}</h3>` +
      `<p class="small muted">${space.dim}-d · ${space.role}</p>` +
      "</div>" +
      `<span class="space-badge">${space.short}</span>` +
      "</div>" +
      '<div class="space-metrics">' +
      `<div class="space-metric ${purityClass}"><span class="k">purity @ ${state.k}</span><span class="v">${pct(
        result.purity
      )}</span></div>` +
      `<div class="space-metric"><span class="k">mean cosine</span><span class="v">${fmt(
        result.meanCosine,
        3
      )}</span></div>` +
      `<div class="space-metric ${voteClass}"><span class="k">vote</span><span class="v small-v">${
        result.vote.label === null ? "–" : CLASS_NAMES[result.vote.label]
      }</span><span class="s">${result.vote.count}/${state.k}</span></div>` +
      "</div>" +
      `<div class="thumb-grid lens-neighbors" id="neighbors-${space.key}"></div>` +
      `<p class="small muted space-note">${space.note}</p>` +
      `<p class="tiny muted">whole subset: k-NN @20 ${pct(knn, 1)} · mean-direction ratio ${fmt(
        ratio,
        3
      )}</p>`;

    const grid = $(`neighbors-${space.key}`);
    result.neighbors.forEach((neighbor, rank) => {
      const index = neighbor.index;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "thumb-button";
      if (DATA.labels[index] !== label) button.classList.add("mismatch");
      button.title = `#${rank + 1} · ${CLASS_NAMES[DATA.labels[index]]} · cosine ${fmt(
        neighbor.score,
        4
      )} · test index ${DATA.indices[index]} · click to select`;
      button.appendChild(UI.thumbNode(sprites.images, IMAGE_SPRITE, index, 42, "thumb"));
      const caption = document.createElement("span");
      caption.textContent = `${CLASS_NAMES[DATA.labels[index]]} ${neighbor.score.toFixed(2)}`;
      button.appendChild(caption);
      button.addEventListener("click", () => setImage(index));
      grid.appendChild(button);
    });
  }

  function intersectionCount(a, b) {
    const set = new Set(b);
    let shared = 0;
    for (const value of a) if (set.has(value)) shared += 1;
    return shared;
  }

  function renderOverlap(results) {
    const container = $("overlapMatrix");
    container.innerHTML = "";
    const corner = document.createElement("div");
    corner.className = "lens-matrix-corner";
    corner.textContent = "space";
    container.appendChild(corner);
    for (const space of SPACES) {
      const head = document.createElement("div");
      head.className = "lens-matrix-head";
      head.textContent = space.short;
      container.appendChild(head);
    }
    let best = null;
    for (const rowSpace of SPACES) {
      const rowHead = document.createElement("div");
      rowHead.className = "lens-matrix-row-head";
      rowHead.textContent = rowSpace.short;
      container.appendChild(rowHead);
      for (const columnSpace of SPACES) {
        const a = results[rowSpace.key].indices;
        const b = results[columnSpace.key].indices;
        const shared = intersectionCount(a, b);
        const jaccard = Core.jaccard(a, b);
        const cell = document.createElement("div");
        cell.className = "lens-matrix-cell";
        cell.textContent = rowSpace === columnSpace ? "—" : pct(jaccard);
        cell.title =
          rowSpace === columnSpace
            ? "the same list"
            : `${shared} of ${state.k} neighbours shared · Jaccard ${jaccard.toFixed(3)}`;
        if (rowSpace !== columnSpace) {
          const strength = Math.min(1, jaccard);
          cell.style.background = `rgba(36, 88, 214, ${0.06 + 0.5 * strength})`;
          cell.style.color = strength > 0.55 ? "#ffffff" : "#18202a";
          if (best === null || jaccard > best.jaccard) {
            best = { rowSpace, columnSpace, jaccard, shared };
          }
        } else {
          cell.style.background = "#eef1f5";
          cell.style.color = "#8a95a3";
        }
        container.appendChild(cell);
      }
    }
    $("overlapK").textContent = String(state.k);
    $("overlapNote").innerHTML =
      best === null
        ? "The three lists share no neighbours."
        : `Closest agreement: <strong>${best.rowSpace.short}</strong> and ` +
          `<strong>${best.columnSpace.short}</strong> share <span class="mono">${best.shared}/${state.k}</span> ` +
          `neighbours (${pct(best.jaccard)} Jaccard).`;
  }

  function buildConsensus(results) {
    const entries = new Map();
    for (const space of SPACES) {
      results[space.key].neighbors.forEach((neighbor, rank) => {
        let entry = entries.get(neighbor.index);
        if (!entry) {
          entry = {
            index: neighbor.index,
            spaces: [],
            bestRank: Infinity,
            bestScore: -Infinity,
          };
          entries.set(neighbor.index, entry);
        }
        entry.spaces.push(space);
        entry.bestRank = Math.min(entry.bestRank, rank + 1);
        entry.bestScore = Math.max(entry.bestScore, neighbor.score);
      });
    }
    return Array.from(entries.values())
      .filter((entry) => entry.spaces.length > 1)
      .sort(
        (a, b) =>
          b.spaces.length - a.spaces.length ||
          a.bestRank - b.bestRank ||
          b.bestScore - a.bestScore
      );
  }

  function renderConsensus(results) {
    const grid = $("consensusGrid");
    const consensus = buildConsensus(results);
    grid.innerHTML = "";
    const label = DATA.labels[state.image];
    if (consensus.length === 0) {
      $("consensusNote").textContent =
        "No image appears in more than one neighbour list at this k. Lower k or try mean-centered features.";
      return;
    }
    for (const entry of consensus.slice(0, 12)) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "thumb-button";
      if (DATA.labels[entry.index] !== label) button.classList.add("mismatch");
      button.title = `${CLASS_NAMES[DATA.labels[entry.index]]} · found by ${entry.spaces
        .map((space) => space.short)
        .join(", ")} · best rank ${entry.bestRank} · test index ${DATA.indices[entry.index]}`;
      button.appendChild(UI.thumbNode(sprites.images, IMAGE_SPRITE, entry.index, 42, "thumb"));
      const badge = document.createElement("span");
      badge.className = "consensus-badge";
      badge.textContent = `${entry.spaces.length}/3`;
      button.appendChild(badge);
      const caption = document.createElement("span");
      caption.textContent = `${CLASS_NAMES[DATA.labels[entry.index]]} · ${entry.spaces
        .map((space) => space.short)
        .join("+")}`;
      button.appendChild(caption);
      button.addEventListener("click", () => setImage(entry.index));
      grid.appendChild(button);
    }
    $("consensusNote").textContent =
      `${consensus.length} image${consensus.length === 1 ? "" : "s"} retrieved by at least two ` +
      `spaces at k = ${state.k}. The badge counts the spaces; the caption names them.`;
  }

  function renderTable(results) {
    const rows = SPACES.map((space) => {
      const result = results[space.key];
      const knn = DATA.meta.knn_accuracy[space.key][state.centered ? "centered" : "raw"];
      const ratio = DATA.meta.mean_direction_ratio[space.key];
      return (
        "<tr>" +
        `<td><span class="space-inline"><span class="space-dot" style="--space-color:${space.color}"></span>${space.short}</span></td>` +
        `<td>${space.dim}</td>` +
        `<td>${pct(knn, 1)}</td>` +
        `<td>${fmt(ratio, 3)}</td>` +
        `<td>${pct(result.purity)}</td>` +
        `<td>${fmt(result.meanCosine, 3)}</td>` +
        `<td>${result.vote.label === null ? "–" : CLASS_NAMES[result.vote.label]} (${result.vote.count}/${state.k})</td>` +
        "</tr>"
      );
    });
    $("comparisonBody").innerHTML = rows.join("");

    const purityLeader = SPACES.slice().sort(
      (a, b) => results[b.key].purity - results[a.key].purity
    )[0];
    const cosineLeader = SPACES.slice().sort(
      (a, b) => results[b.key].meanCosine - results[a.key].meanCosine
    )[0];
    const centeredText = state.centered
      ? "mean-centered features"
      : "raw features, so the shared direction is still in the cosine";
    $("verdictNote").innerHTML =
      `At <span class="mono">k = ${state.k}</span>, <strong>${purityLeader.label}</strong> has the ` +
      `highest class purity (${pct(results[purityLeader.key].purity)}); ` +
      `<strong>${cosineLeader.label}</strong> has the highest mean cosine ` +
      `(${fmt(results[cosineLeader.key].meanCosine, 3)}) on ${centeredText}. ` +
      (state.centered
        ? "Purity and the neighbour vote are the better summary of whether the neighbourhood is semantic."
        : "Switch to mean-centered features to compare rankings without the shared offset.");
  }

  function renderPicker() {
    const grid = $("pickGrid");
    grid.innerHTML = "";
    for (let index = 0; index < N; index += 1) {
      if (DATA.labels[index] !== state.classFilter) continue;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "thumb-button";
      if (index === state.image) button.classList.add("active-a");
      button.title = `test index ${DATA.indices[index]} · predicted ${CLASS_NAMES[DATA.preds[index]]}`;
      button.appendChild(UI.thumbNode(sprites.images, IMAGE_SPRITE, index, 42, "thumb"));
      const caption = document.createElement("span");
      caption.textContent = `#${DATA.indices[index]}`;
      button.appendChild(caption);
      button.addEventListener("click", () => setImage(index));
      grid.appendChild(button);
    }
    const active = grid.querySelector(".active-a");
    if (active && active.scrollIntoView) active.scrollIntoView({ block: "nearest" });
  }

  function syncControls() {
    $("classPick").value = String(state.classFilter);
    for (const button of $("centering").querySelectorAll("button")) {
      button.setAttribute(
        "aria-pressed",
        String(button.dataset.centered === String(state.centered))
      );
    }
    for (const button of $("kSelect").querySelectorAll("button")) {
      button.setAttribute("aria-pressed", String(Number(button.dataset.k) === state.k));
    }
    $("openAtlas").href = `../cifar_atlas/index.html?image=${state.image}`;
    $("openCard").href = `../per_image/index.html?image=${state.image}&b=${(state.image + 1) % N}`;
  }

  function render() {
    const results = {};
    for (const space of SPACES) results[space.key] = computeLens(space);
    syncControls();
    renderStats(results);
    renderBanner();
    for (const space of SPACES) renderSpaceCard(results[space.key]);
    renderOverlap(results);
    renderConsensus(results);
    renderTable(results);
    updateUrl();
  }

  function setImage(index) {
    state.image = Math.max(0, Math.min(N - 1, index));
    state.classFilter = DATA.labels[state.image];
    renderPicker();
    render();
  }

  function imagesOfClass(classIndex) {
    const out = [];
    for (let index = 0; index < N; index += 1) {
      if (DATA.labels[index] === classIndex) out.push(index);
    }
    return out;
  }

  function step(direction) {
    const images = imagesOfClass(state.classFilter);
    if (!images.length) return;
    const position = images.indexOf(state.image);
    const next = ((position < 0 ? 0 : position) + direction + images.length) % images.length;
    setImage(images[next]);
  }

  function updateUrl() {
    try {
      const url = new URL(window.location.href);
      url.searchParams.set("image", String(state.image));
      url.searchParams.set("center", state.centered ? "1" : "0");
      url.searchParams.set("k", String(state.k));
      window.history.replaceState(null, "", url.toString());
    } catch (error) {
      /* file:// in some browsers refuses replaceState; the page still works. */
    }
  }

  function readUrl() {
    const params = new URLSearchParams(window.location.search);
    const requested = Number(params.get("image"));
    if (Number.isInteger(requested) && requested >= 0 && requested < N) {
      state.image = requested;
    } else {
      let firstError = null;
      for (let index = 0; index < N; index += 1) {
        if (DATA.labels[index] !== DATA.preds[index]) {
          firstError = index;
          break;
        }
      }
      state.image = firstError === null ? 0 : firstError;
    }
    const center = params.get("center");
    if (center === "0" || center === "1") state.centered = center === "1";
    const k = Number(params.get("k"));
    if ([5, 8, 12].includes(k)) state.k = k;
    state.classFilter = DATA.labels[state.image];
  }

  function bind() {
    $("classPick").addEventListener("change", (event) => {
      state.classFilter = Number(event.target.value);
      renderPicker();
    });
    $("prevPick").addEventListener("click", () => step(-1));
    $("nextPick").addEventListener("click", () => step(1));
    $("randomPick").addEventListener("click", () => {
      const images = imagesOfClass(state.classFilter);
      if (images.length) setImage(images[Math.floor(Math.random() * images.length)]);
    });
    for (const button of $("centering").querySelectorAll("button")) {
      button.addEventListener("click", () => {
        state.centered = button.dataset.centered === "true";
        render();
      });
    }
    for (const button of $("kSelect").querySelectorAll("button")) {
      button.addEventListener("click", () => {
        state.k = Number(button.dataset.k);
        render();
      });
    }
    $("resetView").addEventListener("click", () => {
      state.centered = true;
      state.k = 8;
      render();
    });
    document.addEventListener("keydown", (event) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement) {
        return;
      }
      if (event.key === "ArrowLeft") step(-1);
      else if (event.key === "ArrowRight") step(1);
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
    readUrl();
    syncControls();
    bind();
    UI.loadSprites(DATA, (error, loaded) => {
      if (error) {
        $("selectedBanner").innerHTML = `<p class="small muted">Could not load the sprite sheets: ${error.message}</p>`;
        return;
      }
      Object.assign(sprites, loaded);
      renderPicker();
      render();
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
