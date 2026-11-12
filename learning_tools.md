# Learning tools: interactive demos for understanding LeJEPA

Status: public roadmap for this repository (2026-11-10). The learning layer
lives here, separate from the C/CUDA implementation in
[fast_JEPA](https://github.com/iiEliJas/fast_JEPA). The goal is a public-facing,
interactive learning layer around the existing PyTorch ground truth: formulas,
real numbers, and hands-on controls, not only static plots.

Implementation status (2026-11-12): SIGReg Lab, LeJEPA Loss Calculator, CIFAR
Embedding Atlas, Per-Image Embedding Card, Three-Space Lens and the landing page
are available. The atlas ships a balanced 2,000-image test subset from the
400-epoch CIFAR run; the checkpoint and raw CIFAR data remain local. The
model-only checkpoint is published in the `v0.1.0` GitHub release.

Existing material to build on:

- `tools/analyze_run.py` and `docs/experiments/002_cifar10_lejepa_analysis.md`
  (eight static figures plus a written analysis).
- `pysrc/lejepa/analysis.py` (PCA, spectra, effective rank, k-NN, sliced
  Gaussianity, prediction loss, cosine statistics).
- `pysrc/lejepa/sigreg.py` (reference SIGReg and exact quadrature).
- `pysrc/lejepa/loss.py` (exact LeJEPA objective).
- `pysrc/lejepa/model.py` and `pysrc/lejepa/train.py` (encoder, projector,
  progressive augmentation recipe).

Model facts that shape the demos:

- CIFAR config: 32x32 input, patch 4 -> 8x8 = 64 patches + CLS.
- Three spaces worth exposing: raw ViT CLS (192-d in the CIFAR config), the
  512-d `proj_head` feature used by the probe, and the 16-d projector output
  used by the prediction loss and SIGReg.
- Four views per image, per-view SIGReg over the N images of each view.
- The projector has BatchNorm1d, so train-mode and eval-mode embeddings differ.
- The trained CIFAR checkpoint referenced in the report
  (`runs/lejepa_cifar10/ckpt.pt`) is gitignored and absent in this checkout.

---

## Design principles

1. **Static first.** Each demo is a self-contained HTML page plus small
   classic-JS files. No runtime server, no mandatory Python dependency, no
   CDN. It works by opening `index.html` and can be published on GitHub Pages.
2. **Real code, not a parallel story.** Numbers must come from the same
   formulas as `pysrc/`. Where JavaScript recomputes a formula, add a Node
   fixture test against PyTorch-generated values.
3. **Explain every number.** Each panel has a one-line "what am I looking at",
   the formula with the protocol symbols, and a "try this" prompt.
4. **Small data.** Ship a 1-2k image subset and quantized embeddings for the
   public atlas; generate the full set locally from `data/` and `ckpt.pt`.
   Do not commit large arrays or checkpoints.
5. **Optional live mode.** A separate `[learning-live]` Gradio extra may serve
   arbitrary image upload and softmax/attention inference. The static demos
   remain the default.
6. **No changes to `csrc/` or the training path** for the learning tools. Only
   new files in this repository, tests under `tests/`, and docs.

---

## Tier 1: understand the objective (no checkpoint needed)

### 1. SIGReg Lab

**What you can do.** Drag a 2-D point cloud or choose a preset (standard
Gaussian, collapsed, anisotropic, two clusters, uniform, correlated). Change
`N`, the number of slices, the number of quadrature knots and `t_max`. Resample
directions. See:

- sampled directions on the unit circle and the resulting 1-D projections;
- the empirical characteristic function `C(t), S(t)` against
  `phi_0(t) = exp(-t^2/2)` for a selected slice;
- the integrand `(C - phi_0)^2 + S^2` times the window, with quadrature nodes;
- per-slice `T_s` and the final `L_SIGReg`;
- the per-point negative gradient field (`-dL/dz`) for the current cloud;
- a "Gaussian null" marker: what `L_SIGReg` a same-size Gaussian draw usually
  gives, so a value near 1 is not a failure;
- a "batch-size bias" panel: `L_SIGReg` vs `N` for Gaussian draws.

**What it teaches.** Cramer-Wold, sliced Epps-Pulley, the exact one-sided
quadrature, why directions are resampled every step, why finite-batch values do
not go to zero, and how the regularizer pushes a collapsed cloud outward.

**Needs.** None. Runs on CPU in the browser.

**Effort.** M.

### 2. LeJEPA Loss Calculator

**What you can do.** Edit a tiny `z` tensor (`V=4, N=3, K=2` by default) in a
number grid. Live-compute:

- `mu = mean(z[:V_g])`, the residual `mu - z`, and `L_pred`;
- `L_pred` with `mean_over_k` vs `sum_over_k` (show the factor K);
- per-view SIGReg values, the mean, and the weighted total;
- each term's share of the total as `lambda` changes;
- a "show the arithmetic" panel with the actual substituted values.

Presets: collapsed, perfectly invariant but diverse, noisy views, one outlier,
non-Gaussian marginals.

**What it teaches.** The exact objective, the `(1 - lambda)` factor, `V_g`,
the mean-over-K convention, why a large raw `L_sig` is not automatically a
problem, and what changes when `lambda` moves.

**Needs.** None. CPU only.

**Effort.** S.

### 3. Collapse Simulator

**What you can do.** Train a tiny 2-layer encoder + predictor on synthetic
two-view 2-D data for a few hundred CPU steps. Toggle SIGReg on/off and scrub
through training. Watch the embedding cloud collapse to a point or spread out;
plot `L_pred`, `L_sig`, effective rank and per-dimension std.

**What it teaches.** Why JEPA needs an anti-collapse term, what collapse looks
like in the diagnostics before the loss shows it, and why SIGReg is not just a
metric.

**Needs.** CPU only.

**Effort.** M.

---

## Tier 2: explore the trained CIFAR model (needs a checkpoint)

### 4. CIFAR Embedding Atlas (available)

**What you can do.** A 2-D map of a balanced test subset (2,000 images by
default; the full build can use all 10k) colored by class. Hover for a
thumbnail; click to lock a selection. Toggle:

- encoder 512-d vs projector 16-d vs raw ViT CLS;
- raw vs mean-centered features;
- PCA vs optional precomputed t-SNE/UMAP;
- all points vs misclassified-only vs a class filter.

Clicking a point opens an image card (see item 5). A confusion-matrix panel
links each cell to the actual images behind that error.

**What it teaches.** Class geometry, semantic confusions, anisotropy, why raw
cosine is dominated by the mean direction, and how the probe feature differs
from the projector.

**Needs.** `ckpt.pt` + CIFAR-10 test images to regenerate the export. The
public page runs on the included assets.

**Effort.** L.

### 5. Per-Image Embedding Card (available)

**What you can do.** For one image, show its four augmented views, their points
in a local 2-D view, the mean `mu`, arrows from each view to `mu`, the pairwise
cosine matrix, and each view's contribution to `L_pred`. Compare image A vs B.
A "prediction loss" readout shows paired vs shuffled views for the selected
image.

**What it teaches.** View invariance, what the prediction term does on a real
example, why `mu` is not detached, and why the projector is trained for
prediction rather than classification.

**Needs.** The same export as the atlas. Runs on the included assets.

**Effort.** M.

### 6. Three-Space Lens (available)

**What you can do.** The same image and neighbourhood shown in three spaces:
raw ViT CLS, 512-d head, 16-d projector. Side-by-side nearest neighbours, class
purity, neighbour vote and mean cosine, plus a cross-space overlap matrix and a
consensus strip of images retrieved by more than one space. Toggle raw vs
mean-centered features and `k`; the selection is kept in the URL.

**What it teaches.** Which space is semantic, which space is for
prediction/SIGReg, and why the projector looks more diffuse. The overlap matrix
separates "same class" from "same neighbourhood": two spaces can agree on the
label while retrieving different images.

**Needs.** The same export as the atlas. Runs on the included assets.

**Effort.** S/M.

### 7. Real-Data SIGReg Inspector

**What you can do.** Run the SIGReg Lab panels on actual model batches. Compare
model vs Gaussian, `per_view` vs `joint`, train mode vs eval mode (BatchNorm
statistics), per-slice `T_s`, KS distance and skew/kurtosis.

**What it teaches.** The regularizer on real embeddings; explains the report's
training `2.47`, evaluation `2.84`, and Gaussian-null `1.05`.

**Needs.** Checkpoint.

**Effort.** M.

### 8. 16-Dimension Inspector

**What you can do.** A grid of 16 histograms (model vs `N(0,1)`), per-dimension
mean/std/skew/kurtosis, a correlation matrix, and a "resample batch" button.

**What it teaches.** Isotropy, per-dimension collapse, and what "the projector
is nearly isotropic" means.

**Needs.** Checkpoint.

**Effort.** S.

### 9. Forward-Pass X-Ray

**What you can do.** One image through the network with real shapes and values
at every stage: pixel stats, 8x8 patches, patch embedding, CLS, each block's
token norms and attention entropy, 512-d head, 16-d projector, probe logits.
Click a stage to see a histogram or heatmap.

**What it teaches.** The architecture end to end, where dimensions change,
where LayerNorm/BatchNorm sit, and what "example values" look like.

**Needs.** Checkpoint.

**Effort.** M.

### 10. Patch & Attention Explorer

**What you can do.** 8x8 patch grid with per-patch norms, CLS attention maps per
layer/head, attention rollout, and a hovered-patch attention view.

**What it teaches.** How the ViT builds the CLS embedding, whether attention is
on objects or background, and layer-wise specialization.

**Needs.** Checkpoint. Requires a small hook and a test because timm may use
fused attention by default.

**Effort.** M/L.

### 11. Augmentation Playground

**What you can do.** Sliders/buttons for crop scale, color jitter, blur,
solarize and flip. See the augmented views and where their embeddings move;
compare pairwise cosine and `L_pred`.

**What it teaches.** What the four views actually are, how strong the
invariance is, and which augmentations move the embedding most.

**Needs.** Checkpoint; live arbitrary images need the optional Gradio mode.

**Effort.** M.

---

## Tier 3: training and research dashboards

### 12. Run Comparator

Load multiple run directories and overlay loss/probe/rank/Gaussianity curves.
Compare configs (`lambda`, slices, `V`, `sigreg_mode`). Mostly reuses the JSONL
logs.

**Effort.** S/M.

### 13. Training Replay

Scrub epochs and watch the embedding atlas morph, with rank/std/probe in sync.
Needs per-epoch checkpoints. Current `save_every` writes only `ckpt_last.pt`, so
this needs an export/record mode or a change to the save policy.

**Effort.** M.

### 14. Guided Tour

A landing page that ties the demos into a course: JEPA in one screen -> loss
calculator -> SIGReg lab -> forward x-ray -> CIFAR atlas -> attention explorer,
with formulas and "try this" prompts.

**Effort.** S.

---

## Recommended first milestone

The first pass is implemented:

1. SIGReg Lab
2. LeJEPA Loss Calculator
3. CIFAR Embedding Atlas
4. Per-Image Embedding Card
5. Three-Space Lens
6. A tiny landing page (item 14)

The atlas, per-image card and lens share the export layer in
`cifar_atlas/generate.py`; the landing page links all five demos. The next
conceptual demo is the collapse simulator (item 3), followed by the combined
real-data SIGReg / 16-D inspector.

---

## Technical layout

```
index.html                   landing page / guided tour
README.md                    how to open, generate assets, run tests
learning_tools.md            this roadmap
assets/
  learning.css               shared style
  canvas.js                  canvas/plot helpers
  sigreg_core.js             pure SIGReg math (require-able from Node)
sigreg_lab/
  index.html
  app.js                     DOM and canvas
loss_calculator/
  index.html
  loss_core.js
  app.js
cifar_atlas/
  generate.py                export model outputs + sprites
  index.html
  app.js                     DOM, canvas, confusion drill-down
  atlas_data.js              generated (included)
per_image/
  index.html
  app.js                     four views, local layout, L_pred
space_lens/
  index.html
  app.js                     same image in CLS, head and projector space
assets/
  atlas_core.js              pure decode/neighbour/prediction math
  atlas_ui.js                sprite and marker helpers
  atlas/*.webp               generated thumbnail sheets
tests/
  *.test.js                  Node fixture tests
  test_site.py               structure and self-containment test
```

Implementation rules:

- Classic `<script>` files, not ES modules, so double-clicking `index.html`
  works under `file://`.
- Pure math lives in `*_core.js` with a small UMD wrapper (`module.exports` for
  Node, `window.X` for the browser).
- Canvas only; no plotting library, no CDN.
- All colors/labels accessible; no information conveyed by color alone.
- Every demo has a footer with the formula reference and the reproduction
  command.

## Checkpoint and asset plan for items 4-5

- Export a compact model-only checkpoint (`enc`, `proj`, `probe`, config,
  git commit) so the site does not need optimizer state.
- Precompute both encoder and projector embeddings in eval mode for the test
  set, plus the probe logits.
- Keep a 1-2k image subset and quantized embeddings for the public site; a full
  local build can use all 10k.
- Validate exported tensors against a PyTorch re-run in
  `tests/test_learning_export.py` (shapes, dtypes, a few values, no NaNs).
- The site must state which checkpoint/commit produced the assets.

## Open questions

- Which checkpoint will be used (the 100-epoch run if available, or the new
  short run currently being trained)?
- Static site only, or also the optional live Gradio app for image upload?
- PCA only, or optional t-SNE/UMAP? (UMAP adds a dependency; t-SNE is available
  in scikit-learn if we add an optional extra.)
- Should the public site ship a subset of CIFAR images (license/attribution) or
  generate everything locally?
