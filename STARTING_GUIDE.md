# Start here

This repository is the static learning layer for
[FastJEPA](https://github.com/iiEliJas/fast_JEPA): interactive pages that turn
the LeJEPA objective and a trained CIFAR-10 run into something you can click and
inspect. There is no build step, no CDN and no runtime server.

## 1. Take the five-minute tour

Open [`index.html`](index.html) in a browser and follow the suggested path:

| Page | What it answers |
|---|---|
| [`loss_calculator/index.html`](loss_calculator/index.html) | What exactly are `L_pred`, `L_SIGReg` and `lambda` computing? |
| [`sigreg_lab/index.html`](sigreg_lab/index.html) | Why does the sliced Epps–Pulley statistic push a cloud toward a Gaussian? |
| [`cifar_atlas/index.html`](cifar_atlas/index.html) | Where does the trained encoder put each test image, and which errors are semantic? |
| [`per_image/index.html`](per_image/index.html) | What does the prediction term do to four augmented views of one image? |

The atlas and per-image card run from assets committed to this repository. You
only need the model checkpoint if you want to regenerate those assets.

## 2. Run the pages

The simplest way is to open the landing page directly:

```bash
xdg-open index.html      # Linux
open index.html          # macOS
```

If your browser is strict about `file://` relative paths, serve the directory:

```bash
python3 -m http.server 8000
# visit http://localhost:8000
```

The pages are classic scripts. Do not open them through a bundler or an IDE
preview that rewrites relative paths.

## 3. Download the CIFAR model (optional)

The trained model is published as a GitHub release asset, not committed to git.
The release is
[`v0.1.0`](https://github.com/iiEliJas/learn_jepa/releases/tag/v0.1.0).

| Asset | Size | SHA-256 |
|---|---|---|
| `cifar10_lejepa_400ep_model.pt` | 13.4 MB | `25e81591e9cf2757ccb86954352b1b3b3d03450e44ca88a2473d0f04f90c4d17` |
| `SHA256SUMS` | — | checksum file for the asset above |

```bash
mkdir -p checkpoints/cifar_400epochs

curl -L \
  -o checkpoints/cifar_400epochs/cifar10_lejepa_400ep_model.pt \
  https://github.com/iiEliJas/learn_jepa/releases/download/v0.1.0/cifar10_lejepa_400ep_model.pt

curl -L \
  -o /tmp/SHA256SUMS \
  https://github.com/iiEliJas/learn_jepa/releases/download/v0.1.0/SHA256SUMS

# Check the checksum from the directory containing the model.
cd checkpoints/cifar_400epochs
sha256sum -c /tmp/SHA256SUMS
```

On macOS use `shasum -a 256 -c /tmp/SHA256SUMS` if `sha256sum` is unavailable.

### What is inside the asset

The asset is a **model-only** checkpoint. The full training checkpoint also
contains optimizer, scheduler and RNG state; those are stripped because the
learning pages only need inference.

It contains:

- `enc`: the ViT encoder plus the 512-d `proj_head`;
- `proj`: the `512 → 512 → 16` projector;
- `probe`: `LayerNorm + Linear(512, 10)`;
- `config`, `git_commit`, `epoch`, `step` and `metrics`;
- `format: "fastjepa.model_only/1"` and the source checkpoint SHA-256.

Model facts: CIFAR-10, 32×32 input, patch size 4, embed dimension 192, depth 6,
3 heads, 16-d projector, 400 epochs (final epoch index 399), FastJEPA commit
`0b769eea158860f888ee34c2bde1242340c69379`. The best probe accuracy recorded in
the checkpoint is `77.11%`; the balanced 2,000-image test subset used by the
atlas scores `74.45%` top-1 in eval mode.

To recreate the release asset from a full checkpoint:

```bash
python cifar_atlas/package_model.py \
  --checkpoint checkpoints/cifar_400epochs/ckpt.pt \
  --out dist/cifar10_lejepa_400ep_model.pt
```

## 4. Regenerate the atlas assets

You need the FastJEPA Python environment (`torch`, `timm`, `torchvision`, `PIL`)
and the CIFAR-10 Python batches. With the release model:

```bash
~/projects/fast_jepa/.venv/bin/python cifar_atlas/generate.py \
  --checkpoint checkpoints/cifar_400epochs/cifar10_lejepa_400ep_model.pt \
  --data-root ../fast_jepa/data \
  --repo-root .
```

That rewrites:

- `cifar_atlas/atlas_data.js` — compact base64 tensors and projections;
- `cifar_atlas/atlas_check.json` — a small PyTorch parity fixture;
- `assets/atlas/cifar_test.webp` — one thumbnail per selected test image;
- `assets/atlas/cifar_views.webp` — four augmented-view thumbnails per image.

Use `--per-class 100` for a smaller 1,000-image build or `--skip-tsne` for a
faster PCA-only build. The checkpoint and CIFAR-10 data stay gitignored.

## 5. Run the tests

```bash
node --test tests/*.test.js
python3 -m pytest tests/test_site.py -q
~/projects/fast_jepa/.venv/bin/python -m pytest tests/test_learning_export.py -q
```

`test_site.py` runs the Node suite and checks that the pages are self-contained.
`test_learning_export.py` validates the shipped tensors on its own, then re-runs
a few images through PyTorch when the local checkpoint and CIFAR-10 data exist.

## 6. Repository map

```text
index.html                  landing page / guided tour
STARTING_GUIDE.md           this file
learning_tools.md           roadmap and design notes
assets/
  learning.css              shared style
  canvas.js                 canvas/plot helpers
  sigreg_core.js            pure SIGReg math
  atlas_core.js             decode, neighbours, prediction terms
  atlas_ui.js               sprite and marker helpers
sigreg_lab/                 interactive SIGReg lab
loss_calculator/            LeJEPA loss calculator
cifar_atlas/
  generate.py               model → static asset export
  package_model.py          full checkpoint → model-only release asset
  index.html, app.js        embedding atlas
per_image/                  one image, four views
tests/                      Node and Python tests
```

## 7. Troubleshooting

- **Blank atlas or card.** Check the browser console. The sprite sheets are
  loaded relative to the page, so the whole repository must be present; opening
  a copied single HTML file will not work.
- **`history.replaceState` warnings under `file://`.** Some browsers restrict
  URL updates from local files. The pages catch the error and keep working;
  use `python3 -m http.server` if you want the URL to update.
- **`ModuleNotFoundError: torch` or `timm`.** Use the FastJEPA virtualenv shown
  above, not the system Python. The static pages themselves need no Python.
- **Checksum mismatch.** Re-download the asset; do not rename it before running
  `sha256sum -c` unless you also edit `SHA256SUMS`.
- **CIFAR-10 batches missing.** Point `--data-root` at the directory containing
  `cifar-10-batches-py`.

## License and attribution

The code in this repository is Apache-2.0. The included thumbnails are a subset
of CIFAR-10 (Krizhevsky, 2009), included only to make the embedding demos
concrete; the dataset's own terms apply. The model asset is provided for
learning and analysis, not as a benchmark claim.
