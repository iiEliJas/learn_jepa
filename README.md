# learn_jepa

[![CI](https://github.com/iiEliJas/learn_jepa/actions/workflows/ci.yml/badge.svg)](https://github.com/iiEliJas/learn_jepa/actions/workflows/ci.yml)

Interactive, dependency-free demos for understanding JEPA, LeJEPA, SIGReg and the
CIFAR-10 representation.

These pages are the learning layer for
**[fast_JEPA](https://github.com/iiEliJas/fast_JEPA)**, a from-scratch C/CUDA
implementation of LeJEPA. The main repository carries the protocol, the kernels
and the training runs; this repository turns the same equations and the same
model outputs into things you can drag, click and inspect. The math here is
checked against the Python reference in `pysrc/lejepa/`.

New here? Read **[STARTING_GUIDE.md](STARTING_GUIDE.md)** for the five-minute
tour, the optional model download and the regeneration workflow.

## Open the demos

Either open the landing page directly:

```bash
xdg-open index.html      # or your file manager
```

or serve the repo:

```bash
python3 -m http.server 8000
# then visit http://localhost:8000
```

No build step, no CDN, no runtime server, no extra Python dependency. The pages
use classic scripts so they work from `file://`.

## What exists

| Page | Status | To regenerate |
|---|---|---|
| `sigreg_lab/index.html` | available | nothing |
| `loss_calculator/index.html` | available | nothing |
| `cifar_atlas/index.html` | available | model + CIFAR-10 |
| `per_image/index.html` | available | model + CIFAR-10 |
| `space_lens/index.html` | available | model + CIFAR-10 |
| `projector_diagnostics/index.html` | available | model + CIFAR-10 |
| `collapse_sim/index.html` | available | nothing |
| forward-pass x-ray, attention, augmentation | planned | model |

The atlas, per-image card, three-space lens and projector diagnostics run from
the included assets. The collapse simulator trains a toy model on synthetic data.
The public atlas ships a balanced 2,000-image test subset; the checkpoint and the
CIFAR-10 data stay local and are gitignored. The trained model is published as a
GitHub release asset instead of being committed. The full roadmap is in
[`learning_tools.md`](learning_tools.md).

## Tests

```bash
python3 -m pip install -r requirements-dev.txt
node --test tests/*.test.js
python3 -m pytest tests/test_site.py -q
~/projects/fast_jepa/.venv/bin/python -m pytest tests/test_learning_export.py -q
```

The Node suite needs no Python. `test_site.py` skips the Node suite if Node is
not installed, and checks page structure, local links and self-containment. The
export test validates the shipped tensors on its own, and re-runs a few images
through PyTorch when the local checkpoint and CIFAR-10 batches are present. CI
runs the Node suite and the PyTorch-free Python checks on every push; the
parity test skips there because the checkpoint and dataset are not committed.

## Contributing

Small fixes and new demo ideas are welcome. Keep the static-first rule: classic
scripts, no CDN, no bundler and no runtime server. Run the test commands above
before opening a pull request; CI runs the same Node and Python checks. For a
new page, start from [`STARTING_GUIDE.md`](STARTING_GUIDE.md#6-repository-map)
and add the export or core module under `assets/` rather than duplicating math
inside the page.

## Regenerating the CIFAR assets

The trained model is not in git. The
[`v0.1.0` release](https://github.com/iiEliJas/learn_jepa/releases/tag/v0.1.0)
includes `cifar10_lejepa_400ep_model.pt` (13.4 MB), a model-only checkpoint
containing `enc`, `proj`, `probe` and the config. Download and checksum
instructions are in
[`STARTING_GUIDE.md`](STARTING_GUIDE.md#3-download-the-cifar-model-optional).

The atlas, per-image card, three-space lens and projector diagnostics share
`cifar_atlas/generate.py`. It:

1. loads the model-only parts of a FastJEPA checkpoint (`enc`, `proj`, `probe`);
2. selects a deterministic, balanced test subset (200 per class by default);
3. runs the encoder, 512-d head, projector and probe in eval mode;
4. generates four training-recipe augmented views per image and their projector
   outputs;
5. writes compact base64 tensors, PCA/t-SNE coordinates, two WebP sprite sheets
   and a small PyTorch fixture under `cifar_atlas/atlas_check.json`.

From the repository root, using the fast_JEPA virtualenv:

```bash
~/projects/fast_jepa/.venv/bin/python cifar_atlas/generate.py \
  --checkpoint checkpoints/cifar_400epochs/ckpt.pt \
  --data-root ../fast_jepa/data
```

The `--checkpoint` path may point at the full training checkpoint or at the
model-only release asset. To create the model-only asset from a full checkpoint:

```bash
python cifar_atlas/package_model.py \
  --checkpoint checkpoints/cifar_400epochs/ckpt.pt \
  --out dist/cifar10_lejepa_400ep_model.pt
```

Use `--per-class 100` for a 1,000-image build or `--skip-tsne` for a faster
PCA-only build. The exported page states the checkpoint SHA-256 and the FastJEPA
commit. The checkpoint and the CIFAR-10 batches are deliberately not committed;
the exported `atlas_data.js` and sprite sheets are.

`tests/test_learning_export.py` checks the exported shapes and decodes the
tensors without any ML dependency, then (when the local checkpoint and data are
present) re-runs a handful of images through PyTorch and compares the values.

## Dataset attribution

The thumbnail sheets contain a 2,000-image subset of the CIFAR-10 test set
(Krizhevsky, 2009), included only to make the embedding demos concrete. The
dataset's own terms apply to those images; the code in this repository is
Apache-2.0.

## License

Apache-2.0
