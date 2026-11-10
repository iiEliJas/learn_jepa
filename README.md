# learn_jepa

Interactive, dependency-free demos for understanding JEPA, LeJEPA, SIGReg and the
CIFAR-10 representation.

These pages are the learning layer for
**[fast_JEPA](https://github.com/iiEliJas/fast_JEPA)**, a from-scratch C/CUDA
implementation of LeJEPA. The main repository carries the protocol, the kernels
and the training runs; this repository turns the same equations and the same
model outputs into things you can drag, click and inspect. The math here is
checked against the Python reference in `pysrc/lejepa/`.

## Open the demos

Either open the landing page directly:

```bash
xdg-open index.html      # or your file manager
```

or serve the repository (useful on a tablet or for relative-path edge cases):

```bash
python -m http.server 8000
# then visit http://localhost:8000
```

No build step, no CDN, no runtime server, no extra Python dependency. The pages
use classic scripts so they work from `file://`.

## What exists

| Page | Status | Needs a checkpoint |
|---|---|---|
| `sigreg_lab/index.html` | available | no |
| `loss_calculator/index.html` | available | no |
| CIFAR embedding atlas | planned | yes |
| per-image embedding card | planned | yes |
| collapse simulator | planned | no |
| forward-pass x-ray, attention, augmentation | planned | yes |

The full roadmap is in [`learning_tools.md`](learning_tools.md).

## Math

The JavaScript core mirrors the Python reference exactly:

- `assets/sigreg_core.js`: one-sided quadrature on `[0, t_max]`, odd number of
  knots, endpoint weights `dt`, interior `2*dt`, all multiplied by
  `exp(-t^2/2)`. The statistic is `T_s = N * sum_t err(t) * weight(t)`,
  averaged over slices. The analytic gradient is checked against central finite
  differences in the Node tests.
- `loss_calculator/loss_core.js`: `mu`, `L_pred` with
  `mean_over_k` / `sum_over_k`, per-view or joint SIGReg, and the weighted
  total. It uses the same core as the lab.

The tests pin the JavaScript SIGReg value to a FP64 PyTorch value computed with
[`pysrc/lejepa/sigreg.py`](https://github.com/iiEliJas/fast_JEPA/blob/main/pysrc/lejepa/sigreg.py)
on the same points and directions. The formula reference is section 5.2 of the
[fast_JEPA protocol](https://github.com/iiEliJas/fast_JEPA/blob/main/fast_JEPA_protocol.md).

## Tests

```bash
node --test tests/*.test.js
python -m pytest tests/test_site.py -q
```

The Python test skips the Node suite if Node is not installed, and also checks
that the pages do not depend on external URLs.

## Adding the CIFAR atlas later

The atlas and per-image card share an export layer that will:

1. load `ckpt.pt` and the CIFAR-10 test set;
2. run the encoder and projector in eval mode;
3. write a compact `assets.json` (or binary) with a 1-2k image subset and both
   embedding spaces plus probe logits;
4. render a self-contained page from the exported data.

Nothing in the current pages depends on the training code, so the export layer
can be added under `cifar_atlas/` without touching the lab or calculator.

## License

Apache-2.0, matching fast_JEPA.
