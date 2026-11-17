# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2026-11-17

### Added

- GitHub Actions CI: JavaScript syntax checks, the Node math suite, site checks
  and the export contract test.
- A manual GitHub Pages workflow that publishes the repository root without a
  build step.
- Link and landing-page coverage checks in `tests/test_site.py`.
- `requirements-dev.txt` and contributing notes for outside contributors.
- This changelog.

### Changed

- Reworked the landing page: a six-stop suggested path, a roadmap that lists
  only planned demos, and an "About the demos" section in place of the old
  meta-commentary.
- Simplified demo tags to `no checkpoint` and `uses CIFAR assets`.
- Updated the GitHub Actions used by CI and Pages to their current majors.
- The suggested path is now a fixed three-column grid, dropping to two and one
  column on narrower screens.

### Fixed

- Removed the duplicated step numbers from the suggested-path cards; the CSS
  badge already supplies the number.

## [0.2.0] - 2026-11-13

### Added

- Three-Space Lens: one image in raw CLS, 512-d `proj_head` and 16-d projector
  space, with neighbour overlap and consensus retrieval.
- Projector Diagnostics: 16 marginal histograms, moments, KS distance, a 16×16
  correlation matrix and real-data SIGReg per slice.
- Collapse Simulator: a browser-trained 2→8→2 LeJEPA on synthetic two-view 2-D
  data, with effective-rank and per-dimension spread diagnostics.

## [0.1.0] - 2026-11-12

### Added

- SIGReg Lab and the LeJEPA Loss Calculator.
- CIFAR Embedding Atlas over a balanced 2,000-image test subset.
- Per-Image Embedding Card with four augmented views and the prediction term.
- `STARTING_GUIDE.md`, the shared export pipeline in `cifar_atlas/generate.py`,
  and Node/Python tests for the JavaScript math.
- A model-only checkpoint release asset, `cifar10_lejepa_400ep_model.pt`.

[Unreleased]: https://github.com/iiEliJas/learn_jepa/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/iiEliJas/learn_jepa/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/iiEliJas/learn_jepa/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/iiEliJas/learn_jepa/releases/tag/v0.1.0
