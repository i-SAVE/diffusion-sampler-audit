# diffusion-sampler-audit v1.0.0

Code and measurement data accompanying *Sampler implementation errors and
measurement resolution in small-scale diffusion studies* by Sharip M. Isaev,
Ilya A. Kistin and Marina V. Bolsunovskaya.

The archive includes:

- CPU diagnostics for single-step updates on subsampled grids (class A) and
  clipping-dependent differences between algebraically equivalent sampler
  forms (class B), including synthetic oracle sweeps.
- A generalized PyTorch sampler with re-derived noise after clipping and a
  QR/SVD-based FID implementation, with a repaired SciPy `gesvd` fallback.
- Fourteen unchanged measurement CSVs for MNIST and CIFAR-10 sampler,
  process-length, seed-variation and sample-size comparisons.
- Scripts that rebuild and check derived tables and export Figures 1–4
  as EPS, PDF and 600 dpi TIFF, plus pinned validation dependencies and CI.

All included diagnostics, table checks, figure assertions, FID and sampler
self-tests, and fallback regression checks were executed during preparation.
See `docs/VALIDATION.md` for the environment and exact scope of verification.

The archive reproduces calculations from saved measurements and synthetic
diagnostics. It does not include the training notebooks, model weights, raw
images or feature matrices required to independently repeat the original
training and FID measurements. Missing values and recorded divergent runs are
retained. Arial is requested by the figure script; font substitution changes
typography when Arial is unavailable.

License: MIT. Scientific authors and ORCIDs: `CITATION.cff`.

This is the first release of this repository. The earlier package discussed
in `CHANGELOG.md` remains a separate archived research artifact.
