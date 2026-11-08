# Release validation record

Validation date: 2026-11-08. This records checks executed during release
preparation; publication and DOI registration are separate operations.

The v1.0.0 release was subsequently published on 2026-11-08 and archived in
[Zenodo record 24046208](https://zenodo.org/records/24046208). Its version DOI is
[10.5281/zenodo.24046208](https://doi.org/10.5281/zenodo.24046208); its concept DOI
is [10.5281/zenodo.24046207](https://doi.org/10.5281/zenodo.24046207). Both DOI
links and the public record resolved successfully. Software type, version,
MIT license, all three creators, affiliations, ORCIDs and description matched
the approved metadata. The downloaded Zenodo ZIP matched the GitHub Release
API zipball byte for byte (SHA-256
`e4e2714e5f8dd69cbe1e86d537a2f9fa7a55b46ad418a2482f444def7d0237e3`),
and all 34 file hashes matched the validated release candidate.

## Environment

Linux x86_64, Python 3.12.14, NumPy 2.3.5, pandas 2.2.3, Matplotlib 3.10.8,
Pillow 13.0.0, SciPy 1.17.0 and PyTorch 2.9.1+cpu. No CUDA device was used.
NumPy reports OpenBLAS 0.3.30. The runs set `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`.

The Python environment inherited the base scientific packages and installed
CPU PyTorch into an isolated virtual environment. The release commit was then
validated with a fresh dependency installation on a GitHub runner. The dependency files pin the directly used
packages; they are not a complete transitive or operating-system lock file.

## Executed checks

| Command or check | Result and scope |
|---|---|
| `python -m compileall -q src diagnostics tables figures tests` | PASS; Python syntax compiled |
| `python diagnostics/grid_consistency.py` | Completed; the full chain gives 31.62 required/realized rescaling, while a 50-step subsample traverses about 5% in the defective update |
| `python diagnostics/form_identity.py` | Completed full sweep; 20,000 index-pair identity residual 2.831e-13; oracle seeds 0, 1, 2 |
| `python tables/make_tables.py` | PASS in all eight output blocks; the script's expected values match the reconstructed results |
| `python tables/check_raw_files.py` | All 50 checks PASS, including paired comparisons, process-length summaries and sample-size offsets |
| `python figures/make_figures.py` | All numeric and export-size assertions PASS; Figures 1–4 exported to EPS/PDF/TIFF, each 17.50 × 6.00 cm; TIFF at 600 dpi in CMYK |
| `python src/fid_robust.py` | All checks passed, including reference agreement, degenerate directions, identical inputs and the n=10,000, d=2,048 run |
| `python src/samplers_fixed.py` | All included synthetic self-tests passed on CPU |
| `python -m unittest discover -s tests -v` | Three regression checks PASS: SciPy driver recovery, final fallback and a known FID under translation with rank-deficient features |
| `CITATION.cff` against official CFF 1.2.0 JSON Schema | PASS; all three authors and ORCIDs retained |
| CSV parsing, row widths and exact duplicate rows | PASS for 14 files, 402 data rows, no malformed or duplicate full rows |
| Unique configuration/seed keys in the nine directly keyed measurement files | PASS; expected row counts retained |
| Original experimental CSVs and MIT license | SHA-256 matches the initial repository byte for byte |

The diagnostics, table scripts and figure script ran on the original source.
The changed FID and sampler self-test scripts were rerun after the technical
fixes; syntax and the new regressions were also checked after editing. The
remaining numerical scripts and all CSV inputs were unchanged. The same checks
passed on GitHub Actions for both
[main](https://github.com/i-SAVE/diffusion-sampler-audit/actions/runs/42431303783)
and the [v1.0.0 tag](https://github.com/i-SAVE/diffusion-sampler-audit/actions/runs/42431500068),
each at release commit `f38366a54351c437154bc4c7b360644e33eca3a5`.

Arial was not installed. Matplotlib reported font substitution; all four
rendered figures were inspected, but exact manuscript typography is unverified.
Generated figures are ignored by Git and rebuilt from the released script.

## Evidence boundaries

| Claim or result | Source | Evidence | Status / limitation |
|---|---|---|---|
| Grid-inconsistent update under-traverses the chain | README, erratum, Figure 1 | Executed schedule calculation and table checks | Verified calculation |
| Clipping separates the update forms under gain error | README, erratum | Executed identity test and three-seed synthetic oracle sweep | Verified for the included synthetic setup; no bound on a trained network |
| MNIST/CIFAR-10 table values and sample-size shifts | README file map, Figures 2–4 | Included CSVs, executed aggregation and 50 raw-file checks | Derived values verified; original training and feature extraction were not rerun |
| Gain-error reversal on a trained network | README, Figure 3 | `e1d_gain.csv` and executed plot assertions | Saved values and transformation verified; model measurements not independently repeated |
| Seed-resolution estimates | Table scripts | Included per-seed or per-arm summaries and executed formulas | Approximate resolution calculations, not a prospective statistical-power analysis; paired MNIST sampler differences are unavailable |
| Historical failure cause and runtime overhead of FID | Source commentary, erratum | Statements retained from the research package | Historical logs and feature matrices absent; the audit does not establish the claimed LAPACK failure cause or reproduce historical timings |

The full manuscript is not included, so agreement with manuscript text means
agreement with the expected constants embedded in the scripts; the submitted
document itself was not checked. Training notebooks, model weights, generated
images and Inception features are absent. External Kaggle notebook identifiers
are recorded but were not fetched or audited.

Missingness is explicit and preserved: ten missing FID measurements in
`e1d_fid.csv` have non-finite-output failure statuses; one DDPM cell at T=200,
NFE=100 in `E1c_results_SimpleUNet.csv` is absent; ten infinite peaks in
`e1d_peak.csv` record divergence. No original result was replaced or imputed.

CIFAR-10 sampler seeds are 0, 1, 2; process-length seeds are 0, 1, 2, 3, 4.
`E1c_seeds3_SimpleUNet.csv` provides arm means and standard deviations rather
than individual seed values, so paired differences cannot be recovered there.
