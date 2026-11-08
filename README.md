# Sampler implementation errors and measurement resolution — code and data

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Companion repository for

> Sh. M. Isaev, I. A. Kistin, M. V. Bolsunovskaya. *Sampler Implementation Errors
> and Measurement Resolution in Small-Scale Diffusion Studies.* Submitted to
> **Pattern Recognition and Image Analysis**.

After installing the dependencies below, the included checks run on CPU without
model weights or dataset downloads. The two diagnostics take milliseconds and about a minute
respectively; the table checks take a few minutes.

```
diagnostics/grid_consistency.py   class A: is the update on the grid it was derived for?
diagnostics/form_identity.py      class B: does a neutral rearrangement change the output?
src/samplers_fixed.py             corrected sampler, both error classes fixed
src/fid_robust.py                 FID via QR factorisation, no covariance product
tables/make_tables.py             rebuilds the tables of the MNIST and oracle experiments, PASS/FAIL
tables/check_raw_files.py         checks every number that rests on the CIFAR-10 sampler grid and
                                  the MNIST process-length runs, PASS/FAIL
figures/make_figures.py           redraws Figs. 1-4 from results/ and asserts every plotted number
results/                          the measurement files everything above is built from
CHANGELOG.md                      erratum to the previously released package
```

## Start here

```bash
python diagnostics/grid_consistency.py     # Table 1
python diagnostics/form_identity.py        # Table 4
python tables/make_tables.py               # Tables 1, 2, 4, 5 and the MNIST numbers of Sections 3-7
python tables/check_raw_files.py           # Table 3, Table 6 and the CIFAR-10 / process-length numbers
python figures/make_figures.py             # Figs. 1-4 (EPS, PDF, 600 dpi TIFF)
```

Both table scripts exit non-zero if any recomputed value disagrees with the
manuscript.

## Where each number of the paper comes from

`tables/make_tables.py` prints its tables under the numbering of the longer
working version of the manuscript (16 tables). The correspondence is:

| paper | script and output block | result files |
|---|---|---|
| Table 1, Fig. 1 | `grid_consistency.py`; `make_tables.py` block 1 | closed form |
| Table 2, Fig. 4a; ratios 3.47-6.71 and 1.87 in Section 3 | `make_tables.py` blocks 2 and 4 | `E1c_results_SimpleUNet.csv`, `e1d_fid.csv` |
| Table 3; factor 4.1-5.3 (Section 3); Section 6.1 CIFAR-10 test; Section 7 | `check_raw_files.py` | `e2c_cifar.csv` |
| Section 4 "Cost on a trained network", Fig. 2 | `make_tables.py` block 3 | `e1d_fid.csv` |
| Table 4 | `form_identity.py`; `make_tables.py` blocks 6-7 | oracle, no data |
| Fig. 3 | `make_tables.py` block 8 | `e1d_gain.csv` |
| Table 5; 12 and 19 seeds (Section 6.1) | `make_tables.py` blocks 9-10 | `E1c_seeds3_SimpleUNet.csv` |
| MNIST process length, 44-91 and 641 seeds (Section 6.1) | `check_raw_files.py` | `e3a_mnist.csv` |
| Table 6, Fig. 4b | `check_raw_files.py`; `make_tables.py` block 14 | `e3a_samplesize.csv`, `e2c_fid_samplesize.csv`, `E3b_samplesize.csv` |
| CIFAR-10 process length, three of ten two-seed pairs (Section 7) | `make_tables.py` block 14 | `E3b_cifar_5seeds.csv` |

## The two checks, in one paragraph each

**Class A — the update is applied on a grid it was not derived for.** A
single-step DDPM update advances the state by one chain position. Run it on a
grid that skips positions and the gap is never accounted for: on a 1000-step
chain sampled at 50 steps only 5 % of the required denoising is applied and the
output stays near noise. The check is an identity on the schedule alone —
`sum_{i in S} log(1/sqrt(alpha_i))` must equal `log(1/sqrt(alpha_bar_T))` — and
it costs as many multiplications as there are steps. It is worth running at the
step budget you actually report, because the error vanishes exactly when the
grid is the full chain, which is what a casual sanity check exercises.

**Class B — an algebraically neutral rearrangement is not numerically
neutral.** The generalised update can be written through `eps` or through the
current state. The two are identical until `x0hat` is clipped to the data
range; after clipping, the first form multiplies the state by more than one at
every step and can compensate only through the network output, so a systematic
gain error compounds. Replacing the denoiser with an oracle that returns the
true noise scaled by `1+g` isolates this from model quality. The fix is one
line: re-derive `eps` from the clipped `x0hat`. Apply it unconditionally — the
sweep detects the mechanism but does not bound it, because a real network's
gain error is not constant along the trajectory and on ours it reverses sign.

## What the results files contain

| file | what it is |
|---|---|
| `e1d_fid.csv` | form (I) against form (II), MNIST, 10 000 samples, both `eta`, with a `status` column recording non-finite output |
| `e1d_gain.csv` | measured gain error `g(t)` of the trained network and the fraction of output energy a multiplicative gain does not explain |
| `e1d_peak.csv` | peak `max|x|` along the trajectory — reported for reference; see the note in `src/samplers_fixed.py` about why the peak is a poor severity measure |
| `E1c_results_SimpleUNet.csv` | the MNIST grid: defective sampler (Eq. (2) of the paper) against Eqs. (3), (4) |
| `E1c_seeds3_SimpleUNet.csv` | the same at three training seeds, per-arm mean and s.d. |
| `e2c_cifar.csv` | CIFAR-10 sampler grid: four samplers (`DDPM_repo` = Eq. (2)), T = 200 and 1000, 10 and 50 steps, three training seeds, 10 000 samples, one row per seed |
| `e2c_fid_samplesize.csv` | the same runs (T = 1000, 50 steps) scored at 2 000 and 10 000 samples |
| `e3a_mnist.csv` | MNIST process length: T = 100-1000, five training seeds, 10, 30 and 60 epochs, one row per seed |
| `e3a_samplesize.csv` | the same runs (T = 1000, 60 epochs) scored at 3 000 and 10 000 samples |
| `E3b_cifar_5seeds.csv` | CIFAR-10 process length, five training seeds |
| `E3b_samplesize.csv` | the same configuration at 2 000 and 10 000 samples |
| `E1b_seeds_3.csv`, `E1b_terminal_noise.csv`, `E1_results_mnist.csv` | earlier MNIST runs, kept for provenance |

The four `e2c_*` and `e3a_*` files are the unmodified outputs of Kaggle notebooks
`notebook8222b01e68` (version 2) and `notebook73c642a5b7` (version 4).

Per-seed values are given wherever we have them. `E1c_seeds3_SimpleUNet.csv`
records only a mean and a standard deviation per arm, so for the MNIST sampler
comparison the standard deviation of the per-seed *differences* cannot be
recovered and seed requirements are given under two readings rather than one.

## Reporting discipline

Three lines, none of which costs a run:

1. State the fraction of the trajectory the sampler traverses at the step
   budget reported. `grid_consistency.py` prints it.
2. State the number of generated samples every FID value was computed from, and
   use the same number in every cell being compared.
3. State the run-to-run variation next to every difference claimed, and the
   number of seeds that would be needed to resolve it.

## Relation to the previously released package

The earlier reproducibility package is left archived and unchanged, so that the
numbers published from it remain reproducible exactly as published. Correcting
an archived artefact in place makes the record of what was published harder to
recover, not easier. `CHANGELOG.md` here identifies every table of that paper
affected by each error class and states which of its conclusions do not survive
re-measurement.

## Requirements

The release checks were run with Python 3.12 on Linux, using the versions in
`requirements.txt`. Create an isolated environment and install them with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
# Needed only for src/samplers_fixed.py and its self-test:
python -m pip install -r requirements-sampler.txt --index-url https://download.pytorch.org/whl/cpu
```

`numpy` for the diagnostics and `tables/make_tables.py`; `tables/check_raw_files.py` needs only the Python standard library;
`pandas`, `matplotlib` and `pillow` for `figures/make_figures.py`; `torch` (CPU is fine) for
`src/samplers_fixed.py`; `scipy` is optional and only used as a fallback SVD
driver in `src/fid_robust.py`.

Additional implementation checks:

```bash
python -m unittest discover -s tests -v
python src/fid_robust.py
python src/samplers_fixed.py
```

The figure script requests Arial. If Arial is unavailable, Matplotlib substitutes
a local font: plotted values and export dimensions can be checked, but typography
will differ. Generated figures and local environments are excluded from Git.

## Reproducibility scope and provenance

This archive reproduces the analytical diagnostics, derived tables and figures
from the included measurement CSVs, and synthetic sampler/FID checks. It does
not contain the training notebooks, trained weights, original generated images
or Inception features needed to independently rerun the MNIST/CIFAR-10 training
and original FID measurements. Kaggle notebook identifiers are recorded above
and in `tables/check_raw_files.py`; their external contents were not validated
as part of this release audit.

The CSV files are preserved unchanged. Ten blank FID entries in `e1d_fid.csv`
are accompanied by explicit non-finite-output failure statuses. The blank DDPM
cell at T = 200, NFE = 100 in `E1c_results_SimpleUNet.csv` is unmeasured here;
the ten infinite peaks in `e1d_peak.csv` record divergence. None is imputed.

See [the validation record](docs/VALIDATION.md) and
[authorship and tool assistance](docs/PROVENANCE.md). The three scientific
authors and their ORCIDs are preserved in `CITATION.cff`.
