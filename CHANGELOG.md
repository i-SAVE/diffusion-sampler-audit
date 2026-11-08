# Changelog

## Unreleased

- Link the verified v1.0.0 version DOI and the all-versions concept DOI in the
  README, add the version DOI to `CITATION.cff`, and record the public archive
  and GitHub Actions verification. These documentation updates postdate the
  v1.0.0 tag; the archived source and experimental data remain unchanged.

## 1.0.0 — initial release of diffusion-sampler-audit

- Include the sampler diagnostics, corrected sampler, robust FID implementation,
  measurement CSVs and scripts that rebuild the paper's tables and figures.
- Fix the SciPy SVD fallback: `scipy.linalg.svdvals` does not accept
  `lapack_driver`; use `scipy.linalg.svd(..., compute_uv=False,
  lapack_driver='gesvd')` so the intended independent driver is actually tried.
- Make the sampler and FID self-tests exit non-zero when a check fails.
- Add regression checks for the FID fallback cascade, pinned CPU validation
  dependencies, and a GitHub Actions validation workflow.
- Document the scope of reproducibility, missing measurements, font substitution
  and human scientific authorship. Preserve all original experimental CSVs and
  the MIT license.
- Correct two-seed-pair wording in the README and report the CIFAR-10 ten-step
  gap consistently with the raw-file check (14.9 FID points after rounding).

This is the first release of this repository. The erratum below concerns the
separate earlier research package, not an earlier tag in this repository.

# Erratum to the previously released reproducibility package

This file records the defects found in the reproducibility package accompanying
our earlier study, the scope of each, and which of that paper's results do not
survive re-measurement.

**The archived release of that package is left unchanged.** Its numbers remain
reproducible exactly as published. Correcting an archived artefact in place
would make the record of what was published harder to recover, not easier. The
corrections live here instead.

Three defects were found, all by applying the checks in this repository to our
own code. Two are in the sampler and one is in the metric.

---

## Defect 1 — single-step update applied to a subsampled grid

`sample_ddpm` applied the ancestral DDPM update

    x <- (1/sqrt(alpha_i)) * (x - beta_i/sqrt(1-alpha_bar_i) * eps) + sqrt(beta_i)*z

which is derived for the transition between **adjacent** chain positions, to
the subsampled grid produced by `make_steps`. When sampling with
`T_infer < T`, consecutive grid entries are `T/T_infer` positions apart and the
gap was unaccounted for.

Magnitude. Each application rescales the state by `1/sqrt(alpha_i)`; traversing
the chain requires a cumulative factor of `1/sqrt(alpha_bar_T)`, which is 31.62
for our schedule. Over the retained indices only:

| steps retained | realised | required | fraction of trajectory |
|---:|---:|---:|---:|
| 1000 | 31.62 | 31.62 | 100 % |
| 200 | 1.99 | 31.62 | 20 % |
| 100 | 1.41 | 31.62 | 10 % |
| 50 | 1.19 | 31.62 | 5 % |
| 10 | 1.04 | 31.62 | 1 % |

At 50 steps the procedure applies 5 % of the required denoising and the output
stays close to the terminal noise distribution. The defect vanishes exactly
when `T_infer == T`, which is why full-chain inspection did not reveal it.

The damage is a dose rather than a constant penalty: it tracks `NFE/T`. Where
that fraction is at or below 10 % the FID of the defective sampler sits between
521 and 536 and barely moves; at 25 % it falls to 383.5 and at 50 % to 184.4.

`sample_ddim` was **not** affected: it already used the generalised form.

Detect it with `diagnostics/grid_consistency.py`. Fix in `src/samplers_fixed.py`.

## Defect 2 — numerically unstable rearrangement of a correct update

The generalised update can be written through the predicted noise or through
the current state. The two forms are algebraically identical — verified to
2.8e-13 over 20 000 random index pairs — but clipping the predicted clean image
to [-1, 1] breaks the identity. The noise form then multiplies the state by
`1/sqrt(alpha_i) > 1` at every step, compensating only through the network
output, so any systematic gain error in the network compounds.

Measured with an oracle denoiser returning the true noise scaled by `1+g`, so
that `g` is the only imperfection present. Ratio of final-sample RMS error,
noise form over state form, T = 1000, mean of three seeds:

| eta | g | NFE 50 | 100 | 200 | 500 | 1000 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.01 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| 1 | 0.05 | 1.0 | 1.0 | 1.2 | 848 | 1391 |
| 0 | 0.01 | 1.0 | 1.0 | 1.1 | 10.2 | 241 |
| 0 | 0.05 | 2.4 | 31.1 | 710 | 1628 | 1911 |

At `g = 0` the two forms agree to 1e-13, but that control checks the harness
rather than the mechanism: with an exact denoiser the clipping never fires. The
separation of causes comes from the identity check, which holds for arbitrary
inputs as long as the estimate is not clipped.

Three points. There is no step budget that is safe independent of the model:
the onset is set jointly by `g` and the step count. Deterministic sampling
(`eta = 0`) is the more vulnerable case. And unlike defect 1, this one is
absent on coarse grids and appears only as the grid is refined — so no single
sanity check catches both.

**On the trained network the cost is larger than the oracle sweep predicts.**
Measured directly, paired, 10 000 samples, one training seed: at T = 1000 and
100 ancestral steps the unstable form costs 61.45 FID points, and 320.97 points
under deterministic sampling; beyond those budgets it returns non-finite
samples. The oracle places the onset too late because it sweeps a *constant*
gain, and the network's actual gain error is not constant: it is below 0.4 % in
magnitude at high noise, reverses sign around `t ≈ 750`, and reaches 36.7 % at
`t = 10` — outside the swept range and on the other side of zero. Use the sweep
to understand the mechanism, not to decide that the fix can be omitted.

Fix, one line — after clipping, re-derive the noise estimate from the clipped
value:

    eps <- (x - sqrt(alpha_bar_i) * x0_clipped) / sqrt(1 - alpha_bar_i)

This reproduces the DDPM posterior form at `eta = 1` and is stable for any
`eta`. It is what the GLIDE/guided-diffusion reference implementation does
unconditionally. In `diffusers`, `DDPMScheduler` computes the posterior mean
directly and is unaffected, while `DDIMScheduler` gates the re-derivation
behind `use_clipped_model_output`, which defaults to `False` while
`clip_sample` defaults to `True`. A pretrained pipeline shipping its own
`scheduler_config.json` may switch the clipping off — Stable Diffusion v1.5
sets `clip_sample: false` — in which case the identity is never broken.

Detect it with `diagnostics/form_identity.py`.

## Defect 3 — FID computation

Independently of the sampler, the FID routine failed on a majority of
measurements. In one re-measurement run 41 of 75 FID values were lost, each
with the same pair of errors:

    LinAlgError: Eigenvalues did not converge      (numpy eigvalsh)
    LinAlgError: SVD did not converge              (fallback numpy svd)

Non-finiteness checks on the generated samples and on the Inception features
passed in every case, so the inputs were clean; the decomposition itself was
failing. The cause is the near-singular matrix square root of the reference
covariance: Inception features on MNIST contain many post-ReLU directions with
zero variance, and both `eigvalsh` and `svd` use divide-and-conquer drivers
that do not converge in that regime — which is why falling back from one to the
other did not help.

Rewritten to avoid forming the covariance product at all. With centred feature
matrices `Xc = Q1 R1` and `Yc = Q2 R2`,

    tr((S1 S2)^(1/2)) = sum(svdvals(R1 R2')) / sqrt((n1-1)(n2-1))

Agreement with the direct evaluation: 1e-14 at full rank, 2e-8 at low rank;
`FID(A, A)` is exactly 0. Cost about +9 % per measurement. A driver cascade
(`gesdd` → `gesvd` → `eigh(A'A)`) guards the remaining cases. See
`src/fid_robust.py`; run it directly for the self-test.

## FID sample size

Metrics were computed at 2 000 samples on CIFAR-10 and 3 000 on MNIST.
Re-measured at 10 000 the values move in the same direction in all sixteen
model–sampler–seed combinations: by 12.5 and 15.6 points on CIFAR-10 in two
independent experiments, and by 3.27 points on MNIST. On CIFAR-10 that offset
is of the same order as the entire difference between correct samplers.

---

## Results of the earlier paper that are withdrawn

- **Table 3** was produced with defect 1 and is withdrawn. Its reported DDPM
  value of 494.8 is reproduced by the defective sampler at the same
  configuration: 494.85, 494.15 and 490.90 across three seeds, a mean of 493.3
  with a standard deviation of 2.1. The corrected sampler in the same cell
  gives 87.74, 96.87 and 94.69.
- **Figures 5 and 7** are derived from Table 3 and are withdrawn.
- **Tables 1 and 2** were computed with a single seed at 2 000 samples. They
  remain in the archive for provenance and should not be cited as measurements.

## A claim that does not reproduce

The manuscript reported a non-monotonic relationship between process length and
FID on CIFAR-10, with a minimum at `T = 500`. Re-measured with the corrected
sampler at 10 000 samples across five training seeds:

| T | published (2 seeds, n = 2000) | re-measured (5 seeds, n = 10000) |
|---:|---|---|
| 200 | 131.5 ± 5.0 | 216.3 ± 46.8 |
| 500 | **113.6 ± 2.5** | 181.3 ± 53.1 |
| 1000 | 117.0 ± 4.7 | **167.3 ± 46.2** |

The minimum moves to the longest process tested. Across all ten pairs of seeds
drawn from the five, "minimum at `T = 500`" occurs in three. The published
conclusion was within run-to-run variation.

## A second claim, with a different outcome

The paper also reported an advantage of deterministic over stochastic sampling
at reduced step budgets. That comparison ran between a defective stochastic
sampler and a correct deterministic one, so it was not a comparison of `eta`.
With both implemented correctly the published direction does not survive as
stated, and what replaces it depends on the benchmark: on MNIST the ordering
reverses in all four cells measured, none of them resolved at three seeds; on
CIFAR-10 the published direction does appear at ten steps, by 14.9 and 15.7
points and resolved at three seeds, and disappears at fifty.

---

## A dead end, recorded so it costs no one else time

Our first diagnostic for defect 2 recorded the peak state magnitude along the
trajectory. As an alarm it works — wherever the two forms differ measurably,
the arm with the larger peak is the arm with the larger damage — but it does
not measure severity, and we initially read it as if it did. At `T = 1000` with
100 ancestral steps the peak reaches 2.5e22 while FID rises only from 85.97 to
147.42: twenty-two orders of magnitude in the instrument against a factor of
1.7 in the output. The final step returns the state to the clipped estimate,
which lies in [-1, 1] by construction, so an excursion in the interior of the
trajectory is bounded before it reaches the output.

We also, at one point, concluded from the oracle sweep that defect 2 could not
explain a degradation observed in our own step-count results. The direct paired
measurement on the trained network refuted that. The oracle sweep assumes one
constant gain characterises a network; ours does not.
