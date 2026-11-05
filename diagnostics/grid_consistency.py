"""
Class A diagnostic: is the update rule being applied on the grid it was derived for?

No training, no sampling, no GPU, no model weights. Pure arithmetic on the
noise schedule; runs in milliseconds.

The check
---------
A single-step DDPM update multiplies the state by 1/sqrt(alpha_i) at each
application. Traversing the whole chain requires a cumulative factor of
1/sqrt(alpha_bar_T). So for any step grid S drawn from a schedule {alpha_i},
an implementation that advances the state by exactly one chain position per
iteration satisfies

    sum_{i in S} log(1 / sqrt(alpha_i))  ==  log(1 / sqrt(alpha_bar_T))

and this holds only when S is the full chain. On a subsampled grid the left
side falls short, and the ratio of the two sides is the fraction of the
required denoising the implementation actually applies.

Interpreting the output
-----------------------
fraction == 1.0    the loop and the update agree.
fraction  < 1.0    the update advances one chain position while the loop
                   advances several; the sampler leaves (1 - fraction) of the
                   denoising unapplied and the output stays close to noise.

The failure is invisible at S == full chain, which is why a full-chain sanity
check does not detect it. Run this at the step budget you actually report.

Usage
-----
    python grid_consistency.py                    # reproduces Table 1
    python grid_consistency.py --T 1000 --nfe 50  # one grid

Or import check_grid() and call it on your own schedule.
"""
from __future__ import annotations

import argparse
import numpy as np


def linear_schedule(T: int, beta_1: float = 1e-4, alpha_bar_T: float = 1e-3):
    """Linear beta with beta_end solved so that alpha_bar_T hits the target.

    This is the schedule family used in the accompanying paper. Any schedule
    works: the diagnostic only needs the per-step alphas.
    """
    low, high = beta_1, 0.999
    for _ in range(200):
        mid = 0.5 * (low + high)
        if np.cumprod(1.0 - np.linspace(beta_1, mid, T))[-1] > alpha_bar_T:
            low = mid
        else:
            high = mid
    betas = np.linspace(beta_1, high, T)
    return 1.0 - betas


def uniform_grid(T: int, nfe: int):
    """The usual reduced-budget grid: nfe indices spread over the chain."""
    return np.unique(np.linspace(0, T - 1, min(nfe, T)).astype(int))


def check_grid(alphas: np.ndarray, steps) -> dict:
    """Evaluate the identity on one grid.

    alphas : per-step alpha_i of the training chain, length T.
    steps  : the indices the inference loop will visit.

    Returns realised and required log-factors, their exponentials, and the
    fraction of the trajectory the implementation traverses.
    """
    steps = np.asarray(list(steps), dtype=int)
    if steps.min() < 0 or steps.max() >= len(alphas):
        raise ValueError("step index outside the schedule")

    log_realised = float(np.sum(-0.5 * np.log(alphas[steps])))
    log_required = float(-0.5 * np.log(np.prod(alphas)))
    return dict(
        n_steps=len(steps),
        realised=float(np.exp(log_realised)),
        required=float(np.exp(log_required)),
        fraction=log_realised / log_required,
        consistent=bool(np.isclose(log_realised, log_required, rtol=1e-9)),
    )


def table_1(T: int = 1000, alpha_bar_T: float = 1e-3,
            budgets=(1000, 500, 200, 100, 50, 10)) -> None:
    alphas = linear_schedule(T, alpha_bar_T=alpha_bar_T)
    print(f"chain T = {T}, alpha_bar_T = {np.prod(alphas):.3e}\n")
    print(f"{'steps':>7}  {'realised':>9}  {'required':>9}  {'fraction':>9}  ok")
    for nfe in budgets:
        r = check_grid(alphas, uniform_grid(T, nfe))
        print(f"{r['n_steps']:>7}  {r['realised']:>9.2f}  {r['required']:>9.2f}"
              f"  {r['fraction']:>8.1%}  {'yes' if r['consistent'] else 'NO'}")
    print("\nA row with fraction < 100 % means the sampler applies that share of")
    print("the required denoising. Only the full chain is self-consistent, which")
    print("is why sampling once at T_infer = T proves nothing about the budgets")
    print("actually reported.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--nfe", type=int, default=None,
                    help="single step budget; omit to reproduce Table 1")
    ap.add_argument("--alpha-bar-T", type=float, default=1e-3)
    a = ap.parse_args()

    if a.nfe is None:
        table_1(a.T, a.alpha_bar_T)
        return

    alphas = linear_schedule(a.T, alpha_bar_T=a.alpha_bar_T)
    r = check_grid(alphas, uniform_grid(a.T, a.nfe))
    print(f"steps kept      {r['n_steps']}")
    print(f"realised factor {r['realised']:.4f}")
    print(f"required factor {r['required']:.4f}")
    print(f"fraction        {r['fraction']:.2%}")
    print("consistent      " + ("yes" if r["consistent"] else
                                "NO - the update is on the wrong grid"))


if __name__ == "__main__":
    main()
