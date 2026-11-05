"""
Class B diagnostic: does an algebraically neutral rearrangement change the output?

No training, no sampling from a real model, no GPU, no weights. NumPy on CPU,
about a minute for the full sweep.

The check
---------
The generalised update admits two algebraically identical forms:

    (I)   x <- sqrt(ab_j) * x0hat + sqrt(1 - ab_j - s^2) * eps + s * z
    (II)  x <- [sqrt(ab_j)(1 - ab_i/ab_j)/(1 - ab_i)] * x0hat
              + [sqrt(ab_i/ab_j)(1 - ab_j)/(1 - ab_i)] * x + s * z

Substituting x = sqrt(ab_i) * x0hat + sqrt(1 - ab_i) * eps turns one into the
other exactly. Part 1 below verifies that numerically.

The identity holds only while x0hat is the value derived from eps. Every
practical implementation clips x0hat to the data range, and clipping breaks it.
After clipping, form (I) reuses the network's original eps while form (II) is
expressed in terms of the current state; form (I) then multiplies the state by
a factor above one at every step and can compensate only through eps, so any
systematic error in the network output compounds.

To measure that without entangling it with model quality, the denoiser is
replaced by an oracle that returns the true noise scaled by (1 + g). The only
imperfection in the system is then g, and the two forms can be compared on
equal terms against a known x0. The substitution trick follows Tang et al.,
ICML 2026, who use an exact posterior to isolate sampler behaviour in discrete
diffusion.

Interpreting the output
-----------------------
Part 1  residual ~1e-13 : the two forms are the same computation. If your
        implementation fails here, the defect is in the algebra, not in clipping.
Part 3  ratio ~1.0      : the two forms agree at that (eta, g, step count).
        ratio >> 1.0    : the arm that reuses eps is diverging.
        ratio inf       : it overflowed.

A study that runs only at coarse grids will see 1.0 everywhere. The error
appears as the grid is refined, which is the opposite of the class A error and
the reason a single sanity check catches neither.

What this does NOT tell you
---------------------------
The sweep assumes one constant g characterises a network. On the model measured
in the accompanying paper the gain error reverses sign along the trajectory and
reaches 0.367 at t = 10 - outside this sweep and on the other side of zero.
The diagnostic therefore detects the mechanism but does not bound it. Apply the
one-line fix unconditionally rather than using this sweep to decide it is safe
to omit. The fix: after clipping x0hat, re-derive eps from the clipped value,

    eps <- (x - sqrt(ab_i) * x0hat) / sqrt(1 - ab_i)

Usage
-----
    python form_identity.py                 # identity, coefficients, full sweep
    python form_identity.py --quick         # identity and coefficients only
"""
from __future__ import annotations

import argparse
import numpy as np

f32 = np.float32


def make_schedule(T: int, beta_1: float = 1e-4, target: float = 1e-3):
    """Linear beta, beta_end solved so that alpha_bar_T hits the target."""
    low, high = beta_1, 0.999
    for _ in range(100):
        mid = 0.5 * (low + high)
        if np.cumprod(1 - np.linspace(beta_1, mid, T))[-1] > target:
            low = mid
        else:
            high = mid
    betas = np.linspace(beta_1, high, T).astype(f32)
    return betas, np.cumprod(1 - betas).astype(f32)


def make_steps(T: int, nfe: int):
    return list(reversed(np.linspace(0, T - 1, min(nfe, T)).astype(int).tolist()))


def check_identity(ab, n: int = 20_000, seed: int = 0) -> float:
    """Largest discrepancy between the two forms over random index pairs.

    Unclipped, so the identity must hold to floating-point precision.
    """
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(n):
        i = rng.integers(1, len(ab))
        j = rng.integers(0, i)
        ai, aj = float(ab[i]), float(ab[j])
        x0, eps = rng.normal(), rng.normal()
        x = np.sqrt(ai) * x0 + np.sqrt(1 - ai) * eps
        s2 = (1 - aj) * (1 - ai / aj) / (1 - ai)
        form_I = np.sqrt(aj) * x0 + np.sqrt(max(0.0, 1 - aj - s2)) * eps
        form_II = ((np.sqrt(aj) * (1 - ai / aj) / (1 - ai)) * x0
                   + (np.sqrt(ai / aj) * (1 - aj) / (1 - ai)) * x)
        worst = max(worst, abs(form_I - form_II))
    return worst


def coefficients(betas, ab):
    """Per-step multiplier on x for each form, and its product over the chain."""
    T = len(ab)
    alphas = 1 - betas
    per_step = [(i, float(np.sqrt(ab[i - 1] / ab[i])),
                 float(np.sqrt(alphas[i]) * (1 - ab[i - 1]) / (1 - ab[i])))
                for i in [T - 1, T // 2, T // 10, 1]]
    pI = float(np.prod([np.sqrt(ab[i - 1] / ab[i]) for i in range(T - 1, 0, -1)]))
    pII = float(np.prod([np.sqrt(alphas[i]) * (1 - ab[i - 1]) / (1 - ab[i])
                         for i in range(T - 1, 0, -1)]))
    return per_step, pI, pII


def sample(ab, T, nfe, gain_error, recompute, eta=1.0, n=4096, seed=0):
    """Run one form against an oracle denoiser with a constant gain error.

    Returns (peak |x| along the trajectory, RMS error of the output vs x0).

    The peak is reported for reference only. It is a poor severity measure:
    the final step returns the state to the clipped x0hat, so a large excursion
    mid-trajectory can be bounded before it reaches the output. Judge by the
    output error.
    """
    rng = np.random.default_rng(seed)
    st = make_steps(T, nfe)
    x0_true = np.clip(rng.normal(size=n), -1, 1).astype(f32)
    ab0 = ab[st[0]]
    x = (np.sqrt(ab0) * x0_true + np.sqrt(1 - ab0) * rng.normal(size=n)).astype(f32)
    peak = float(np.abs(x).max())
    for k, i in enumerate(st):
        last = (k == len(st) - 1)
        ai = ab[i]
        aj = f32(1.0) if last else ab[st[k + 1]]
        eps = (((x - np.sqrt(ai) * x0_true) / np.sqrt(1 - ai))
               * (1 + gain_error)).astype(f32)
        x0 = np.clip(((x - np.sqrt(1 - ai) * eps) / np.sqrt(ai)).astype(f32), -1, 1)
        if recompute:                                   # <- the one-line fix
            eps = ((x - np.sqrt(ai) * x0) / np.sqrt(1 - ai)).astype(f32)
        s2 = f32(0.0) if last else (eta ** 2) * (1 - aj) * (1 - ai / aj) / (1 - ai)
        x = (np.sqrt(aj) * x0 + np.sqrt(max(f32(0), 1 - aj - s2)) * eps).astype(f32)
        if not last and s2 > 0:
            x = (x + np.sqrt(s2) * rng.normal(size=n).astype(f32)).astype(f32)
        if not np.isfinite(x).all():
            return np.inf, np.inf
        peak = max(peak, float(np.abs(x).max()))
    return peak, float(np.sqrt(np.mean((np.clip(x, -1, 1) - x0_true) ** 2)))


def ratio(ab, T, nfe, g, eta, seeds=(0, 1, 2)) -> float:
    """Output RMS error of form (I) over form (II), averaged over seeds."""
    rs = []
    for s in seeds:
        _, a = sample(ab, T, nfe, g, False, eta, seed=s)
        _, b = sample(ab, T, nfe, g, True, eta, seed=s)
        rs.append(np.inf if not np.isfinite(a) else a / b)
    return np.inf if any(not np.isfinite(r) for r in rs) else float(np.mean(rs))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--T", type=int, default=1000)
    ap.add_argument("--quick", action="store_true",
                    help="skip the sweep (parts 1 and 2 only)")
    a = ap.parse_args()

    T = a.T
    betas, ab = make_schedule(T)
    print(f"schedule: T={T}, beta_end={betas[-1]:.5f}, alpha_bar_T={ab[-1]:.3e}\n")

    print("=" * 72)
    print("1. The two forms are algebraically identical")
    print("=" * 72)
    print(f"   largest discrepancy over 20 000 index pairs: {check_identity(ab):.3e}")
    print("   -> they diverge only because clipping x0hat breaks the substitution")
    print("      x = sqrt(ab_i)*x0hat + sqrt(1-ab_i)*eps\n")

    print("=" * 72)
    print("2. The multiplier each form applies to the state")
    print("=" * 72)
    per_step, pI, pII = coefficients(betas, ab)
    print(f"   {'step i':>8s} {'form I':>12s} {'form II':>12s}")
    for i, x, y in per_step:
        print(f"   {i:8d} {x:12.6f} {y:12.6f}")
    print(f"\n   product over the chain: form I = {pI:.4g}, form II = {pII:.4g}")
    print("   -> form (I) amplifies the state and can compensate only through")
    print("      the eps term, that is, through the network output\n")

    if a.quick:
        return

    print("=" * 72)
    print("3. Output error ratio, form (I) over form (II), oracle denoiser")
    print("=" * 72)
    budgets = [50, 100, 200, 500, 1000]
    print(f"   {'eta':>4s} {'g':>6s}" + "".join(f"{n:>9d}" for n in budgets))
    for eta in (1.0, 0.0):
        for g in (0.01, 0.02, 0.05, 0.10):
            row = [ratio(ab, T, n, g, eta) for n in budgets]
            cells = "".join("      inf" if not np.isfinite(r) else f"{r:9.1f}"
                            for r in row)
            print(f"   {eta:>4.0f} {g:>6.2f}{cells}")
    print("\n   1.0 means indistinguishable; large values mean form (I) is")
    print("   diverging; inf means it overflowed single precision.")
    print("   Note that the deterministic case (eta = 0) fails earlier, which is")
    print("   the opposite of what a reading based on injected noise predicts.")


if __name__ == "__main__":
    main()
