"""
Regenerate the paper's tables from the released result files, and check each
against the value printed in the manuscript.

Every table this script can build is rebuilt from a CSV in ../results/ or
recomputed from first principles; nothing is transcribed. Each rebuilt table is
compared against the numbers in the submitted manuscript and marked PASS or
FAIL. Tables whose source file is not in this release are listed explicitly at
the end rather than silently skipped.

    python tables/make_tables.py            # all tables, with checks
    python tables/make_tables.py --quiet    # PASS/FAIL summary only

Exit status is non-zero if any check fails.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
sys.path.insert(0, os.path.join(HERE, "..", "diagnostics"))

FAILURES: list[str] = []
QUIET = False


def load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def say(*a):
    if not QUIET:
        print(*a)


def check(label, got, want, tol=0.02):
    """Compare a rebuilt value against the manuscript. tol is absolute."""
    ok = (got is not None and want is not None
          and math.isfinite(got) and abs(got - want) <= tol)
    if not ok:
        FAILURES.append(f"{label}: rebuilt {got}, manuscript {want}")
    return ok


def banner(n, title, source):
    say(f"\n{'=' * 74}\nTable {n}. {title}\n  source: {source}\n{'=' * 74}")


# --------------------------------------------------------------------------- #
# Table 1 - cumulative rescaling on a subsampled grid (analytic)
# --------------------------------------------------------------------------- #
def table_1():
    from grid_consistency import linear_schedule, uniform_grid, check_grid
    banner(1, "Cumulative rescaling applied by Eq. (4)", "closed form, T = 1000")
    alphas = linear_schedule(1000, alpha_bar_T=1e-3)
    want = {1000: (31.62, 1.00), 500: (5.61, 0.50), 200: (1.99, 0.20),
            100: (1.41, 0.10), 50: (1.19, 0.05), 10: (1.04, 0.01)}
    say(f"{'steps':>7} {'realised':>9} {'required':>9} {'fraction':>9}")
    ok = True
    for nfe, (w_real, w_frac) in want.items():
        r = check_grid(alphas, uniform_grid(1000, nfe))
        say(f"{nfe:>7} {r['realised']:>9.2f} {r['required']:>9.2f} {r['fraction']:>8.1%}")
        ok &= check(f"T1 realised@{nfe}", r["realised"], w_real, 0.01)
        ok &= check(f"T1 fraction@{nfe}", r["fraction"], w_frac, 0.001)
    return ok


# --------------------------------------------------------------------------- #
# Table 2 - Eq. (4) against Eq. (5) on MNIST
# --------------------------------------------------------------------------- #
def table_2():
    banner(2, "FID on MNIST under Eq. (4) and Eq. (5)",
           "results/E1c_results_SimpleUNet.csv")
    rows = load("E1c_results_SimpleUNet.csv")
    want = {(200, 5): 3.5, (200, 10): 4.6, (200, 20): 5.2, (200, 50): 3.7,
            (700, 5): 3.9, (700, 10): 4.8, (700, 20): 5.3, (700, 50): 5.3,
            (1000, 5): 4.1, (1000, 10): 5.5, (1000, 20): 6.2, (1000, 50): 5.9,
            (700, 100): 2.8, (1000, 100): 3.6}
    say(f"{'T':>5} {'NFE':>5} {'Eq.(4)':>9} {'Eq.(5)':>9} {'ratio':>7}")
    ok = True
    for r in rows:
        T, nfe = int(r["T"]), int(r["nfe"])
        num = float(r["DDPM_repo"])
        den = float(r["DDPM"]) if r["DDPM"] else None
        ratio = num / den if den else None
        say(f"{T:>5} {nfe:>5} {num:>9.1f} "
            f"{den if den is None else f'{den:>9.1f}'} "
            f"{'      -' if ratio is None else f'{ratio:>7.1f}'}")
        if (T, nfe) in want:
            ok &= check(f"T2 ratio {T}/{nfe}", ratio, want[(T, nfe)], 0.06)
    return ok


# --------------------------------------------------------------------------- #
# Table 3 - form (I) against form (II) on the trained network
# --------------------------------------------------------------------------- #
def _e1d():
    out = {}
    for r in load("e1d_fid.csv"):
        key = (int(r["T_train"]), int(r["nfe"]), float(r["eta"]), r["form"])
        out[key] = (float(r["fid"]) if r["status"] == "ok" else None, r["status"])
    return out


def table_3():
    banner(3, "form (I) against form (II), MNIST, 10 000 samples",
           "results/e1d_fid.csv")
    d = _e1d()
    FI, FII = "I_без_пересчёта", "II_с_пересчётом"
    want = {(200, 10, 1.0, FI): 116.94, (200, 50, 1.0, FII): 96.25,
            (1000, 50, 1.0, FII): 79.57, (1000, 100, 1.0, FI): 147.42,
            (1000, 100, 1.0, FII): 85.97, (1000, 50, 0.0, FI): 214.35,
            (1000, 100, 0.0, FI): 429.23, (1000, 100, 0.0, FII): 108.26}
    say(f"{'T':>5} {'NFE':>5} {'eta=1 (I)':>11} {'eta=1 (II)':>11} "
        f"{'eta=0 (I)':>11} {'eta=0 (II)':>11}")
    ok = True
    for T in (200, 1000):
        for nfe in (10, 50, 100, 200, 500, 1000):
            if nfe > T or (T, nfe, 1.0, FII) not in d:
                continue
            cells = []
            for eta in (1.0, 0.0):
                for form in (FI, FII):
                    v, st = d.get((T, nfe, eta, form), (None, "missing"))
                    cells.append("       fail" if v is None else f"{v:>11.2f}")
            say(f"{T:>5} {nfe:>5} " + " ".join(cells))
    for k, w in want.items():
        ok &= check(f"T3 {k[0]}/{k[1]}/eta={k[2]}/{'I' if k[3] == FI else 'II'}",
                    d.get(k, (None, ""))[0], w, 0.02)
    # the paired costs quoted in the abstract
    ok &= check("cost eta=1 NFE=100", d[(1000, 100, 1.0, FI)][0]
                - d[(1000, 100, 1.0, FII)][0], 61.45, 0.02)
    ok &= check("cost eta=0 NFE=100", d[(1000, 100, 0.0, FI)][0]
                - d[(1000, 100, 0.0, FII)][0], 320.97, 0.02)
    say("\n  paired cost of form (I): 61.45 points at eta=1, "
        "320.97 at eta=0 (T=1000, 100 steps)")
    return ok


# --------------------------------------------------------------------------- #
# Table 4 - class A ratios recomputed against form (II)
# --------------------------------------------------------------------------- #
def table_4():
    banner(4, "Table 2 ratios recomputed against the corrected form",
           "results/E1c_results_SimpleUNet.csv + results/e1d_fid.csv")
    repo = {(int(r["T"]), int(r["nfe"])): float(r["DDPM_repo"])
            for r in load("E1c_results_SimpleUNet.csv")}
    d = _e1d()
    FII = "II_с_пересчётом"
    want = {(200, 5): (3.5, 3.47), (200, 10): (4.6, 4.58), (200, 20): (5.2, 5.29),
            (200, 50): (3.7, 3.98), (200, 100): (None, 1.87),
            (700, 5): (3.9, 3.87), (700, 10): (4.8, 4.80), (700, 20): (5.3, 5.46),
            (700, 50): (5.3, 5.83), (700, 100): (2.8, 5.17),
            (1000, 5): (4.1, 4.10), (1000, 10): (5.5, 5.59), (1000, 20): (6.2, 6.35),
            (1000, 50): (5.9, 6.71), (1000, 100): (3.6, 6.16)}
    say(f"{'T':>5} {'NFE':>5} {'Eq.(4)':>9} {'form (II)':>10} "
        f"{'old':>6} {'new':>6}")
    ok = True
    missing = []
    for (T, nfe), num in sorted(repo.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        cell = d.get((T, nfe, 1.0, FII))
        if cell is None or cell[0] is None:
            missing.append(f"T={T}/NFE={nfe}")
            continue
        new = num / cell[0]
        old = want.get((T, nfe), (None, None))[0]
        say(f"{T:>5} {nfe:>5} {num:>9.1f} {cell[0]:>10.2f} "
            f"{'     -' if old is None else f'{old:>6.1f}'} {new:>6.2f}")
        if (T, nfe) in want:
            ok &= check(f"T4 ratio {T}/{nfe}", new, want[(T, nfe)][1], 0.02)
    if missing:
        say("\n  form (II) not measured at: " + ", ".join(missing))
        say("  These are the cells the top-up run closes; the manuscript states")
        say("  the corrected range over the five cells it does cover.")
    return ok


# --------------------------------------------------------------------------- #
# Tables 6 and 7 - coefficients and the oracle sweep (analytic)
# --------------------------------------------------------------------------- #
def tables_6_7():
    from form_identity import make_schedule, coefficients, check_identity, ratio
    betas, ab = make_schedule(1000)
    banner(6, "The multiplier each form applies to the state",
           "closed form, T = 1000")
    _, pI, pII = coefficients(betas, ab)
    say(f"  form (I)  product over the chain: {pI:.2f}")
    say(f"  form (II) product over the chain: {pII:.3g}")
    ok = check("T6 form I product", pI, 31.62, 0.01)
    ok &= check("T6 form II product", pII * 1e6, 3.2, 0.05)
    resid = check_identity(ab)
    say(f"  identity residual over 20 000 index pairs: {resid:.3e}")
    ok &= check("identity residual exponent", math.log10(resid), -12.5, 0.6)

    banner(7, "Ratio of final-sample RMS error, form (I) over form (II)",
           "diagnostics/form_identity.py, oracle denoiser")
    want = {(1.0, 0.05): [1.0, 1.0, 1.2, 848.0, 1391.0],
            (0.0, 0.05): [2.4, 31.1, 710.0, 1628.0, 1911.0]}
    budgets = [50, 100, 200, 500, 1000]
    say(f"  {'eta':>4} {'g':>6}" + "".join(f"{n:>9}" for n in budgets))
    for eta in (1.0, 0.0):
        for g in (0.01, 0.02, 0.05, 0.10):
            row = [ratio(ab, 1000, n, g, eta) for n in budgets]
            say(f"  {eta:>4.0f} {g:>6.2f}"
                + "".join("      inf" if not math.isfinite(r) else f"{r:9.1f}"
                          for r in row))
            if (eta, g) in want:
                for n, r, w in zip(budgets, row, want[(eta, g)]):
                    ok &= check(f"T7 eta={eta} g={g} nfe={n}", r, w,
                                max(0.1, 0.03 * w))
    return ok


# --------------------------------------------------------------------------- #
# Table 8 - measured gain error
# --------------------------------------------------------------------------- #
def table_8():
    banner(8, "Measured gain error on the trained MNIST network",
           "results/e1d_gain.csv")
    rows = {int(r["t"]): (float(r["g"]), float(r["resid_frac"]))
            for r in load("e1d_gain.csv")}
    say(f"{'t':>6} {'g':>10} {'resid_frac':>11}")
    for t in sorted(rows, reverse=True):
        say(f"{t:>6} {rows[t][0]:>+10.5f} {rows[t][1]:>11.4f}")
    ok = check("T8 g at t=800", rows[800][0], 0.00166, 1e-4)
    ok &= check("T8 g at t=700", rows[700][0], -0.00336, 1e-4)
    ok &= check("T8 g at t=10", rows[10][0], -0.36655, 1e-4)
    ok &= check("T8 resid at t=10", rows[10][1], 0.3069, 1e-3)
    say("\n  sign reversal between t = 800 and t = 700; magnitude 36.7 % at t = 10")
    return ok


# --------------------------------------------------------------------------- #
# Tables 9 and 10 - MNIST sampler comparison and its resolution limit
# --------------------------------------------------------------------------- #
def tables_9_10():
    banner(9, "FID on MNIST, mean +- s.d. over three seeds",
           "results/E1c_seeds3_SimpleUNet.csv")
    cells = {}
    for r in load("E1c_seeds3_SimpleUNet.csv"):
        cells.setdefault((int(r["nfe"]), int(r["T"])), []).append(
            (r["solver"], float(r["mean"]), float(r["sd"])))
    say(f"{'NFE':>5} {'T':>6}  " + "  ".join(f"{s:>16}" for s in ("DDPM", "DDIM", "UniPC")))
    for k in sorted(cells):
        by = {s: (m, sd) for s, m, sd in cells[k]}
        say(f"{k[0]:>5} {k[1]:>6}  "
            + "  ".join(f"{by[s][0]:>10.1f} +-{by[s][1]:>4.1f}"
                        for s in ("DDPM", "DDIM", "UniPC")))

    banner(10, "Seeds required to resolve best against second best",
            "computed from Table 9; s = mean of the two arms' s.d.")
    want = {(10, 200): (12, 24), (10, 1000): (19, 37),
            (50, 200): (26, 51), (50, 1000): (1169, 2337)}
    say(f"{'NFE':>5} {'T':>6} {'comparison':>18} {'gap':>7} {'s':>7} "
        f"{'paired':>8} {'unpaired':>9}")
    ok = True
    for k in sorted(cells):
        v = sorted(cells[k], key=lambda x: x[1])
        (a, ma, sa), (b, mb, sb) = v[0], v[1]
        gap, s = mb - ma, (sa + sb) / 2
        n_p = math.ceil(4 * s * s / gap / gap)
        n_u = math.ceil(8 * s * s / gap / gap)
        say(f"{k[0]:>5} {k[1]:>6} {f'{a} vs {b}':>18} {gap:>7.2f} {s:>7.2f} "
            f"{n_p:>8} {n_u:>9}")
        ok &= check(f"T10 paired {k}", n_p, want[k][0], 0)
        ok &= check(f"T10 unpaired {k}", n_u, want[k][1], 0)
    say("\n  No cell is resolvable at three seeds. The NFE = 50 rows additionally")
    say("  carry class B in the DDPM arm - see Section 6.1 of the manuscript.")
    return ok


# --------------------------------------------------------------------------- #
# Table 14 - CIFAR-10 process length
# --------------------------------------------------------------------------- #
def table_14():
    banner(14, "FID on CIFAR-10 by process length, five training seeds",
           "results/E3b_cifar_5seeds.csv")
    d = {}
    for r in load("E3b_cifar_5seeds.csv"):
        d.setdefault(int(r["T"]), {})[int(r["seed"])] = float(r["fid"])
    want = {200: (216.3, 46.8, 107.4), 500: (181.3, 53.1, 108.3),
            1000: (167.3, 46.2, 108.7)}
    say(f"{'T':>6} {'mean':>8} {'s.d.':>7} {'min':>8} {'max':>8} {'spread':>8}")
    ok = True
    for T in sorted(d):
        v = list(d[T].values())
        mean = sum(v) / len(v)
        sd = math.sqrt(sum((x - mean) ** 2 for x in v) / (len(v) - 1))
        say(f"{T:>6} {mean:>8.1f} {sd:>7.1f} {min(v):>8.1f} {max(v):>8.1f} "
            f"{max(v) - min(v):>8.1f}")
        ok &= check(f"T14 mean {T}", mean, want[T][0], 0.05)
        ok &= check(f"T14 sd {T}", sd, want[T][1], 0.05)
        ok &= check(f"T14 spread {T}", max(v) - min(v), want[T][2], 0.05)

    # the two-seed enumeration behind "three of ten pairs"
    seeds = sorted(d[200])
    hits = 0
    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            m = {T: (d[T][seeds[i]] + d[T][seeds[j]]) / 2 for T in d}
            hits += (min(m, key=m.get) == 500)
    say(f"\n  two-seed draws reproducing the published optimum at T = 500: "
        f"{hits} of 10")
    ok &= check("T14 pair count", hits, 3, 0)
    return ok


MISSING = """
Table 3, Table 6 and the CIFAR-10 sampler and MNIST process-length numbers
-----------------------------------------------------------------------
  are checked by tables/check_raw_files.py from results/e2c_cifar.csv,
  results/e2c_fid_samplesize.csv, results/e3a_mnist.csv and
  results/e3a_samplesize.csv (one row per training seed).
"""


def main() -> int:
    global QUIET
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--quiet", action="store_true")
    QUIET = ap.parse_args().quiet

    results = {
        "Table 1": table_1(),
        "Table 2": table_2(),
        "Table 3": table_3(),
        "Table 4": table_4(),
        "Tables 6-7": tables_6_7(),
        "Table 8": table_8(),
        "Tables 9-10": tables_9_10(),
        "Table 14": table_14(),
    }
    print(MISSING)
    print("=" * 74)
    for name, ok in results.items():
        print(f"  {name:<14} {'PASS' if ok else 'FAIL'}")
    if FAILURES:
        print("\nmismatches against the manuscript:")
        for f in FAILURES:
            print("  " + f)
        return 1
    print("\nEvery rebuilt value matches the manuscript.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
