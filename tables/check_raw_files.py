"""Check every number of the PRIA manuscript that rests on the two raw result files
added in v1.0.0: results/e2c_cifar.csv (CIFAR-10 sampler grid, three training seeds)
and results/e3a_mnist.csv (MNIST process length, five training seeds), together with
their sample-size companions e2c_fid_samplesize.csv and e3a_samplesize.csv.

Run:  python tables/check_raw_files.py        (standard library only, CPU, < 1 s)
Exits non-zero if any value disagrees with the manuscript.

Provenance: outputs of Kaggle notebooks notebook8222b01e68 (version 2, script
version 342706933) and notebook73c642a5b7 (version 4, script version 342706439),
copied unchanged from the platform's output storage on November 5, 2026.
"""
import csv, math, os, sys
from statistics import mean, stdev

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
T_CRIT_DF2 = 4.302653   # two-sided 95% Student t, 2 degrees of freedom (three seeds)
FAIL = []

def load(name):
    with open(os.path.join(RES, name), newline="") as f:
        return list(csv.DictReader(f))

def check(label, got, want, tol=0.0):
    ok = (got == want) if isinstance(want, (str, bool, int)) and tol == 0 else abs(got - want) <= tol + 1e-9
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: computed {got}, manuscript {want}")
    if not ok: FAIL.append(label)

def r1(x): return float(f"{x:.1f}")

# ------------------------------------------------------------------ CIFAR-10 sampler grid
c = load("e2c_cifar.csv")
cell = {}
for r in c:
    cell.setdefault((int(r["T_train"]), int(r["nfe"]), r["solver"]), {})[int(r["seed"])] = float(r["fid"])
CELLS = [(200, 10), (1000, 10), (200, 50), (1000, 50)]
SOLV = ["DDPM_repo", "DDPM", "DDIM", "UniPC"]          # DDPM_repo = Eq. (2); DDPM = Eqs. (3), (4)
TABLE3 = {(200, 10): ["495.9 ± 4.7", "117.3 ± 6.3", "102.3 ± 5.0", "86.6 ± 3.5"],
          (1000, 10): ["510.7 ± 0.5", "124.0 ± 14.3", "108.3 ± 8.8", "90.0 ± 4.3"],
          (200, 50): ["394.0 ± 1.1", "90.7 ± 5.3", "91.1 ± 6.0", "85.6 ± 5.4"],
          (1000, 50): ["493.3 ± 2.1", "93.1 ± 4.8", "91.6 ± 4.5", "84.2 ± 1.7"]}
print("Table 3. FID on CIFAR-10, mean ± s.d. over three training seeds")
for k in CELLS:
    for s, want in zip(SOLV, TABLE3[k]):
        v = list(cell[(k[0], k[1], s)].values())
        check(f"T={k[0]} NFE={k[1]} {s}", f"{mean(v):.1f} ± {stdev(v):.1f}", want)

print("\nSection 3. Cost of the first error class on CIFAR-10 (ratio of cell means)")
ratios = [mean(cell[(T, n, 'DDPM_repo')].values()) / mean(cell[(T, n, 'DDPM')].values()) for T, n in CELLS]
check("lowest ratio", r1(min(ratios)), 4.1, 0.0); check("highest ratio", r1(max(ratios)), 5.3, 0.0)
v = cell[(1000, 50, 'DDPM_repo')]
check("archived 494.8 reproduced by seed 1", round(v[1], 1), 494.9, 0.0)
check("archived 494.8 reproduced by seed 2", round(v[2], 1), 494.1, 0.0)

def paired(a, b, T, n):
    d = [cell[(T, n, a)][s] - cell[(T, n, b)][s] for s in (0, 1, 2)]
    m, sd = mean(d), stdev(d)
    return d, m, sd, m / (sd / math.sqrt(3))

print("\nSection 6.1. UniPC against the next-best correct sampler, paired 95% test over three seeds")
sep = 0
for T, n in CELLS:
    means = {s: mean(cell[(T, n, s)].values()) for s in ("DDPM", "DDIM", "UniPC")}
    nb = min(("DDPM", "DDIM"), key=means.get)
    d, m, sd, t = paired(nb, "UniPC", T, n)
    sep += abs(t) > T_CRIT_DF2
    print(f"    T={T} NFE={n}: next best {nb}, gap {m:.2f}, s.d. of paired differences {sd:.2f}, t = {t:.2f}")
check("cells separated (paired 95% test)", sep, 3)

print("\nSection 7. Deterministic against ancestral sampling on CIFAR-10")
for T, want in ((200, 14.9), (1000, 15.7)):
    d, m, sd, t = paired("DDPM", "DDIM", T, 10)
    check(f"DDIM lead at NFE=10, T={T}", r1(m), want, 0.0)
    check(f"  resolved at three seeds (paired 95% test), T={T}", abs(t) > T_CRIT_DF2, True)
gaps50 = [mean(cell[(T, 50, 'DDPM')].values()) - mean(cell[(T, 50, 'DDIM')].values()) for T in (200, 1000)]
check("NFE=50: |DDPM - DDIM| below 1.5 in both cells", max(abs(g) for g in gaps50) < 1.5, True)
check("NFE=50: opposite directions", gaps50[0] * gaps50[1] < 0, True)

print("\nSection 6.1. Seed noise across experiments and benchmarks")
corr = [(T, n, s) for T, n in CELLS for s in ("DDPM", "DDIM", "UniPC")]
sd_cifar = mean(stdev(cell[k].values()) for k in corr)
cv_cifar = mean(stdev(cell[k].values()) / mean(cell[k].values()) for k in corr)
e3b = {}
for r in load("E3b_cifar_5seeds.csv"): e3b.setdefault(int(r["T"]), []).append(float(r["fid"]))
sd_e3b = mean(stdev(v) for v in e3b.values()); cv_e3b = mean(stdev(v) / mean(v) for v in e3b.values())
mn = [r for r in load("E1c_seeds3_SimpleUNet.csv") if r["solver"] in ("DDPM", "DDIM", "UniPC")]
sd_mn = mean(float(r["sd"]) for r in mn); cv_mn = mean(float(r["sd"]) / float(r["mean"]) for r in mn)
check("factor between the two CIFAR-10 experiments (s.d.)", r1(sd_e3b / sd_cifar), 8.4, 0.0)
check("factor between the benchmarks, MNIST / CIFAR-10 (s.d.)", r1(sd_mn / sd_cifar), 3.6, 0.0)
check("same in coefficients of variation, CIFAR-10 experiments", r1(cv_e3b / cv_cifar), 4.5, 0.0)
check("same in coefficients of variation, benchmarks", r1(cv_mn / cv_cifar), 3.6, 0.0)

# ------------------------------------------------------------------ MNIST process length
print("\nSection 6.1. MNIST process length, five training seeds")
e = {}
for r in load("e3a_mnist.csv"):
    e.setdefault(int(r["epochs"]), {}).setdefault(int(r["T_train"]), {})[int(r["seed"])] = float(r["fid"])
best, need_pair, need_appr, need_unp = [], [], [], []
for ep in sorted(e):
    m = {T: mean(v.values()) for T, v in e[ep].items()}
    b1, b2 = sorted(m, key=m.get)[:2]; gap = m[b2] - m[b1]
    s = (stdev(e[ep][b1].values()) + stdev(e[ep][b2].values())) / 2
    d = [e[ep][b2][k] - e[ep][b1][k] for k in range(5)]; sdd = stdev(d)
    best.append(b1); need_pair.append(math.ceil(4 * sdd ** 2 / gap ** 2))
    need_appr.append(math.ceil(4 * s ** 2 / gap ** 2)); need_unp.append(math.ceil(8 * s ** 2 / gap ** 2))
    print(f"    {ep} epochs: best T={b1}, second T={b2}, gap {gap:.2f}, mean s.d. {s:.2f}, "
          f"s.d. of paired differences {sdd:.2f}; not resolved: {gap < 2 * sdd / math.sqrt(5)}")
check("the optimal T changes with the training budget", len(set(best)) > 1, True)
check("seeds needed, paired reading (lowest)", min(need_pair), 44)
check("seeds needed, paired reading (highest)", max(need_pair), 91)
check("seeds needed, unpaired reading (highest)", max(need_unp), 641)

# ------------------------------------------------------------------ sample size (Table 6)
print("\nTable 6. Sample-size offset, FID at the smaller sample minus FID at 10 000")
s = load("e3a_samplesize.csv"); by = {}
for r in s: by.setdefault(int(r["seed"]), {})[int(r["n"])] = float(r["fid"])
d = [by[k][3000] - by[k][10000] for k in sorted(by)]
check("MNIST, length: mean", round(mean(d), 2), 3.27); check("MNIST, length: s.d.", round(stdev(d), 2), 0.74)
check("MNIST, length: positive", f"{sum(x > 0 for x in d)} / {len(d)}", "5 / 5")
check("MNIST range low", round(min(d), 2), 2.11); check("MNIST range high", round(max(d), 2), 3.88)
s = load("e2c_fid_samplesize.csv"); by = {}
for r in s: by.setdefault((int(r["seed"]), r["solver"]), {})[int(r["n"])] = float(r["fid"])
d = [v[2000] - v[10000] for v in by.values()]
check("CIFAR-10, samplers: mean", round(mean(d), 2), 15.56); check("CIFAR-10, samplers: s.d.", round(stdev(d), 2), 0.79)
check("CIFAR-10, samplers: positive", f"{sum(x > 0 for x in d)} / {len(d)}", "6 / 6")
check("CIFAR-10 range high", round(max(d), 2), 16.21)
for k, v in by.items():
    check(f"10 000-sample value equals Table 3 run, seed {k[0]} {k[1]}", round(cell[(1000, 50, k[1])][k[0]], 2), v[10000], 0.0)

print("\n" + ("ALL CHECKS PASS" if not FAIL else f"{len(FAIL)} CHECK(S) FAILED: " + "; ".join(FAIL)))
sys.exit(1 if FAIL else 0)
