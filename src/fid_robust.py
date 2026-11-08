"""
Numerically robust FID. Drop-in replacement for the usual fid_calc.

Why
---
One run of the process-length experiment lost 41 of 75 measurements. Every
failure was the same:

    LinAlgError: Eigenvalues did not converge      (np.linalg.eigvalsh)
    LinAlgError: SVD did not converge              (np.linalg.svd fallback)

The non-finite checks on samples and on Inception features never fired, so the
data were clean. The decomposition itself was failing.

Cause
-----
The usual formulation computes

    s1h = sqrt(S_real)                    # matrix square root of one covariance
    M   = s1h @ S_gen @ s1h
    ev  = eigvalsh(M)

`s1h` is close to singular: Inception features on MNIST contain many dead
post-ReLU directions with zero variance. Adding 1e-6*I makes the matrix
formally positive definite but leaves the condition number of M enormous.
Both `eigvalsh` and `svd` in NumPy use divide-and-conquer drivers (`syevd`,
`gesdd`) that fail to converge in that regime - and they fail together, so
falling back from one to the other does not help.

Fix
---
Do not form the product of covariances at all.

Let Xc and Yc be the centred feature matrices (n x d), so that
S1 = Xc'Xc/(n1-1) and S2 = Yc'Yc/(n2-1). With thin QR factorisations
Xc = Q1 R1 and Yc = Q2 R2, and the identity eig(AB) = eig(BA),

    tr((S1 S2)^(1/2)) = sum(svdvals(R1 R2')) / sqrt((n1-1)(n2-1))

The Q factors are orthogonal and do not affect singular values. No covariance
product is formed, no matrix square root is taken, nothing is inverted. QR and
SVD are the two most stable decompositions available.

Verified: agrees with the direct evaluation (`eigvals(S1@S2)`) to 1e-14 at full
rank and 2e-8 at low rank; agrees with the previous scheme, wherever that
scheme ran at all, to 2.4e-8 relative.

Cost: about 12 s against 3.4 s at n=10000, d=2048. Within a full FID
evaluation of roughly 98 s that is a 9 % increase.

Self-test: python fid_robust.py
"""
import numpy as np


def _svdvals(A):
    """Singular values with a cascade of fallbacks.

    gesdd (NumPy's default) is faster but is exactly the driver that fails to
    converge on hard matrices. gesvd is based on QR iterations: slower, but it
    converges in practice.
    """
    try:
        return np.linalg.svd(A, compute_uv=False), 'gesdd'
    except np.linalg.LinAlgError:
        pass
    try:
        import scipy.linalg
        return scipy.linalg.svd(A, compute_uv=False, lapack_driver='gesvd'), 'gesvd'
    except Exception:
        pass
    # last resort: singular values of A via the eigenvalues of A'A
    w = np.linalg.eigvalsh((A.T @ A + (A.T @ A).T) / 2)
    return np.sqrt(np.clip(w, 0, None)), 'eigh(A^T A)'


def feature_ref(F):
    """Reduce an (n x d) feature matrix to the little that FID needs.

    Returns (mean, R from QR, trace of the covariance, n). The feature matrix
    itself need not be kept: R is d x d.
    """
    F = np.asarray(F, dtype=np.float64)
    n = len(F)
    m = F.mean(0)
    Fc = F - m
    R = np.linalg.qr(Fc, mode='r')
    tr = float((Fc ** 2).sum() / (n - 1))
    return m, R, tr, n


def fid_from_refs(ref1, ref2, return_path=False):
    """FID from two feature_ref reductions."""
    m1, R1, tr1, n1 = ref1
    m2, R2, tr2, n2 = ref2
    sv, path = _svdvals(R1 @ R2.T)
    tr_sqrt = float(sv.sum() / np.sqrt((n1 - 1) * (n2 - 1)))
    d = m1 - m2
    val = float(d @ d + tr1 + tr2 - 2 * tr_sqrt)
    return (val, path) if return_path else val


def fid_from_features(F_real, F_gen, return_path=False):
    """Convenience wrapper: FID straight from two feature matrices."""
    return fid_from_refs(feature_ref(F_real), feature_ref(F_gen), return_path)


# --------------------------------------------------------------------------- #
# Self-test: python fid_robust.py
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    import time

    def fid_reference(X, Y):
        """Slow, honest reference via eigvals of the covariance product."""
        m1, m2 = X.mean(0), Y.mean(0)
        s1, s2 = np.cov(X, rowvar=False), np.cov(Y, rowvar=False)
        ev = np.linalg.eigvals(s1 @ s2)
        tr = np.sum(np.sqrt(np.clip(ev.real, 0, None)))
        d = m1 - m2
        return float(d @ d + np.trace(s1) + np.trace(s2) - 2 * tr)

    rng = np.random.default_rng(0)
    print('Test 1. Agreement with the reference.\n')
    print(f'   {"d":>5s} {"n":>6s} {"rank":>5s} {"reference":>14s} {"robust":>14s}'
          f' {"rel.":>9s} {"path":>8s}')
    ok = True
    for d, n, rank in [(128, 2000, 128), (256, 3000, 80),
                       (512, 4000, 150), (512, 4000, 512)]:
        A = rng.normal(size=(n, rank)) @ rng.normal(size=(rank, d))
        B = rng.normal(size=(n, rank)) @ rng.normal(size=(rank, d)) * 2.0 + 0.3
        a = fid_reference(A, B)
        b, path = fid_from_features(A, B, return_path=True)
        rel = abs(a - b) / abs(a)
        ok &= rel < 1e-6
        print(f'   {d:5d} {n:6d} {rank:5d} {a:14.5f} {b:14.5f} {rel:9.2e} {path:>8s}')

    print('\nTest 2. Degenerate directions (dead ReLU features).')
    d, n = 512, 3000
    A = rng.normal(size=(n, 60)) @ rng.normal(size=(60, d))
    B = rng.normal(size=(n, 60)) @ rng.normal(size=(60, d)) * 1.7
    A[:, ::3] = 0.0          # a third of the features identically zero
    B[:, ::3] = 0.0
    a = fid_reference(A, B)
    b, path = fid_from_features(A, B, return_path=True)
    good = abs(a - b) / abs(a) < 1e-6
    ok &= good
    print(f'   reference {a:.5f}  robust {b:.5f}  path {path}  '
          f'{"OK" if good else "FAILED"}')

    print('\nTest 3. Identical samples must give FID = 0.')
    A = rng.normal(size=(2000, 256))
    v = fid_from_features(A, A)
    good = abs(v) < 1e-6
    ok &= good
    print(f'   FID(A, A) = {v:.3e}  {"OK" if good else "FAILED"}')

    print('\nTest 4. Speed at realistic scale (n=10000, d=2048).')
    X = rng.normal(size=(10000, 300)) @ rng.normal(size=(300, 2048))
    Y = rng.normal(size=(10000, 300)) @ rng.normal(size=(300, 2048)) * 1.5
    t = time.time(); v, path = fid_from_features(X, Y, return_path=True)
    print(f'   FID = {v:.3f} in {time.time()-t:.1f} s, path {path}')

    print('\nRESULT:', 'all checks passed' if ok else 'PROBLEMS FOUND')
    raise SystemExit(0 if ok else 1)
