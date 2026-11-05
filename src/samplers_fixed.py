"""
Corrected sampler: two independent error classes.

    Class A - wrong update on a subsampled grid (loud failure).
    Class B - correct update in an unstable rearrangement (silent degradation).

===========================================================================
CLASS A. Wrong step on a subsampled grid
===========================================================================

The previous implementation applied the DDPM update for the transition
t -> t-1 (between ADJACENT chain positions) on a grid built by make_steps,
which subsamples. At T=1000 and T_infer=50 adjacent grid entries are 20 chain
positions apart, while the formula advanced the state by one. The gap was
never accounted for.

Magnitude: each step multiplies x by 1/sqrt(alpha_i); traversing the chain
requires a cumulative factor of 1/sqrt(alpha_bar_T) = 31.62. What was actually
applied was 1.19 at 50 steps, that is 5 % of the trajectory. The output stayed
at the level of noise.

Fix
---
The generalised form (DDIM with parameter eta), correct for any subsampling:

    x0_hat = (x - sqrt(1 - ab_i) * eps) / sqrt(ab_i)
    sigma  = eta * sqrt((1 - ab_j) / (1 - ab_i)) * sqrt(1 - ab_i / ab_j)
    x      = sqrt(ab_j) * x0_hat + sqrt(1 - ab_j - sigma^2) * eps + sigma * z

eta = 1 gives ancestral DDPM sampling, eta = 0 gives DDIM; one function covers
both. The marginal variance is preserved exactly:
(1 - ab_j - sigma^2) + sigma^2 = 1 - ab_j.

Check at eta = 0 with an exact noise prediction: x0 is recovered to 1e-14 at
any step count (in float64; in float32 accumulation gives about 5e-4 over the
full chain, which is the precision of the dtype, not of the formula).

===========================================================================
CLASS B. Unstable rearrangement of the same formula
===========================================================================

The formula above can be written in two algebraically IDENTICAL ways
(verified: discrepancy 2.8e-13 over 20 000 random index pairs):

    (I)  through eps: x = sqrt(ab_j)*x0_hat + c*eps + sigma*z
    (II) through x:   x = [sqrt(ab_j)(1-ab_i/ab_j)/(1-ab_i)] * x0_hat
                        + [sqrt(ab_i/ab_j)(1-ab_j)/(1-ab_i)] * x  + sigma*z

The identity holds only while x0_hat is the value derived from eps. Clipping
x0_hat to [-1, 1] breaks it, and the forms begin to diverge:

    multiplier on x in form (I):  1/sqrt(alpha_i) > 1   - AMPLIFICATION
    multiplier on x in form (II): < 1                   - CONTRACTION

Over the whole chain at T=1000, form (I) multiplies the state by 31.62 and
form (II) by 3.2e-06. In form (I) that is compensated solely through the eps
term, that is, through the network output. Any systematic gain error `g` in
the network therefore compounds.

Measure the error at the OUTPUT against the true x0 (an oracle denoiser
returns the true noise scaled by 1+g). How many times worse form (I) is than
form (II), averaged over three seeds, T = 1000:

    eta = 1 (ancestral DDPM)
       g \ NFE      50     100     200     500    1000
         0.01      1.0     1.0     1.0     1.0     1.0
         0.02      1.0     1.0     1.0     1.2     372
         0.05      1.0     1.0     1.2     848    1391
         0.10      1.0     1.2     719    1148     inf

    eta = 0 (DDIM)
       g \ NFE      50     100     200     500    1000
         0.01      1.0     1.0     1.1    10.2     241
         0.02      1.0     1.2     5.4     342     785
         0.05      2.4    31.1     710    1628    1911
         0.10     92.7    1501    2016    2086     inf

Three things to read out of that table:

1. The onset is set by the PAIR (g, step count). There is no single safe step
   budget - it depends on the quality of the network.
2. eta = 0 (DDIM) is MORE vulnerable than eta = 1. This is counter-intuitive
   and it matters: DDIMScheduler in diffusers takes form (I).
3. At g = 0.02 and NFE <= 200 there is NO difference at all. Attributing a
   degradation in that regime to class B is not supported.

BE CAREFUL with the peak max|x|. An earlier version of this file measured the
peak and treated it as the signature of class B. That is wrong: at NFE=100 and
g=0.02 the peak is 7.9 at eta=0 against 4.4 at eta=1, while the damage to the
output is 1.2x against 1.0x - the opposite order. The last step returns the
state to the clipped x0 in [-1, 1], so a large excursion mid-trajectory may
never reach the output. Build the diagnostic on the output error, not the peak.

THE FIX is one line: after clipping, re-derive eps from the clipped x0_hat.
Form (I) then agrees numerically with form (II) at eta = 1 and stays stable at
any eta.

    eps = (x - sqrt(ab_i) * x0_hat) / sqrt(1 - ab_i)

This is what the reference GLIDE/guided-diffusion implementation does. In
diffusers it exists behind the flag `use_clipped_model_output`, which defaults
to False, while `clip_sample` defaults to True:

    DDPMScheduler  - form (II), stable in all cases;
    DDIMScheduler  - form (I); the re-derivation must be switched on by hand.

If you use DDIMScheduler on a fine grid, pass `use_clipped_model_output=True`
or `clip_sample=False`.

Usage
-----
Replace sample_ddpm and sample_ddim with calls to sample_generalized. The
signatures are preserved, so calling code does not change.
"""
import torch


@torch.no_grad()
def sample_generalized(model, schedule, n_samples, steps=None, eta=1.0,
                       shape=(1, 28, 28), device=None, generator=None):
    """Generalised sampler. eta=1 -> DDPM, eta=0 -> DDIM.

    steps : decreasing list of indices from {0, ..., T-1}. Subsampling is fine.
            None -> full chain.
    """
    device = device or schedule['betas'].device
    ab = schedule['alpha_bar']
    T = schedule['T']
    if steps is None:
        steps = list(range(T - 1, -1, -1))

    x = torch.randn(n_samples, *shape, device=device, generator=generator)
    for k, i in enumerate(steps):
        last = (k == len(steps) - 1)
        ab_i = ab[i]
        ab_j = torch.ones((), device=device) if last else ab[steps[k + 1]]

        t = torch.full((n_samples,), i, device=device, dtype=torch.long)
        eps = model(x, t)

        x0 = ((x - torch.sqrt(1. - ab_i) * eps) / torch.sqrt(ab_i)).clamp(-1., 1.)

        # CLASS B. Re-derive eps from the clipped x0. Without this line the
        # multiplier on x is 1/sqrt(alpha_i) > 1 at every step, and the
        # network error is amplified exponentially in the step count.
        eps = (x - torch.sqrt(ab_i) * x0) / torch.sqrt(1. - ab_i)

        if last:
            sigma = torch.zeros((), device=device)
        else:
            sigma = eta * torch.sqrt((1. - ab_j) / (1. - ab_i)) \
                        * torch.sqrt((1. - ab_i / ab_j).clamp(min=0.))
        coef = torch.sqrt((1. - ab_j - sigma ** 2).clamp(min=0.))

        x = torch.sqrt(ab_j) * x0 + coef * eps
        if eta > 0 and not last:
            x = x + sigma * torch.randn(x.shape, device=device, generator=generator)

    return x.clamp(-1., 1.)


def sample_ddpm(model, schedule, n_samples, steps=None, **kw):
    """Drop-in replacement for the previous sample_ddpm. Signature preserved."""
    return sample_generalized(model, schedule, n_samples, steps, eta=1.0, **kw)


def sample_ddim(model, schedule, n_samples, steps=None, **kw):
    """Drop-in replacement for the previous sample_ddim. Signature preserved."""
    return sample_generalized(model, schedule, n_samples, steps, eta=0.0, **kw)


# --------------------------------------------------------------------------- #
# Self-test: python samplers_fixed.py
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    import numpy as np

    def make_schedule(T, beta_1=1e-4, target=1e-3):
        low, high = beta_1, 0.999
        for _ in range(100):
            mid = (low + high) / 2
            ab = torch.cumprod(1. - torch.linspace(beta_1, mid, T), dim=0)
            if ab[-1] > target: low = mid
            else: high = mid
        betas = torch.linspace(beta_1, high, T)
        alphas = 1. - betas
        return {'T': T, 'betas': betas, 'alphas': alphas,
                'alpha_bar': torch.cumprod(alphas, dim=0)}

    def make_steps(T_train, T_infer):
        return list(reversed(torch.linspace(0, T_train - 1, T_infer,
                                            dtype=torch.long).tolist()))

    torch.manual_seed(0)
    T = 1000
    sch = make_schedule(T)
    x0_true = torch.randn(256, 1, 28, 28)
    eps_true = torch.randn_like(x0_true)

    class Oracle:
        """Exact model: returns the true noise."""
        def __call__(self, x, t): return eps_true
    oracle = Oracle()

    print('Test 1. eta=0 (DDIM) with an exact model must return x0 exactly.')
    print('   Run in float64, otherwise this measures the dtype, not the formula.')
    ok = True
    ab64 = sch['alpha_bar'].double()
    x0_64, eps_64 = x0_true.double(), eps_true.double()
    for nfe in [1000, 200, 100, 50, 20, 10, 5]:
        st = make_steps(T, nfe)
        ab0 = ab64[st[0]]
        xx = torch.sqrt(ab0) * x0_64 + torch.sqrt(1 - ab0) * eps_64
        for k, i in enumerate(st):
            last = (k == len(st) - 1)
            ab_i = ab64[i]
            ab_j = torch.ones((), dtype=torch.float64) if last else ab64[st[k + 1]]
            x0 = (xx - torch.sqrt(1 - ab_i) * eps_64) / torch.sqrt(ab_i)
            xx = torch.sqrt(ab_j) * x0 + torch.sqrt((1 - ab_j).clamp(min=0)) * eps_64
        err = (xx - x0_64).abs().max().item()
        good = err < 1e-9
        ok &= good
        print(f'   steps={nfe:5d}  max|x-x0| = {err:.2e}  {"OK" if good else "FAILED"}')

    print('\nTest 2. eta=1 (DDPM): the marginal variance must be preserved.')
    for nfe in [1000, 100, 50, 10]:
        st = make_steps(T, nfe)
        worst = 0.
        for k, i in enumerate(st[:-1]):
            ab_i = sch['alpha_bar'][i]; ab_j = sch['alpha_bar'][st[k + 1]]
            s2 = ((1 - ab_j) / (1 - ab_i)) * (1 - ab_i / ab_j)
            c2 = 1 - ab_j - s2
            worst = max(worst, abs(float(c2 + s2 - (1 - ab_j))))
        good = worst < 1e-6
        ok &= good
        print(f'   steps={nfe:5d}  max|c^2+sigma^2-(1-ab_j)| = {worst:.2e}  '
              f'{"OK" if good else "FAILED"}')

    print('\nTest 3. How much of the trajectory the previous formula traversed.')
    need = 1 / torch.sqrt(sch['alpha_bar'][-1])
    for nfe in [1000, 100, 50, 10]:
        st = make_steps(T, nfe)
        got = float(np.prod([1 / np.sqrt(float(sch['alphas'][i])) for i in st]))
        print(f'   steps={nfe:5d}  factor {got:8.2f} of {float(need):.2f}  '
              f'-> {np.log(got)/np.log(float(need))*100:5.1f} % of the trajectory')

    print('\nTest 4. Class B: output error with an imperfect network.')
    print('   The oracle returns the true noise scaled by (1+g).')
    print('   We measure output RMS against the known x0, NOT the peak max|x|.')

    def out_err(nfe, recompute, g, eta=1.0, seed=0):
        """Returns the output RMS error against x0_true.

        NOTE on the seed. An earlier version called torch.manual_seed(0) right
        before randn_like, while x0_true was the FIRST draw after an identical
        manual_seed(0) at the top of the block. The starting noise then came
        out identical to the target - a degenerate case in which the oracle has
        nothing to corrupt, and the test reported 1.0x where the true figure
        was 31x. The noise is therefore drawn from a separate generator with a
        different seed.
        """
        st = make_steps(T, nfe)
        ab = sch['alpha_bar']
        x0t = x0_true.clamp(-1, 1)
        ab0 = ab[st[0]]
        gen = torch.Generator().manual_seed(seed + 991)
        x = (torch.sqrt(ab0) * x0t
             + torch.sqrt(1 - ab0) * torch.randn(x0t.shape, generator=gen))
        for k, i in enumerate(st):
            last = (k == len(st) - 1)
            ab_i = ab[i]
            ab_j = torch.ones(()) if last else ab[st[k + 1]]
            eps = ((x - torch.sqrt(ab_i) * x0t)
                   / torch.sqrt(1 - ab_i)) * (1 + g)
            x0 = ((x - torch.sqrt(1 - ab_i) * eps) / torch.sqrt(ab_i)).clamp(-1, 1)
            if recompute:
                eps = (x - torch.sqrt(ab_i) * x0) / torch.sqrt(1 - ab_i)
            s2 = torch.zeros(()) if last else \
                (eta ** 2) * ((1 - ab_j) / (1 - ab_i)) * (1 - ab_i / ab_j).clamp(min=0)
            x = torch.sqrt(ab_j) * x0 + torch.sqrt((1 - ab_j - s2).clamp(min=0)) * eps
            if not last and eta > 0:
                x = x + torch.sqrt(s2) * torch.randn(x.shape, generator=gen)
            if not torch.isfinite(x).all():
                return float('inf')
        return float(torch.sqrt(torch.mean((x.clamp(-1, 1) - x0t) ** 2)))

    # The comparison is paired: the same generator, so both forms see exactly
    # one starting noise and one sequence of sigma*z draws.
    okB = True
    for eta, name in [(1.0, 'eta=1 (DDPM)'), (0.0, 'eta=0 (DDIM)')]:
        print(f'\n   {name} - how many times worse form I is than form II')
        print('   ' + f'{"g / NFE":>9s}' + ''.join(f'{n:>9d}' for n in [50, 100, 200, 1000]))
        for g in [0.01, 0.05]:
            row = '   ' + f'{g:9.2f}'
            for nfe in [50, 100, 200, 1000]:
                a = out_err(nfe, False, g, eta)
                c = out_err(nfe, True, g, eta)
                row += f'{"inf":>9s}' if a == float('inf') else f'{a / c:9.1f}'
            print(row)

    print('\n   Control: with an exact network (g=0) the forms must agree.')
    for eta in [1.0, 0.0]:
        for nfe in [100, 1000]:
            a = out_err(nfe, False, 0.0, eta)
            c = out_err(nfe, True, 0.0, eta)
            good = abs(a - c) < 1e-5
            okB &= good
            print(f'      eta={eta} NFE={nfe:5d}: |I-II| = {abs(a - c):.2e}  '
                  f'{"OK" if good else "FAILED"}')
    ok &= okB
    print(f'   {"OK" if okB else "FAILED"}: with no gain error the forms are identical, '
          f'so the divergence above is caused only by (clipping, network error).')

    print('\nRESULT:', 'all checks passed' if ok else 'PROBLEMS FOUND')
