"""
Gate 2 demo: how good does an optimized EMA crossover look on PURE NOISE,
and what does an in-sample permutation test do about it?

Setup
- 8,760 bars = 4 years of 4h candles, Gaussian log-returns (sigma 0.008). No fees, no slippage.
- Strategy: EMA fast/slow crossover, long/short flip. Grid: fast 5-55 (step 5), slow 20-190 (step 10).
- Return per bar = position[t-1] * log_return[t]. Profit factor (PF) measured per bar.
- Permutation test: shuffle the bar returns (same distribution, same start/end, no patterns),
  RE-OPTIMIZE the whole grid on every shuffle, count how often noise beats the real best PF.

Run:  pip install numpy scipy && python permutation_demo.py     (about 2 minutes)
"""
import numpy as np
from scipy.signal import lfilter

N_BARS, SIGMA = 8760, 0.008
FASTS = list(range(5, 60, 5))
SLOWS = list(range(20, 200, 10))
GRID = [(f, s) for f in FASTS for s in SLOWS if f < s]
SPANS = sorted(set(FASTS + SLOWS))


def ema(x, span):
    a = 2.0 / (span + 1.0)
    y, _ = lfilter([a], [1, a - 1], x, zi=[(1 - a) * x[0]])
    return y


def best_pf(r):
    logp = np.cumsum(r)
    emas = {sp: ema(logp, sp) for sp in SPANS}
    best = 0.0
    for f, s in GRID:
        strat = np.sign(emas[f] - emas[s])[:-1] * r[1:]
        loss = -strat[strat < 0].sum()
        if loss > 0:
            best = max(best, strat[strat > 0].sum() / loss)
    return best


def perm_p(r, n_perm, rng):
    real = best_pf(r)
    beats = sum(best_pf(rng.permutation(r)) >= real for _ in range(n_perm))
    return real, (beats + 1) / (n_perm + 1)


if __name__ == "__main__":
    # Experiment 1: best-of-grid PF across 300 pure-noise price histories
    rng = np.random.default_rng(42)
    pfs = np.array([best_pf(rng.normal(0, SIGMA, N_BARS)) for _ in range(300)])
    print(f"[300 noise series] best PF: median {np.median(pfs):.3f} | 90th pct {np.percentile(pfs, 90):.3f} | max {pfs.max():.3f}")
    print(f"[300 noise series] best PF >= 1.05: {(pfs >= 1.05).mean():.1%}")

    # Experiment 2: full permutation test (200 shuffles, re-optimized) on 20 noise series
    out = []
    for seed in range(100, 120):
        r = np.random.default_rng(seed).normal(0, SIGMA, N_BARS)
        out.append(perm_p(r, 200, np.random.default_rng(seed + 1000)))
    reals, ps = np.array([a for a, _ in out]), np.array([b for _, b in out])
    print(f"[20 noise series] best PF >= 1.05 (looks like an edge): {(reals >= 1.05).sum()} of 20")
    print(f"[20 noise series] passed permutation test at p < 0.05:  {(ps < 0.05).sum()} of 20 | median p {np.median(ps):.2f}")
