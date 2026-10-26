"""Exact Shapley values over data sources, with bootstrap confidence intervals.

v(S) is the walk-forward forecast skill using only sources in S, so a
source's Shapley value is its average marginal skill across all subsets of
the other sources. Exact computation needs 2^n model runs; with up to ~10
sources that is affordable.

Confidence intervals resample the out-of-sample forecasts, not the models:
each of the 2^n coalitions is run once, then skill is recomputed on
resampled rows. Resampling is clustered (whole firms, or whole quarters)
because forecasts for one firm, or one quarter, share errors.
"""
from itertools import chain, combinations
from math import factorial

import numpy as np

from .model import COVID, evaluate, walk_forward


def shapley(sources, value):
    n = len(sources)
    cache = {}

    def v(subset):
        key = frozenset(subset)
        if key not in cache:
            cache[key] = value(sorted(key))
        return cache[key]

    phi = {}
    for s in sources:
        others = [o for o in sources if o != s]
        total = 0.0
        for k in range(n):
            w = factorial(k) * factorial(n - k - 1) / factorial(n)
            for subset in combinations(others, k):
                total += w * (v(subset + (s,)) - v(subset))
        phi[s] = total
    return phi, cache


def source_shapley(panel, sources, model="huber", metric="skill"):
    return shapley(sources, lambda s: evaluate(panel, s, model)[metric])


def coalition_errors(panel, sources, model="huber", holdout=False):
    """Absolute out-of-sample errors for every subset of sources, on identical rows."""
    subsets = list(chain.from_iterable(combinations(sources, k) for k in range(len(sources) + 1)))
    errors, base = {}, None
    for subset in subsets:
        p = walk_forward(panel, list(subset), model, holdout)
        if base is None:
            base = p
            naive_err = (p["naive"] - p["yoy_growth"]).abs().to_numpy()
        assert (p["ticker"].to_numpy() == base["ticker"].to_numpy()).all()
        errors[frozenset(subset)] = (p["pred"] - p["yoy_growth"]).abs().to_numpy()
    return errors, naive_err, base[["ticker", "end", "nowcast_date"]].reset_index(drop=True)


def _skill(err, naive_err, idx):
    return 1 - err[idx].mean() / naive_err[idx].mean()


def bootstrap_shapley(panel, sources, model="huber", metric="skill", n_boot=1000,
                      clusters=("ticker", "quarter"), seed=0, holdout=False):
    """Point Shapley values plus 95% intervals, one interval per clustering.

    Returns {source: {"value": v, "ticker": (low, high), "quarter": (low, high)}}.
    """
    errors, naive_err, rows = coalition_errors(panel, sources, model, holdout)
    keep = np.arange(len(rows))
    if metric == "skill_ex_covid":
        keep = keep[~rows["end"].between(*COVID).to_numpy()]
    rows = rows.iloc[keep].reset_index(drop=True)
    errors = {k: e[keep] for k, e in errors.items()}
    naive_err = naive_err[keep]

    def phi(idx):
        return shapley(sources, lambda s: _skill(errors[frozenset(s)], naive_err, idx))[0]

    point = phi(np.arange(len(rows)))
    out = {s: {"value": point[s]} for s in sources}
    for cluster in clusters:
        groups = rows["nowcast_date"].dt.to_period("Q") if cluster == "quarter" else rows[cluster]
        codes = groups.astype(str).to_numpy()
        members = [np.flatnonzero(codes == g) for g in np.unique(codes)]
        rng = np.random.default_rng(seed)
        draws = {s: [] for s in sources}
        for _ in range(n_boot):
            pick = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[i] for i in pick])
            for s, v in phi(idx).items():
                draws[s].append(v)
        for s in sources:
            out[s][cluster] = tuple(np.percentile(draws[s], [2.5, 97.5]))
    return out


def verdict(r):
    """'helps' or 'hurts' only when both clusterings put the interval on one side of zero."""
    lows, highs = zip(r["ticker"], r["quarter"])
    if min(lows) > 0:
        return "helps"
    if max(highs) < 0:
        return "hurts"
    return "no effect"
