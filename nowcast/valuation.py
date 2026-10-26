"""Exact Shapley values over data sources.

v(S) is the walk-forward forecast skill using only sources in S, so a
source's Shapley value is its average marginal skill across all subsets of
the other sources. Exact computation needs 2^n model runs; with up to ~10
sources that is affordable.
"""
from itertools import combinations
from math import factorial

from .model import evaluate


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


def source_shapley(panel, sources, model="ridge"):
    return shapley(sources, lambda s: evaluate(panel, s, model)["skill"])
