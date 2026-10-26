"""Diebold-Mariano test for equal forecast accuracy on a panel.

The loss differential (naive error minus model error) is averaged across
firms within each quarter, giving one time series. Its mean is tested
against zero with a Newey-West standard error, which allows quarters close
in time to have correlated differentials.
"""
from math import erf, sqrt

import numpy as np


def newey_west_se(x, lags):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    var = x @ x / n
    for k in range(1, lags + 1):
        w = 1 - k / (lags + 1)
        var += 2 * w * (x[k:] @ x[:-k]) / n
    return sqrt(var / n)


def diebold_mariano(preds, lags=4):
    """Positive t means the model beats naive. Returns (mean differential, t, two-sided p)."""
    d = (preds["naive"] - preds["yoy_growth"]).abs() - (preds["pred"] - preds["yoy_growth"]).abs()
    series = d.groupby(preds["nowcast_date"].dt.to_period("Q")).mean().sort_index().to_numpy()
    se = newey_west_se(series, min(lags, len(series) - 1))
    t = series.mean() / se if se > 0 else np.nan
    p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))) if np.isfinite(t) else np.nan
    return series.mean(), t, p
