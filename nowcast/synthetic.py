"""Synthetic firms where revenue growth is partly driven by web attention.

Used for tests and to sanity-check the pipeline before real data: a
correct pipeline must find the planted signal and must not find signal
when it is removed.
"""
import numpy as np
import pandas as pd


def make_firms(n_firms=15, start="2015-01-01", end="2025-12-31", signal=0.7, seed=0):
    rng = np.random.default_rng(seed)
    quarters = pd.period_range(start, end, freq="Q")
    revenue, views = [], []
    for i in range(n_firms):
        ticker = f"F{i:02d}"
        level = np.cumsum(rng.normal(0.02, 0.08, len(quarters)))
        base = rng.uniform(7, 11)
        for j, q in enumerate(quarters):
            q_start, q_end = q.start_time.normalize(), q.end_time.normalize()
            days = pd.date_range(q_start, q_end, freq="D")
            v = np.exp(base + level[j] + rng.normal(0, 0.15, len(days)))
            views.append(pd.DataFrame({"ticker": ticker, "article": ticker, "date": days, "views": v.round()}))
            if j >= 4:
                yoy = signal * (level[j] - level[j - 4]) + rng.normal(0.03, 0.03)
                revenue.append({
                    "ticker": ticker,
                    "start": q_start,
                    "end": q_end,
                    "filed": q_end + pd.Timedelta(days=int(rng.integers(25, 45))),
                    "yoy_growth": yoy,
                })
    views = pd.concat(views, ignore_index=True)
    views["available_at"] = views["date"] + pd.Timedelta(days=1)
    return pd.DataFrame(revenue), views
