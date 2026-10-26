"""Point-in-time feature panel: one row per firm-quarter.

Every feature for a quarter is computed as of its nowcast date, using only
rows whose knowable date (`filed` or `available_at`) is on or before it.
Feature columns are named `<source>__<name>` so models and the data
valuation can select whole sources at once.
"""
import numpy as np
import pandas as pd

NOWCAST_LAG = pd.Timedelta(days=7)  # forecast a week after quarter end, weeks before earnings
MIN_DAYS = 60  # a quarter window needs this many observed days to count
# Wikipedia pageviews start 2015-07-01; 2016 gives every quarter a full year-ago window.
# All models are trained and scored on the same rows, so sources are compared fairly.
STUDY_START = pd.Timestamp("2016-01-01")


def quarter_panel(revenue, lag=NOWCAST_LAG):
    """Targets plus the most recent revenue growth that was public at the nowcast date."""
    q = revenue[["ticker", "start", "end", "filed", "yoy_growth"]].copy()
    q["nowcast_date"] = q["end"] + lag
    q = q[q["filed"] > q["nowcast_date"]]  # already reported by then: nothing to nowcast

    known = (
        revenue.dropna(subset=["yoy_growth"])[["ticker", "filed", "end", "yoy_growth"]]
        .rename(columns={"filed": "known_at", "end": "known_end", "yoy_growth": "rev__last_yoy"})
    )
    panel = pd.merge_asof(
        q.sort_values("nowcast_date"),
        known.sort_values("known_at"),
        left_on="nowcast_date",
        right_on="known_at",
        by="ticker",
    )
    stale = panel["known_end"] >= panel["end"]
    panel.loc[stale, "rev__last_yoy"] = np.nan
    return panel.drop(columns=["known_at", "known_end"]).reset_index(drop=True)


def _window_mean(dates, values, lo, hi):
    mask = (dates >= lo) & (dates <= hi)
    return values[mask].mean() if mask.sum() >= MIN_DAYS else np.nan


def daily_source_features(panel, daily, source):
    """Quarterly features from a daily series with columns ticker, date, value, available_at.

    For each firm-quarter: log growth vs the same window a year earlier and
    vs the previous quarter. Levels are left out: raw attention is not
    comparable across firms (Walmart's page always outdraws Wingstop's).
    """
    year, quarter = pd.DateOffset(years=1), pd.Timedelta(days=91)
    out = []
    for ticker, rows in panel.groupby("ticker"):
        d = daily[daily["ticker"] == ticker]
        for r in rows.itertuples():
            seen = d[d["available_at"] <= r.nowcast_date]
            dates, vals = seen["date"], seen["value"]
            cur = _window_mean(dates, vals, r.start, r.end)
            last_year = _window_mean(dates, vals, r.start - year, r.end - year)
            last_q = _window_mean(dates, vals, r.start - quarter, r.start - pd.Timedelta(days=1))
            out.append({
                "ticker": ticker,
                "end": r.end,
                f"{source}__yoy": np.log1p(cur) - np.log1p(last_year),
                f"{source}__qoq": np.log1p(cur) - np.log1p(last_q),
            })
    return panel.merge(pd.DataFrame(out), on=["ticker", "end"], how="left")


def pageviews_daily(pageviews):
    """Sum a firm's articles into one daily series."""
    return (
        pageviews.groupby(["ticker", "date"], as_index=False)
        .agg(value=("views", "sum"), available_at=("available_at", "max"))
    )


def log_growth(revenue):
    """ln(revenue / revenue a year earlier). Percent growth from a near-zero base
    (cruise lines and casinos in 2021) is unbounded; log growth keeps a crash and its
    recovery symmetric. Non-positive revenue, or a zero year-ago base (new listings),
    is treated as missing."""
    r = revenue.copy()
    g = r["yoy_growth"]
    r["yoy_growth"] = np.log1p(g.where(np.isfinite(g) & (g > -1)))
    return r


def build_panel(revenue, pageviews=None, study_start=STUDY_START):
    panel = quarter_panel(log_growth(revenue))
    panel = panel[panel["start"] >= study_start].reset_index(drop=True)
    if pageviews is not None and not pageviews.empty:
        panel = daily_source_features(panel, pageviews_daily(pageviews), "wiki")
    return panel
