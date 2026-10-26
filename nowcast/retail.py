"""Census advance retail sales, as they were known at each nowcast date.

ALFRED keeps every vintage of every FRED series: each row is a value plus the
window [realtime_start, realtime_end] during which it was the published
number. Asking "what did the March figure say on April 20?" means taking the
row whose window contains April 20. That excludes both later revisions and
months not yet released.
"""
import os
import re

import numpy as np
import pandas as pd
import requests

URL = "https://api.stlouisfed.org/fred/series/observations"

# GICS sub-industry pattern -> Census category (monthly, seasonally adjusted, $M).
# First match wins; anything unmatched uses total retail and food services.
CATEGORY_RULES = [
    (r"restaurant", "RSFSDP"),
    (r"food distributor", "RSFSDP"),  # their customers are restaurants
    (r"apparel retail", "RSCCAS"),
    (r"apparel|footwear|luxury", "RSCCAS"),
    (r"broadline retail|internet", "RSNSR"),
    (r"consumer staples merchandise|hypermarket|general merchandise", "RSGMS"),
    (r"food retail", "RSGCS"),
    (r"drug retail|personal care", "RSHPCS"),
    (r"packaged food|soft drink|brewer|distiller|vintner|beverage|tobacco", "RSDBS"),
    (r"household products", "RSGMS"),
    (r"home improvement|homebuilding|building products", "RSBMGESD"),
    (r"homefurnishing|home furnishing", "RSFHFS"),
    (r"computer & electronics retail|consumer electronics", "RSEAS"),
    (r"automobile|automotive|motorcycle|distributors", "RSMVPD"),
    (r"leisure products", "RSSGHBMS"),
    (r"specialty retail|other specialty", "RSMSR"),
]
DEFAULT_SERIES = "RSAFS"


def category_for(sub_industry):
    s = str(sub_industry).lower()
    for pattern, series in CATEGORY_RULES:
        if re.search(pattern, s):
            return series
    return DEFAULT_SERIES


def map_universe(universe):
    return universe.assign(retail_series=universe["sub_industry"].map(category_for))


def fetch_vintages(client, series, refresh=False):
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("Set FRED_API_KEY (free at fred.stlouisfed.org > My Account > API Keys)")
    url = (f"{URL}?series_id={series}&api_key={key}&file_type=json"
           "&realtime_start=2014-01-01&realtime_end=9999-12-31&observation_start=2013-01-01")
    try:
        data = client.get_json(url, f"fred/{series}.json", refresh)
    except requests.HTTPError:
        return None
    if not data:
        return None
    df = pd.DataFrame(data["observations"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["realtime_end"] = df["realtime_end"].replace("9999-12-31", "2262-01-01")
    for col in ("date", "realtime_start", "realtime_end"):
        df[col] = pd.to_datetime(df[col])
    return df.dropna(subset=["value"]).assign(series=series)[["series", "date", "value", "realtime_start", "realtime_end"]]


def as_of(vintages, t):
    """Monthly values as published on date t, indexed by month."""
    v = vintages[(vintages["realtime_start"] <= t) & (vintages["realtime_end"] >= t)]
    return v.set_index(v["date"].dt.to_period("M"))["value"]


def quarter_months(start, end):
    # Months with most of their days inside the fiscal quarter (handles 52/53-week calendars).
    return pd.period_range(start + pd.Timedelta(days=14), end - pd.Timedelta(days=14), freq="M")


def _yoy(known, months):
    cur = known.reindex(months)
    prior = known.reindex(months - 12)
    ok = cur.notna().values & prior.notna().values
    if not ok.any():
        return np.nan, 0
    return np.log(cur[ok].sum() / prior[ok].sum()), int(ok.sum())


def retail_features(panel, vintages, mapping):
    """Per firm-quarter: category sales growth vs a year ago over the months published
    by the nowcast date, and its change from the previous quarter (acceleration)."""
    series_of = dict(zip(mapping["ticker"], mapping["retail_series"]))
    by_series = dict(tuple(vintages.groupby("series")))
    cache, rows = {}, []
    for r in panel.itertuples():
        s = series_of.get(r.ticker)
        if s not in by_series:
            continue
        key = (s, r.nowcast_date, r.start, r.end)
        if key not in cache:
            known = as_of(by_series[s], r.nowcast_date)
            months = quarter_months(r.start, r.end)
            yoy, n = _yoy(known, months)
            prev_yoy, _ = _yoy(known, months - 3)
            cache[key] = (yoy, yoy - prev_yoy, n)
        yoy, accel, n = cache[key]
        rows.append({"ticker": r.ticker, "end": r.end, "retail__yoy": yoy,
                     "retail__accel": accel, "retail_months": n})
    feats = pd.DataFrame(rows, columns=["ticker", "end", "retail__yoy", "retail__accel", "retail_months"])
    return panel.merge(feats, on=["ticker", "end"], how="left")


def build_retail(client, universe, refresh=False):
    mapping = map_universe(universe)
    frames, missing = [], []
    for s in sorted(set(mapping["retail_series"])):
        df = fetch_vintages(client, s, refresh)
        if df is None:
            missing.append(s)
        else:
            frames.append(df)
    return pd.concat(frames, ignore_index=True), mapping, missing
