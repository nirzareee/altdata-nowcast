"""Point-in-time quarterly revenue from SEC XBRL company facts.

Each value keeps the date it was first filed, so a model trained "as of"
some date only sees revenue that was public by then. Q4 is rarely filed
on its own; it is derived as fiscal-year revenue minus Q1-Q3.
"""
import pandas as pd

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
]

QUARTER_DAYS = (80, 100)
YEAR_DAYS = (350, 380)


def ticker_to_cik(client):
    data = client.get_json(TICKERS_URL, "sec/company_tickers.json")
    return {row["ticker"]: int(row["cik_str"]) for row in data.values()}


def fetch_facts(client, cik, refresh=False):
    return client.get_json(FACTS_URL.format(cik=cik), f"sec/facts_{cik:010d}.json", refresh)


def revenue_rows(facts):
    """Every reported (non-dimensional) revenue value, one row per filing."""
    gaap = facts.get("facts", {}).get("us-gaap", {})
    rows = [
        {
            "tag": tag,
            "start": f["start"],
            "end": f["end"],
            "value": f["val"],
            "filed": f["filed"],
            "form": f["form"],
        }
        for tag in REVENUE_TAGS
        for f in gaap.get(tag, {}).get("units", {}).get("USD", [])
        if "start" in f
    ]
    df = pd.DataFrame(rows, columns=["tag", "start", "end", "value", "filed", "form"])
    for col in ("start", "end", "filed"):
        df[col] = pd.to_datetime(df[col])
    df["days"] = (df["end"] - df["start"]).dt.days
    return df


def first_reported(rows):
    """Collapse to one value per period: the earliest filing, plus the latest restatement.

    When several tags cover the same period, the largest is kept; company
    facts exclude segment breakdowns, so a smaller value is usually a
    narrower concept rather than a different total.
    """
    rows = rows.sort_values("filed")
    per_tag = rows.groupby(["tag", "start", "end"]).agg(
        value=("value", "first"),
        value_latest=("value", "last"),
        filed=("filed", "first"),
        form=("form", "first"),
        days=("days", "first"),
    ).reset_index()
    best = per_tag.sort_values("value", ascending=False).drop_duplicates(["start", "end"])
    return best.sort_values("end").reset_index(drop=True)


def _between(s, bounds):
    return s.between(*bounds)


def quarterly_revenue(facts):
    periods = first_reported(revenue_rows(facts))
    if periods.empty:
        return periods

    quarters = periods[_between(periods["days"], QUARTER_DAYS)].copy()
    quarters["source"] = "reported"
    years = periods[_between(periods["days"], YEAR_DAYS)]

    derived = []
    for y in years.itertuples():
        inside = quarters[(quarters["start"] >= y.start) & (quarters["end"] <= y.end)]
        if len(inside) != 3 or (inside["end"] == y.end).any():
            continue
        derived.append({
            "tag": y.tag,
            "start": inside["end"].max() + pd.Timedelta(days=1),
            "end": y.end,
            "value": y.value - inside["value"].sum(),
            "value_latest": y.value_latest - inside["value_latest"].sum(),
            "filed": y.filed,
            "form": y.form,
            "source": "derived_q4",
        })

    out = pd.concat([quarters, pd.DataFrame(derived)], ignore_index=True)
    out["days"] = (out["end"] - out["start"]).dt.days
    out = out.sort_values("end").drop_duplicates("end", keep="first")
    return add_yoy(out.reset_index(drop=True))


def add_yoy(q):
    """Year-over-year growth against the quarter ending ~1 year earlier."""
    q = q.sort_values("end")
    prior = q[["end", "value"]].rename(columns={"end": "prior_end", "value": "prior_value"})
    q["target_end"] = q["end"] - pd.DateOffset(years=1)
    merged = pd.merge_asof(
        q.sort_values("target_end"),
        prior.sort_values("prior_end"),
        left_on="target_end",
        right_on="prior_end",
        direction="nearest",
        tolerance=pd.Timedelta(days=10),
    )
    merged["yoy_growth"] = merged["value"] / merged["prior_value"] - 1
    return merged.drop(columns=["target_end", "prior_end"]).sort_values("end").reset_index(drop=True)


def build_targets(client, tickers, refresh=False, ciks=None):
    ciks = ciks or ticker_to_cik(client)
    frames, missing = [], []
    for t in tickers:
        cik = ciks.get(t)
        facts = fetch_facts(client, cik, refresh) if cik else None
        q = quarterly_revenue(facts) if facts else pd.DataFrame()
        if q.empty:
            missing.append(t)
            continue
        frames.append(q.assign(ticker=t, cik=cik))
    return pd.concat(frames, ignore_index=True), missing
