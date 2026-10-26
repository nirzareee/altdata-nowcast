import numpy as np
import pandas as pd
import pytest

from nowcast.retail import category_for, retail_features


def vintages(revised_factor=1.0, revision_lag_years=1):
    """Monthly sales growing 1%/month. Each month is first published on the 15th of
    the next month, then revised by `revised_factor` after `revision_lag_years`."""
    rows = []
    for k, month in enumerate(pd.period_range("2014-01", "2017-12", freq="M")):
        value = 100 * 1.01 ** k
        first = month.end_time.normalize() + pd.Timedelta(days=15)
        revised = first + pd.DateOffset(years=revision_lag_years)
        rows.append({"date": month.start_time, "value": value,
                     "realtime_start": first, "realtime_end": revised - pd.Timedelta(days=1)})
        rows.append({"date": month.start_time, "value": value * revised_factor,
                     "realtime_start": revised, "realtime_end": pd.Timestamp("2262-01-01")})
    return pd.DataFrame(rows).assign(series="RSFSDP")


PANEL = pd.DataFrame({
    "ticker": ["A"],
    "start": [pd.Timestamp("2016-01-01")],
    "end": [pd.Timestamp("2016-03-31")],
    "nowcast_date": [pd.Timestamp("2016-04-07")],
})
MAP = pd.DataFrame({"ticker": ["A"], "retail_series": ["RSFSDP"]})


def test_only_months_published_by_nowcast_date():
    row = retail_features(PANEL, vintages(), MAP).iloc[0]
    assert row["retail_months"] == 2  # March is published Apr 15, after the Apr 7 nowcast
    assert row["retail__yoy"] == pytest.approx(12 * np.log(1.01))


def test_revisions_after_nowcast_date_are_ignored():
    # Revisions land 3 years later, after the Apr 2016 nowcast: feature must not change.
    first = retail_features(PANEL, vintages(1.0, 3), MAP).iloc[0]
    revised = retail_features(PANEL, vintages(1.5, 3), MAP).iloc[0]
    assert first["retail__yoy"] == pytest.approx(revised["retail__yoy"])


def test_revisions_before_nowcast_date_are_used():
    # Year-ago months (early 2015) were revised in early 2016, before the nowcast:
    # an analyst on Apr 7 2016 saw revised year-ago values against first-print current ones.
    row = retail_features(PANEL, vintages(1.5, 1), MAP).iloc[0]
    assert row["retail__yoy"] == pytest.approx(12 * np.log(1.01) - np.log(1.5))


def test_unmapped_firm_gets_no_features():
    out = retail_features(PANEL.assign(ticker="Z"), vintages(), MAP)
    assert out["retail__yoy"].isna().all()


@pytest.mark.parametrize("sub_industry, series", [
    ("Restaurants", "RSFSDP"),
    ("Apparel Retail", "RSCCAS"),
    ("Consumer Staples Merchandise Retail", "RSGMS"),
    ("Broadline Retail", "RSNSR"),
    ("Packaged Foods & Meats", "RSDBS"),
    ("Home Improvement Retail", "RSBMGESD"),
    ("Hotels, Resorts & Cruise Lines", "RSAFS"),
    ("Interactive Media & Services", "RSAFS"),
])
def test_category_mapping(sub_industry, series):
    assert category_for(sub_industry) == series


def test_parse_fred_response(monkeypatch):
    from nowcast.retail import fetch_vintages

    class FakeClient:
        def get_json(self, url, cache_key=None, refresh=False):
            assert "api_key=k" in url and cache_key == "fred/RSFSDP.json"  # key never in cache path
            return {"observations": [
                {"realtime_start": "2016-02-12", "realtime_end": "2016-03-14", "date": "2016-01-01", "value": "100.5"},
                {"realtime_start": "2016-03-15", "realtime_end": "9999-12-31", "date": "2016-01-01", "value": "101.0"},
                {"realtime_start": "2016-03-15", "realtime_end": "9999-12-31", "date": "2016-02-01", "value": "."},
            ]}

    monkeypatch.setenv("FRED_API_KEY", "k")
    df = fetch_vintages(FakeClient(), "RSFSDP")
    assert len(df) == 2  # "." (missing) dropped
    assert df["realtime_end"].max() == pd.Timestamp("2262-01-01")
