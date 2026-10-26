import numpy as np
import pandas as pd

from nowcast.experiments import revision_bias, write_revision_bias
from nowcast.features import build_panel
from nowcast.synthetic import make_firms


def random_vintages(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    level = 100.0
    for month in pd.period_range("2013-01", "2026-12", freq="M"):
        level *= np.exp(rng.normal(0.003, 0.01))
        first = month.end_time.normalize() + pd.Timedelta(days=15)
        rows.append({"date": month.start_time, "value": level, "realtime_start": first,
                     "realtime_end": first + pd.Timedelta(days=364)})
        rows.append({"date": month.start_time, "value": level * rng.uniform(0.98, 1.02),
                     "realtime_start": first + pd.Timedelta(days=365), "realtime_end": pd.Timestamp("2262-01-01")})
    return pd.DataFrame(rows).assign(series="RSAFS")


def test_revision_bias_runs_and_reports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    revenue, views = make_firms()
    retail = random_vintages()
    mapping = pd.DataFrame({"ticker": revenue["ticker"].unique(), "retail_series": "RSAFS"})
    panel = build_panel(revenue, views, retail, mapping)
    df = revision_bias(panel, retail, mapping, ["rev", "wiki", "retail"], n_boot=20)
    assert len(df) == 6
    assert (df.loc[df["data"] == "realtime", "inflation vs honest"] == 0).all()
    path = write_revision_bias(df)
    assert "revised_no_lag" in path.read_text()
