import pandas as pd
import pytest

from nowcast.features import build_panel
from nowcast.model import evaluate, walk_forward
from nowcast.synthetic import make_firms
from nowcast.valuation import source_shapley

SOURCES = ["rev", "wiki"]


@pytest.fixture(scope="module")
def data():
    return make_firms()


@pytest.fixture(scope="module")
def panel(data):
    return build_panel(*data)


def test_features_ignore_future_data(data, panel):
    """Changing anything published after a cutoff must not change earlier features."""
    revenue, views = data
    cutoff = pd.Timestamp("2020-01-01")
    views2 = views.copy()
    views2.loc[views2["available_at"] > cutoff, "views"] *= 100
    revenue2 = revenue.copy()
    revenue2.loc[revenue2["filed"] > cutoff, "yoy_growth"] += 5

    before = panel[panel["nowcast_date"] <= cutoff].set_index(["ticker", "end"])
    after = build_panel(revenue2, views2).set_index(["ticker", "end"]).loc[before.index]
    cols = [c for c in panel.columns if "__" in c]
    pd.testing.assert_frame_equal(before[cols], after[cols])


def test_trains_only_on_reported_quarters(panel):
    preds = walk_forward(panel, SOURCES)
    assert (preds["train_last_filed"] < preds["nowcast_date"]).all()
    assert (preds["nowcast_date"] < pd.Timestamp("2024-07-01")).all()


def test_finds_planted_signal(panel):
    assert evaluate(panel, SOURCES)["skill"] > 0.3


def test_no_signal_when_none_planted():
    panel = build_panel(*make_firms(signal=0.0))
    gain = evaluate(panel, ["wiki"])["skill"] - evaluate(panel, [])["skill"]
    assert abs(gain) < 0.03


def test_shapley_credits_the_informative_source(panel):
    phi, values = source_shapley(panel, SOURCES)
    assert phi["wiki"] > phi["rev"]
    assert sum(phi.values()) == pytest.approx(values[frozenset(SOURCES)] - values[frozenset()])
