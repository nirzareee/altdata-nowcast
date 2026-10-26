import numpy as np
import pytest

from nowcast.features import build_panel
from nowcast.model import walk_forward
from nowcast.stats import diebold_mariano, newey_west_se
from nowcast.synthetic import make_firms
from nowcast.valuation import bootstrap_shapley, source_shapley

SOURCES = ["rev", "wiki"]


@pytest.fixture(scope="module")
def signal_panel():
    return build_panel(*make_firms(signal=0.7))


@pytest.fixture(scope="module")
def null_panel():
    return build_panel(*make_firms(signal=0.0))


def test_bootstrap_point_matches_exact_shapley(signal_panel):
    exact, _ = source_shapley(signal_panel, SOURCES)
    boot = bootstrap_shapley(signal_panel, SOURCES, n_boot=50)
    for s in SOURCES:
        assert boot[s]["value"] == pytest.approx(exact[s])
        lo, hi = boot[s]["ticker"]
        assert lo <= boot[s]["value"] <= hi


def test_planted_signal_interval_excludes_zero(signal_panel):
    boot = bootstrap_shapley(signal_panel, SOURCES, n_boot=200)
    assert boot["wiki"]["ticker"][0] > 0 and boot["wiki"]["quarter"][0] > 0


def test_no_signal_interval_includes_zero(null_panel):
    boot = bootstrap_shapley(null_panel, SOURCES, n_boot=200)
    lo, hi = boot["wiki"]["ticker"]
    assert lo < 0 < hi


def test_newey_west_matches_iid_se_without_lags():
    x = np.random.default_rng(0).normal(size=400)
    assert newey_west_se(x, 0) == pytest.approx(x.std() / np.sqrt(len(x)))


def test_diebold_mariano(signal_panel, null_panel):
    _, t, p = diebold_mariano(walk_forward(signal_panel, SOURCES))
    assert t > 0 and p < 0.01
    _, t, p = diebold_mariano(walk_forward(null_panel, ["wiki"]))
    assert p > 0.01
