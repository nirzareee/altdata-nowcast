import numpy as np
import pytest

from nowcast.features import build_panel
from nowcast.report import build_report
from nowcast.synthetic import make_firms


def test_holdout_is_scored_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    panel = build_panel(*make_firms())
    panel["retail__yoy"] = np.random.default_rng(1).normal(0, 0.05, len(panel))
    summary = build_report(panel, ["rev", "wiki", "retail"], n_boot=20)
    assert summary.exists() and (summary.parent / "source_values.png").exists()
    with pytest.raises(SystemExit):
        build_report(panel, ["rev", "wiki", "retail"], n_boot=20)
    build_report(panel, ["rev", "wiki", "retail"], n_boot=20, force=True)
