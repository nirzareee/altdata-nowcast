"""Revision-bias experiment: how much does using revised macro data, instead of
the figures published at the time, inflate the measured value of a source?

Only the retail features change between runs; revenue, Wikipedia, models and
evaluation rows are identical. Development period only; the holdout is untouched.
"""
from pathlib import Path

import pandas as pd

from .retail import MODES, retail_features
from .valuation import bootstrap_shapley, verdict

OUT = Path("results")
DESCRIPTIONS = {
    "realtime": "Figures as published on the nowcast date (honest)",
    "revised": "Today's revised figures, release timing respected",
    "revised_no_lag": "Today's revised figures, including months not yet released",
}


def with_retail_mode(panel, retail, retail_map, mode):
    base = panel.drop(columns=[c for c in panel.columns if c.startswith("retail")])
    return retail_features(base, retail, retail_map, mode)


def revision_bias(panel, retail, retail_map, sources, model="huber", n_boot=1000):
    rows = []
    for mode in MODES:
        p = with_retail_mode(panel, retail, retail_map, mode)
        for metric in ("skill", "skill_ex_covid"):
            r = bootstrap_shapley(p, sources, model, metric, n_boot)["retail"]
            lo = min(r["ticker"][0], r["quarter"][0])
            hi = max(r["ticker"][1], r["quarter"][1])
            rows.append({"data": mode, "quarters": "all" if metric == "skill" else "outside COVID",
                         "retail value": r["value"], "low": lo, "high": hi, "verdict": verdict(r)})
    df = pd.DataFrame(rows)
    honest = df[df["data"] == "realtime"].set_index("quarters")["retail value"]
    df["inflation vs honest"] = df["retail value"] - df["quarters"].map(honest)
    return df


def write_revision_bias(df, path=OUT / "revision_bias.md"):
    OUT.mkdir(exist_ok=True)
    table = df.copy()
    for c in ("retail value", "low", "high", "inflation vs honest"):
        table[c] = table[c].map("{:+.3f}".format)
    table["data"] = table["data"].map(lambda m: f"{m}: {DESCRIPTIONS[m]}")
    header = "| " + " | ".join(table.columns) + " |\n| " + " | ".join("---" for _ in table.columns) + " |\n"
    body = "\n".join("| " + " | ".join(map(str, r)) + " |" for r in table.itertuples(index=False))
    path.write_text(
        "# Revision bias in the value of Census retail sales\n\n"
        "Shapley value of retail sales (95% CI, wider of firm- and quarter-clustered bootstrap) "
        "when the same backtest uses different versions of the data. Development period only.\n\n"
        + header + body + "\n"
    )
    return path
