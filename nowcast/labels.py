"""Hand-labeled benchmark for guidance extraction.

`make_sample` draws releases and writes a labeling sheet. Fill one row per
guidance item; for a release with no revenue guidance, leave one row with
has_guidance = n. `score` compares a model's extractions with the labels.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .guidance import outlook_passages

LABELS = Path("data/labels")
LABEL_COLUMNS = ["has_guidance", "period_type", "fiscal_year", "fiscal_quarter", "kind",
                 "low", "high", "scale", "basis", "notes"]
KEY = ["accession", "period_type", "fiscal_year", "fiscal_quarter", "kind"]


def make_sample(releases, n=200, seed=0):
    """Random releases, spread across years so early and recent formats both appear."""
    r = releases.assign(year=pd.to_datetime(releases["filed"]).dt.year)
    per_year = int(np.ceil(n / r["year"].nunique()))
    sample = (r.groupby("year", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per_year), random_state=seed))
                .sample(frac=1, random_state=seed).head(n))
    (LABELS / "passages").mkdir(parents=True, exist_ok=True)
    for row in sample.itertuples():
        text = Path(row.path).read_text()
        (LABELS / "passages" / f"{row.accession}.txt").write_text(
            f"{row.ticker}  filed {pd.Timestamp(row.filed):%Y-%m-%d}  full text: {row.path}\n\n"
            f"--- outlook passages ---\n{outlook_passages(text) or '(none found by filter)'}\n")
    sheet = sample[["accession", "ticker", "filed"]].copy()
    for c in LABEL_COLUMNS:
        sheet[c] = ""
    path = LABELS / "labels.csv"
    sheet.to_csv(path, index=False)
    return path


def _items(df):
    d = df.copy()
    d["fiscal_quarter"] = pd.to_numeric(d["fiscal_quarter"], errors="coerce").fillna(0).astype(int)
    d["fiscal_year"] = pd.to_numeric(d["fiscal_year"], errors="coerce").fillna(0).astype(int)
    d["low"] = pd.to_numeric(d["low"], errors="coerce")
    d["high"] = pd.to_numeric(d["high"], errors="coerce")
    return d


def score(labels, preds):
    """Release-level detection plus item-level value accuracy, on labeled releases only."""
    labels = labels.copy()
    labels["has_guidance"] = labels["has_guidance"].astype(str).str.strip().str.lower().str.startswith("y")
    sample = set(labels["accession"])
    preds = preds[preds["accession"].isin(sample)]
    if "none" in preds:
        preds = preds[preds["none"].isna() | (preds["none"] != True)]  # noqa: E712

    truth = labels.groupby("accession")["has_guidance"].any()
    found = truth.index.to_series().isin(set(preds["accession"]))
    tp, fp = int((truth & found).sum()), int((~truth & found).sum())
    fn = int((truth & ~found).sum())
    precision = tp / (tp + fp) if tp + fp else np.nan
    recall = tp / (tp + fn) if tp + fn else np.nan

    gold = _items(labels[labels["has_guidance"]])
    pred = _items(preds)
    matched = gold.merge(pred, on=KEY, suffixes=("_gold", "_pred"))
    exact = (np.isclose(matched["low_gold"], matched["low_pred"]) &
             np.isclose(matched["high_gold"], matched["high_pred"]))
    return {
        "releases labeled": len(truth),
        "with guidance": int(truth.sum()),
        "detection precision": precision,
        "detection recall": recall,
        "item recall": len(matched.drop_duplicates(KEY)) / len(gold) if len(gold) else np.nan,
        "value accuracy (matched items)": exact.mean() if len(matched) else np.nan,
        "verified share of predictions": pred["verified"].mean() if "verified" in pred and len(pred) else np.nan,
    }
