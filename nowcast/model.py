"""Walk-forward nowcasting of year-over-year revenue growth.

For each calendar quarter of nowcast dates, a model is trained only on
firm-quarters whose revenue had been filed before the earliest nowcast
date in that quarter, then predicts the quarter. Two references:
"naive" (growth stays at the last reported rate) is reported for context;
the no-source model (historical mean) is the zero point for valuing sources.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MODELS = {
    "ridge": lambda: make_pipeline(SimpleImputer(), StandardScaler(), Ridge(alpha=1.0)),
    "gbm": lambda: HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=200, random_state=0),
}
HOLDOUT_START = pd.Timestamp("2024-07-01")


def source_columns(panel, sources):
    return [c for c in panel.columns if "__" in c and c.split("__")[0] in sources]


def eval_rows(panel, holdout=False):
    p = panel.dropna(subset=["yoy_growth", "rev__last_yoy"])
    in_holdout = p["nowcast_date"] >= HOLDOUT_START
    return p[in_holdout if holdout else ~in_holdout].sort_values("nowcast_date")


def walk_forward(panel, sources, model="ridge", holdout=False, min_train=100):
    """Out-of-sample predictions. Returns test rows with `pred` and `naive`."""
    labeled = panel.dropna(subset=["yoy_growth", "rev__last_yoy"])
    test_rows = eval_rows(panel, holdout)
    cols = source_columns(panel, sources)

    preds = []
    for _, test in test_rows.groupby(test_rows["nowcast_date"].dt.to_period("Q")):
        cutoff = test["nowcast_date"].min()
        train = labeled[labeled["filed"] < cutoff]
        if len(train) < min_train:
            continue
        test = test.assign(naive=test["rev__last_yoy"], train_last_filed=train["filed"].max())
        if cols:
            m = MODELS[model]()
            m.fit(train[cols], train["yoy_growth"])
            test["pred"] = m.predict(test[cols])
        else:
            # No sources: historical mean. Growth mean-reverts, so this already beats
            # naive; valuing sources against it stops that gain being credited to data.
            test["pred"] = train["yoy_growth"].mean()
        preds.append(test)
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()


def score(preds):
    err, base = (preds["pred"] - preds["yoy_growth"]).abs(), (preds["naive"] - preds["yoy_growth"]).abs()
    # Direction of change in growth: will this quarter accelerate or decelerate vs last reported?
    actual_dir = np.sign(preds["yoy_growth"] - preds["naive"])
    pred_dir = np.sign(preds["pred"] - preds["naive"])
    moved = pred_dir != 0
    return {
        "n": len(preds),
        "mae": err.mean(),
        "naive_mae": base.mean(),
        "skill": 1 - err.mean() / base.mean(),
        "accel_hit_rate": (actual_dir[moved] == pred_dir[moved]).mean() if moved.any() else np.nan,
    }


def evaluate(panel, sources, model="ridge", holdout=False):
    return score(walk_forward(panel, sources, model, holdout))
