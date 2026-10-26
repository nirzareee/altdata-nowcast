import argparse
from pathlib import Path

import pandas as pd

OUT = Path("data/processed")
REVENUE = OUT / "revenue_quarterly.parquet"
PAGEVIEWS = OUT / "pageviews_daily.parquet"
PANEL = OUT / "panel.parquet"
RETAIL = OUT / "retail_vintages.parquet"
RETAIL_MAP = OUT / "retail_map.csv"
UNIVERSE = "config/universe_sp500.csv"


def report_missing(missing):
    if missing:
        print(f"{len(missing)} not found:")
        for m in missing:
            print("  ", m)


def read_universe(args):
    if not Path(args.universe).exists():
        raise SystemExit(f"{args.universe} not found; run: python -m nowcast.cli universe")
    return pd.read_csv(args.universe, comment="#")


def cmd_universe(args):
    from .http import Client
    from .universe import build_universe

    df = build_universe(Client(), args.refresh)
    df.to_csv(UNIVERSE, index=False)
    print(f"wrote {len(df)} firms to {UNIVERSE}")
    print(df["sector"].value_counts().to_string())
    report_missing(df.loc[df["wiki_articles"].isna(), "ticker"].tolist())


def cmd_targets(args):
    from .edgar import build_targets
    from .http import Client

    universe = read_universe(args)
    ciks = dict(zip(universe["ticker"], universe["cik"])) if "cik" in universe else None
    df, missing = build_targets(Client(), universe["ticker"].tolist(), args.refresh, ciks)
    df.to_parquet(REVENUE, index=False)
    print(f"wrote {len(df):,} firm-quarters to {REVENUE}")
    report_missing(missing)


def cmd_pageviews(args):
    from .http import Client
    from .pageviews import build_pageviews

    universe = read_universe(args)
    df, missing = build_pageviews(Client(), universe, args.start, args.end, args.refresh)
    df.to_parquet(PAGEVIEWS, index=False)
    print(f"wrote {len(df):,} daily rows to {PAGEVIEWS}")
    report_missing(missing)


def cmd_retail(args):
    from .http import Client
    from .retail import build_retail

    df, mapping, missing = build_retail(Client(), read_universe(args), args.refresh)
    df.to_parquet(RETAIL, index=False)
    mapping[["ticker", "sub_industry", "retail_series"]].to_csv(RETAIL_MAP, index=False)
    print(f"wrote {len(df):,} vintage rows for {df['series'].nunique()} series to {RETAIL}")
    print(f"firm -> category map in {RETAIL_MAP}:")
    print(mapping["retail_series"].value_counts().to_string())
    report_missing(missing)


def load_data(args):
    if args.synthetic:
        from .synthetic import make_firms
        return make_firms()
    pv = pd.read_parquet(PAGEVIEWS) if PAGEVIEWS.exists() else None
    retail = pd.read_parquet(RETAIL) if RETAIL.exists() else None
    retail_map = pd.read_csv(RETAIL_MAP) if RETAIL_MAP.exists() else None
    return pd.read_parquet(REVENUE), pv, retail, retail_map


def cmd_panel(args):
    from .features import build_panel

    panel = build_panel(*load_data(args))
    panel.to_parquet(PANEL, index=False)
    print(f"wrote {len(panel):,} rows x {panel.shape[1]} cols to {PANEL}")
    if "retail_months" in panel:
        print("retail months published by nowcast date:", panel["retail_months"].value_counts().sort_index().to_dict())
    feats = [c for c in panel.columns if "__" in c]
    print("feature coverage:", ", ".join(f"{c} {panel[c].notna().mean():.0%}" for c in feats))
    extreme = panel.reindex(panel["yoy_growth"].abs().sort_values(ascending=False).index).head(5)
    print("most extreme quarters (log growth):")
    print(extreme[["ticker", "end", "yoy_growth"]].to_string(index=False, float_format="%.2f"))


def load_panel(args):
    """Panel joined to universe columns (sector, ...), optionally filtered by --where col=value."""
    panel = pd.read_parquet(PANEL)
    if not (args.where or getattr(args, "by", None)):
        return panel
    info = read_universe(args).drop(columns=["wiki_articles", "company", "cik"], errors="ignore")
    panel = panel.merge(info, on="ticker", how="left")
    if args.where:
        col, value = args.where.split("=", 1)
        panel = panel[panel[col] == value]
        print(f"{args.where}: {panel['ticker'].nunique()} firms")
    return panel


def cmd_evaluate(args):
    from .model import score, walk_forward
    from .stats import diebold_mariano

    panel = load_panel(args)
    rows = []
    for sources in ([], *[[s] for s in args.sources], args.sources):
        preds = walk_forward(panel, sources, args.model, args.holdout, train_covid=args.train_covid)
        _, dm_t, dm_p = diebold_mariano(preds)
        rows.append({"sources": "+".join(sources) or "(mean)", **score(preds), "dm_t": dm_t, "dm_p": dm_p})
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("dm_t > 0: beats naive; dm_p: Diebold-Mariano p-value (quarterly loss differentials, Newey-West SE)")


def cmd_report(args):
    from .report import build_report

    path = build_report(load_panel(args), args.sources, args.model, args.n_boot, args.force)
    print(f"wrote {path} and charts in {path.parent}/")


def cmd_value(args):
    from .model import evaluate
    from .valuation import bootstrap_shapley, source_shapley, verdict

    panel = load_panel(args)
    groups = [("all", panel)]
    if args.by:
        groups += list(panel.groupby(args.by))
    rows = []
    for name, g in groups:
        base = {"group": name, "firms": g["ticker"].nunique(), "n": evaluate(g, [], args.model)["n"]}
        if not args.ci:
            phi, _ = source_shapley(g, args.sources, args.model, args.metric)
            rows.append({**base, **{s: f"{v:+.3f}" for s, v in phi.items()}})
            continue
        res = bootstrap_shapley(g, args.sources, args.model, args.metric, args.n_boot)
        for s, r in res.items():
            rows.append({**base, "source": s, "value": f"{r['value']:+.3f}",
                         "95% CI, firms": "[{:+.3f}, {:+.3f}]".format(*r["ticker"]),
                         "95% CI, quarters": "[{:+.3f}, {:+.3f}]".format(*r["quarter"]),
                         "verdict": verdict(r)})
    print(pd.DataFrame(rows).to_string(index=False))
    if args.ci:
        print("CIs resample whole firms or whole quarters; a verdict needs both intervals on one side of 0.")


def main():
    p = argparse.ArgumentParser(prog="nowcast")
    p.add_argument("--universe", default=UNIVERSE)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("universe")
    s.add_argument("--refresh", action="store_true", help="re-download the constituents list")

    for name in ("targets", "pageviews"):
        s = sub.add_parser(name)
        s.add_argument("--refresh", action="store_true", help="ignore cached downloads")
        s.add_argument("--start", default="2015-07-01")
        s.add_argument("--end", default=str((pd.Timestamp.today() - pd.Timedelta(days=2)).date()))

    s = sub.add_parser("report", help="final results, including the one-time holdout run")
    s.add_argument("--sources", nargs="+", default=["rev", "wiki", "retail"])
    s.add_argument("--model", choices=["huber", "ridge", "gbm"], default="huber")
    s.add_argument("--n-boot", type=int, default=1000)
    s.add_argument("--force", action="store_true", help="rerun even though the holdout was already scored")
    s.set_defaults(where=None)

    s = sub.add_parser("retail")
    s.add_argument("--refresh", action="store_true", help="re-download all vintages")

    s = sub.add_parser("panel")
    s.add_argument("--synthetic", action="store_true", help="use simulated firms instead of downloads")

    for name in ("evaluate", "value"):
        s = sub.add_parser(name)
        s.add_argument("--sources", nargs="+", default=["rev", "wiki", "retail"])
        s.add_argument("--model", choices=["huber", "ridge", "gbm"], default="huber")
        s.add_argument("--where", help="filter by a universe column, e.g. sector='Consumer Staples'")
        if name == "value":
            s.add_argument("--by", help="also value sources within each group of this column, e.g. sector")
            s.add_argument("--metric", choices=["skill", "skill_ex_covid"], default="skill",
                           help="forecast skill measure the sources are valued on")
            s.add_argument("--ci", action="store_true", help="add 95%% cluster-bootstrap confidence intervals")
            s.add_argument("--n-boot", type=int, default=1000)
        if name == "evaluate":
            s.add_argument("--holdout", action="store_true", help="score the untouched holdout period")
            s.add_argument("--train-covid", action="store_true", help="also train on Apr 2020 - Dec 2021")

    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"report": cmd_report, "universe": cmd_universe, "retail": cmd_retail, "targets": cmd_targets, "pageviews": cmd_pageviews, "panel": cmd_panel,
     "evaluate": cmd_evaluate, "value": cmd_value}[args.command](args)


if __name__ == "__main__":
    main()
