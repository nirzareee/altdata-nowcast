import argparse
from pathlib import Path

import pandas as pd

OUT = Path("data/processed")
REVENUE = OUT / "revenue_quarterly.parquet"
PAGEVIEWS = OUT / "pageviews_daily.parquet"
PANEL = OUT / "panel.parquet"


def report_missing(missing):
    if missing:
        print(f"{len(missing)} not found:")
        for m in missing:
            print("  ", m)


def cmd_targets(args):
    from .edgar import build_targets
    from .http import Client

    universe = pd.read_csv(args.universe, comment="#")
    df, missing = build_targets(Client(), universe["ticker"].tolist(), args.refresh)
    df.to_parquet(REVENUE, index=False)
    print(f"wrote {len(df):,} firm-quarters to {REVENUE}")
    report_missing(missing)


def cmd_pageviews(args):
    from .http import Client
    from .pageviews import build_pageviews

    universe = pd.read_csv(args.universe, comment="#")
    df, missing = build_pageviews(Client(), universe, args.start, args.end, args.refresh)
    df.to_parquet(PAGEVIEWS, index=False)
    print(f"wrote {len(df):,} daily rows to {PAGEVIEWS}")
    report_missing(missing)


def load_data(args):
    if args.synthetic:
        from .synthetic import make_firms
        return make_firms()
    pv = pd.read_parquet(PAGEVIEWS) if PAGEVIEWS.exists() else None
    return pd.read_parquet(REVENUE), pv


def cmd_panel(args):
    from .features import build_panel

    panel = build_panel(*load_data(args))
    panel.to_parquet(PANEL, index=False)
    print(f"wrote {len(panel):,} rows x {panel.shape[1]} cols to {PANEL}")


def cmd_evaluate(args):
    from .model import evaluate

    panel = pd.read_parquet(PANEL)
    rows = []
    for sources in ([], *[[s] for s in args.sources], args.sources):
        r = evaluate(panel, sources, args.model, args.holdout, args.train_covid)
        rows.append({"sources": "+".join(sources) or "(mean)", **r})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def cmd_value(args):
    from .valuation import source_shapley

    phi, _ = source_shapley(pd.read_parquet(PANEL), args.sources, args.model)
    for s, v in sorted(phi.items(), key=lambda kv: -kv[1]):
        print(f"{s:10} {v:+.3f}")


def main():
    p = argparse.ArgumentParser(prog="nowcast")
    p.add_argument("--universe", default="config/universe.csv")
    sub = p.add_subparsers(dest="command", required=True)

    for name in ("targets", "pageviews"):
        s = sub.add_parser(name)
        s.add_argument("--refresh", action="store_true", help="ignore cached downloads")
        s.add_argument("--start", default="2015-07-01")
        s.add_argument("--end", default=str((pd.Timestamp.today() - pd.Timedelta(days=2)).date()))

    s = sub.add_parser("panel")
    s.add_argument("--synthetic", action="store_true", help="use simulated firms instead of downloads")

    for name in ("evaluate", "value"):
        s = sub.add_parser(name)
        s.add_argument("--sources", nargs="+", default=["rev", "wiki"])
        s.add_argument("--model", choices=["huber", "ridge", "gbm"], default="huber")
        if name == "evaluate":
            s.add_argument("--holdout", action="store_true", help="score the untouched holdout period")
            s.add_argument("--train-covid", action="store_true", help="also train on Apr 2020 - Dec 2021")

    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"targets": cmd_targets, "pageviews": cmd_pageviews, "panel": cmd_panel,
     "evaluate": cmd_evaluate, "value": cmd_value}[args.command](args)


if __name__ == "__main__":
    main()
