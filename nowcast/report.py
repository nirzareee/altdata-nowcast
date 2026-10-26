"""Final results: development-period and holdout tables, source values with
confidence intervals, and the summary chart.

The holdout (nowcasts from 2024-07-01) is scored once. A lock file records the
run; a second run needs --force, so the holdout can't quietly become a tuning set.
"""
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .model import score, walk_forward
from .stats import diebold_mariano
from .valuation import bootstrap_shapley, verdict

OUT = Path("results")
LOCK = OUT / "HOLDOUT_SCORED"
COLORS = {"rev": "#2a78d6", "wiki": "#eb6834", "retail": "#1baf7a"}  # validated categorical slots 1-3
LABELS = {"rev": "Revenue history", "wiki": "Wikipedia pageviews", "retail": "Census retail sales"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e4e3df"


def accuracy_table(panel, sources, model, holdout):
    rows = []
    for subset in ([], *[[s] for s in sources], sources):
        preds = walk_forward(panel, subset, model, holdout)
        _, t, p = diebold_mariano(preds)
        r = score(preds)
        rows.append({"sources": " + ".join(subset) or "(none)", "n": r["n"], "MAE": r["mae"],
                     "naive MAE": r["naive_mae"], "skill": r["skill"],
                     "skill ex-COVID": r["skill_ex_covid"], "direction hit rate": r["accel_hit_rate"],
                     "DM t": t, "DM p": p})
    return pd.DataFrame(rows)


def value_table(panel, sources, model, n_boot):
    settings = [
        ("Development, all quarters", dict(metric="skill")),
        ("Development, outside COVID", dict(metric="skill_ex_covid")),
        ("Holdout, Jul 2024 on", dict(metric="skill", holdout=True)),
    ]
    rows = []
    for name, kw in settings:
        for s, r in bootstrap_shapley(panel, sources, model, n_boot=n_boot, **kw).items():
            lo = min(r["ticker"][0], r["quarter"][0])
            hi = max(r["ticker"][1], r["quarter"][1])
            rows.append({"setting": name, "source": s, "value": r["value"], "low": lo, "high": hi,
                         "verdict": verdict(r)})
    return pd.DataFrame(rows)


def plot_values(values, path):
    """Dot-and-whisker per source, one panel per evaluation setting, shared x axis."""
    settings = list(dict.fromkeys(values["setting"]))
    sources = list(dict.fromkeys(values["source"]))
    fig, axes = plt.subplots(1, len(settings), figsize=(11, 3.2), sharex=True, sharey=True)
    for ax, setting in zip(axes, settings):
        d = values[values["setting"] == setting].set_index("source").loc[sources]
        y = range(len(sources))[::-1]
        ax.axvline(0, color=MUTED, lw=1)
        for yi, (s, r) in zip(y, d.iterrows()):
            ax.plot([r["low"], r["high"]], [yi, yi], color=COLORS.get(s, MUTED), lw=2, solid_capstyle="round")
            ax.plot(r["value"], yi, "o", ms=8, color=COLORS.get(s, MUTED), mec="#fcfcfb", mew=1)
            ax.annotate(f"{r['value']:+.3f}", (r["value"], yi), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8, color=INK)
        ax.set_title(setting, fontsize=10, color=INK, loc="left")
        ax.set_yticks(list(y), [LABELS.get(s, s) for s in sources], fontsize=9, color=INK)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="x", colors=MUTED, labelsize=8)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, len(sources) - 0.3)
    fig.supxlabel("Shapley value: share of forecast skill vs. naive contributed by the source "
                  "(95% CI, wider of firm- and quarter-clustered)", fontsize=9, color=MUTED)
    fig.patch.set_facecolor("#fcfcfb")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


def _md(df, floats="{:+.3f}"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            fmt = "{:.3f}" if c in ("MAE", "naive MAE", "DM p", "direction hit rate") else floats
            df[c] = df[c].map(lambda x: fmt.format(x) if pd.notna(x) else "")
    header = "| " + " | ".join(df.columns) + " |\n| " + " | ".join("---" for _ in df.columns) + " |\n"
    return header + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))


def build_report(panel, sources, model="huber", n_boot=1000, force=False):
    OUT.mkdir(exist_ok=True)
    if LOCK.exists() and not force:
        raise SystemExit(f"The holdout was already scored ({LOCK.read_text().strip()}). "
                         "Rerunning invites tuning on it; pass --force only to regenerate unchanged results.")
    dev = accuracy_table(panel, sources, model, holdout=False)
    hold = accuracy_table(panel, sources, model, holdout=True)
    values = value_table(panel, sources, model, n_boot)
    values.to_csv(OUT / "source_values.csv", index=False)
    plot_values(values, OUT / "source_values.png")

    summary = OUT / "summary.md"
    summary.write_text(
        f"# Results\n\nModel: {model}. Sources: {', '.join(sources)}. "
        f"{panel['ticker'].nunique()} firms.\n\n"
        "![Source values](source_values.png)\n\n"
        f"## Source values (Shapley, 95% CI)\n\n{_md(values)}\n\n"
        f"## Accuracy, development period (to Jun 2024)\n\n{_md(dev)}\n\n"
        f"## Accuracy, holdout (Jul 2024 on, scored once)\n\n{_md(hold)}\n"
    )
    LOCK.write_text(f"first scored {datetime.now():%Y-%m-%d %H:%M}")
    return summary
