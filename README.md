# What Is Data Worth?

Valuing alternative data sources for probabilistic revenue nowcasting.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export NOWCAST_USER_AGENT="Your Name you@stonybrook.edu"   # required by SEC and Wikimedia
```

## Pipeline

```bash
python -m nowcast.cli targets        # SEC revenue      -> data/processed/revenue_quarterly.parquet
python -m nowcast.cli pageviews      # Wikipedia views  -> data/processed/pageviews_daily.parquet
python -m nowcast.cli panel          # features         -> data/processed/panel.parquet
python -m nowcast.cli evaluate       # walk-forward skill per source combination
python -m nowcast.cli value          # Shapley value of each source
pytest -q
```

Try the last three on simulated firms first: `python -m nowcast.cli panel --synthetic`.

`evaluate --holdout` scores the period from 2024-07-01 on. Run it once, at the end.

## How a forecast is made

1. **Nowcast date** = quarter end + 7 days, weeks before most firms report.
2. **Features** use only data knowable by that date: revenue by `filed`, pageviews by `available_at`.
3. **Training** uses only firm-quarters whose revenue was filed before the nowcast date.
4. **Target** is year-over-year revenue growth.

## Reading the results

| Column | Meaning |
| --- | --- |
| `skill` | 1 − MAE / naive MAE. Naive = growth stays at the last reported rate |
| `accel_hit_rate` | How often the model gets the direction of change in growth right |
| `(mean)` row | Model with no data: predicts historical average growth |

Source value is measured against the `(mean)` row, not against naive. Growth mean-reverts, so the historical mean already beats naive; measuring against naive would credit that gain to whatever data the model uses.

## Point-in-time rules

| Data | Value kept | Knowable at |
| --- | --- | --- |
| Quarterly revenue | First reported (`value`); latest restatement in `value_latest` | `filed` date of first filing |
| Q4 revenue | Fiscal year minus Q1–Q3 (`source = derived_q4`) | 10-K `filed` date |
| Pageviews | Daily human views | `available_at` = day after |

`tests/test_pipeline.py` checks this directly: it changes every value published after a cutoff and asserts earlier features don't move.

## Known issues to audit

- Revenue tags: when several revenue tags cover one period, the largest is kept. Check `tag` for firms that switched around 2018 (ASC 606).
- Wikipedia titles in `config/universe.csv` are a hand-made v0. Renamed pages lose history; redirects only count direct hits.
- 27 firms is a pilot. Results need 100+ firms before they mean much.
