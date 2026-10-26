# What Is Data Worth?

**Measuring how much each alternative data source adds to forecasts of company revenue.**

Funds and companies buy external data (web traffic, search trends, app usage, job postings) to see business performance before it is reported. Most of these sources overlap, and it is rarely measured what any one of them adds on top of the others. This project builds a leakage-free forecasting pipeline and uses it to put a number, with uncertainty, on the value of each source.

The test case: nowcast quarterly revenue growth for US consumer companies a week after each quarter ends, weeks before the company reports it.

## How it works

1. **Collect** revenue from SEC filings and daily signals from public sources.
2. **Align in time.** Every value carries the date it became public. A forecast made on date `t` sees only data public before `t`.
3. **Forecast** with walk-forward models: for each quarter, train only on quarters already reported, then predict.
4. **Value each source** with Shapley values: a source's average contribution to forecast skill across every combination of the other sources. Overlapping sources share credit instead of each claiming it.

## Current status

Pilot: 27 consumer firms, 2016–2026, two sources (revenue history, Wikipedia pageviews).

Early findings, before tuning:

- **Wikipedia attention tracks revenue for platforms, not big-box retail.** Per-firm correlation between pageview growth and revenue growth is 0.5–0.6 for Airbnb, Lyft, Etsy and Uber, and near zero or negative for Walmart, Target, Costco, McDonald's and Yum.
- **COVID dominates raw error.** Forecast errors in 2020–21 are 4–6× other years, so results are reported with and without those quarters.
- **Predicting growth from scratch underperformed the naive forecast** in normal years. Models now predict the change from last quarter's growth instead.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export NOWCAST_USER_AGENT="Your Name you@email.com"   # required by SEC and Wikimedia
pytest -q
```

## Usage

```bash
python -m nowcast.cli targets        # SEC revenue      -> data/processed/revenue_quarterly.parquet
python -m nowcast.cli pageviews      # Wikipedia views  -> data/processed/pageviews_daily.parquet
python -m nowcast.cli panel          # features         -> data/processed/panel.parquet
python -m nowcast.cli evaluate       # forecast skill per source combination
python -m nowcast.cli value          # Shapley value of each source
```

`python -m nowcast.cli panel --synthetic` builds simulated firms with a planted signal; the pipeline should find it.

`evaluate --holdout` scores the period from 2024-07-01 on. It is run once, at the end.

### Reading `evaluate`

| Column | Meaning |
| --- | --- |
| `skill` | 1 − MAE / naive MAE. Naive = growth stays at the last reported rate. Above 0 beats naive |
| `skill_ex_covid` | Same, excluding quarters ending Apr 2020 – Dec 2021 |
| `accel_hit_rate` | How often the forecast gets the direction of change in growth right |
| `(mean)` row | No data: naive plus the average historical change |

## Design decisions

| Decision | Why |
| --- | --- |
| Keep first-reported revenue, not restated | Restatements were not known at forecast time |
| Derive Q4 as fiscal year minus Q1–Q3 | Q4 is only reported inside the 10-K |
| Target year-over-year growth | Comparable across firm sizes; removes seasonality |
| Predict change from last growth | Small, noisy panel; anchoring on naive avoids fitting noise |
| Growth features only, no levels | Raw pageviews are not comparable across firms |
| Study starts 2016 | Pageviews begin July 2015; every quarter needs a year-ago window |
| Exact Shapley over sources | 2ⁿ model runs is affordable for up to ~10 sources |

`tests/test_pipeline.py` checks the time alignment directly: it changes every value published after a cutoff and asserts earlier features don't move.

## Roadmap

- [x] SEC revenue, Wikipedia pageviews, walk-forward models, Shapley valuation
- [ ] Google Trends
- [ ] Expand to ~100 firms selected by industry code
- [ ] LLM entity resolution: map firms to brands, apps and pages automatically
- [ ] Job postings, Reddit/news mentions, earnings-call guidance
- [ ] Hierarchical Bayesian model; conformal prediction intervals
- [ ] Bootstrap confidence intervals on source values; value decay over time; value by sector
- [ ] Comparison with analyst consensus; earnings-announcement event study

## Known issues

- When several revenue tags cover one period, the largest is kept. Audit the `tag` column for firms that switched around 2018 (ASC 606).
- Wikipedia titles in `config/universe.csv` are hand-picked. Renamed pages lose history; redirects count only direct hits.
- 27 firms is a pilot; conclusions need the full universe.
