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

### Step 2 results (99 S&P 500 consumer firms, Huber model, log growth)

| Group | Firms | n | Revenue history: Shapley value | Wikipedia: Shapley value |
| --- | --- | --- | --- | --- |
| All | 99 | 2,809 | +0.060 | −0.003 |
| Consumer Staples | 33 | 893 | +0.083 | −0.002 |
| Consumer Discretionary | 47 | 1,357 | +0.051 | 0.000 |
| Communication Services | 19 | 366 | +0.048 | −0.001 |

- **The pilot result replicates at 4× the scale.** Revenue history beats the naive forecast by 6% (5% outside COVID) and calls the direction of change 63% of the time.
- **Wikipedia pageviews add no value in any sector.** With 90% feature coverage across 99 firms, this is a robust negative result.
- **Past growth is most informative for consumer staples**, whose growth is steadiest.

### Pilot results (27 firms, Huber model, percent growth, COVID window to Dec 2021)

| Firms | Revenue history: skill ex-COVID | Wikipedia: Shapley value | n |
| --- | --- | --- | --- |
| All | +0.062 | −0.001 | 609 |
| Digital (12) | +0.098 | −0.004 | 160 |
| Physical (15) | +0.049 | +0.001 | 339 |

- **Wikipedia pageviews add no forecasting value**, pooled or split by business model. Per-firm correlations with revenue growth reach 0.5–0.6 for Airbnb, Lyft, Etsy and Uber, but that co-movement doesn't survive into out-of-sample forecasts.
- **Revenue history beats the naive forecast by ~5–10%** outside COVID and gets the direction of change right ~62% of the time.

### What the pilot taught about method

- **COVID dominates raw error.** Errors in 2020–21 are 4–6× other years; results are reported with and without those quarters.
- **Predicting growth from scratch underperformed naive.** Models now predict the change from last quarter's growth.
- **Squared-error models let COVID set the rules.** Ridge trained on the 2020–21 crash-and-rebound over-corrected calm quarters (skill −0.34 outside COVID). Huber loss and clipping fixed it; excluding COVID from training made little further difference.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export NOWCAST_USER_AGENT="Your Name you@email.com"   # required by SEC and Wikimedia
pytest -q
```

## Usage

```bash
python -m nowcast.cli universe       # firm list        -> config/universe_sp500.csv
python -m nowcast.cli targets        # SEC revenue      -> data/processed/revenue_quarterly.parquet
python -m nowcast.cli pageviews      # Wikipedia views  -> data/processed/pageviews_daily.parquet
python -m nowcast.cli panel          # features         -> data/processed/panel.parquet
python -m nowcast.cli evaluate       # forecast skill per source combination
python -m nowcast.cli value          # Shapley value of each source
python -m nowcast.cli value --by sector                       # ...overall and within each sector
python -m nowcast.cli evaluate --where "sector=Consumer Staples"
```

**Universe rule:** every S&P 500 member in Consumer Discretionary, Consumer Staples or Communication Services (103 firms). CIK and Wikipedia article come from the constituents table, so nothing is hand-picked. The 27-firm pilot list is kept in `config/universe.csv` (`--universe config/universe.csv`).

`python -m nowcast.cli panel --synthetic` builds simulated firms with a planted signal; the pipeline should find it.

`evaluate --holdout` scores the period from 2024-07-01 on. It is run once, at the end.

Models: `--model huber` (default, robust linear), `ridge`, `gbm`. `--train-covid` includes Apr 2020–Dec 2022 in training, for comparison.

### Reading `evaluate`

| Column | Meaning |
| --- | --- |
| `skill` | 1 − MAE / naive MAE. Naive = growth stays at the last reported rate. Above 0 beats naive |
| `skill_ex_covid` | Same, excluding quarters ending Apr 2020 – Dec 2022 (2022 is compared against COVID-year revenue) |
| `accel_hit_rate` | How often the forecast gets the direction of change in growth right |
| `(mean)` row | No data: naive plus the average historical change |

## Design decisions

| Decision | Why |
| --- | --- |
| Keep first-reported revenue, not restated | Restatements were not known at forecast time |
| Derive Q4 as fiscal year minus Q1–Q3 | Q4 is only reported inside the 10-K |
| Target year-over-year log growth, ln(revenue / year-ago revenue) | Comparable across firm sizes; removes seasonality; stays bounded when the year-ago base is near zero (cruise lines, casinos in 2021) |
| Predict change from last growth | Small, noisy panel; anchoring on naive avoids fitting noise |
| Don't train on COVID quarters; Huber loss; clip at ±100 pts | One extreme regime shouldn't set the rules for every other quarter |
| Growth features only, no levels | Raw pageviews are not comparable across firms |
| Study starts 2016 | Pageviews begin July 2015; every quarter needs a year-ago window |
| Exact Shapley over sources | 2ⁿ model runs is affordable for up to ~10 sources |

`tests/test_pipeline.py` checks the time alignment directly: it changes every value published after a cutoff and asserts earlier features don't move.

## Roadmap

- [x] **Step 1.** SEC revenue, Wikipedia pageviews, walk-forward models, Shapley valuation
- [x] **Step 2.** Expand to S&P 500 consumer sectors (103 firms), Wikipedia mapping taken from the constituents table
- [ ] **Step 3.** Google Trends
- [ ] **Step 4.** LLM entity resolution: map firms to brands, apps and pages
- [ ] Job postings, Reddit/news mentions, earnings-call guidance
- [ ] Hierarchical Bayesian model; conformal prediction intervals
- [ ] Bootstrap confidence intervals on source values; value decay over time; value by sector
- [ ] Comparison with analyst consensus; earnings-announcement event study

## Known issues

- The universe uses today's S&P 500 members, so firms that shrank and left the index are missing (survivorship bias). Revenue forecasts are less exposed to this than return studies, but the event study in a later step will need historical membership.
- When several revenue tags cover one period, the largest is kept. Audit the `tag` column for firms that switched around 2018 (ASC 606).
- Wikipedia titles in `config/universe.csv` are hand-picked. Renamed pages lose history; redirects count only direct hits.
- 27 firms is a pilot; conclusions need the full universe.
