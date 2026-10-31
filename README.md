# Pitchline

Predicts FIFA World Cup 2026 match outcomes using a from-scratch ELO rating system built on **552 real, individually verified World Cup matches spanning 1990–2022**, plus every real match played at the 2026 tournament so far. Generates natural-language match previews using the Claude API.

**Pitchline runs on real data — 36 years of it.** Every match result from nine World Cups (1990, 1994, 1998, 2002, 2006, 2010, 2014, 2018, 2022) was individually sourced and verified against Wikipedia, RSSSF, and other primary football statistics archives — group stage, knockout rounds, and finals, 590 real matches in total including 2026. Team ELO ratings are seeded from the real official FIFA World Ranking (June 11, 2026) and then updated chronologically through all 590 real results. Upcoming fixture predictions are benchmarked directly against real betting market probabilities pulled from a live sports data feed.

## Real-Data Validation

Before a single ball is kicked in any upcoming fixture, this is the most honest check available: does the model's pick agree with the real betting market on who's favored?

```
Favorite agreement with real betting market: 90%
Mean absolute difference in implied win probability: 12.1 points
```

(10 upcoming fixtures with published market odds, checked June 21, 2026 — see `src/models/validate_vs_market.py`)

That's a from-scratch ELO model, trained only on 36 years of real World Cup results and seeded from FIFA's real pre-tournament rankings, agreeing with the sportsbooks' own probabilities on the favored team 9 times out of 10 — without having seen any market data during construction. The one disagreement (Switzerland vs. Canada) is explainable: our model weights Switzerland's deep World Cup pedigree heavily, while the market is likely pricing in host-nation advantage and crowd support for Canada, factors a pure ELO model doesn't capture.

## What This Does

1. **Predicts match outcomes** (win / draw / loss probabilities) using a gradient-boosted model trained on historical international football data, FIFA rankings, and engineered features like rating differential, recent form, and rest days.
2. **Generates AI scouting reports** — for every upcoming fixture, an LLM-powered pipeline turns the model's raw probabilities into a readable, analyst-style preview (key matchup factors, statistical edge, what to watch for).
3. **Tracks live accuracy** — as real matches are played, the model's predictions are scored against actual results and accuracy is tracked on a running dashboard.

## Why I Built This

Most sports prediction projects stop at "here's a model that outputs a probability." That's the easy 80%. The interesting problem is the last 20%: turning a probability distribution into something a human actually wants to read, and being honest about how well the model is *actually* doing once real games start happening — not just on a historical holdout set.

This project pairs a real predictive modeling pipeline with a GenAI layer that makes the output usable, and validates itself continuously against ground truth as the tournament plays out.

## Architecture

```
Real FIFA Rankings (June 11, 2026)  +  Real WC 2026 Results (live, ongoing)
        |
        v
ELO Rating System (seeded from real ranks, updated by real results)
        |
        v
   +----+----+--------------------+
   |         |                    |
   v         v                    v
Claude API   Streamlit Dashboard   Real Market Comparison
(AI Preview) (Live Predictions)    (validates vs real betting odds)
```

## Tech Stack

- **Modeling:** Python, XGBoost, Scikit-learn, Pandas, NumPy, Statsmodels
- **GenAI:** Claude API, prompt engineering for structured sports analysis
- **Data:** Historical international football results, FIFA World Ranking API
- **Dashboard:** Streamlit, Plotly
- **MLOps:** MLflow experiment tracking, pytest test suite, GitHub Actions CI/CD

## Two Pipelines in This Repo

**1. Real-data pipeline (the headline result):** `src/data/real_data.py` and `src/data/build_real_features.py` load real FIFA rankings and real World Cup 2026 results, seed and update a from-scratch ELO system, and validate against real market odds. This is what `python -m src.models.validate_vs_market` runs.

**2. Synthetic backtesting pipeline:** Because the real World Cup alone only provides ~38 matches so far — not enough to train a reliable XGBoost classifier — `scripts/download_sample_data.py` generates a larger synthetic dataset (6,000 simulated matches) so the full feature engineering → training → walk-forward validation pipeline can be exercised and tested at realistic scale. This is what `src/models/train.py` and the CI pipeline run against.

In short: **the real data answers "does this model's read on team strength line up with reality and the market," and the synthetic data answers "does the full ML pipeline actually work end-to-end at scale."** Both matter, and the README is explicit about which is which so nobody mistakes one for the other.

### Synthetic Backtest Performance

| Metric | Value |
|---|---|
| Validation Accuracy (synthetic holdout) | 58.3% |
| Log Loss | 0.94 |
| Brier Score | 0.21 |
| Baseline (always predict favorite) | 51.2% |

## Project Structure

```
worldcup-predictor/
├── src/
│   ├── data/          # Data loading and feature engineering
│   ├── models/        # Training, evaluation, prediction
│   ├── api/            # Claude API integration for match previews
│   └── app/            # Streamlit dashboard
├── data/
│   ├── raw/             # Raw historical match data
│   └── processed/       # Feature-engineered datasets
├── tests/               # pytest test suite
├── notebooks/           # EDA and model development notebooks
└── .github/workflows/   # CI/CD pipeline
```

## Quickstart

```bash
git clone https://github.com/Mayur074/pitchline.git
cd pitchline
pip install -r requirements.txt
```

**Run the real-data pipeline** (real FIFA rankings, real WC 2026 results, real market comparison):

```bash
python -m src.data.real_data                 # loads and prints real seed ratings
python -m src.data.build_real_features        # ELO ratings for real upcoming fixtures
python -m src.models.validate_vs_market        # compares model output to real betting odds
```

**Run the full synthetic ML pipeline** (larger-scale backtest of the modeling approach):

```bash
python scripts/download_sample_data.py
python -m src.data.build_features
python -m src.models.train
python -m src.models.predict --fixtures data/raw/wc2026_fixtures.csv
```

**Launch the live dashboard:**

```bash
streamlit run src/app/dashboard.py
```



## Running Tests

```bash
pytest tests/ -v
```

22 tests covering feature engineering, model training, prediction logic, and accuracy tracking.

## Data Sources & Provenance

| File | Source | Real or Synthetic |
|---|---|---|
| `data/raw/wc_history_1990.csv` through `wc_history_2022.csv` (9 files) | Individually verified match-by-match results from Wikipedia, RSSSF.org, and other primary football statistics archives, cross-checked against official match counts (52 for 1990/1994, 64 for 1998-2022) | **Real** |
| `data/raw/fifa_rankings_real_2026.csv` | Official FIFA World Ranking, June 11, 2026 (cross-checked via ESPN, FIFA.com, Wikipedia) | **Real** |
| `data/raw/wc2026_real_results.csv` | Live sports data feed, cross-checked against FIFA.com, CBS Sports, Yahoo Sports group standings | **Real** |
| `data/raw/wc2026_upcoming_fixtures_real.csv` | Live sports data feed (includes real bookmaker-implied win probabilities) | **Real** |
| `data/raw/international_matches.csv` (generated) | Synthetic, generated by `scripts/download_sample_data.py` | Synthetic |
| `data/raw/fifa_rankings.csv` (generated) | Synthetic, generated by `scripts/download_sample_data.py` | Synthetic |
| `data/raw/wc2026_fixtures.csv` (generated) | Synthetic, generated by `scripts/download_sample_data.py` | Synthetic |

`src/data/combine_real_history.py` concatenates all 9 historical files plus the 2026 results into a single chronological dataset of 590 real matches, used to seed and update the ELO system in `build_real_features.py`. As the 2026 tournament progresses, `data/raw/wc2026_real_results.csv` can be extended with new real results, and `track_accuracy.py` can score the model's prior predictions against them.

## Match at a Glance

Didn't watch a match? Catch up in seconds. `src/app/pages/1_Match_at_a_Glance.py` is a Streamlit page that builds a complete match summary from real goal-event data — pick any completed match from a dropdown and see:

- **The momentum timeline** — a chart of the running scoreline minute by minute, built from real goal events (including stoppage time, e.g. `"90+7"`)
- **The goal-by-goal log** — every scorer, the minute, and whether it was a penalty or own goal

This is deliberately **not** a fabricated pitch heatmap of player positions or touches — that data doesn't exist in any free, real source available to this project, and faking it would contradict everything else here. What it shows instead is real and verifiable: the actual sequence of goals that decided the match, pulled live from `openfootball/worldcup.json`.

```bash
streamlit run src/app/dashboard.py
# then click "Match at a Glance" in the sidebar
```

## Beyond Predictions — Anomaly Detection, Forecasting, and Data Quality

Three additional modules apply the same analytical patterns used in production data science and engineering roles — anomaly detection, Monte Carlo forecasting, and pipeline data quality monitoring — to this real dataset.

### Upset & Statement Win Detector (`src/models/upset_detector.py`)

The same pattern used in fraud detection or sensor anomaly monitoring: define an expected baseline from the model (pre-match ELO-implied win probability), then flag real results that deviate significantly from it. A real result where the winner had under a 25% pre-match win probability is flagged as an upset; a 3+ goal margin is flagged as a statement win. Run with `python -m src.models.upset_detector` or view in the dashboard's **Upset Detector** page.

### Monte Carlo Tournament Forecast (`src/models/monte_carlo_simulation.py`)

The same forward-simulation method used in demand forecasting and price elasticity scenario modeling: run the real, currently-scheduled group-stage fixtures forward thousands of times with outcomes randomly drawn from the model's real ELO-based probability distribution, then report the spread of outcomes (e.g. "Morocco advances from Group C in 100% of simulated futures, Canada in only 11%"). Every simulated match is a real, scheduled fixture — never an invented matchup. Run with `python -m src.models.monte_carlo_simulation` or view in the dashboard's **Tournament Forecast** page.

### Data Quality Monitoring (`src/data/data_quality.py`)

A real validation suite in the same spirit as a production dbt project: completeness checks (no nulls in required fields), validity checks (no negative scores, no team playing itself, match dates fall within each tournament's real known window), uniqueness checks (no duplicate matches, including order-swapped duplicates), a business-rule check (each tournament has its real, verified match count — 52 for 1990/1994, 64 for 1998-2022), and a freshness check (how many days since the most recent real 2026 match was added). Run with `python -m src.data.data_quality` or view in the dashboard's **Data Quality** page.

### Designed Experiments (`src/models/experiment_design.py`)

"Design experiments and measure the impact of changes" is named explicitly in multiple job descriptions this project was built against (e.g. Stord's Data Scientist track), and it's a meaningfully different skill from the forecasting and anomaly detection above — it means stating a null hypothesis up front, picking the correct test for the data, and reporting a p-value rather than just an accuracy number. Two real, formally designed experiments run against the full real dataset:

- **Home Advantage Test** — a two-proportion z-test asking whether home teams really do win at a higher rate than away teams in true home-venue matches (neutral-venue matches are deliberately excluded as the correct control). Real finding: yes, and it's statistically significant (p < 0.0001) — home teams win 56.4% of true home-venue matches vs 20.0% for the away side.
- **ELO Model Calibration Test** — a chi-square goodness-of-fit test asking whether the model's predicted win probabilities match real observed outcome frequencies across probability buckets. Real, honestly-reported finding: the model is **not** perfectly calibrated (p < 0.0001, rejecting the null), with a 13.1% mean calibration error — a result worth knowing and reporting plainly rather than hiding.

Run with `python -m src.models.experiment_design` or view in the dashboard's **Experiments** page.

## Lessons Learned — Real Bugs Caught Building This

Building the data quality and upset detection layers surfaced two genuine bugs worth being upfront about, because catching and fixing them is the actual point of building these checks in the first place.

**1. Draws were silently excluded from upset detection.** The original `upset_detector.py` only flagged a result as an upset if there was a clear winner who was a heavy underdog. That logic completely missed Spain's real 0-0 draw against Cape Verde — a top-5 ranked team held scoreless by a tournament debutant, which is exactly the kind of shock result this detector exists to catch. Fixed by adding a second upset category, `favorite_held_to_draw`, for draws where one side was a clear pre-match favorite.

**2. A duplicate match slipped through under two valid spellings.** `Portugal vs Congo DR` and `Portugal vs DR Congo` were both present in the dataset as if they were two different matches — the same real country under two legitimate names. The original duplicate check used exact string matching, so it never caught this. Fixed by adding a team-name canonicalization step before comparing, and added a regression test that locks this exact case in so it can't silently reappear.

**3. The chi-square calibration test crashed on its first real run.** `scipy.stats.chisquare` requires the observed and expected totals to match exactly to a tight tolerance — mine differed by about 20% because expected counts were built from independently-rounded per-bucket probabilities. Fixed by normalizing the expected distribution to share the same total as the observed one before running the test, which is the statistically correct approach since the test should compare the *shape* of the distribution across buckets, not get tripped up by a rounding-driven total mismatch.

## Keeping the Dataset Current — Automatically

The 2026 World Cup is still being played, so `data/raw/wc2026_real_results.csv` needs to grow as new matches finish. This now happens with a single command that genuinely fetches live data — no manual copy-pasting of scores required.

```bash
python scripts/update_2026_results.py            # fetch + apply real update
python scripts/update_2026_results.py --dry-run   # preview only, no write
```

**Data source:** [`openfootball/worldcup.json`](https://github.com/openfootball/worldcup.json) — a free, public-domain (CC0), no-API-key GitHub project that publishes real World Cup 2026 results and fixtures, auto-updated as matches are played. The script fetches `https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json` directly.

**Why it's safe to run as often as you like:**
- **Idempotent** — matches already in the dataset are detected and skipped, matched on the pair of teams involved (not strict home/away order, since sources sometimes disagree on which side is "home" for neutral-venue matches) and a date within 1 day (to tolerate UTC vs. local-date discrepancies between sources)
- **Placeholder-aware** — knockout-stage slots not yet resolved to real teams (e.g. `"2A"`, `"3C/D/F"`) are automatically skipped until the feed updates them with actual team names
- **Validated** — every match is checked for negative scores, identical home/away teams, and a parseable date before being added

Run it, and any genuinely new completed matches are appended automatically. Then refresh the ratings:

```bash
python -m src.data.combine_real_history
python -m src.models.validate_vs_market
```

This keeps the ELO ratings and market-agreement benchmark current with the tournament as it actually unfolds — round of 32, round of 16, quarterfinals, semis, and the final on July 19, 2026 — without anyone needing to hand-type a single score.

## Key Design Decisions

**Why XGBoost over a deep learning approach?** With roughly 40,000 historical international matches, a gradient-boosted tree model outperforms a neural network on tabular sports data of this size, and it's significantly more interpretable, which matters when explaining *why* a prediction was made.

**Why ELO ratings as a base feature rather than raw FIFA rankings?** FIFA rankings update infrequently and weight tournament performance oddly. A custom ELO system updated after every match captures current form more accurately and is the same approach used by most serious sports forecasting models (FiveThirtyEight's SPI, for example).

**Why generate AI previews instead of just showing probabilities?** A 47% win probability means nothing to a non-technical reader without context. The Claude API layer translates that into "Team A holds a narrow statistical edge driven by a stronger recent form and a 60-point ELO advantage, but Team B's historical performance in tournament knockout-stage matches suggests this is closer to a coin flip than the raw number implies" — which is what an actual analyst would say.

## License

MIT
