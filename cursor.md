You are a senior MLE building a production-grade platform to find and backtest profitable college football betting edges (ATS/totals/period markets) and assemble parlays with calibrated probabilities.

High-level goals

Data layer

Ingest historical College Football data via CollegeFootballData (CFBD) (Python client cfbd) — rosters, schedules, game stats, play-by-play (for EPA/WPA-like team strength), recruiting, SP+-style ratings if available.

Ingest historical odds + line movement & book="fanduel" snapshots via The Odds API (including 1Q/1H/full-game spreads/totals).

Optional weather: integrate one provider (e.g., Meteostat or Visual Crossing) to attach temperature, wind, precipitation to each game.

Warehouse & modeling dataset

Normalize keys (season, week, home/away, team_id, game_id, kickoff UTC).

Maintain bronze/silver/gold tables:

bronze: raw ingests; silver: cleaned & conformed; gold: modeling tables with engineered features and labels (ATS result, total hit/miss, 1Q/1H outcomes).

Compute rolling features: EPA per play (off/def), success rate, pace, finishing drives, havoc, explosiveness; injury/roster deltas; rest/travel; odds movement velocity; cross-book disagreement; 1Q/1H vs game deltas; weather impact factors.

Models

Start with binary classifiers for market outcomes (cover ATS? total over? 1Q spread?).

Use gradient boosting (XGBoost/LightGBM/CatBoost) + probability calibration (isotonic/Platt). Report Brier score, log loss, calibration curves.

Output implied probabilities; compare to market implied (vig-removed) to compute edge and Kelly fraction under constraints.

Parlay module: compute combined probability with correlation control (don’t assume independence). Use copula or empirical correlation from history; provide conservative blend.

Backtesting & portfolio

Run season/week backtests with transaction cost + market availability filters.

Track EV, ROI, drawdown, hit rate by market (full/1H/1Q).

Portfolio sizing: flat, fractional Kelly, capped exposure, diversification by team/market/time.

Interfaces

CLI (typer) for: ingest, build-features, train, backtest, scan-live (if later), make-parlays.

Tests (pytest) for data joins, label creation, feature transforms, calibration, and parlay math.

Light docs in README and docstrings.

Constraints & non-negotiables

Python 3.11+, poetry for deps, ruff + black for lint/format, pre-commit hooks.

All secrets via .env (dotenv). No keys hardcoded.

Deterministic training seeds; save artifacts with metadata (git SHA, data snapshot ranges).

DataFrame contracts with pydantic models (or pandera) for schema checks.

Clear unit/integration tests.

Avoid scraping; use public APIs with their TOS.

Tech stack

Data: pandas/polars, pyarrow, sqlite or duckdb (local) → easy to swap to Postgres.

Modeling: xgboost/lightgbm/catboost, sklearn, mapie (conformal optional), scikit-calibration or custom isotonic.

CLI: typer; Config: pydantic-settings.

Weather (pluggable): meteostat or visual-crossing client.

Plotting: matplotlib/plotly for calibration & ROC/PR, backtest equity curve.

External APIs (implement providers with clean interfaces)

CFBD: Use the cfbd Python client. Endpoints: games, rosters, teams, PBP (if available via bigquery or replicate via helpers), ratings if exposed.

The Odds API: sports=americanfootball_ncaaf, bookmakers filter includes "fanduel", markets: spreads/totals for full/1H/1Q if available; include historical snapshots.

Weather: provide WeatherProvider abstraction; start with one provider.

Data contracts (initial)

GameKey: { season:int, week:int, season_type:str, game_id:str, kickoff_utc:datetime, home:str, away:str, home_id:int?, away_id:int? }

OddsRow: { game_id, provider:str, book:str, market:str (spread,total), period:str (game,1H,1Q), fetched_at, home_price, away_price, home_handicap, total_points }

TeamStrength: rolling features per team & date (EPA_off, EPA_def, success_rate_off/def, pace, explosiveness, etc.)

WeatherRow: { game_id, kickoff_utc, temp_c, wind_mps, precip_mm, humidity }

LabelATS: { game_id, period, spread_line, result_cover:bool }

LabelTotal: { game_id, period, total_line, result_over:bool }

Build order (milestones)

Scaffold repo (poetry, pre-commit, folders, config).

Providers: CFBD client, The Odds API client, Weather client; write smoke tests that hit stub/mocked responses.

Ingest & normalize → bronze tables; write idempotent batchers by season/week.

Feature engineering (silver→gold): rolling windows; opponent-adjusted strength; weather join; odds movement features.

Label creation for ATS/Total across game/1H/1Q.

Model training (xgboost + calibration) with config-driven experiments; serialize with model cards.

Backtest across seasons; report metrics & plots.

Parlay optimizer with correlation penalties and Kelly-with-caps.

Deliverables / acceptance criteria

poetry run app ingest --season 2023 --odds-provider theodds --cfbd --weather meteostat runs end-to-end and populates data/bronze & data/silver.

poetry run app build-features --as-of 2024-08-01 generates data/gold/features.parquet with schema checks passing.

poetry run app train --market ats_1Q trains, saves artifacts/model_ats_1Q_*.pkl, outputs calibration curve and Brier score.

poetry run app backtest --season 2022 --market ats_game prints ROI, EV, max DD, and saves equity curve.

poetry run app make-parlays --date 2023-10-21 --top-n 20 --risk 1% returns proposed tickets with expected value and correlation-adjusted hit rates.

pytest -q passes; key transforms have unit tests.