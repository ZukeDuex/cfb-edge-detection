# CFB Edge Detection Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Poetry](https://img.shields.io/badge/poetry-dependency%20management-blue.svg)](https://python-poetry.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-grade platform to find and backtest profitable college football betting edges (ATS/totals/period markets) and assemble parlays with calibrated probabilities.

## Quick Start

1. Install dependencies:
```bash
poetry install
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the platform:
```bash
# Ingest data
poetry run app ingest --season 2023 --odds-provider theodds --cfbd --weather meteostat

# Build features
poetry run app build-features --as-of 2024-08-01

# Train models
poetry run app train --market ats_1Q

# Run backtests
poetry run app backtest --season 2022 --market ats_game

# Generate parlays
poetry run app make-parlays --date 2023-10-21 --top-n 20 --risk 1%
```

## Features

- **Data Ingestion**: Historical CFB data via CFBD API, odds via The Odds API, weather data
- **Feature Engineering**: Rolling team statistics, EPA calculations, odds movement analysis
- **ML Models**: Gradient boosting with probability calibration
- **Backtesting**: Comprehensive evaluation with transaction costs
- **Parlay Optimization**: Correlation-aware parlay assembly with Kelly sizing

## Data Contracts

The platform uses structured data contracts for:
- Game metadata (season, week, teams, kickoff times)
- Odds data (spreads, totals, line movement)
- Team strength metrics (EPA, success rates, pace)
- Weather conditions
- Betting outcomes (ATS covers, totals over/under)

## Architecture

- **Bronze Layer**: Raw API data
- **Silver Layer**: Cleaned and normalized data
- **Gold Layer**: Feature-engineered modeling datasets
- **Models**: Calibrated ML models for edge detection
- **Backtesting**: Historical performance evaluation
- **Parlays**: Multi-leg bet optimization

## Testing

Run the test suite:
```bash
poetry run pytest
```

## Development

Set up pre-commit hooks:
```bash
poetry run pre-commit install
```

Format and lint:
```bash
poetry run ruff check .
poetry run black .
```
