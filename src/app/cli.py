"""Command-line interface for CFB Edge platform."""

import logging
from datetime import datetime
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import settings
from .logging import setup_logging, get_logger
from .pipelines.ingest import IngestPipeline
from .pipelines.normalize import NormalizePipeline
from .pipelines.features import FeaturePipeline
from .pipelines.labels import LabelsPipeline
from .pipelines.backtest import BacktestPipeline
from .pipelines.parlays import ParlayOptimizer
from .models.train import ModelTrainer
from .models.calibration import ModelCalibrator
from .providers.weather_meteostat import MeteostatWeatherProvider
from .providers.weather_weatherapi import WeatherAPIProvider

# Initialize CLI app
app = typer.Typer(help="CFB Edge Detection Platform")
console = Console()

# Initialize pipelines
normalize_pipeline = NormalizePipeline()
feature_pipeline = FeaturePipeline()
labels_pipeline = LabelsPipeline()
backtest_pipeline = BacktestPipeline()
parlay_optimizer = ParlayOptimizer()
model_trainer = ModelTrainer()
model_calibrator = ModelCalibrator()


@app.command()
def ingest(
    season: int = typer.Option(..., help="CFB season year"),
    week: Optional[int] = typer.Option(None, help="Specific week (optional)"),
    odds_provider: str = typer.Option("theodds", help="Odds provider"),
    cfbd: bool = typer.Option(True, help="Include CFBD data"),
    weather: Optional[str] = typer.Option(
        None, help="Weather provider (meteostat, weatherapi)"
    ),
):
    """Ingest data from various sources."""
    setup_logging()
    logger = get_logger(__name__)

    console.print(f"[bold blue]Ingesting data for season {season}[/bold blue]")
    if week:
        console.print(f"Week: {week}")

    weather_provider = None
    if weather:
        if weather.lower() == "meteostat":
            weather_provider = MeteostatWeatherProvider()
        elif weather.lower() == "weatherapi":
            weather_provider = WeatherAPIProvider()
        else:
            raise typer.BadParameter("Unsupported weather provider")

    ingest_pipeline = IngestPipeline(weather_provider=weather_provider)

    try:
        results = ingest_pipeline.run_full_ingest(
            season=season, week=week, include_weather=weather_provider is not None
        )

        # Display results
        table = Table(title="Ingestion Results")
        table.add_column("Source", style="cyan")
        table.add_column("Records", style="green")

        for source, data in results.items():
            if hasattr(data, "__len__"):
                table.add_row(source, str(len(data)))
            else:
                table.add_row(source, "N/A")

        console.print(table)
        console.print("[bold green]Ingestion completed successfully![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error during ingestion: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def build_features(
    as_of: str = typer.Option(..., help="Date to build features as of (YYYY-MM-DD)"),
):
    """Build feature engineering pipeline."""
    setup_logging()
    logger = get_logger(__name__)

    console.print(f"[bold blue]Building features as of {as_of}[/bold blue]")

    try:
        features_df = feature_pipeline.run_feature_engineering(as_of)

        console.print(f"[bold green]Features built successfully![/bold green]")
        console.print(f"Records: {len(features_df)}")
        console.print(f"Features: {len(features_df.columns)}")

    except Exception as e:
        console.print(f"[bold red]Error building features: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def train(
    market: str = typer.Option(..., help="Market to train (e.g., ats_game, total_1H)"),
):
    """Train ML model for a specific market."""
    setup_logging()
    logger = get_logger(__name__)

    console.print(f"[bold blue]Training model for market: {market}[/bold blue]")

    try:
        # Train model
        model_results = model_trainer.train_market_model(market)

        # Display metrics
        table = Table(title=f"Model Performance - {market}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        for metric, value in model_results["metrics"].items():
            table.add_row(metric, f"{value:.4f}")

        console.print(table)

        # Calibrate model
        console.print("[bold blue]Calibrating model...[/bold blue]")
        calibration_results = model_calibrator.calibrate_market_model(market)

        console.print(f"[bold green]Model training completed![/bold green]")
        console.print(
            f"Brier Score: {calibration_results['metrics']['brier_score']:.4f}"
        )

    except Exception as e:
        console.print(f"[bold red]Error training model: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def backtest(
    season: int = typer.Option(..., help="Season to backtest"),
    market: str = typer.Option("ats_game", help="Market to backtest"),
):
    """Run backtest for a specific season and market."""
    setup_logging()
    logger = get_logger(__name__)

    console.print(
        f"[bold blue]Running backtest for season {season}, market {market}[/bold blue]"
    )

    try:
        results = backtest_pipeline.run_backtest(season, market)

        # Display metrics
        table = Table(title=f"Backtest Results - {market}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        metrics = results["metrics"]
        table.add_row("Total Bets", str(metrics["total_bets"]))
        table.add_row("Hit Rate", f"{metrics['hit_rate']:.3f}")
        table.add_row("Total Return", f"${metrics['total_return']:.2f}")
        table.add_row("Total EV", f"${metrics['total_ev']:.2f}")
        table.add_row("Max Drawdown", f"${metrics['max_drawdown']:.2f}")
        table.add_row("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")

        console.print(table)
        console.print("[bold green]Backtest completed successfully![/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error during backtest: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def make_parlays(
    date: str = typer.Option(..., help="Date for parlay recommendations (YYYY-MM-DD)"),
    top_n: int = typer.Option(20, help="Number of top parlays to generate"),
    risk: str = typer.Option("1%", help="Risk percentage"),
):
    """Generate parlay recommendations."""
    setup_logging()
    logger = get_logger(__name__)

    # Parse risk percentage
    risk_percentage = float(risk.replace("%", ""))

    console.print(
        f"[bold blue]Generating parlay recommendations for {date}[/bold blue]"
    )
    console.print(f"Top N: {top_n}, Risk: {risk_percentage}%")

    try:
        recommendations = parlay_optimizer.run_parlay_optimization(
            date=date, top_n=top_n, risk_percentage=risk_percentage
        )

        if not recommendations:
            console.print("[bold yellow]No profitable parlays found[/bold yellow]")
            return

        # Display top recommendations
        table = Table(title=f"Top Parlay Recommendations - {date}")
        table.add_column("Parlay ID", style="cyan")
        table.add_column("Legs", style="green")
        table.add_column("Probability", style="yellow")
        table.add_column("Odds", style="magenta")
        table.add_column("EV", style="red")
        table.add_column("Stake", style="blue")
        table.add_column("Potential Return", style="green")

        for rec in recommendations[:10]:  # Show top 10
            table.add_row(
                rec["parlay_id"],
                str(rec["legs"]),
                f"{rec['parlay_prob']:.3f}",
                f"{rec['parlay_odds']:.2f}",
                f"{rec['ev']:.3f}",
                f"${rec['recommended_stake']:.2f}",
                f"${rec['potential_return']:.2f}",
            )

        console.print(table)
        console.print(
            f"[bold green]Generated {len(recommendations)} parlay recommendations![/bold green]"
        )

    except Exception as e:
        console.print(f"[bold red]Error generating parlays: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def normalize(
    season: int = typer.Option(..., help="CFB season year"),
    week: Optional[int] = typer.Option(None, help="Specific week (optional)"),
):
    """Normalize and clean ingested data."""
    setup_logging()
    logger = get_logger(__name__)

    console.print(f"[bold blue]Normalizing data for season {season}[/bold blue]")
    if week:
        console.print(f"Week: {week}")

    try:
        normalized_df = normalize_pipeline.run_normalization(season, week)

        console.print(f"[bold green]Normalization completed![/bold green]")
        console.print(f"Records: {len(normalized_df)}")
        console.print(f"Columns: {len(normalized_df.columns)}")

    except Exception as e:
        console.print(f"[bold red]Error during normalization: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def create_labels(
    season: int = typer.Option(..., help="CFB season year"),
    week: Optional[int] = typer.Option(None, help="Specific week (optional)"),
):
    """Create betting outcome labels."""
    setup_logging()
    logger = get_logger(__name__)

    console.print(f"[bold blue]Creating labels for season {season}[/bold blue]")
    if week:
        console.print(f"Week: {week}")

    try:
        labels_dict = labels_pipeline.run_label_creation(season, week)

        console.print(f"[bold green]Label creation completed![/bold green]")

        # Display label counts
        table = Table(title="Label Creation Results")
        table.add_column("Market-Period", style="cyan")
        table.add_column("Records", style="green")

        for market_period, labels_df in labels_dict.items():
            table.add_row(market_period, str(len(labels_df)))

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error creating labels: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def status():
    """Show platform status and configuration."""
    console.print("[bold blue]CFB Edge Platform Status[/bold blue]")

    # Configuration
    table = Table(title="Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Data Directory", str(settings.data_dir))
    table.add_row("Artifacts Directory", str(settings.artifacts_dir))
    table.add_row("Random Seed", str(settings.random_seed))
    table.add_row("Test Size", str(settings.test_size))
    table.add_row("Calibration Method", settings.calibration_method)
    table.add_row("Transaction Cost", f"${settings.transaction_cost:.2f}")
    table.add_row("Max Kelly Fraction", f"{settings.max_kelly_fraction:.2f}")

    console.print(table)

    # Check API keys
    api_table = Table(title="API Keys")
    api_table.add_column("Provider", style="cyan")
    api_table.add_column("Status", style="green")

    api_table.add_row("CFBD", "✓" if settings.cfbd_api_key else "✗")
    api_table.add_row("Odds API", "✓" if settings.odds_api_key else "✗")
    api_table.add_row("Weather", "✓" if settings.weather_api_key else "✗")

    console.print(api_table)


if __name__ == "__main__":
    app()
