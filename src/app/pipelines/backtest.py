"""Backtesting pipeline for evaluating model performance."""

from datetime import datetime
from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

from ..config import settings
from ..logging import get_logger
from ..utils.betting import american_to_decimal, kelly_fraction, calculate_ev

logger = get_logger(__name__)


class BacktestPipeline:
    """Pipeline for backtesting model performance."""

    def __init__(self):
        """Initialize backtesting pipeline."""
        self.artifacts_dir = settings.artifacts_dir
        self.artifacts_dir.mkdir(exist_ok=True)
        self.transaction_cost = settings.transaction_cost
        self.max_kelly_fraction = settings.max_kelly_fraction

    def calculate_betting_metrics(self, predictions_df: pd.DataFrame) -> Dict:
        """Calculate betting performance metrics."""
        logger.info("Calculating betting metrics")

        # Calculate expected value
        predictions_df["ev"] = predictions_df.apply(
            lambda row: calculate_ev(row["model_prob"], row["decimal_odds"]), axis=1
        )

        # Calculate Kelly fraction
        predictions_df["kelly_fraction"] = predictions_df.apply(
            lambda row: min(
                kelly_fraction(row["model_prob"], row["decimal_odds"]),
                self.max_kelly_fraction,
            ),
            axis=1,
        )

        # Apply transaction costs
        predictions_df["net_ev"] = predictions_df["ev"] - self.transaction_cost

        # Calculate returns
        predictions_df["bet_size"] = (
            predictions_df["kelly_fraction"] * 100
        )  # Assume $100 bankroll
        predictions_df["win"] = predictions_df["outcome"].astype(int)
        predictions_df["return"] = predictions_df.apply(
            lambda row: (row["bet_size"] * (row["decimal_odds"] - 1))
            if row["win"]
            else -row["bet_size"],
            axis=1,
        )

        # Calculate cumulative metrics
        predictions_df["cumulative_return"] = predictions_df["return"].cumsum()
        predictions_df["cumulative_ev"] = predictions_df["net_ev"].cumsum()

        # Calculate performance metrics
        total_bets = len(predictions_df)
        winning_bets = predictions_df["win"].sum()
        hit_rate = winning_bets / total_bets if total_bets > 0 else 0

        total_return = predictions_df["return"].sum()
        total_ev = predictions_df["net_ev"].sum()

        # Calculate drawdown
        cumulative_returns = predictions_df["cumulative_return"]
        running_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - running_max
        max_drawdown = drawdown.min()

        # Calculate Sharpe ratio (simplified)
        returns_std = predictions_df["return"].std()
        sharpe_ratio = total_return / returns_std if returns_std > 0 else 0

        metrics = {
            "total_bets": total_bets,
            "winning_bets": winning_bets,
            "hit_rate": hit_rate,
            "total_return": total_return,
            "total_ev": total_ev,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "avg_bet_size": predictions_df["bet_size"].mean(),
            "avg_kelly_fraction": predictions_df["kelly_fraction"].mean(),
        }

        logger.info(
            f"Calculated metrics: Hit rate: {hit_rate:.3f}, Total return: ${total_return:.2f}"
        )
        return metrics, predictions_df

    def plot_equity_curve(self, predictions_df: pd.DataFrame, market: str) -> None:
        """Plot equity curve."""
        logger.info(f"Creating equity curve for {market}")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Cumulative returns
        ax1.plot(
            predictions_df.index,
            predictions_df["cumulative_return"],
            label="Cumulative Return",
            linewidth=2,
        )
        ax1.plot(
            predictions_df.index,
            predictions_df["cumulative_ev"],
            label="Cumulative EV",
            linewidth=2,
            alpha=0.7,
        )
        ax1.set_xlabel("Bet Number")
        ax1.set_ylabel("Cumulative Return ($)")
        ax1.set_title(f"Equity Curve - {market}")
        ax1.legend()
        ax1.grid(True)

        # Drawdown
        cumulative_returns = predictions_df["cumulative_return"]
        running_max = cumulative_returns.expanding().max()
        drawdown = cumulative_returns - running_max

        ax2.fill_between(predictions_df.index, drawdown, 0, alpha=0.3, color="red")
        ax2.plot(predictions_df.index, drawdown, color="red", linewidth=1)
        ax2.set_xlabel("Bet Number")
        ax2.set_ylabel("Drawdown ($)")
        ax2.set_title("Drawdown")
        ax2.grid(True)

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"equity_curve_{market}_{timestamp}.png"
        plot_path = self.artifacts_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved equity curve to {plot_path}")

    def plot_calibration_by_decile(
        self, predictions_df: pd.DataFrame, market: str
    ) -> None:
        """Plot calibration by decile."""
        logger.info(f"Creating calibration by decile plot for {market}")

        # Create deciles
        predictions_df["decile"] = pd.qcut(
            predictions_df["model_prob"], 10, labels=False
        )

        # Calculate calibration by decile
        decile_stats = (
            predictions_df.groupby("decile")
            .agg({"model_prob": "mean", "outcome": "mean", "win": "count"})
            .reset_index()
        )

        decile_stats.columns = ["decile", "avg_predicted_prob", "actual_rate", "count"]

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.scatter(
            decile_stats["avg_predicted_prob"],
            decile_stats["actual_rate"],
            s=decile_stats["count"] * 2,
            alpha=0.7,
        )
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

        ax.set_xlabel("Average Predicted Probability")
        ax.set_ylabel("Actual Win Rate")
        ax.set_title(f"Calibration by Decile - {market}")
        ax.legend()
        ax.grid(True)

        # Add count labels
        for _, row in decile_stats.iterrows():
            ax.annotate(
                f"n={row['count']}",
                (row["avg_predicted_prob"], row["actual_rate"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )

        plt.tight_layout()

        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"calibration_decile_{market}_{timestamp}.png"
        plot_path = self.artifacts_dir / plot_filename
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved calibration by decile plot to {plot_path}")

    def run_backtest(
        self, season: int, market: str, predictions_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """Run backtest for a specific season and market.

        Parameters
        ----------
        season : int
            Season year for the backtest.
        market : str
            Betting market identifier (e.g., ``spread`` or ``total``).
        predictions_df : pandas.DataFrame, optional
            DataFrame containing model predictions. Expected columns are:
            ``game_id`` (str), ``model_prob`` (float), ``market_prob`` (float),
            ``american_odds`` (int), and ``outcome`` (int). If not provided,
            predictions are loaded from
            ``artifacts/predictions_{market}_{season}.parquet``.
        """
        logger.info(f"Running backtest for season {season}, market {market}")

        if predictions_df is None:
            predictions_path = (
                self.artifacts_dir / f"predictions_{market}_{season}.parquet"
            )
            if not predictions_path.exists():
                raise FileNotFoundError(
                    f"Predictions file not found at {predictions_path}"
                )
            predictions_df = pd.read_parquet(predictions_path)

        required_cols = {
            "game_id",
            "model_prob",
            "market_prob",
            "american_odds",
            "outcome",
        }
        missing_cols = required_cols - set(predictions_df.columns)
        if missing_cols:
            raise ValueError(
                f"Predictions DataFrame is missing required columns: {missing_cols}"
            )

        # Convert American odds to decimal if not already present
        if "decimal_odds" not in predictions_df.columns:
            predictions_df["decimal_odds"] = predictions_df["american_odds"].apply(
                american_to_decimal
            )

        # Calculate betting metrics
        metrics, predictions_df = self.calculate_betting_metrics(predictions_df)

        # Create plots
        self.plot_equity_curve(predictions_df, market)
        self.plot_calibration_by_decile(predictions_df, market)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"backtest_results_{market}_{season}_{timestamp}.pkl"
        results_path = self.artifacts_dir / results_filename

        results = {
            "season": season,
            "market": market,
            "metrics": metrics,
            "predictions": predictions_df,
            "timestamp": timestamp,
        }

        import pickle

        with open(results_path, "wb") as f:
            pickle.dump(results, f)

        logger.info(f"Backtest completed for {market}. Results saved to {results_path}")
        logger.info(
            f"Hit rate: {metrics['hit_rate']:.3f}, Total return: ${metrics['total_return']:.2f}"
        )

        return results
