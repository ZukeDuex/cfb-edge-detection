"""Labels pipeline for creating betting outcome labels."""

import pandas as pd

from app.config import settings
from app.logging import get_logger
from app.validation.schemas import LabelATS, LabelTotal

logger = get_logger(__name__)


class LabelsPipeline:
    """Pipeline for creating betting outcome labels."""

    def __init__(self):
        """Initialize labels pipeline."""
        self.data_dir = settings.data_dir
        self.gold_dir = self.data_dir / "gold"
        self.gold_dir.mkdir(exist_ok=True)

    def _get_period_scores(self, game: pd.Series, period: str) -> tuple[float, float]:
        """Return home and away scores for a given period.

        Scores are sourced from ``home_score``/``away_score`` for the full game,
        ``home_score_1H``/``away_score_1H`` for first-half lines, and
        ``home_score_1Q``/``away_score_1Q`` for first-quarter lines when those
        columns are available. If the period-specific columns are missing, the
        scores fall back to scaled final scores.
        """

        if period == "game":
            return game.get("home_score", 0), game.get("away_score", 0)

        if period == "1H":
            home_score = game.get("home_score_1H")
            away_score = game.get("away_score_1H")
            if pd.isna(home_score) or pd.isna(away_score):
                home_score = game.get("home_score", 0) * 0.5
                away_score = game.get("away_score", 0) * 0.5
            return home_score, away_score

        if period == "1Q":
            home_score = game.get("home_score_1Q")
            away_score = game.get("away_score_1Q")
            if pd.isna(home_score) or pd.isna(away_score):
                home_score = game.get("home_score", 0) * 0.25
                away_score = game.get("away_score", 0) * 0.25
            return home_score, away_score

        msg = f"Unsupported period: {period}"
        raise ValueError(msg)

    def create_ats_labels(
        self,
        games_df: pd.DataFrame,
        period: str = "game",
    ) -> pd.DataFrame:
        """Create ATS (Against The Spread) labels.

        Source columns: ``home_score``/``away_score`` for game lines,
        ``home_score_1H``/``away_score_1H`` for first-half lines, and
        ``home_score_1Q``/``away_score_1Q`` for first-quarter lines.
        """
        logger.info("Creating ATS labels for %s period", period)

        labels_data = []

        for _, game in games_df.iterrows():
            try:
                # Get spread line for the period
                spread_col = f"home_handicap_{period}_spread"
                if spread_col not in game.index or pd.isna(game[spread_col]):
                    continue

                spread_line = game[spread_col]

                # Get scores for the period
                try:
                    home_score, away_score = self._get_period_scores(game, period)
                except ValueError:
                    continue

                # Calculate ATS result
                # Positive spread means home team is favored
                if spread_line > 0:
                    # Home team is favored
                    result_cover = (home_score - away_score) > spread_line
                else:
                    # Away team is favored
                    result_cover = (away_score - home_score) > abs(spread_line)

                labels_data.append(
                    {
                        "game_id": game["game_id"],
                        "period": period,
                        "spread_line": spread_line,
                        "result_cover": result_cover,
                        "home_score": home_score,
                        "away_score": away_score,
                        "home_team": game["home"],
                        "away_team": game["away"],
                    },
                )

            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Error creating ATS label for game %s: %s",
                    game["game_id"],
                    e,
                )
                continue

        if not labels_data:
            logger.warning("No ATS labels created")
            return pd.DataFrame()

        labels_df = pd.DataFrame(labels_data)

        # Validate using schema
        try:
            for _, row in labels_df.iterrows():
                LabelATS(**row.to_dict())
        except Exception:
            logger.exception("Schema validation failed")
            raise

        logger.info("Created %s ATS labels for %s period", len(labels_df), period)
        return labels_df

    def create_total_labels(
        self,
        games_df: pd.DataFrame,
        period: str = "game",
    ) -> pd.DataFrame:
        """Create totals (over/under) labels.

        Source columns mirror :meth:`create_ats_labels` and come from
        ``home_score``/``away_score``, ``home_score_1H``/``away_score_1H``, and
        ``home_score_1Q``/``away_score_1Q``.
        """
        logger.info("Creating total labels for %s period", period)

        labels_data = []

        for _, game in games_df.iterrows():
            try:
                # Get total line for the period
                total_col = f"total_points_{period}_total"
                if total_col not in game.index or pd.isna(game[total_col]):
                    continue

                total_line = game[total_col]

                # Get scores for the period
                try:
                    home_score, away_score = self._get_period_scores(game, period)
                except ValueError:
                    continue

                # Calculate total result
                total_points = home_score + away_score
                result_over = total_points > total_line

                labels_data.append(
                    {
                        "game_id": game["game_id"],
                        "period": period,
                        "total_line": total_line,
                        "result_over": result_over,
                        "total_points": total_points,
                        "home_score": home_score,
                        "away_score": away_score,
                        "home_team": game["home"],
                        "away_team": game["away"],
                    },
                )

            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Error creating total label for game %s: %s",
                    game["game_id"],
                    e,
                )
                continue

        if not labels_data:
            logger.warning("No total labels created")
            return pd.DataFrame()

        labels_df = pd.DataFrame(labels_data)

        # Validate using schema
        try:
            for _, row in labels_df.iterrows():
                LabelTotal(**row.to_dict())
        except Exception:
            logger.exception("Schema validation failed")
            raise

        logger.info("Created %s total labels for %s period", len(labels_df), period)
        return labels_df

    def create_all_labels(self, games_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Create all possible labels."""
        logger.info("Creating all labels")

        periods = ["game", "1H", "1Q"]
        markets = ["ats", "total"]

        all_labels = {}

        for period in periods:
            for market in markets:
                if market == "ats":
                    labels_df = self.create_ats_labels(games_df, period)
                    key = f"ats_{period}"
                else:
                    labels_df = self.create_total_labels(games_df, period)
                    key = f"total_{period}"

                if not labels_df.empty:
                    all_labels[key] = labels_df

        logger.info("Created labels for %s market-period combinations", len(all_labels))
        return all_labels

    def save_labels(self, labels_dict: dict[str, pd.DataFrame]) -> None:
        """Save labels to gold layer."""
        for market_period, labels_df in labels_dict.items():
            filename = f"labels_{market_period}.parquet"
            filepath = self.gold_dir / filename
            labels_df.to_parquet(filepath, index=False)
            logger.info(
                "Saved %s %s labels to %s",
                len(labels_df),
                market_period,
                filepath,
            )

    def run_label_creation(
        self,
        season: int,
        week: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Run full label creation pipeline."""
        if week is not None:
            logger.info("Starting label creation for season %s, week %s", season, week)
        else:
            logger.info("Starting label creation for season %s", season)

        # Load feature data
        feature_files = list(self.gold_dir.glob("features_*.parquet"))
        if not feature_files:
            msg = "No feature data found in gold layer"
            raise ValueError(msg)

        # Load the most recent feature data
        latest_file = max(feature_files, key=lambda x: x.stat().st_mtime)
        games_df = pd.read_parquet(latest_file)

        # Create all labels
        all_labels = self.create_all_labels(games_df)

        # Save labels
        self.save_labels(all_labels)

        logger.info("Label creation completed successfully")
        return all_labels
