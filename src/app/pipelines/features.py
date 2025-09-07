"""Feature engineering pipeline for gold layer."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..config import settings
from ..logging import get_logger
from ..validation.schemas import FeatureRow

logger = get_logger(__name__)


class FeaturePipeline:
    """Pipeline for engineering features from normalized data."""

    def __init__(self):
        """Initialize feature pipeline."""
        self.data_dir = settings.data_dir
        self.silver_dir = self.data_dir / "silver"
        self.gold_dir = self.data_dir / "gold"
        self.gold_dir.mkdir(exist_ok=True)
        self.rolling_windows = settings.rolling_windows

    def calculate_epa_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate EPA-based features (simplified implementation)."""
        logger.info("Calculating EPA features")

        # This is a simplified EPA calculation
        # In production, you'd use actual play-by-play data

        features_df = games_df.copy()

        # Simulate EPA calculations based on game stats
        # Offensive EPA per play (simplified)
        features_df["epa_off"] = (
            features_df.get("offensive_yards_per_play", 0) * 0.1
            + features_df.get("offensive_points_per_game", 0) * 0.05
        )

        # Defensive EPA per play (simplified)
        features_df["epa_def"] = (
            features_df.get("defensive_yards_per_play", 0) * -0.1
            + features_df.get("defensive_points_per_game", 0) * -0.05
        )

        # Success rate (simplified)
        features_df["success_rate_off"] = (
            features_df.get("offensive_yards_per_play", 0) / 6.0
        )
        features_df["success_rate_def"] = 1 - (
            features_df.get("defensive_yards_per_play", 0) / 6.0
        )

        # Pace (plays per game)
        features_df["pace"] = features_df.get("offensive_plays", 0) + features_df.get(
            "defensive_plays", 0
        )

        # Explosiveness (simplified)
        features_df["explosiveness_off"] = (
            features_df.get("offensive_yards_per_play", 0) * 0.2
        )
        features_df["explosiveness_def"] = (
            features_df.get("defensive_yards_per_play", 0) * -0.2
        )

        # Finishing drives (simplified)
        features_df["finishing_drives_off"] = (
            features_df.get("offensive_points_per_game", 0) / 10.0
        )
        features_df["finishing_drives_def"] = 1 - (
            features_df.get("defensive_points_per_game", 0) / 10.0
        )

        return features_df

    def calculate_rolling_features(self, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling window features for teams."""
        logger.info("Calculating rolling features")

        rolling_features = []

        for team in team_stats_df["team"].unique():
            team_data = team_stats_df[team_stats_df["team"] == team].copy()
            team_data = team_data.sort_values("date")

            for window in self.rolling_windows:
                # fmt: off
                # Rolling EPA
                team_data[f"epa_off_{window}g"] = team_data["epa_off"].rolling(window=window, min_periods=1).mean()
                team_data[f"epa_def_{window}g"] = team_data["epa_def"].rolling(window=window, min_periods=1).mean()

                # Rolling success rate
                team_data[f"success_rate_off_{window}g"] = team_data["success_rate_off"].rolling(window=window, min_periods=1).mean()
                team_data[f"success_rate_def_{window}g"] = team_data["success_rate_def"].rolling(window=window, min_periods=1).mean()

                # Rolling pace
                team_data[f"pace_{window}g"] = team_data["pace"].rolling(window=window, min_periods=1).mean()

                # Rolling explosiveness
                team_data[f"explosiveness_off_{window}g"] = team_data["explosiveness_off"].rolling(window=window, min_periods=1).mean()
                team_data[f"explosiveness_def_{window}g"] = team_data["explosiveness_def"].rolling(window=window, min_periods=1).mean()

                # Rolling finishing drives
                team_data[f"finishing_drives_off_{window}g"] = team_data["finishing_drives_off"].rolling(window=window, min_periods=1).mean()
                team_data[f"finishing_drives_def_{window}g"] = team_data["finishing_drives_def"].rolling(window=window, min_periods=1).mean()
                # fmt: on

            rolling_features.append(team_data)

        result_df = pd.concat(rolling_features, ignore_index=True)
        logger.info(
            f"Calculated rolling features for {len(result_df)} team-game records"
        )
        return result_df

    def calculate_odds_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate odds-based features."""
        logger.info("Calculating odds features")

        features_df = games_df.copy()

        # Line movement (simplified - would need historical odds data)
        features_df["line_movement"] = 0.0  # Placeholder

        # Movement velocity (simplified)
        features_df["movement_velocity"] = 0.0  # Placeholder

        # Cross-book delta (simplified)
        features_df["cross_book_delta"] = 0.0  # Placeholder

        return features_df

    def calculate_weather_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate weather-based features."""
        logger.info("Calculating weather features")

        features_df = games_df.copy()

        # Temperature features
        if "temp_c" in features_df.columns:
            features_df["temp_f"] = features_df["temp_c"] * 9 / 5 + 32
            features_df["is_cold"] = features_df["temp_f"] < 32
            features_df["is_hot"] = features_df["temp_f"] > 85
            features_df["is_extreme_temp"] = (
                features_df["is_cold"] | features_df["is_hot"]
            )
        else:
            features_df["temp_f"] = 70.0  # Default temperature
            features_df["is_cold"] = False
            features_df["is_hot"] = False
            features_df["is_extreme_temp"] = False

        # Wind features
        if "wind_mps" in features_df.columns:
            features_df["wind_mph"] = features_df["wind_mps"] * 2.237
            features_df["is_windy"] = features_df["wind_mph"] > 15
            features_df["is_very_windy"] = features_df["wind_mph"] > 25
        else:
            features_df["wind_mph"] = 5.0  # Default wind
            features_df["is_windy"] = False
            features_df["is_very_windy"] = False

        # Precipitation features
        if "precip_mm" in features_df.columns:
            features_df["is_rainy"] = features_df["precip_mm"] > 0
            features_df["is_heavy_rain"] = features_df["precip_mm"] > 5
        else:
            features_df["is_rainy"] = False
            features_df["is_heavy_rain"] = False

        # Combined weather impact
        features_df["weather_impact"] = (
            features_df["is_extreme_temp"].astype(int)
            + features_df["is_windy"].astype(int)
            + features_df["is_rainy"].astype(int)
        )

        return features_df

    def calculate_rest_travel_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rest and travel features."""
        logger.info("Calculating rest and travel features")

        features_df = games_df.copy()

        # Days rest (simplified - would need previous game dates)
        features_df["days_rest"] = 7  # Default to 7 days

        # Travel distance (simplified - would need stadium coordinates)
        features_df["travel_distance"] = 0.0  # Default to 0 (home game)

        return features_df

    def create_feature_matrix(
        self, games_df: pd.DataFrame, team_stats_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Create final feature matrix."""
        logger.info("Creating feature matrix")

        # Start with games data
        features_df = games_df.copy()

        # Add EPA features
        features_df = self.calculate_epa_features(features_df)

        # Add rolling features (if team stats available)
        if not team_stats_df.empty:
            rolling_features = self.calculate_rolling_features(team_stats_df)
            # Merge rolling features (simplified)
            for col in rolling_features.columns:
                if col not in features_df.columns:
                    features_df[col] = 0.0

        # Add odds features
        features_df = self.calculate_odds_features(features_df)

        # Add weather features
        features_df = self.calculate_weather_features(features_df)

        # Add rest/travel features
        features_df = self.calculate_rest_travel_features(features_df)

        # Ensure all required feature columns exist
        required_features = [
            "epa_off_3g",
            "epa_def_3g",
            "epa_off_5g",
            "epa_def_5g",
            "epa_off_10g",
            "epa_def_10g",
            "success_rate_off_3g",
            "success_rate_def_3g",
            "success_rate_off_5g",
            "success_rate_def_5g",
            "pace_3g",
            "pace_5g",
            "line_movement",
            "movement_velocity",
            "cross_book_delta",
            "temp_c",
            "wind_mps",
            "precip_mm",
            "days_rest",
            "travel_distance",
        ]

        for feature in required_features:
            if feature not in features_df.columns:
                features_df[feature] = 0.0

        # Validate using schema
        try:
            for _, row in features_df.iterrows():
                FeatureRow(**row.to_dict())
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            raise

        logger.info(
            f"Created feature matrix with {len(features_df)} records and {len(features_df.columns)} features"
        )
        return features_df

    def save_gold_data(self, data_df: pd.DataFrame, filename: str) -> None:
        """Save feature data to gold layer."""
        filepath = self.gold_dir / filename
        data_df.to_parquet(filepath, index=False)
        logger.info(f"Saved gold data to {filepath}")

    def run_feature_engineering(self, as_of_date: str) -> pd.DataFrame:
        """Run full feature engineering pipeline."""
        logger.info(f"Starting feature engineering as of {as_of_date}")

        # Load silver data
        silver_files = list(self.silver_dir.glob("normalized_*.parquet"))
        if not silver_files:
            raise ValueError("No normalized data found in silver layer")

        # Load the most recent normalized data
        latest_file = max(silver_files, key=lambda x: x.stat().st_mtime)
        games_df = pd.read_parquet(latest_file)

        # Load team stats (if available)
        team_stats_files = list(self.silver_dir.glob("team_stats_*.parquet"))
        team_stats_df = (
            pd.read_parquet(team_stats_files[0]) if team_stats_files else pd.DataFrame()
        )

        # Create feature matrix
        features_df = self.create_feature_matrix(games_df, team_stats_df)

        # Save to gold layer
        gold_filename = f"features_{as_of_date.replace('-', '')}.parquet"
        self.save_gold_data(features_df, gold_filename)

        logger.info("Feature engineering completed successfully")
        return features_df
