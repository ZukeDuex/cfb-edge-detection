"""Weather data provider using Meteostat."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from meteostat import Point, Hourly

from ..config import settings
from ..logging import get_logger

logger = get_logger(__name__)


class MeteostatWeatherProvider:
    """Weather data provider using Meteostat."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize weather provider."""
        self.api_key = api_key or settings.weather_api_key
        # Meteostat doesn't require API key for basic usage

    def _get_stadium_location(self, team: str) -> Optional[Point]:
        """Get stadium coordinates for a team."""
        # This is a simplified mapping - in production, you'd want a comprehensive database
        stadium_locations = {
            "Alabama": Point(33.2084, -87.5504),  # Bryant-Denny Stadium
            "Auburn": Point(32.6024, -85.4902),  # Jordan-Hare Stadium
            "Georgia": Point(33.9500, -83.3750),  # Sanford Stadium
            "Florida": Point(29.6500, -82.3500),  # Ben Hill Griffin Stadium
            "LSU": Point(30.4115, -91.1899),  # Tiger Stadium
            "Texas": Point(30.2836, -97.7316),  # Darrell K Royal Stadium
            "Oklahoma": Point(35.2058, -97.4456),  # Gaylord Family Stadium
            "Ohio State": Point(40.0019, -83.0197),  # Ohio Stadium
            "Michigan": Point(42.2658, -83.7485),  # Michigan Stadium
            "Penn State": Point(40.8123, -77.8561),  # Beaver Stadium
            "USC": Point(34.2569, -118.2879),  # Los Angeles Memorial Coliseum
            "UCLA": Point(34.0722, -118.2437),  # Rose Bowl
            "Notre Dame": Point(41.6990, -86.2384),  # Notre Dame Stadium
            "Clemson": Point(34.6788, -82.8373),  # Memorial Stadium
            "Florida State": Point(30.4419, -84.2911),  # Doak Campbell Stadium
            "Miami": Point(25.6081, -80.4090),  # Hard Rock Stadium
            "Stanford": Point(37.4341, -122.1637),  # Stanford Stadium
            "Oregon": Point(44.0582, -123.0703),  # Autzen Stadium
            "Washington": Point(47.6500, -122.3031),  # Husky Stadium
            "Wisconsin": Point(43.0708, -89.4062),  # Camp Randall Stadium
            "Iowa": Point(41.6581, -91.5517),  # Kinnick Stadium
            "Nebraska": Point(40.8207, -96.7056),  # Memorial Stadium
            "Michigan State": Point(42.7258, -84.4806),  # Spartan Stadium
            "Minnesota": Point(44.9778, -93.2225),  # Huntington Bank Stadium
            "Northwestern": Point(42.0650, -87.6967),  # Ryan Field
            "Purdue": Point(40.4259, -86.9142),  # Ross-Ade Stadium
            "Indiana": Point(39.1817, -86.5264),  # Memorial Stadium
            "Illinois": Point(40.1020, -88.2272),  # Memorial Stadium
            "Maryland": Point(38.9887, -76.9376),  # Maryland Stadium
            "Rutgers": Point(40.5008, -74.4474),  # SHI Stadium
        }

        return stadium_locations.get(team)

    def fetch_weather(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Fetch weather data for games."""
        weather_data = []

        for _, game in games_df.iterrows():
            try:
                # Get stadium location
                home_location = self._get_stadium_location(game["home"])
                if not home_location:
                    logger.warning(f"No location data for {game['home']}")
                    continue

                # Get weather data around kickoff time
                kickoff = game["kickoff_utc"]
                start_time = kickoff - timedelta(hours=2)
                end_time = kickoff + timedelta(hours=2)

                # Fetch hourly weather data
                weather = Hourly(home_location, start_time, end_time)
                weather = weather.fetch()

                if weather.empty:
                    logger.warning(f"No weather data for {game['home']} at {kickoff}")
                    continue

                # Find closest hour to kickoff
                weather["time_diff"] = abs((weather.index - kickoff).total_seconds())
                closest_hour = weather.loc[weather["time_diff"].idxmin()]

                weather_data.append(
                    {
                        "game_id": game["game_id"],
                        "kickoff_utc": kickoff,
                        "temp_c": closest_hour.get("temp"),
                        "wind_mps": closest_hour.get("wspd"),
                        "precip_mm": closest_hour.get("prcp"),
                        "humidity": closest_hour.get("rhum"),
                        "pressure": closest_hour.get("pres"),
                        "cloud_cover": closest_hour.get("coco"),
                    }
                )

            except Exception as e:
                logger.error(f"Error fetching weather for game {game['game_id']}: {e}")
                continue

        if not weather_data:
            logger.warning("No weather data fetched")
            return pd.DataFrame()

        df = pd.DataFrame(weather_data)
        logger.info(f"Fetched weather data for {len(df)} games")
        return df

    def get_weather_features(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Generate weather-based features."""
        features_df = weather_df.copy()

        # Temperature features
        features_df["temp_f"] = features_df["temp_c"] * 9 / 5 + 32
        features_df["is_cold"] = features_df["temp_f"] < 32
        features_df["is_hot"] = features_df["temp_f"] > 85
        features_df["is_extreme_temp"] = features_df["is_cold"] | features_df["is_hot"]

        # Wind features
        features_df["wind_mph"] = features_df["wind_mps"] * 2.237
        features_df["is_windy"] = features_df["wind_mph"] > 15
        features_df["is_very_windy"] = features_df["wind_mph"] > 25

        # Precipitation features
        features_df["is_rainy"] = features_df["precip_mm"] > 0
        features_df["is_heavy_rain"] = features_df["precip_mm"] > 5

        # Combined weather impact
        features_df["weather_impact"] = (
            features_df["is_extreme_temp"].astype(int)
            + features_df["is_windy"].astype(int)
            + features_df["is_rainy"].astype(int)
        )

        return features_df
