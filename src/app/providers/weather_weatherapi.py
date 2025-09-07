"""Weather data provider using WeatherAPI.com."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

from ..config import settings
from ..logging import get_logger

logger = get_logger(__name__)


class WeatherAPIProvider:
    """Weather data provider using WeatherAPI.com."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize weather provider."""
        self.api_key = api_key or settings.weather_api_key
        self.base_url = "http://api.weatherapi.com/v1"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "CFB-Edge-Platform/1.0"})

    def _get_stadium_location(self, team: str) -> Optional[tuple]:
        """Get stadium coordinates for a team."""
        # This is a simplified mapping - in production, you'd want a comprehensive database
        stadium_locations = {
            "Alabama": (33.2084, -87.5504),  # Bryant-Denny Stadium
            "Auburn": (32.6024, -85.4902),  # Jordan-Hare Stadium
            "Georgia": (33.9500, -83.3750),  # Sanford Stadium
            "Florida": (29.6500, -82.3500),  # Ben Hill Griffin Stadium
            "LSU": (30.4115, -91.1899),  # Tiger Stadium
            "Texas": (30.2836, -97.7316),  # Darrell K Royal Stadium
            "Oklahoma": (35.2058, -97.4456),  # Gaylord Family Stadium
            "Ohio State": (40.0019, -83.0197),  # Ohio Stadium
            "Michigan": (42.2658, -83.7485),  # Michigan Stadium
            "Penn State": (40.8123, -77.8561),  # Beaver Stadium
            "USC": (34.2569, -118.2879),  # Los Angeles Memorial Coliseum
            "UCLA": (34.0722, -118.2437),  # Rose Bowl
            "Notre Dame": (41.6990, -86.2384),  # Notre Dame Stadium
            "Clemson": (34.6788, -82.8373),  # Memorial Stadium
            "Florida State": (30.4419, -84.2911),  # Doak Campbell Stadium
            "Miami": (25.6081, -80.4090),  # Hard Rock Stadium
            "Stanford": (37.4341, -122.1637),  # Stanford Stadium
            "Oregon": (44.0582, -123.0703),  # Autzen Stadium
            "Washington": (47.6500, -122.3031),  # Husky Stadium
            "Wisconsin": (43.0708, -89.4062),  # Camp Randall Stadium
            "Iowa": (41.6581, -91.5517),  # Kinnick Stadium
            "Nebraska": (40.8207, -96.7056),  # Memorial Stadium
            "Michigan State": (42.7258, -84.4806),  # Spartan Stadium
            "Minnesota": (44.9778, -93.2225),  # Huntington Bank Stadium
            "Northwestern": (42.0650, -87.6967),  # Ryan Field
            "Purdue": (40.4259, -86.9142),  # Ross-Ade Stadium
            "Indiana": (39.1817, -86.5264),  # Memorial Stadium
            "Illinois": (40.1020, -88.2272),  # Memorial Stadium
            "Maryland": (38.9887, -76.9376),  # Maryland Stadium
            "Rutgers": (40.5008, -74.4474),  # SHI Stadium
        }

        return stadium_locations.get(team)

    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make API request to WeatherAPI.com."""
        url = f"{self.base_url}/{endpoint}"
        params["key"] = self.api_key

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"WeatherAPI request failed: {e}")
            raise

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

                lat, lon = home_location
                kickoff = game["kickoff_utc"]

                # Format date for WeatherAPI
                date_str = kickoff.strftime("%Y-%m-%d")

                # Fetch weather data for the game date
                params = {
                    "q": f"{lat},{lon}",
                    "dt": date_str,
                    "hour": kickoff.hour,
                }

                data = self._make_request("forecast.json", params)

                if "forecast" not in data or not data["forecast"]["forecastday"]:
                    logger.warning(
                        f"No weather forecast for {game['home']} on {date_str}"
                    )
                    continue

                # Get weather data for the specific hour
                forecast_day = data["forecast"]["forecastday"][0]
                hour_data = None

                for hour in forecast_day["hour"]:
                    if hour["time"].endswith(f"{kickoff.hour:02d}:00"):
                        hour_data = hour
                        break

                if not hour_data:
                    logger.warning(
                        f"No hourly data for {game['home']} at hour {kickoff.hour}"
                    )
                    continue

                weather_data.append(
                    {
                        "game_id": game["game_id"],
                        "kickoff_utc": kickoff,
                        "temp_c": hour_data["temp_c"],
                        "wind_mps": hour_data["wind_kph"] / 3.6,  # Convert km/h to m/s
                        "precip_mm": hour_data["precip_mm"],
                        "humidity": hour_data["humidity"],
                        "pressure": hour_data["pressure_mb"],
                        "cloud_cover": hour_data["cloud"],
                        "condition": hour_data["condition"]["text"],
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

        # Weather condition features
        features_df["is_clear"] = features_df["condition"].str.contains(
            "Clear|Sunny", case=False, na=False
        )
        features_df["is_cloudy"] = features_df["condition"].str.contains(
            "Cloud|Overcast", case=False, na=False
        )
        features_df["is_stormy"] = features_df["condition"].str.contains(
            "Storm|Thunder|Lightning", case=False, na=False
        )

        # Combined weather impact
        features_df["weather_impact"] = (
            features_df["is_extreme_temp"].astype(int)
            + features_df["is_windy"].astype(int)
            + features_df["is_rainy"].astype(int)
            + features_df["is_stormy"].astype(int)
        )

        return features_df
