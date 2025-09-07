"""The Odds API client for fetching betting odds."""

import logging
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests

from ..config import settings
from ..logging import get_logger

logger = get_logger(__name__)


class OddsAPIClient:
    """Client for The Odds API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Odds API client."""
        self.api_key = api_key or settings.odds_api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "CFB-Edge-Platform/1.0"})

    def _make_request(self, endpoint: str, params: Dict) -> Dict:
        """Make API request with rate limiting and retries."""
        url = f"{self.base_url}/{endpoint}"
        params["apiKey"] = self.api_key

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()

                # Rate limiting
                if "x-requests-remaining" in response.headers:
                    remaining = int(response.headers["x-requests-remaining"])
                    if remaining < 10:
                        logger.warning(f"Low API requests remaining: {remaining}")
                        time.sleep(1)

                return response.json()

            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)  # Exponential backoff
                else:
                    raise

    def fetch_odds(
        self,
        season: int,
        week: Optional[int] = None,
        markets: List[str] = None,
        periods: List[str] = None,
        book: str = "fanduel",
    ) -> pd.DataFrame:
        """Fetch odds for CFB games.

        Week filtering uses the ISO calendar week derived from each game's
        ``commence_time``.
        """
        if markets is None:
            markets = ["spreads", "totals"]
        if periods is None:
            periods = ["game", "1H", "1Q"]

        # Map periods to API format
        period_map = {"game": "full_game", "1H": "first_half", "1Q": "first_quarter"}

        odds_data = []

        for market in markets:
            for period in periods:
                try:
                    params = {
                        "markets": market,
                        "regions": "us",
                        "bookmakers": book,
                        "oddsFormat": "american",
                        "dateFormat": "iso",
                    }

                    # Add period-specific parameters
                    if period != "game":
                        params["markets"] = f"{market}_{period_map[period]}"

                    data = self._make_request(
                        "sports/americanfootball_ncaaf/odds", params
                    )

                    for game in data:
                        game_id = str(game["id"])
                        commence_time = datetime.fromisoformat(
                            game["commence_time"].replace("Z", "+00:00")
                        )

                        # Filter by season/week using commence_time
                        if commence_time.year != season:
                            continue
                        game_week = commence_time.isocalendar().week
                        if week is not None and game_week != week:
                            continue

                        for bookmaker in game.get("bookmakers", []):
                            if bookmaker["key"] == book:
                                for market_data in bookmaker.get("markets", []):
                                    for outcome in market_data.get("outcomes", []):
                                        odds_data.append(
                                            {
                                                "game_id": game_id,
                                                "season": season,
                                                "week": game_week,
                                                "season_type": "regular",
                                                "period": period,
                                                "market": market,
                                                "book": book,
                                                "fetched_at": datetime.now(),
                                                "home_team": game["home_team"],
                                                "away_team": game["away_team"],
                                                "commence_time": commence_time,
                                                "outcome_name": outcome["name"],
                                                "price": outcome["price"],
                                                "point": outcome.get("point"),
                                            }
                                        )

                    logger.info(f"Fetched {market} odds for {period} period")

                except Exception as e:
                    logger.error(f"Error fetching {market} odds for {period}: {e}")
                    continue

        if not odds_data:
            logger.warning("No odds data fetched")
            return pd.DataFrame()

        df = pd.DataFrame(odds_data)

        # Normalize the data structure
        normalized_data = []
        for _, row in df.iterrows():
            if row["market"] == "spreads":
                normalized_data.append(
                    {
                        "game_id": row["game_id"],
                        "season": row["season"],
                        "week": row["week"],
                        "season_type": row["season_type"],
                        "period": row["period"],
                        "market": "spread",
                        "book": row["book"],
                        "fetched_at": row["fetched_at"],
                        "home_team": row["home_team"],
                        "away_team": row["away_team"],
                        "home_price": row["price"]
                        if "home" in row["outcome_name"].lower()
                        else None,
                        "away_price": row["price"]
                        if "away" in row["outcome_name"].lower()
                        else None,
                        "home_handicap": row["point"]
                        if "home" in row["outcome_name"].lower()
                        else None,
                        "total_points": None,
                    }
                )
            elif row["market"] == "totals":
                normalized_data.append(
                    {
                        "game_id": row["game_id"],
                        "season": row["season"],
                        "week": row["week"],
                        "season_type": row["season_type"],
                        "period": row["period"],
                        "market": "total",
                        "book": row["book"],
                        "fetched_at": row["fetched_at"],
                        "home_team": row["home_team"],
                        "away_team": row["away_team"],
                        "home_price": row["price"]
                        if "over" in row["outcome_name"].lower()
                        else None,
                        "away_price": row["price"]
                        if "under" in row["outcome_name"].lower()
                        else None,
                        "home_handicap": None,
                        "total_points": row["point"],
                    }
                )

        result_df = pd.DataFrame(normalized_data)
        logger.info(f"Normalized {len(result_df)} odds records")
        return result_df
