"""Data ingestion pipeline for bronze layer."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from ..config import settings
from ..logging import get_logger
from ..providers.cfbd_client import CFBDClient
from ..providers.odds_theoddsapi import OddsAPIClient
from ..providers.weather_meteostat import WeatherProvider

logger = get_logger(__name__)


class IngestPipeline:
    """Pipeline for ingesting raw data from various sources."""
    
    def __init__(self):
        """Initialize ingestion pipeline."""
        self.cfbd_client = CFBDClient()
        self.odds_client = OddsAPIClient()
        self.weather_provider = WeatherProvider()
        self.data_dir = settings.data_dir / "bronze"
    
    def ingest_games(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """Ingest games data from CFBD."""
        logger.info(f"Ingesting games for season {season}" + (f", week {week}" if week else ""))
        
        games_df = self.cfbd_client.fetch_games(season, week)
        
        # Save to bronze layer
        filename = f"games_{season}" + (f"_week_{week}" if week else "") + ".parquet"
        filepath = self.data_dir / filename
        games_df.to_parquet(filepath, index=False)
        
        logger.info(f"Saved {len(games_df)} games to {filepath}")
        return games_df
    
    def ingest_odds(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """Ingest odds data from The Odds API."""
        logger.info(f"Ingesting odds for season {season}" + (f", week {week}" if week else ""))
        
        odds_df = self.odds_client.fetch_odds(season, week)
        
        if not odds_df.empty:
            # Save to bronze layer
            filename = f"odds_{season}" + (f"_week_{week}" if week else "") + ".parquet"
            filepath = self.data_dir / filename
            odds_df.to_parquet(filepath, index=False)
            
            logger.info(f"Saved {len(odds_df)} odds records to {filepath}")
        else:
            logger.warning("No odds data to save")
        
        return odds_df
    
    def ingest_weather(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Ingest weather data for games."""
        logger.info(f"Ingesting weather data for {len(games_df)} games")
        
        weather_df = self.weather_provider.fetch_weather(games_df)
        
        if not weather_df.empty:
            # Save to bronze layer
            filename = f"weather_{datetime.now().strftime('%Y%m%d')}.parquet"
            filepath = self.data_dir / filename
            weather_df.to_parquet(filepath, index=False)
            
            logger.info(f"Saved {len(weather_df)} weather records to {filepath}")
        else:
            logger.warning("No weather data to save")
        
        return weather_df
    
    def ingest_team_stats(self, season: int) -> pd.DataFrame:
        """Ingest team statistics from CFBD."""
        logger.info(f"Ingesting team stats for season {season}")
        
        stats_df = self.cfbd_client.fetch_team_stats(season)
        
        # Save to bronze layer
        filename = f"team_stats_{season}.parquet"
        filepath = self.data_dir / filename
        stats_df.to_parquet(filepath, index=False)
        
        logger.info(f"Saved {len(stats_df)} team stats records to {filepath}")
        return stats_df
    
    def ingest_ratings(self, season: int) -> pd.DataFrame:
        """Ingest team ratings from CFBD."""
        logger.info(f"Ingesting ratings for season {season}")
        
        ratings_df = self.cfbd_client.fetch_ratings(season)
        
        # Save to bronze layer
        filename = f"ratings_{season}.parquet"
        filepath = self.data_dir / filename
        ratings_df.to_parquet(filepath, index=False)
        
        logger.info(f"Saved {len(ratings_df)} ratings records to {filepath}")
        return ratings_df
    
    def run_full_ingest(self, season: int, week: Optional[int] = None, 
                        include_weather: bool = True) -> dict:
        """Run full ingestion pipeline."""
        logger.info(f"Starting full ingest for season {season}" + (f", week {week}" if week else ""))
        
        results = {}
        
        try:
            # Ingest games
            games_df = self.ingest_games(season, week)
            results['games'] = games_df
            
            # Ingest odds
            odds_df = self.ingest_odds(season, week)
            results['odds'] = odds_df
            
            # Ingest weather (if requested and games available)
            if include_weather and not games_df.empty:
                weather_df = self.ingest_weather(games_df)
                results['weather'] = weather_df
            
            # Ingest team stats and ratings (season-level only)
            if week is None:
                stats_df = self.ingest_team_stats(season)
                results['team_stats'] = stats_df
                
                ratings_df = self.ingest_ratings(season)
                results['ratings'] = ratings_df
            
            logger.info("Full ingest completed successfully")
            
        except Exception as e:
            logger.error(f"Error during full ingest: {e}")
            raise
        
        return results
