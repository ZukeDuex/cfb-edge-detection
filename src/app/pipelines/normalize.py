"""Data normalization pipeline for silver layer."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..config import settings
from ..logging import get_logger
from ..validation.schemas import GameKey, OddsRow, WeatherRow

logger = get_logger(__name__)


class NormalizePipeline:
    """Pipeline for normalizing and cleaning raw data."""
    
    def __init__(self):
        """Initialize normalization pipeline."""
        self.data_dir = settings.data_dir
        self.bronze_dir = self.data_dir / "bronze"
        self.silver_dir = self.data_dir / "silver"
        self.silver_dir.mkdir(exist_ok=True)
    
    def normalize_games(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize games data."""
        logger.info(f"Normalizing {len(games_df)} games")
        
        # Clean and standardize data
        normalized = games_df.copy()
        
        # Ensure required columns exist
        required_cols = ['game_id', 'season', 'week', 'season_type', 'kickoff_utc', 'home', 'away']
        missing_cols = [col for col in required_cols if col not in normalized.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert data types
        normalized['game_id'] = normalized['game_id'].astype(str)
        normalized['season'] = normalized['season'].astype(int)
        normalized['week'] = normalized['week'].astype(int)
        normalized['season_type'] = normalized['season_type'].astype(str)
        
        # Ensure kickoff_utc is datetime
        if not pd.api.types.is_datetime64_any_dtype(normalized['kickoff_utc']):
            normalized['kickoff_utc'] = pd.to_datetime(normalized['kickoff_utc'])
        
        # Clean team names
        normalized['home'] = normalized['home'].str.strip()
        normalized['away'] = normalized['away'].str.strip()
        
        # Add normalized columns
        normalized['home_id'] = normalized.get('home_id', None)
        normalized['away_id'] = normalized.get('away_id', None)
        
        # Validate using schema
        try:
            for _, row in normalized.iterrows():
                GameKey(**row.to_dict())
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            raise
        
        logger.info(f"Normalized {len(normalized)} games successfully")
        return normalized
    
    def normalize_odds(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize odds data."""
        logger.info(f"Normalizing {len(odds_df)} odds records")
        
        if odds_df.empty:
            logger.warning("No odds data to normalize")
            return pd.DataFrame()
        
        normalized = odds_df.copy()
        
        # Ensure required columns exist
        required_cols = ['game_id', 'provider', 'book', 'market', 'period', 'fetched_at']
        missing_cols = [col for col in required_cols if col not in normalized.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert data types
        normalized['game_id'] = normalized['game_id'].astype(str)
        normalized['provider'] = normalized['provider'].astype(str)
        normalized['book'] = normalized['book'].astype(str)
        normalized['market'] = normalized['market'].astype(str)
        normalized['period'] = normalized['period'].astype(str)
        
        # Ensure fetched_at is datetime
        if not pd.api.types.is_datetime64_any_dtype(normalized['fetched_at']):
            normalized['fetched_at'] = pd.to_datetime(normalized['fetched_at'])
        
        # Clean numeric columns
        numeric_cols = ['home_price', 'away_price', 'home_handicap', 'total_points']
        for col in numeric_cols:
            if col in normalized.columns:
                normalized[col] = pd.to_numeric(normalized[col], errors='coerce')
        
        # Validate using schema
        try:
            for _, row in normalized.iterrows():
                OddsRow(**row.to_dict())
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            raise
        
        logger.info(f"Normalized {len(normalized)} odds records successfully")
        return normalized
    
    def normalize_weather(self, weather_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize weather data."""
        logger.info(f"Normalizing {len(weather_df)} weather records")
        
        if weather_df.empty:
            logger.warning("No weather data to normalize")
            return pd.DataFrame()
        
        normalized = weather_df.copy()
        
        # Ensure required columns exist
        required_cols = ['game_id', 'kickoff_utc']
        missing_cols = [col for col in required_cols if col not in normalized.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert data types
        normalized['game_id'] = normalized['game_id'].astype(str)
        
        # Ensure kickoff_utc is datetime
        if not pd.api.types.is_datetime64_any_dtype(normalized['kickoff_utc']):
            normalized['kickoff_utc'] = pd.to_datetime(normalized['kickoff_utc'])
        
        # Clean numeric columns
        numeric_cols = ['temp_c', 'wind_mps', 'precip_mm', 'humidity']
        for col in numeric_cols:
            if col in normalized.columns:
                normalized[col] = pd.to_numeric(normalized[col], errors='coerce')
        
        # Validate using schema
        try:
            for _, row in normalized.iterrows():
                WeatherRow(**row.to_dict())
        except Exception as e:
            logger.error(f"Schema validation failed: {e}")
            raise
        
        logger.info(f"Normalized {len(normalized)} weather records successfully")
        return normalized
    
    def join_data(self, games_df: pd.DataFrame, odds_df: pd.DataFrame, 
                  weather_df: pd.DataFrame) -> pd.DataFrame:
        """Join normalized data sources."""
        logger.info("Joining normalized data sources")
        
        # Start with games as base
        joined = games_df.copy()
        
        # Join odds data
        if not odds_df.empty:
            # Group odds by game_id and period to handle multiple markets
            odds_grouped = odds_df.groupby(['game_id', 'period', 'market']).agg({
                'home_price': 'first',
                'away_price': 'first',
                'home_handicap': 'first',
                'total_points': 'first',
                'fetched_at': 'first',
                'book': 'first'
            }).reset_index()
            
            # Pivot to get separate columns for each market/period combination
            odds_pivot = odds_grouped.pivot_table(
                index='game_id',
                columns=['period', 'market'],
                values=['home_price', 'away_price', 'home_handicap', 'total_points'],
                aggfunc='first'
            )
            
            # Flatten column names
            odds_pivot.columns = ['_'.join(col).strip() for col in odds_pivot.columns]
            odds_pivot = odds_pivot.reset_index()
            
            joined = joined.merge(odds_pivot, on='game_id', how='left')
            logger.info(f"Joined odds data: {len(joined)} records")
        
        # Join weather data
        if not weather_df.empty:
            weather_clean = weather_df[['game_id', 'temp_c', 'wind_mps', 'precip_mm', 'humidity']].copy()
            joined = joined.merge(weather_clean, on='game_id', how='left')
            logger.info(f"Joined weather data: {len(joined)} records")
        
        logger.info(f"Final joined dataset: {len(joined)} records")
        return joined
    
    def save_silver_data(self, data_df: pd.DataFrame, filename: str) -> None:
        """Save normalized data to silver layer."""
        filepath = self.silver_dir / filename
        data_df.to_parquet(filepath, index=False)
        logger.info(f"Saved silver data to {filepath}")
    
    def run_normalization(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """Run full normalization pipeline."""
        logger.info(f"Starting normalization for season {season}" + (f", week {week}" if week else ""))
        
        # Load bronze data
        games_file = f"games_{season}" + (f"_week_{week}" if week else "") + ".parquet"
        odds_file = f"odds_{season}" + (f"_week_{week}" if week else "") + ".parquet"
        
        games_df = pd.read_parquet(self.bronze_dir / games_file)
        odds_df = pd.read_parquet(self.bronze_dir / odds_file) if (self.bronze_dir / odds_file).exists() else pd.DataFrame()
        
        # Try to load weather data
        weather_files = list(self.bronze_dir.glob("weather_*.parquet"))
        weather_df = pd.read_parquet(weather_files[0]) if weather_files else pd.DataFrame()
        
        # Normalize each dataset
        normalized_games = self.normalize_games(games_df)
        normalized_odds = self.normalize_odds(odds_df)
        normalized_weather = self.normalize_weather(weather_df)
        
        # Join data
        joined_data = self.join_data(normalized_games, normalized_odds, normalized_weather)
        
        # Save to silver layer
        silver_filename = f"normalized_{season}" + (f"_week_{week}" if week else "") + ".parquet"
        self.save_silver_data(joined_data, silver_filename)
        
        logger.info("Normalization completed successfully")
        return joined_data
