"""
Shared utilities for CFB betting analysis
Consolidates common functions used across multiple scripts
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
import os

logger = logging.getLogger(__name__)

class CFBDataUtils:
    """Common utilities for CFB data processing"""
    
    @staticmethod
    def load_data_robust(file_path: str, **kwargs) -> pd.DataFrame:
        """Load data with robust path handling"""
        if not os.path.isabs(file_path):
            possible_paths = [
                file_path,
                f"data/{file_path}",
                f"analysis/data/{file_path}",
                f"../data/{file_path}",
                f"../analysis/data/{file_path}"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    file_path = path
                    break
            else:
                raise FileNotFoundError(f"Could not find data file: {file_path}")
        
        logger.info(f"Loading data from: {file_path}")
        return pd.read_csv(file_path, **kwargs)
    
    @staticmethod
    def get_standard_feature_columns() -> List[str]:
        """Get standard feature columns used across models"""
        return [
            'season', 'week', 'is_home_game', 'is_conference_game', 'is_neutral_site', 
            'attendance', 'homePregameElo', 'awayPregameElo', 'elo_difference',
            'homePostgameWinProbability', 'awayPostgameWinProbability', 'excitementIndex',
            'is_early_season', 'is_mid_season', 'is_late_season', 'is_postseason'
        ]
    
    @staticmethod
    def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality and return issues"""
        issues = {}
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        if missing_counts.any():
            issues['missing_values'] = missing_counts[missing_counts > 0].to_dict()
        
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues['duplicates'] = duplicates
        
        return issues

class BettingStrategyConfig:
    """Configurable betting strategy parameters"""
    
    def __init__(self, config_dict: Dict[str, Any] = None):
        self.min_edge_points = 3.0
        self.bet_amount = 100
        self.min_confidence = 0.6
        self.max_bet_percentage = 0.1
        self.max_daily_bets = 5
        
        if config_dict:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'min_edge_points': self.min_edge_points,
            'bet_amount': self.bet_amount,
            'min_confidence': self.min_confidence,
            'max_bet_percentage': self.max_bet_percentage,
            'max_daily_bets': self.max_daily_bets
        }
    
    @classmethod
    def load_config(cls, file_path: str):
        """Load configuration from file"""
        import json
        with open(file_path, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
