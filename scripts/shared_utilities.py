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
