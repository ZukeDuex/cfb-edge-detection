"""College Football Data API client."""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import cfbd
import pandas as pd

from ..config import settings
from ..logging import get_logger

logger = get_logger(__name__)


class CFBDClient:
    """Client for College Football Data API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize CFBD client."""
        self.api_key = api_key or settings.cfbd_api_key
        cfbd.Configuration().api_key['Authorization'] = self.api_key
        cfbd.Configuration().api_key_prefix['Authorization'] = 'Bearer'
        
        self.games_api = cfbd.GamesApi()
        self.teams_api = cfbd.TeamsApi()
        self.stats_api = cfbd.StatsApi()
        self.ratings_api = cfbd.RatingsApi()
    
    def fetch_games(self, season: int, week: Optional[int] = None) -> pd.DataFrame:
        """Fetch games for a season and optionally specific week."""
        try:
            if week:
                games = self.games_api.get_games(year=season, week=week)
            else:
                games = self.games_api.get_games(year=season)
            
            games_data = []
            for game in games:
                games_data.append({
                    'game_id': str(game.id),
                    'season': season,
                    'week': game.week,
                    'season_type': game.season_type,
                    'kickoff_utc': datetime.fromisoformat(game.start_date.replace('Z', '+00:00')),
                    'home': game.home_team,
                    'away': game.away_team,
                    'home_id': game.home_id,
                    'away_id': game.away_id,
                    'home_score': game.home_points,
                    'away_score': game.away_points,
                    'completed': game.completed,
                    'neutral_site': game.neutral_site,
                    'conference_game': game.conference_game,
                })
            
            df = pd.DataFrame(games_data)
            logger.info(f"Fetched {len(df)} games for season {season}" + (f", week {week}" if week else ""))
            return df
            
        except Exception as e:
            logger.error(f"Error fetching games: {e}")
            raise
    
    def fetch_team_stats(self, season: int, team: Optional[str] = None) -> pd.DataFrame:
        """Fetch team statistics for a season."""
        try:
            stats = self.stats_api.get_team_season_stats(year=season, team=team)
            
            stats_data = []
            for team_stat in stats:
                stats_data.append({
                    'team': team_stat.team,
                    'season': season,
                    'games': team_stat.games,
                    'plays': team_stat.plays,
                    'offensive_plays': team_stat.offensive_plays,
                    'defensive_plays': team_stat.defensive_plays,
                    'offensive_yards': team_stat.offensive_yards,
                    'defensive_yards': team_stat.defensive_yards,
                    'offensive_yards_per_play': team_stat.offensive_yards_per_play,
                    'defensive_yards_per_play': team_stat.defensive_yards_per_play,
                    'offensive_points': team_stat.offensive_points,
                    'defensive_points': team_stat.defensive_points,
                    'offensive_points_per_game': team_stat.offensive_points_per_game,
                    'defensive_points_per_game': team_stat.defensive_points_per_game,
                })
            
            df = pd.DataFrame(stats_data)
            logger.info(f"Fetched stats for {len(df)} teams in season {season}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching team stats: {e}")
            raise
    
    def fetch_ratings(self, season: int) -> pd.DataFrame:
        """Fetch SP+ style ratings for a season."""
        try:
            ratings = self.ratings_api.get_sp_ratings(year=season)
            
            ratings_data = []
            for rating in ratings:
                ratings_data.append({
                    'team': rating.team,
                    'season': season,
                    'sp_rating': rating.rating,
                    'sp_offense': rating.offense,
                    'sp_defense': rating.defense,
                    'sp_special_teams': rating.special_teams,
                })
            
            df = pd.DataFrame(ratings_data)
            logger.info(f"Fetched SP+ ratings for {len(df)} teams in season {season}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching ratings: {e}")
            raise
    
    def fetch_teams(self) -> pd.DataFrame:
        """Fetch all teams."""
        try:
            teams = self.teams_api.get_teams()
            
            teams_data = []
            for team in teams:
                teams_data.append({
                    'team_id': team.id,
                    'team': team.name,
                    'mascot': team.mascot,
                    'conference': team.conference,
                    'division': team.division,
                    'color': team.color,
                    'alt_color': team.alt_color,
                    'logo': team.logo,
                })
            
            df = pd.DataFrame(teams_data)
            logger.info(f"Fetched {len(df)} teams")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching teams: {e}")
            raise
