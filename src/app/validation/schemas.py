"""Data validation schemas using Pydantic."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class GameKey(BaseModel):
    """Schema for game identification."""
    season: int = Field(..., description="CFB season year")
    week: int = Field(..., description="Week number in season")
    season_type: str = Field(default="regular", description="Season type (regular, postseason)")
    game_id: str = Field(..., description="Unique game identifier")
    kickoff_utc: datetime = Field(..., description="Game kickoff time in UTC")
    home: str = Field(..., description="Home team name")
    away: str = Field(..., description="Away team name")
    home_id: Optional[int] = Field(None, description="Home team ID")
    away_id: Optional[int] = Field(None, description="Away team ID")


class OddsRow(BaseModel):
    """Schema for odds data."""
    game_id: str = Field(..., description="Game identifier")
    provider: str = Field(..., description="Odds provider name")
    book: str = Field(..., description="Sportsbook name")
    market: str = Field(..., description="Market type (spread, total)")
    period: str = Field(..., description="Period (game, 1H, 1Q)")
    fetched_at: datetime = Field(..., description="When odds were fetched")
    home_price: int = Field(..., description="Home team American odds")
    away_price: int = Field(..., description="Away team American odds")
    home_handicap: Optional[float] = Field(None, description="Home team spread")
    total_points: Optional[float] = Field(None, description="Total points line")


class TeamStrength(BaseModel):
    """Schema for team strength metrics."""
    team_id: int = Field(..., description="Team identifier")
    date: datetime = Field(..., description="Date of metrics")
    epa_off: float = Field(..., description="Offensive EPA per play")
    epa_def: float = Field(..., description="Defensive EPA per play")
    success_rate_off: float = Field(..., description="Offensive success rate")
    success_rate_def: float = Field(..., description="Defensive success rate")
    pace: float = Field(..., description="Plays per game")
    explosiveness_off: float = Field(..., description="Offensive explosiveness")
    explosiveness_def: float = Field(..., description="Defensive explosiveness")
    finishing_drives_off: float = Field(..., description="Offensive finishing drives")
    finishing_drives_def: float = Field(..., description="Defensive finishing drives")


class WeatherRow(BaseModel):
    """Schema for weather data."""
    game_id: str = Field(..., description="Game identifier")
    kickoff_utc: datetime = Field(..., description="Game kickoff time")
    temp_c: Optional[float] = Field(None, description="Temperature in Celsius")
    wind_mps: Optional[float] = Field(None, description="Wind speed in m/s")
    precip_mm: Optional[float] = Field(None, description="Precipitation in mm")
    humidity: Optional[float] = Field(None, description="Humidity percentage")


class LabelATS(BaseModel):
    """Schema for ATS (Against The Spread) labels."""
    game_id: str = Field(..., description="Game identifier")
    period: str = Field(..., description="Period (game, 1H, 1Q)")
    spread_line: float = Field(..., description="Spread line")
    result_cover: bool = Field(..., description="Did the favorite cover?")


class LabelTotal(BaseModel):
    """Schema for totals labels."""
    game_id: str = Field(..., description="Game identifier")
    period: str = Field(..., description="Period (game, 1H, 1Q)")
    total_line: float = Field(..., description="Total points line")
    result_over: bool = Field(..., description="Did the total go over?")


class FeatureRow(BaseModel):
    """Schema for engineered features."""
    game_id: str = Field(..., description="Game identifier")
    team: str = Field(..., description="Team name")
    opponent: str = Field(..., description="Opponent name")
    is_home: bool = Field(..., description="Is this team home?")
    date: datetime = Field(..., description="Game date")
    
    # Rolling features
    epa_off_3g: float = Field(..., description="3-game rolling offensive EPA")
    epa_def_3g: float = Field(..., description="3-game rolling defensive EPA")
    epa_off_5g: float = Field(..., description="5-game rolling offensive EPA")
    epa_def_5g: float = Field(..., description="5-game rolling defensive EPA")
    epa_off_10g: float = Field(..., description="10-game rolling offensive EPA")
    epa_def_10g: float = Field(..., description="10-game rolling defensive EPA")
    
    success_rate_off_3g: float = Field(..., description="3-game rolling offensive success rate")
    success_rate_def_3g: float = Field(..., description="3-game rolling defensive success rate")
    success_rate_off_5g: float = Field(..., description="5-game rolling offensive success rate")
    success_rate_def_5g: float = Field(..., description="5-game rolling defensive success rate")
    
    pace_3g: float = Field(..., description="3-game rolling pace")
    pace_5g: float = Field(..., description="5-game rolling pace")
    
    # Odds features
    line_movement: Optional[float] = Field(None, description="Line movement from open to close")
    movement_velocity: Optional[float] = Field(None, description="Rate of line movement")
    cross_book_delta: Optional[float] = Field(None, description="Spread difference across books")
    
    # Weather features
    temp_c: Optional[float] = Field(None, description="Temperature in Celsius")
    wind_mps: Optional[float] = Field(None, description="Wind speed in m/s")
    precip_mm: Optional[float] = Field(None, description="Precipitation in mm")
    
    # Rest and travel
    days_rest: int = Field(..., description="Days since last game")
    travel_distance: Optional[float] = Field(None, description="Travel distance in miles")
