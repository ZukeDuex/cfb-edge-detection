"""Tests for data validation schemas."""

import pytest
from datetime import datetime
from app.validation.schemas import (
    GameKey,
    OddsRow,
    WeatherRow,
    LabelATS,
    LabelTotal,
    FeatureRow,
)


def test_game_key_schema():
    """Test GameKey schema validation."""
    valid_data = {
        "season": 2023,
        "week": 1,
        "season_type": "regular",
        "game_id": "game_123",
        "kickoff_utc": datetime.now(),
        "home": "Alabama",
        "away": "Auburn",
        "home_id": 1,
        "away_id": 2,
    }
    
    game_key = GameKey(**valid_data)
    assert game_key.season == 2023
    assert game_key.home == "Alabama"


def test_odds_row_schema():
    """Test OddsRow schema validation."""
    valid_data = {
        "game_id": "game_123",
        "provider": "theodds",
        "book": "fanduel",
        "market": "spread",
        "period": "game",
        "fetched_at": datetime.now(),
        "home_price": -110,
        "away_price": -110,
        "home_handicap": -7.5,
        "total_points": None,
    }
    
    odds_row = OddsRow(**valid_data)
    assert odds_row.game_id == "game_123"
    assert odds_row.home_price == -110


def test_weather_row_schema():
    """Test WeatherRow schema validation."""
    valid_data = {
        "game_id": "game_123",
        "kickoff_utc": datetime.now(),
        "temp_c": 20.0,
        "wind_mps": 5.0,
        "precip_mm": 0.0,
        "humidity": 60.0,
    }
    
    weather_row = WeatherRow(**valid_data)
    assert weather_row.temp_c == 20.0
    assert weather_row.wind_mps == 5.0


def test_label_ats_schema():
    """Test LabelATS schema validation."""
    valid_data = {
        "game_id": "game_123",
        "period": "game",
        "spread_line": -7.5,
        "result_cover": True,
    }
    
    label_ats = LabelATS(**valid_data)
    assert label_ats.game_id == "game_123"
    assert label_ats.result_cover is True


def test_label_total_schema():
    """Test LabelTotal schema validation."""
    valid_data = {
        "game_id": "game_123",
        "period": "game",
        "total_line": 55.5,
        "result_over": False,
    }
    
    label_total = LabelTotal(**valid_data)
    assert label_total.game_id == "game_123"
    assert label_total.result_over is False


def test_feature_row_schema():
    """Test FeatureRow schema validation."""
    valid_data = {
        "game_id": "game_123",
        "team": "Alabama",
        "opponent": "Auburn",
        "is_home": True,
        "date": datetime.now(),
        "epa_off_3g": 0.1,
        "epa_def_3g": -0.1,
        "epa_off_5g": 0.15,
        "epa_def_5g": -0.15,
        "epa_off_10g": 0.2,
        "epa_def_10g": -0.2,
        "success_rate_off_3g": 0.6,
        "success_rate_def_3g": 0.4,
        "success_rate_off_5g": 0.65,
        "success_rate_def_5g": 0.35,
        "pace_3g": 70.0,
        "pace_5g": 72.0,
        "line_movement": 0.5,
        "movement_velocity": 0.1,
        "cross_book_delta": 0.0,
        "temp_c": 20.0,
        "wind_mps": 5.0,
        "precip_mm": 0.0,
        "days_rest": 7,
        "travel_distance": 0.0,
    }
    
    feature_row = FeatureRow(**valid_data)
    assert feature_row.game_id == "game_123"
    assert feature_row.team == "Alabama"
