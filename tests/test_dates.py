"""Tests for date utilities."""

import pytest
from datetime import datetime, timezone
from app.utils.dates import parse_date, get_season_week, is_game_time


def test_parse_date_string():
    """Test parsing date strings."""
    # Test various formats
    date_str = "2023-09-15"
    parsed = parse_date(date_str)
    assert isinstance(parsed, datetime)
    assert parsed.tzinfo == timezone.utc
    
    date_str = "2023-09-15T18:30:00"
    parsed = parse_date(date_str)
    assert parsed.hour == 18
    assert parsed.minute == 30


def test_parse_date_datetime():
    """Test parsing datetime objects."""
    dt = datetime(2023, 9, 15, 18, 30)
    parsed = parse_date(dt)
    assert parsed.tzinfo == timezone.utc
    
    dt_with_tz = datetime(2023, 9, 15, 18, 30, tzinfo=timezone.utc)
    parsed = parse_date(dt_with_tz)
    assert parsed == dt_with_tz


def test_get_season_week():
    """Test season and week calculation."""
    # August game (start of season)
    date = datetime(2023, 8, 26)
    season, week = get_season_week(date)
    assert season == 2023
    assert week == 1
    
    # September game
    date = datetime(2023, 9, 15)
    season, week = get_season_week(date)
    assert season == 2023
    assert week >= 1


def test_is_game_time():
    """Test game time detection."""
    # Evening game time
    evening_game = datetime(2023, 9, 15, 19, 30)
    assert is_game_time(evening_game) is True
    
    # Morning time
    morning = datetime(2023, 9, 15, 10, 30)
    assert is_game_time(morning) is False
    
    # Late night
    late_night = datetime(2023, 9, 15, 23, 30)
    assert is_game_time(late_night) is True
