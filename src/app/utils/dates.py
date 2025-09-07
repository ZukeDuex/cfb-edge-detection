"""Utility functions for date handling."""

from datetime import datetime, timezone
from typing import Union

import pandas as pd


def parse_date(date_str: Union[str, datetime]) -> datetime:
    """Parse a date string or datetime object to UTC datetime."""
    if isinstance(date_str, datetime):
        if date_str.tzinfo is None:
            return date_str.replace(tzinfo=timezone.utc)
        return date_str.astimezone(timezone.utc)
    
    if isinstance(date_str, str):
        # Try common formats
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue
        
        # Try pandas parsing as fallback
        try:
            return pd.to_datetime(date_str, utc=True).to_pydatetime()
        except Exception:
            pass
    
    raise ValueError(f"Unable to parse date: {date_str}")


def get_season_week(date: datetime) -> tuple[int, int]:
    """Get season and week for a given date."""
    year = date.year
    month = date.month
    
    # CFB season typically runs August to January
    if month >= 8:
        season = year
    else:
        season = year - 1
    
    # Rough week calculation (this would need refinement based on actual CFB schedule)
    if month == 8:
        week = 1
    elif month == 9:
        week = min(4, (date.day - 1) // 7 + 1)
    elif month == 10:
        week = min(8, ((date.day - 1) // 7 + 1) + 4)
    elif month == 11:
        week = min(12, ((date.day - 1) // 7 + 1) + 8)
    elif month == 12:
        week = min(15, ((date.day - 1) // 7 + 1) + 12)
    else:
        week = 1
    
    return season, week


def is_game_time(date: datetime) -> bool:
    """Check if a date/time falls during typical CFB game hours."""
    hour = date.hour
    # CFB games typically run Thursday-Saturday, 6PM-11PM local time
    return 18 <= hour <= 23
