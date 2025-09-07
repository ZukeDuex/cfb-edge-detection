"""Betting utility functions."""

from typing import Tuple


def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    return -odds / (-odds + 100)


def american_to_decimal(odds: int) -> float:
    """Convert American odds to decimal odds."""
    if odds > 0:
        return (odds / 100) + 1
    return (100 / -odds) + 1


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American odds."""
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    return int(-100 / (decimal_odds - 1))


def remove_vig(p_home: float, p_away: float) -> Tuple[float, float]:
    """Remove vig from implied probabilities."""
    total_prob = p_home + p_away
    return p_home / total_prob, p_away / total_prob


def kelly_fraction(p: float, decimal_odds: float, b: float = None) -> float:
    """Calculate Kelly fraction for optimal bet sizing."""
    if b is None:
        b = decimal_odds - 1
    return max(0.0, (p * b - (1 - p)) / b)


def calculate_edge(model_prob: float, market_prob: float) -> float:
    """Calculate betting edge (model prob - market prob)."""
    return model_prob - market_prob


def calculate_ev(model_prob: float, decimal_odds: float) -> float:
    """Calculate expected value of a bet."""
    return (model_prob * decimal_odds) - 1
