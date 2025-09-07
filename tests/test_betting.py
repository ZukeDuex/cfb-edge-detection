"""Tests for betting utilities."""

import pytest
from app.utils.betting import (
    american_to_prob,
    american_to_decimal,
    decimal_to_american,
    remove_vig,
    kelly_fraction,
    calculate_edge,
    calculate_ev,
)


def test_american_to_prob():
    """Test American odds to probability conversion."""
    # Positive odds
    assert american_to_prob(110) == pytest.approx(100 / 210, rel=1e-3)
    assert american_to_prob(200) == pytest.approx(100 / 300, rel=1e-3)
    
    # Negative odds
    assert american_to_prob(-110) == pytest.approx(110 / 210, rel=1e-3)
    assert american_to_prob(-200) == pytest.approx(200 / 300, rel=1e-3)


def test_american_to_decimal():
    """Test American odds to decimal odds conversion."""
    # Positive odds
    assert american_to_decimal(110) == pytest.approx(2.1, rel=1e-3)
    assert american_to_decimal(200) == pytest.approx(3.0, rel=1e-3)
    
    # Negative odds
    assert american_to_decimal(-110) == pytest.approx(1.909, rel=1e-3)
    assert american_to_decimal(-200) == pytest.approx(1.5, rel=1e-3)


def test_decimal_to_american():
    """Test decimal odds to American odds conversion."""
    assert decimal_to_american(2.1) == 110
    assert decimal_to_american(3.0) == 200
    assert decimal_to_american(1.909) == -110
    assert decimal_to_american(1.5) == -200


def test_remove_vig():
    """Test vig removal."""
    p_home, p_away = remove_vig(0.55, 0.50)
    assert p_home + p_away == pytest.approx(1.0, rel=1e-3)
    assert p_home > 0.5
    assert p_away < 0.5


def test_kelly_fraction():
    """Test Kelly fraction calculation."""
    # Positive edge
    kelly = kelly_fraction(0.6, 2.0)  # 60% prob, 2.0 decimal odds
    assert kelly > 0
    
    # No edge
    kelly = kelly_fraction(0.5, 2.0)  # 50% prob, 2.0 decimal odds
    assert kelly == 0
    
    # Negative edge
    kelly = kelly_fraction(0.4, 2.0)  # 40% prob, 2.0 decimal odds
    assert kelly == 0


def test_calculate_edge():
    """Test edge calculation."""
    edge = calculate_edge(0.6, 0.5)
    assert edge == 0.1
    
    edge = calculate_edge(0.4, 0.5)
    assert edge == -0.1


def test_calculate_ev():
    """Test expected value calculation."""
    ev = calculate_ev(0.6, 2.0)  # 60% prob, 2.0 decimal odds
    assert ev == pytest.approx(0.2, rel=1e-3)
    
    ev = calculate_ev(0.5, 2.0)  # 50% prob, 2.0 decimal odds
    assert ev == 0.0
