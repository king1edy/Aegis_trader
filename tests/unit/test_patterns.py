"""
Unit Tests — Pattern Recognizer
================================
Tests for: src/strategies/patterns.py

Covers: bullish/bearish engulfing, hammer, shooting star, pin bar, scan.
"""

import pandas as pd
import pytest

from src.strategies.patterns import PatternRecognizer

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Bullish Engulfing
# ---------------------------------------------------------------------------

class TestBullishEngulfing:
    """Tests for PatternRecognizer.is_bullish_engulfing"""

    def test_true_positive(self):
        """A bearish candle followed by a larger bullish candle should match."""
        previous = pd.Series({"open": 2050, "high": 2055, "low": 2040, "close": 2042})
        current = pd.Series({"open": 2041, "high": 2060, "low": 2039, "close": 2055})
        assert PatternRecognizer.is_bullish_engulfing(current, previous) is True

    def test_false_when_both_bullish(self):
        """Two bullish candles should NOT form a bullish engulfing."""
        previous = pd.Series({"open": 2040, "high": 2055, "low": 2038, "close": 2050})
        current = pd.Series({"open": 2045, "high": 2060, "low": 2044, "close": 2058})
        assert PatternRecognizer.is_bullish_engulfing(current, previous) is False

    # TODO: test_false_when_no_engulfing — current body smaller than previous body
    # TODO: test_exact_boundary — current.open == previous.close (edge case)


# ---------------------------------------------------------------------------
# Bearish Engulfing
# ---------------------------------------------------------------------------

class TestBearishEngulfing:
    """Tests for PatternRecognizer.is_bearish_engulfing"""

    def test_true_positive(self):
        """A bullish candle followed by a larger bearish candle should match."""
        previous = pd.Series({"open": 2040, "high": 2055, "low": 2038, "close": 2050})
        current = pd.Series({"open": 2052, "high": 2058, "low": 2035, "close": 2038})
        assert PatternRecognizer.is_bearish_engulfing(current, previous) is True

    # TODO: test_false_when_both_bearish
    # TODO: test_false_when_no_engulfing


# ---------------------------------------------------------------------------
# Hammer
# ---------------------------------------------------------------------------

class TestHammer:
    """Tests for PatternRecognizer.is_hammer"""

    def test_classic_hammer(self):
        """Long lower wick, small upper wick, body in upper half → hammer."""
        candle = pd.Series({"open": 2048, "high": 2050, "low": 2030, "close": 2049})
        assert PatternRecognizer.is_hammer(candle) is True

    # TODO: test_rejected_when_upper_wick_too_long
    # TODO: test_rejected_when_body_in_lower_half
    # TODO: test_rejected_when_doji (body_size == 0)


# ---------------------------------------------------------------------------
# Shooting Star
# ---------------------------------------------------------------------------

class TestShootingStar:
    """Tests for PatternRecognizer.is_shooting_star"""

    def test_classic_shooting_star(self):
        """Long upper wick, small lower wick, body in lower half → shooting star."""
        candle = pd.Series({"open": 2032, "high": 2050, "low": 2030, "close": 2031})
        assert PatternRecognizer.is_shooting_star(candle) is True

    # TODO: test_rejected_when_lower_wick_too_long
    # TODO: test_rejected_when_body_in_upper_half


# ---------------------------------------------------------------------------
# Pin Bar & Confidence
# ---------------------------------------------------------------------------

class TestPinBar:
    """Tests for PatternRecognizer.is_pin_bar"""

    # TODO: test_bullish_pin_bar_delegates_to_hammer
    # TODO: test_bearish_pin_bar_delegates_to_shooting_star
    # TODO: test_invalid_direction_returns_false
    pass


class TestPatternConfidence:
    """Tests for PatternRecognizer.get_pattern_confidence"""

    def test_known_patterns(self):
        assert PatternRecognizer.get_pattern_confidence("bullish_engulfing") == 0.8
        assert PatternRecognizer.get_pattern_confidence("hammer") == 0.75

    # TODO: test_unknown_pattern_returns_default_0_5


# ---------------------------------------------------------------------------
# Scan / Get All
# ---------------------------------------------------------------------------

class TestScanReversalPatterns:
    """Tests for PatternRecognizer.scan_reversal_patterns"""

    # TODO: test_finds_bullish_pattern_in_dataframe
    # TODO: test_returns_none_on_insufficient_data
    # TODO: test_lookback_limits_search_range
    pass
