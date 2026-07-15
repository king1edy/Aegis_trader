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

    def test_false_when_no_engulfing(self):
        """Current body smaller than previous body should not match."""
        previous = pd.Series({"open": 2050, "high": 2055, "low": 2040, "close": 2042})
        current = pd.Series({"open": 2045, "high": 2048, "low": 2042, "close": 2047})
        assert PatternRecognizer.is_bullish_engulfing(current, previous) is False

    def test_exact_boundary(self):
        """Current.open == previous.close (edge case) should still be engulfing."""
        previous = pd.Series({"open": 2050, "high": 2055, "low": 2040, "close": 2045})
        current = pd.Series({"open": 2045, "high": 2060, "low": 2040, "close": 2055})
        assert PatternRecognizer.is_bullish_engulfing(current, previous) is True


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

    def test_false_when_both_bearish(self):
        """Two bearish candles should NOT form a bearish engulfing."""
        previous = pd.Series({"open": 2050, "high": 2055, "low": 2040, "close": 2042})
        current = pd.Series({"open": 2045, "high": 2048, "low": 2038, "close": 2040})
        assert PatternRecognizer.is_bearish_engulfing(current, previous) is False

    def test_false_when_no_engulfing(self):
        """Current body smaller than previous body should not match."""
        previous = pd.Series({"open": 2040, "high": 2055, "low": 2038, "close": 2050})
        current = pd.Series({"open": 2048, "high": 2052, "low": 2042, "close": 2045})
        assert PatternRecognizer.is_bearish_engulfing(current, previous) is False


# ---------------------------------------------------------------------------
# Hammer
# ---------------------------------------------------------------------------

class TestHammer:
    """Tests for PatternRecognizer.is_hammer"""

    def test_classic_hammer(self):
        """Long lower wick, small upper wick, body in upper half → hammer."""
        candle = pd.Series({"open": 2048, "high": 2050, "low": 2030, "close": 2049})
        assert PatternRecognizer.is_hammer(candle) is True

    def test_rejected_when_upper_wick_too_long(self):
        """Upper wick too long should reject hammer."""
        candle = pd.Series({"open": 2048, "high": 2060, "low": 2030, "close": 2049})
        assert PatternRecognizer.is_hammer(candle) is False

    def test_rejected_when_body_in_lower_half(self):
        """Body in lower half should reject hammer."""
        candle = pd.Series({"open": 2030, "high": 2050, "low": 2020, "close": 2032})
        assert PatternRecognizer.is_hammer(candle) is False

    def test_rejected_when_doji(self):
        """Doji (body_size == 0) should reject hammer."""
        candle = pd.Series({"open": 2040, "high": 2050, "low": 2030, "close": 2040})
        assert PatternRecognizer.is_hammer(candle) is False


# ---------------------------------------------------------------------------
# Shooting Star
# ---------------------------------------------------------------------------

class TestShootingStar:
    """Tests for PatternRecognizer.is_shooting_star"""

    def test_classic_shooting_star(self):
        """Long upper wick, small lower wick, body in lower half → shooting star."""
        candle = pd.Series({"open": 2032, "high": 2050, "low": 2030, "close": 2031})
        assert PatternRecognizer.is_shooting_star(candle) is True

    def test_rejected_when_lower_wick_too_long(self):
        """Lower wick too long should reject shooting star."""
        candle = pd.Series({"open": 2032, "high": 2050, "low": 2020, "close": 2031})
        assert PatternRecognizer.is_shooting_star(candle) is False

    def test_rejected_when_body_in_upper_half(self):
        """Body in upper half should reject shooting star."""
        candle = pd.Series({"open": 2048, "high": 2060, "low": 2040, "close": 2049})
        assert PatternRecognizer.is_shooting_star(candle) is False


# ---------------------------------------------------------------------------
# Pin Bar & Confidence
# ---------------------------------------------------------------------------

class TestPinBar:
    """Tests for PatternRecognizer.is_pin_bar"""

    def test_bullish_pin_bar_delegates_to_hammer(self):
        """Bullish pin bar should call hammer logic."""
        candle = pd.Series({"open": 2048, "high": 2050, "low": 2030, "close": 2049})
        assert PatternRecognizer.is_pin_bar(candle, "bullish") is True

    def test_bearish_pin_bar_delegates_to_shooting_star(self):
        """Bearish pin bar should call shooting star logic."""
        candle = pd.Series({"open": 2032, "high": 2050, "low": 2030, "close": 2031})
        assert PatternRecognizer.is_pin_bar(candle, "bearish") is True

    def test_invalid_direction_returns_false(self):
        """Invalid direction should return False."""
        candle = pd.Series({"open": 2048, "high": 2050, "low": 2030, "close": 2049})
        assert PatternRecognizer.is_pin_bar(candle, "invalid") is False


class TestPatternConfidence:
    """Tests for PatternRecognizer.get_pattern_confidence"""

    def test_known_patterns(self):
        assert PatternRecognizer.get_pattern_confidence("bullish_engulfing") == 0.8
        assert PatternRecognizer.get_pattern_confidence("hammer") == 0.75
        assert PatternRecognizer.get_pattern_confidence("bearish_engulfing") == 0.8
        assert PatternRecognizer.get_pattern_confidence("shooting_star") == 0.75
        assert PatternRecognizer.get_pattern_confidence("pin_bar") == 0.75

    def test_unknown_pattern_returns_default_0_5(self):
        """Unknown pattern should return default 0.5 confidence."""
        assert PatternRecognizer.get_pattern_confidence("unknown_pattern") == 0.5
        assert PatternRecognizer.get_pattern_confidence("") == 0.5


# ---------------------------------------------------------------------------
# Scan / Get All
# ---------------------------------------------------------------------------

class TestScanReversalPatterns:
    """Tests for PatternRecognizer.scan_reversal_patterns"""

    def test_finds_bullish_pattern_in_dataframe(self):
        """Should find bullish engulfing pattern in dataframe."""
        df = pd.DataFrame({
            "open": [2050, 2041, 2035],
            "high": [2055, 2060, 2045],
            "low": [2040, 2039, 2030],
            "close": [2042, 2055, 2040]
        })
        result = PatternRecognizer.scan_reversal_patterns(df, "bullish", lookback=2)
        assert result is not None
        assert result.pattern_name == "bullish_engulfing"
        assert result.confidence == 0.8

    def test_returns_none_on_insufficient_data(self):
        """Should return None when insufficient data."""
        df = pd.DataFrame({
            "open": [2050],
            "high": [2055],
            "low": [2040],
            "close": [2042]
        })
        result = PatternRecognizer.scan_reversal_patterns(df, "bullish")
        assert result is None

    def test_lookback_limits_search_range(self):
        """Should only search within lookback range."""
        df = pd.DataFrame({
            "open": [2050, 2041, 2035, 2040],
            "high": [2055, 2060, 2045, 2050],
            "low": [2040, 2039, 2030, 2035],
            "close": [2042, 2055, 2040, 2045]
        })
        # With lookback=1, only last bar is checked (no pattern)
        result = PatternRecognizer.scan_reversal_patterns(df, "bullish", lookback=1)
        assert result is None

        # With lookback=3, should find pattern at index 1 (within lookback range)
        result = PatternRecognizer.scan_reversal_patterns(df, "bullish", lookback=3)
        assert result is not None
        assert result.candle_index == 1


# ---------------------------------------------------------------------------
# Get All Patterns
# ---------------------------------------------------------------------------

class TestGetAllPatterns:
    """Tests for PatternRecognizer.get_all_patterns"""

    def test_returns_all_patterns_in_dataframe(self):
        """Should find all reversal patterns in dataframe."""
        df = pd.DataFrame({
            "open": [2050, 2041, 2032, 2040],
            "high": [2055, 2060, 2050, 2055],
            "low": [2040, 2039, 2030, 2035],
            "close": [2042, 2055, 2031, 2045]
        })
        patterns = PatternRecognizer.get_all_patterns(df, lookback=3)
        assert len(patterns) >= 1
        
        # Should find bullish engulfing at index 1
        bullish_found = any(p.pattern_name == "bullish_engulfing" for p in patterns)
        assert bullish_found is True

    def test_returns_empty_list_on_insufficient_data(self):
        """Should return empty list when insufficient data."""
        df = pd.DataFrame({
            "open": [2050],
            "high": [2055],
            "low": [2040],
            "close": [2042]
        })
        patterns = PatternRecognizer.get_all_patterns(df)
        assert patterns == []