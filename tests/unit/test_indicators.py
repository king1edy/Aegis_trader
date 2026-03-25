"""
Unit Tests — Indicator Calculator
=================================
Tests for: src/strategies/indicators.py

Covers: EMA, WMA, Hull MA, RSI, ATR, swing detection, dataframe validation.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategies.indicators import IndicatorCalculator, IndicatorConfig
from src.core.exceptions import InsufficientDataError

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

class TestCalculateEMA:
    """Tests for IndicatorCalculator.calculate_ema"""

    def test_ema_output_length_matches_input(self, sample_ohlcv_df):
        """EMA series should have the same length as the input series."""
        result = IndicatorCalculator.calculate_ema(sample_ohlcv_df["close"], period=20)
        assert len(result) == len(sample_ohlcv_df)

    def test_ema_constant_series_equals_constant(self):
        """EMA of a constant series should equal that constant."""
        series = pd.Series([100.0] * 50)
        result = IndicatorCalculator.calculate_ema(series, period=14)
        assert result.iloc[-1] == pytest.approx(100.0)

    # TODO: test_ema_reacts_to_price_change — verify EMA moves toward new price
    # TODO: test_ema_known_value — compare against a hand-calculated value


# ---------------------------------------------------------------------------
# WMA
# ---------------------------------------------------------------------------

class TestCalculateWMA:
    """Tests for IndicatorCalculator.calculate_wma"""

    def test_wma_initial_period_is_nan(self):
        """First (period - 1) values should be NaN."""
        series = pd.Series(range(1, 21), dtype=float)
        result = IndicatorCalculator.calculate_wma(series, period=5)
        assert result.iloc[:4].isna().all()
        assert not np.isnan(result.iloc[4])

    # TODO: test_wma_known_value — verify against hand-calculated WMA
    # TODO: test_wma_output_length — should match input length


# ---------------------------------------------------------------------------
# Hull MA
# ---------------------------------------------------------------------------

class TestCalculateHullMA:
    """Tests for IndicatorCalculator.calculate_hull_ma"""

    def test_hull_ma_output_length(self, sample_ohlcv_df):
        """Hull MA output should match input length."""
        result = IndicatorCalculator.calculate_hull_ma(
            sample_ohlcv_df["close"], period=55
        )
        assert len(result) == len(sample_ohlcv_df)

    # TODO: test_hull_ma_less_lag_than_ema — Hull MA should follow price changes faster
    # TODO: test_hull_ma_constant_series — should equal the constant


# ---------------------------------------------------------------------------
# RSI
# ---------------------------------------------------------------------------

class TestCalculateRSI:
    """Tests for IndicatorCalculator.calculate_rsi"""

    def test_rsi_all_gains_near_100(self):
        """Monotonically increasing prices should produce RSI near 100."""
        series = pd.Series(np.linspace(100, 200, 100))
        result = IndicatorCalculator.calculate_rsi(series, period=14)
        assert result.iloc[-1] > 95.0

    def test_rsi_bounded_0_to_100(self, sample_ohlcv_df):
        """RSI values should stay within [0, 100]."""
        result = IndicatorCalculator.calculate_rsi(
            sample_ohlcv_df["close"], period=14
        )
        valid = result.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    # TODO: test_rsi_all_losses_near_0
    # TODO: test_rsi_flat_series_is_50


# ---------------------------------------------------------------------------
# ATR
# ---------------------------------------------------------------------------

class TestCalculateATR:
    """Tests for IndicatorCalculator.calculate_atr"""

    def test_atr_non_negative(self, sample_ohlcv_df):
        """ATR should never be negative."""
        result = IndicatorCalculator.calculate_atr(sample_ohlcv_df, period=14)
        valid = result.dropna()
        assert (valid >= 0).all()

    # TODO: test_atr_constant_range_converges — if high-low is always X, ATR → X
    # TODO: test_atr_output_length


# ---------------------------------------------------------------------------
# Swing Detection
# ---------------------------------------------------------------------------

class TestFindSwings:
    """Tests for IndicatorCalculator.find_swings"""

    def test_detects_known_peak(self):
        """A clear V-shaped high should be detected as a swing high."""
        highs = [1, 2, 3, 4, 5, 4, 3, 2, 1, 2, 3]
        lows = [x - 0.5 for x in highs]
        df = pd.DataFrame({"high": highs, "low": lows})
        swing_highs, _ = IndicatorCalculator.find_swings(df, lookback=2)
        assert 4 in swing_highs  # index of the peak (value 5)

    # TODO: test_detects_known_trough
    # TODO: test_no_swings_in_flat_data


# ---------------------------------------------------------------------------
# calculate_all & validate_dataframe
# ---------------------------------------------------------------------------

class TestCalculateAll:
    """Tests for IndicatorCalculator.calculate_all"""

    def test_raises_on_insufficient_data(self):
        """Should raise InsufficientDataError when bars < minimum required."""
        small_df = pd.DataFrame({
            "open": [1.0] * 10,
            "high": [1.1] * 10,
            "low": [0.9] * 10,
            "close": [1.0] * 10,
            "volume": [100] * 10,
        })
        config = IndicatorConfig()
        with pytest.raises(InsufficientDataError):
            IndicatorCalculator.calculate_all(small_df, config)

    # TODO: test_adds_all_expected_columns — call on valid df, verify column names
    # TODO: test_does_not_modify_original — ensure original df unchanged


class TestValidateDataframe:
    """Tests for IndicatorCalculator.validate_dataframe"""

    def test_raises_on_missing_column(self):
        """Should raise when a required column is absent."""
        df = pd.DataFrame({"open": [1], "high": [2], "low": [0.5], "close": [1]})
        with pytest.raises(InsufficientDataError):
            IndicatorCalculator.validate_dataframe(df, "XAUUSD")

    # TODO: test_raises_on_empty_df
    # TODO: test_raises_on_all_nan_column
    # TODO: test_passes_on_valid_df
