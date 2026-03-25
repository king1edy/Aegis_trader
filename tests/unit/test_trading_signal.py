"""
Unit Tests — Trading Signal
============================
Tests for: src/strategies/base_strategy.py (TradingSignal dataclass)

Covers: signal validation, risk/reward, serialization.
"""

from datetime import datetime, timezone

import pytest

from src.execution.mt5_connector import OrderDirection
from src.strategies.base_strategy import TradingSignal

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_buy_signal(**overrides) -> TradingSignal:
    """Create a valid BUY signal with sensible defaults, applying overrides."""
    defaults = dict(
        timestamp=datetime.now(timezone.utc),
        symbol="XAUUSD",
        direction=OrderDirection.BUY,
        entry_price=2050.0,
        stop_loss=2040.0,
        take_profit_1=2060.0,
        take_profit_2=2070.0,
        confidence=0.8,
        reason="test",
    )
    defaults.update(overrides)
    return TradingSignal(**defaults)


def _make_sell_signal(**overrides) -> TradingSignal:
    """Create a valid SELL signal with sensible defaults, applying overrides."""
    defaults = dict(
        timestamp=datetime.now(timezone.utc),
        symbol="XAUUSD",
        direction=OrderDirection.SELL,
        entry_price=2050.0,
        stop_loss=2060.0,
        take_profit_1=2040.0,
        take_profit_2=2030.0,
        confidence=0.8,
        reason="test",
    )
    defaults.update(overrides)
    return TradingSignal(**defaults)


# ---------------------------------------------------------------------------
# Valid Signals
# ---------------------------------------------------------------------------

class TestValidSignals:
    """Verify that correctly-formed signals pass validation."""

    def test_valid_buy_signal(self):
        signal = _make_buy_signal()
        assert signal.direction == OrderDirection.BUY
        assert signal.stop_loss < signal.entry_price < signal.take_profit_1

    def test_valid_sell_signal(self):
        signal = _make_sell_signal()
        assert signal.direction == OrderDirection.SELL
        assert signal.stop_loss > signal.entry_price > signal.take_profit_1

    def test_confidence_at_boundary_0(self):
        """Confidence at 0 should be accepted."""
        signal = _make_buy_signal(confidence=0.0)
        assert signal.confidence == 0.0

    def test_confidence_at_boundary_1(self):
        """Confidence at 1 should be accepted."""
        signal = _make_buy_signal(confidence=1.0)
        assert signal.confidence == 1.0


# ---------------------------------------------------------------------------
# Invalid Signals
# ---------------------------------------------------------------------------

class TestInvalidSignals:
    """Verify that bad inputs are rejected at construction."""

    def test_buy_rejects_sl_above_entry(self):
        with pytest.raises(ValueError, match="Stop loss must be below entry"):
            _make_buy_signal(stop_loss=2060.0)

    def test_sell_rejects_sl_below_entry(self):
        with pytest.raises(ValueError, match="Stop loss must be above entry"):
            _make_sell_signal(stop_loss=2040.0)

    def test_rejects_confidence_above_1(self):
        with pytest.raises(ValueError):
            _make_buy_signal(confidence=1.5)

    def test_rejects_confidence_below_0(self):
        """Confidence below 0 should be rejected."""
        with pytest.raises(ValueError):
            _make_buy_signal(confidence=-0.5)

    def test_buy_rejects_tp_below_entry(self):
        """Buy signal with TP below entry should be rejected."""
        with pytest.raises(ValueError, match="Take profit must be above entry"):
            _make_buy_signal(take_profit_1=2040.0)

    def test_sell_rejects_tp_above_entry(self):
        """Sell signal with TP above entry should be rejected."""
        with pytest.raises(ValueError, match="Take profit must be below entry"):
            _make_sell_signal(take_profit_1=2060.0)


# ---------------------------------------------------------------------------
# Risk / Reward
# ---------------------------------------------------------------------------

class TestRiskReward:
    """Tests for TradingSignal.get_risk_reward_ratio"""

    def test_known_rr_ratio(self):
        signal = _make_buy_signal(
            entry_price=2050.0, stop_loss=2040.0, take_profit_1=2070.0
        )
        # sl_dist = 10, tp_dist = 20 → RR = 2.0
        assert signal.get_risk_reward_ratio() == pytest.approx(2.0)

    def test_rr_for_sell_signal(self):
        """Test risk/reward ratio for sell signal."""
        signal = _make_sell_signal(
            entry_price=2050.0, stop_loss=2060.0, take_profit_1=2030.0
        )
        # sl_dist = 10, tp_dist = 20 → RR = 2.0
        assert signal.get_risk_reward_ratio() == pytest.approx(2.0)

    def test_rr_when_sl_equals_entry(self):
        """Risk/reward should be 0 when stop loss equals entry."""
        # Since validation prevents creating signal with SL == entry,
        # we test that it raises ValueError
        with pytest.raises(ValueError, match="Stop loss must be below entry"):
            _make_buy_signal(
                entry_price=2050.0, stop_loss=2050.0, take_profit_1=2070.0
            )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    """Tests for TradingSignal.to_dict"""

    def test_to_dict_has_required_keys(self):
        signal = _make_buy_signal()
        d = signal.to_dict()
        for key in ("symbol", "direction", "entry_price", "stop_loss", "confidence"):
            assert key in d

    def test_direction_serialized_as_string(self):
        """Direction should be serialized as "BUY" or "SELL" string."""
        buy_signal = _make_buy_signal()
        buy_dict = buy_signal.to_dict()
        assert buy_dict["direction"] == "BUY"
        
        sell_signal = _make_sell_signal()
        sell_dict = sell_signal.to_dict()
        assert sell_dict["direction"] == "SELL"

    def test_timestamp_is_iso_format(self):
        """Timestamp should be serialized as ISO format string."""
        signal = _make_buy_signal()
        d = signal.to_dict()
        assert isinstance(d["timestamp"], str)
        # Check if it's ISO format (contains T and Z or +)
        assert "T" in d["timestamp"] or "+" in d["timestamp"]


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_confidence(self):
        """Zero confidence should be accepted."""
        signal = _make_buy_signal(confidence=0.0)
        assert signal.confidence == 0.0
        assert signal.get_risk_reward_ratio() > 0

    def test_max_confidence(self):
        """Maximum confidence (1.0) should be accepted."""
        signal = _make_buy_signal(confidence=1.0)
        assert signal.confidence == 1.0

    def test_tp2_optional(self):
        """Take profit 2 should be optional."""
        signal = TradingSignal(
            timestamp=datetime.now(timezone.utc),
            symbol="XAUUSD",
            direction=OrderDirection.BUY,
            entry_price=2050.0,
            stop_loss=2040.0,
            take_profit_1=2060.0,
            take_profit_2=None,
            confidence=0.8,
            reason="test"
        )
        assert signal.take_profit_2 is None
        assert signal.get_risk_reward_ratio() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Market Context
# ---------------------------------------------------------------------------

class TestMarketContext:
    """Tests for market context handling."""

    def test_market_context_in_to_dict(self):
        """Market context should be included in serialization."""
        signal = _make_buy_signal(
            market_context={
                "trend": "bullish",
                "volatility": 0.15,
                "session": "london"
            }
        )
        d = signal.to_dict()
        assert "market_context" in d
        assert d["market_context"]["trend"] == "bullish"
        assert d["market_context"]["volatility"] == 0.15

    def test_empty_market_context(self):
        """Empty market context should be handled gracefully."""
        signal = _make_buy_signal(market_context={})
        d = signal.to_dict()
        assert "market_context" in d
        assert d["market_context"] == {}


# ---------------------------------------------------------------------------
# String Representation
# ---------------------------------------------------------------------------

class TestStringRepresentation:
    """Tests for string representation of TradingSignal."""

    def test_str_contains_key_info(self):
        """String representation should contain key information."""
        signal = _make_buy_signal()
        str_repr = str(signal)
        assert "BUY" in str_repr
        assert "XAUUSD" in str_repr
        assert "2050.0" in str_repr or "2050" in str_repr