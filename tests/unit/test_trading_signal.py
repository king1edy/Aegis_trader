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

    # TODO: test_confidence_at_boundary_0 — should be accepted
    # TODO: test_confidence_at_boundary_1 — should be accepted


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

    # TODO: test_rejects_confidence_below_0
    # TODO: test_buy_rejects_tp_below_entry
    # TODO: test_sell_rejects_tp_above_entry


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

    # TODO: test_rr_for_sell_signal
    # TODO: test_rr_when_sl_equals_entry — should return 0


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

    # TODO: test_direction_serialized_as_string — d["direction"] should be "BUY"/"SELL"
    # TODO: test_timestamp_is_iso_format
