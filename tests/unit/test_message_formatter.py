"""
Unit Tests — Message Formatter
===============================
Tests for: src/notifications/message_formatter.py

Covers: trade opened/closed, signal, risk warning, daily summary, system status.
"""

import pytest

from src.notifications.message_formatter import MessageFormatter

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Trade Opened
# ---------------------------------------------------------------------------

class TestFormatTradeOpened:
    """Tests for MessageFormatter.format_trade_opened"""

    def test_buy_contains_green_emoji(self):
        msg = MessageFormatter.format_trade_opened(
            symbol="XAUUSD",
            direction="BUY",
            entry_price=2050.0,
            stop_loss=2040.0,
            take_profit_1=2060.0,
            lot_size=0.05,
        )
        assert "🟢" in msg
        assert "BUY" in msg
        assert "2050" in msg

    def test_sell_contains_red_emoji(self):
        msg = MessageFormatter.format_trade_opened(
            symbol="XAUUSD",
            direction="SELL",
            entry_price=2050.0,
            stop_loss=2060.0,
            take_profit_1=2040.0,
        )
        assert "🔴" in msg

    # TODO: test_includes_tp2_when_provided
    # TODO: test_includes_confidence_when_nonzero
    # TODO: test_rr_ratio_calculated_correctly


# ---------------------------------------------------------------------------
# Trade Closed
# ---------------------------------------------------------------------------

class TestFormatTradeClosed:
    """Tests for MessageFormatter.format_trade_closed"""

    def test_profit_uses_money_emoji(self):
        msg = MessageFormatter.format_trade_closed(
            symbol="XAUUSD",
            direction="BUY",
            entry_price=2040.0,
            exit_price=2060.0,
            profit_loss=200.0,
            lot_size=0.1,
            duration_minutes=90,
        )
        assert "💰" in msg
        assert "+$200" in msg

    # TODO: test_loss_uses_loss_emoji
    # TODO: test_duration_formatted_as_hours_when_over_60_minutes
    # TODO: test_partial_close_shows_percentage


# ---------------------------------------------------------------------------
# Risk Warning
# ---------------------------------------------------------------------------

class TestFormatRiskWarning:
    """Tests for MessageFormatter.format_risk_warning"""

    def test_drawdown_warning_title(self):
        msg = MessageFormatter.format_risk_warning(
            warning_type="drawdown",
            current_value=8.5,
            limit_value=10.0,
        )
        assert "DRAWDOWN WARNING" in msg

    # TODO: test_daily_loss_title
    # TODO: test_unknown_warning_type_uses_raw_name


# ---------------------------------------------------------------------------
# Daily Summary
# ---------------------------------------------------------------------------

class TestFormatDailySummary:
    """Tests for MessageFormatter.format_daily_summary"""

    # TODO: test_contains_all_stats (date, trades, win_rate, net_pnl)
    # TODO: test_positive_balance_change_has_plus_sign
    # TODO: test_negative_net_pnl_uses_loss_emoji
    pass


# ---------------------------------------------------------------------------
# System Status
# ---------------------------------------------------------------------------

class TestFormatSystemStatus:
    """Tests for MessageFormatter.format_system_status"""

    def test_started_uses_rocket_emoji(self):
        msg = MessageFormatter.format_system_status("started")
        assert "🚀" in msg
        assert "SYSTEM STARTED" in msg

    # TODO: test_stopped_status
    # TODO: test_includes_details_when_provided (balance, version, broker)