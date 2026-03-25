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

    def test_includes_tp2_when_provided(self):
        """Test that TP2 is included when provided"""
        msg = MessageFormatter.format_trade_opened(
            symbol="XAUUSD",
            direction="BUY",
            entry_price=2050.0,
            stop_loss=2040.0,
            take_profit_1=2060.0,
            take_profit_2=2070.0,
        )
        assert "TP2" in msg
        assert "2070" in msg

    def test_includes_confidence_when_nonzero(self):
        """Test that confidence is included when > 0"""
        msg = MessageFormatter.format_trade_opened(
            symbol="XAUUSD",
            direction="BUY",
            entry_price=2050.0,
            stop_loss=2040.0,
            take_profit_1=2060.0,
            confidence=85.5,
        )
        assert "Confidence" in msg
        # Confidence is rounded to integer (86% from 85.5)
        assert "86%" in msg

    def test_rr_ratio_calculated_correctly(self):
        """Test that risk-reward ratio is calculated correctly"""
        msg = MessageFormatter.format_trade_opened(
            symbol="XAUUSD",
            direction="BUY",
            entry_price=2050.0,
            stop_loss=2040.0,  # 10 point risk
            take_profit_1=2070.0,  # 20 point reward
        )
        # R:R should be 1:2
        assert "R:R:" in msg
        assert "1:2.0" in msg or "2.0" in msg


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

    def test_loss_uses_loss_emoji(self):
        """Test that loss uses the loss emoji"""
        msg = MessageFormatter.format_trade_closed(
            symbol="XAUUSD",
            direction="BUY",
            entry_price=2060.0,
            exit_price=2040.0,
            profit_loss=-150.0,
            lot_size=0.1,
            duration_minutes=60,
        )
        assert "💸" in msg
        # The format uses `$-150.00` not `-$150`
        assert "$-150" in msg

    def test_duration_formatted_as_hours_when_over_60_minutes(self):
        """Test that duration shows hours when over 60 minutes"""
        msg = MessageFormatter.format_trade_closed(
            symbol="XAUUSD",
            direction="BUY",
            entry_price=2040.0,
            exit_price=2060.0,
            profit_loss=200.0,
            lot_size=0.1,
            duration_minutes=125,
        )
        # 125 minutes = 2 hours 5 minutes
        assert "2h" in msg
        assert "5m" in msg

    def test_partial_close_shows_percentage(self):
        """Test that partial close shows percentage closed"""
        msg = MessageFormatter.format_trade_closed(
            symbol="XAUUSD",
            direction="BUY",
            entry_price=2040.0,
            exit_price=2060.0,
            profit_loss=100.0,
            lot_size=0.05,
            duration_minutes=45,
            is_partial=True,
            partial_percent=50,
        )
        assert "PARTIAL CLOSE" in msg
        assert "50%" in msg


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

    def test_daily_loss_title(self):
        """Test that daily loss warning has correct title"""
        msg = MessageFormatter.format_risk_warning(
            warning_type="daily_loss",
            current_value=5.0,
            limit_value=3.0,
        )
        assert "DAILY LOSS LIMIT" in msg

    def test_unknown_warning_type_uses_raw_name(self):
        """Test that unknown warning type uses uppercase name"""
        msg = MessageFormatter.format_risk_warning(
            warning_type="custom_warning",
            current_value=10.0,
            limit_value=20.0,
        )
        assert "CUSTOM_WARNING" in msg


# ---------------------------------------------------------------------------
# Daily Summary
# ---------------------------------------------------------------------------

class TestFormatDailySummary:
    """Tests for MessageFormatter.format_daily_summary"""

    def test_contains_all_stats(self):
        """Test that daily summary contains all required statistics"""
        msg = MessageFormatter.format_daily_summary(
            date="2025-03-25",
            total_trades=10,
            winning_trades=6,
            losing_trades=4,
            gross_profit=500.0,
            gross_loss=200.0,
            net_pnl=300.0,
            win_rate=60.0,
            starting_balance=10000.0,
            ending_balance=10300.0,
        )
        assert "2025-03-25" in msg
        assert "10" in msg
        assert "6" in msg
        assert "4" in msg
        assert "500" in msg
        assert "200" in msg
        assert "300" in msg
        assert "60.0%" in msg

    def test_positive_balance_change_has_plus_sign(self):
        """Test that positive balance change shows plus sign"""
        msg = MessageFormatter.format_daily_summary(
            date="2025-03-25",
            total_trades=5,
            winning_trades=3,
            losing_trades=2,
            gross_profit=300.0,
            gross_loss=100.0,
            net_pnl=200.0,
            win_rate=60.0,
            starting_balance=10000.0,
            ending_balance=10200.0,
        )
        assert "+$200" in msg

    def test_negative_net_pnl_uses_loss_emoji(self):
        """Test that negative net P&L uses loss emoji"""
        msg = MessageFormatter.format_daily_summary(
            date="2025-03-25",
            total_trades=5,
            winning_trades=2,
            losing_trades=3,
            gross_profit=100.0,
            gross_loss=300.0,
            net_pnl=-200.0,
            win_rate=40.0,
            starting_balance=10000.0,
            ending_balance=9800.0,
        )
        assert "💸" in msg
        # The format uses `$-200.00` not `-$200`
        assert "$-200" in msg


# ---------------------------------------------------------------------------
# System Status
# ---------------------------------------------------------------------------

class TestFormatSystemStatus:
    """Tests for MessageFormatter.format_system_status"""

    def test_started_uses_rocket_emoji(self):
        msg = MessageFormatter.format_system_status("started")
        assert "🚀" in msg
        assert "SYSTEM STARTED" in msg

    def test_stopped_status(self):
        """Test that stopped status uses stop emoji"""
        msg = MessageFormatter.format_system_status("stopped")
        assert "🛑" in msg
        assert "SYSTEM STOPPED" in msg

    def test_includes_details_when_provided(self):
        """Test that details are included when provided"""
        msg = MessageFormatter.format_system_status(
            "started",
            details={
                "version": "1.0.0",
                "balance": 10000.0,
                "broker": "Paper Trading"
            }
        )
        assert "1.0.0" in msg
        assert "10000" in msg
        assert "Paper Trading" in msg


# ---------------------------------------------------------------------------
# Signal Generated
# ---------------------------------------------------------------------------

class TestFormatSignalGenerated:
    """Tests for MessageFormatter.format_signal_generated"""

    def test_signal_generated_contains_emoji(self):
        """Test that signal generated has correct emoji"""
        msg = MessageFormatter.format_signal_generated(
            symbol="XAUUSD",
            direction="BUY",
            entry_price=2050.0,
            stop_loss=2040.0,
            take_profit=2060.0,
            confidence=75.0,
        )
        assert "📊" in msg
        assert "GENERATED" in msg

    def test_signal_executed_shows_executed_status(self):
        """Test that executed signal shows executed status"""
        msg = MessageFormatter.format_signal_generated(
            symbol="XAUUSD",
            direction="BUY",
            entry_price=2050.0,
            stop_loss=2040.0,
            take_profit=2060.0,
            confidence=75.0,
            was_executed=True,
        )
        assert "EXECUTED" in msg


# ---------------------------------------------------------------------------
# Signal Rejected
# ---------------------------------------------------------------------------

class TestFormatSignalRejected:
    """Tests for MessageFormatter.format_signal_rejected"""

    def test_signal_rejected_format(self):
        """Test that signal rejected message is formatted correctly"""
        msg = MessageFormatter.format_signal_rejected(
            symbol="XAUUSD",
            direction="BUY",
            reason="Insufficient data"
        )
        assert "SIGNAL REJECTED" in msg
        assert "XAUUSD" in msg
        assert "BUY" in msg
        assert "Insufficient data" in msg