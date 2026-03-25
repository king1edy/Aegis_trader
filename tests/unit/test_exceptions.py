"""
Unit Tests — Custom Exceptions
==============================
Tests for: src/core/exceptions.py

Covers: exception hierarchy, message formatting, detail storage.
"""

import pytest

from src.core.exceptions import (
    TradingSystemError,
    ConfigurationError,
    BrokerConnectionError,
    MT5ConnectionError,
    TradingError,
    InsufficientMarginError,
    OrderExecutionError,
    PositionNotFoundError,
    RiskManagementError,
    RiskLimitExceededError,
    MaxDrawdownExceededError,
    DailyLossLimitError,
    StrategyError,
    InsufficientDataError,
    IndicatorCalculationError,
    StaleDataError,
    ManualOverrideAttemptError,
    CooldownPeriodError,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Base Exception
# ---------------------------------------------------------------------------

class TestTradingSystemError:
    """Tests for the base TradingSystemError."""

    def test_message_stored(self):
        err = TradingSystemError("something broke")
        assert err.message == "something broke"

    def test_str_without_details(self):
        err = TradingSystemError("oops")
        assert str(err) == "oops"

    def test_str_with_details(self):
        err = TradingSystemError("oops", details={"code": 42})
        assert "42" in str(err)

    def test_details_default_to_empty_dict(self):
        """Test that details defaults to empty dictionary when not provided"""
        err = TradingSystemError("something broke")
        assert err.details == {}
        assert isinstance(err.details, dict)
        
        err2 = TradingSystemError("something broke", details=None)
        assert err2.details == {}


# ---------------------------------------------------------------------------
# Hierarchy
# ---------------------------------------------------------------------------

class TestExceptionHierarchy:
    """Verify subclass relationships are correct."""

    def test_mt5_connection_is_broker_connection(self):
        err = MT5ConnectionError("fail")
        assert isinstance(err, BrokerConnectionError)
        assert isinstance(err, TradingSystemError)

    def test_insufficient_margin_is_trading_error(self):
        err = InsufficientMarginError(
            required_margin=5000, available_margin=1000, symbol="XAUUSD"
        )
        assert isinstance(err, TradingError)

    def test_risk_limit_exceeded_is_risk_management_error(self):
        """Test that RiskLimitExceededError is a subclass of RiskManagementError"""
        err = RiskLimitExceededError(
            limit_type="daily",
            current_value=0.05,
            limit_value=0.03
        )
        assert isinstance(err, RiskManagementError)
        assert isinstance(err, TradingSystemError)

    def test_indicator_calculation_error_is_strategy_error(self):
        """Test that IndicatorCalculationError is a subclass of StrategyError"""
        err = IndicatorCalculationError(indicator_name="RSI", reason="Not enough data")
        assert isinstance(err, StrategyError)
        assert isinstance(err, TradingSystemError)

    def test_manual_override_is_behavioral_safeguard_error(self):
        """Test that ManualOverrideAttemptError is a subclass of BehavioralSafeguardError"""
        err = ManualOverrideAttemptError(action="close_position")
        assert isinstance(err, TradingSystemError)
        assert err.details["action"] == "close_position"


# ---------------------------------------------------------------------------
# Specialized Exceptions — Attribute Storage
# ---------------------------------------------------------------------------

class TestSpecializedExceptions:
    """Verify specialized exceptions store the correct details."""

    def test_insufficient_margin_details(self):
        err = InsufficientMarginError(
            required_margin=5000, available_margin=1000, symbol="XAUUSD"
        )
        assert err.details["required_margin"] == 5000
        assert err.details["available_margin"] == 1000
        assert err.details["symbol"] == "XAUUSD"

    def test_position_not_found_details(self):
        err = PositionNotFoundError(ticket=123456)
        assert err.details["ticket"] == 123456

    def test_order_execution_error_stores_error_code(self):
        """Test that OrderExecutionError stores error_code and broker_message"""
        err = OrderExecutionError(
            order_type="BUY",
            symbol="XAUUSD",
            error_code=10004,
            broker_message="Invalid volume"
        )
        assert err.details["order_type"] == "BUY"
        assert err.details["symbol"] == "XAUUSD"
        assert err.details["error_code"] == 10004
        assert err.details["broker_message"] == "Invalid volume"

    def test_risk_limit_exceeded_stores_limit_type(self):
        """Test that RiskLimitExceededError stores limit_type, current_value, limit_value"""
        err = RiskLimitExceededError(
            limit_type="daily_risk",
            current_value=0.05,
            limit_value=0.03
        )
        assert err.details["limit_type"] == "daily_risk"
        assert err.details["current_value"] == 0.05
        assert err.details["limit_value"] == 0.03

    def test_stale_data_error_stores_ages(self):
        """Test that StaleDataError stores data_age and max_age"""
        err = StaleDataError(
            symbol="XAUUSD",
            data_age_seconds=120.5,
            max_age_seconds=60.0
        )
        assert err.details["symbol"] == "XAUUSD"
        assert err.details["data_age_seconds"] == 120.5
        assert err.details["max_age_seconds"] == 60.0

    def test_cooldown_period_stores_remaining_minutes(self):
        """Test that CooldownPeriodError stores remaining_minutes and reason"""
        err = CooldownPeriodError(remaining_minutes=15, reason="Max consecutive losses")
        assert err.details["remaining_minutes"] == 15
        assert err.details["reason"] == "Max consecutive losses"


# ---------------------------------------------------------------------------
# Additional Exception Tests
# ---------------------------------------------------------------------------

class TestAdditionalExceptions:
    """Tests for other specialized exceptions."""

    def test_max_drawdown_exceeded_stores_details(self):
        """Test MaxDrawdownExceededError stores drawdown values"""
        err = MaxDrawdownExceededError(
            current_drawdown=0.15,
            max_drawdown=0.10
        )
        assert err.details["current_drawdown"] == 0.15
        assert err.details["max_drawdown"] == 0.10

    def test_daily_loss_limit_stores_details(self):
        """Test DailyLossLimitError stores loss values"""
        err = DailyLossLimitError(
            daily_loss=500.0,
            daily_limit=1000.0
        )
        assert err.details["daily_loss"] == 500.0
        assert err.details["daily_limit"] == 1000.0

    def test_insufficient_data_error(self):
        """Test InsufficientDataError (no custom attributes)"""
        err = InsufficientDataError("Not enough bars for calculation")
        assert err.message == "Not enough bars for calculation"
        assert isinstance(err, StrategyError)

    def test_indicator_calculation_error_works(self):
        """Test IndicatorCalculationError with both parameters"""
        err = IndicatorCalculationError(
            indicator_name="EMA50",
            reason="Insufficient lookback period"
        )
        assert err.details["indicator_name"] == "EMA50"
        assert err.details["reason"] == "Insufficient lookback period"

    def test_manual_override_error(self):
        """Test ManualOverrideAttemptError stores action"""
        err = ManualOverrideAttemptError(action="close_position")
        assert err.details["action"] == "close_position"
        assert "Manual override attempted" in str(err)


# ---------------------------------------------------------------------------
# Exception String Representations
# ---------------------------------------------------------------------------

class TestExceptionStringRepresentation:
    """Test that exceptions produce readable string representations."""

    def test_trading_system_error_with_details(self):
        err = TradingSystemError("Error", details={"code": 500})
        assert "Error" in str(err)
        assert "500" in str(err)

    def test_insufficient_margin_string(self):
        err = InsufficientMarginError(
            required_margin=5000,
            available_margin=1000,
            symbol="XAUUSD"
        )
        str_repr = str(err)
        assert "Insufficient margin" in str_repr
        assert "XAUUSD" in str_repr

    def test_cooldown_period_string(self):
        err = CooldownPeriodError(remaining_minutes=15, reason="Max consecutive losses")
        str_repr = str(err)
        assert "Trading paused" in str_repr
        assert "15 minutes" in str_repr


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestExceptionEdgeCases:
    """Test edge cases for exceptions."""

    def test_zero_remaining_minutes_cooldown(self):
        """Test cooldown with zero remaining minutes"""
        err = CooldownPeriodError(remaining_minutes=0, reason="Test reason")
        assert err.details["remaining_minutes"] == 0

    def test_large_ticket_number(self):
        """Test position not found with large ticket number"""
        large_ticket = 999999999
        err = PositionNotFoundError(ticket=large_ticket)
        assert err.details["ticket"] == large_ticket
        assert str(large_ticket) in str(err)