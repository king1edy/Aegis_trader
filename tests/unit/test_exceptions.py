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

    # TODO: test_details_default_to_empty_dict


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

    # TODO: test_risk_limit_exceeded_is_risk_management_error
    # TODO: test_indicator_calculation_error_is_strategy_error
    # TODO: test_manual_override_is_behavioral_safeguard_error


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
        assert err.details["symbol"] == "XAUUSD"

    def test_position_not_found_details(self):
        err = PositionNotFoundError(ticket=123456)
        assert err.details["ticket"] == 123456

    # TODO: test_order_execution_error_stores_error_code
    # TODO: test_risk_limit_exceeded_stores_limit_type
    # TODO: test_stale_data_error_stores_ages
    # TODO: test_cooldown_period_stores_remaining_minutes
