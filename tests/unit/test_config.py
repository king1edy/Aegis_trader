"""
Unit Tests — Configuration
==========================
Tests for: src/core/config.py

Covers: default values, validators, computed properties, trading warnings.
"""

import pytest
from pydantic import ValidationError

from src.core.config import Settings, Environment, TradingSession

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Default Values
# ---------------------------------------------------------------------------

class TestSettingsDefaults:
    """Verify critical defaults are sane."""

    def test_default_risk_per_trade(self, mock_settings):
        assert mock_settings.max_risk_per_trade == 0.01

    def test_default_max_daily_trades(self, mock_settings):
        assert mock_settings.max_daily_trades == 3

    def test_manual_override_disabled_by_default(self, mock_settings):
        assert mock_settings.enable_manual_override is False

    # TODO: test_default_symbol_is_xauusd
    # TODO: test_default_lot_sizes


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

class TestSettingsValidators:
    """Tests for field validators."""

    def test_valid_log_level_accepted(self, mock_settings):
        s = Settings(log_level="debug", mt5_login=0, mt5_password="t")
        assert s.log_level == "DEBUG"  # should be uppercased

    def test_invalid_log_level_rejected(self):
        with pytest.raises(ValidationError):
            Settings(log_level="INVALID_LEVEL", mt5_login=0, mt5_password="t")

    # TODO: test_risk_per_trade_rejects_negative
    # TODO: test_risk_per_trade_rejects_above_5_percent


# ---------------------------------------------------------------------------
# Computed Properties
# ---------------------------------------------------------------------------

class TestComputedProperties:
    """Tests for @property methods on Settings."""

    def test_db_url_constructed(self, mock_settings):
        url = mock_settings.db_url
        assert url.startswith("postgresql://")
        assert mock_settings.postgres_user in url

    def test_async_db_url_uses_asyncpg(self, mock_settings):
        assert "asyncpg" in mock_settings.async_db_url

    def test_active_sessions_parses_csv(self, mock_settings):
        sessions = mock_settings.active_sessions
        assert TradingSession.LONDON in sessions
        assert TradingSession.NEWYORK in sessions

    # TODO: test_is_production_true_for_production_env
    # TODO: test_redis_connection_url_constructed


# ---------------------------------------------------------------------------
# Trading Config Validation
# ---------------------------------------------------------------------------

class TestTradingConfigWarnings:
    """Tests for Settings.validate_trading_config()"""

    def test_warns_on_high_risk(self):
        s = Settings(max_risk_per_trade=0.03, mt5_login=0, mt5_password="t")
        warnings = s.validate_trading_config()
        assert any("recommended 2%" in w for w in warnings)

    # TODO: test_warns_on_production_with_manual_override
    # TODO: test_warns_on_missing_mt5_credentials
    # TODO: test_no_warnings_on_healthy_config
