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
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings():
    """Create a mock settings instance with default values."""
    return Settings(
        mt5_login=12345,
        mt5_password="test_password",
        app_env=Environment.DEVELOPMENT
    )


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

    # test_default_symbol_is_xauusd#

    def test_default_symbol_is_xauusd(self, mock_settings):
        """Test that default symbol is XAUUSD"""
        assert mock_settings.default_symbol == "XAUUSD"


    # test_default_lot_sizes#

    def test_default_lot_sizes(self, mock_settings):
        """Test that default lot sizes are correct"""
        assert mock_settings.default_lot_size == 0.01
        assert mock_settings.max_lot_size == 0.05
        assert mock_settings.default_lot_size <= mock_settings.max_lot_size

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

    # test_risk_per_trade_rejects_negative#

    def test_risk_per_trade_rejects_negative(self):
        """Test that negative risk per trade is rejected"""
        with pytest.raises(ValidationError):
            Settings(
                max_risk_per_trade=-0.01,
                mt5_login=12345,
                mt5_password="test"
            )

    def test_risk_per_trade_rejects_above_5_percent(self):
        """Test that risk per trade above 5% is rejected"""
        with pytest.raises(ValidationError):
            Settings(
                max_risk_per_trade=0.06,  # 6% > 5%
                mt5_login=12345,
                mt5_password="test"
            )
        
    # EXTRA: Testing valid range ensures the validator correctly accepts boundary values
    # Added to verify that the field validation works for all acceptable inputs, not just rejects invalid ones

    # test_risk_per_trade_rejects_above_5_percent#

    def test_risk_per_trade_accepts_valid_range(self):
        """Test that valid risk percentages are accepted"""
        # Test lower bound (0.001 = 0.1%)
        s1 = Settings(max_risk_per_trade=0.001, mt5_login=0, mt5_password="t")
        assert s1.max_risk_per_trade == 0.001
        
        # Test upper bound (0.05 = 5%)
        s2 = Settings(max_risk_per_trade=0.05, mt5_login=0, mt5_password="t")
        assert s2.max_risk_per_trade == 0.05
        
        # Test middle value (0.02 = 2%)
        s3 = Settings(max_risk_per_trade=0.02, mt5_login=0, mt5_password="t")
        assert s3.max_risk_per_trade == 0.02


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

    # test_is_production_true_for_production_env#

    def test_is_production_true_for_production_env(self):
        """Test that is_production returns True for production environment"""
        s = Settings(
            app_env=Environment.PRODUCTION,
            mt5_login=0,
            mt5_password="test"
        )
        assert s.is_production is True
        
        # Should be False for development
        s_dev = Settings(
            app_env=Environment.DEVELOPMENT,
            mt5_login=0,
            mt5_password="test"
        )
        assert s_dev.is_production is False
        
        # Should be False for staging
        s_staging = Settings(
            app_env=Environment.STAGING,
            mt5_login=0,
            mt5_password="test"
        )
        assert s_staging.is_production is False
    # test_redis_connection_url_constructed#

    def test_redis_connection_url_constructed(self, mock_settings):
        """Test that Redis URL is constructed correctly"""
        url = mock_settings.redis_connection_url
        assert url.startswith("redis://")
        assert mock_settings.redis_host in url
        assert str(mock_settings.redis_port) in url
        assert str(mock_settings.redis_db) in url

    # EXTRA: Testing custom session strings ensures the CSV parsing handles various formats
    # Added to verify the active_sessions property correctly parses different session combinations
    def test_active_sessions_with_custom_sessions(self):
        """Test that custom session strings are parsed correctly"""
        s = Settings(
            trade_sessions="london",
            mt5_login=0,
            mt5_password="test"
        )
        sessions = s.active_sessions
        assert len(sessions) == 1
        assert TradingSession.LONDON in sessions
        
        s2 = Settings(
            trade_sessions="london,newyork,asian",
            mt5_login=0,
            mt5_password="test"
        )
        sessions2 = s2.active_sessions
        assert len(sessions2) == 3
        assert TradingSession.ASIAN in sessions2
        assert TradingSession.LONDON in sessions2
        assert TradingSession.NEWYORK in sessions2

# ---------------------------------------------------------------------------
# Trading Config Validation
# ---------------------------------------------------------------------------

class TestTradingConfigWarnings:
    """Tests for Settings.validate_trading_config()"""

    def test_warns_on_high_risk(self):
        s = Settings(max_risk_per_trade=0.03, mt5_login=0, mt5_password="t")
        warnings = s.validate_trading_config()
        assert any("recommended 2%" in w for w in warnings)

    # test_warns_on_production_with_manual_override#
    def test_warns_on_production_with_manual_override(self):
        """Test that production with manual override enabled generates warning"""
        s = Settings(
            app_env=Environment.PRODUCTION,
            enable_manual_override=True,
            mt5_login=12345,
            mt5_password="test"
        )
        warnings = s.validate_trading_config()
        assert any("Manual override is enabled in production" in w for w in warnings)

    # test_warns_on_missing_mt5_credentials#

    def test_warns_on_missing_mt5_credentials(self):
        s = Settings(
            mt5_login=0,
            mt5_password="",
            app_env=Environment.DEVELOPMENT
        )
        warnings = s.validate_trading_config()
        assert any("MT5 credentials not configured" in w for w in warnings)
    # test_no_warnings_on_healthy_config#
    def test_no_warnings_on_healthy_config(self):
        s = Settings(
            max_risk_per_trade=0.01,  # 1% (within recommended)
            max_lot_size=0.05,
            default_lot_size=0.01,
            app_env=Environment.DEVELOPMENT,
            enable_manual_override=False,
            mt5_login=12345,
            mt5_password="test_password",
            telegram_enabled=True
        )
        warnings = s.validate_trading_config()
        assert len(warnings) == 0

    # EXTRA: Production debug mode is a critical safety issue that should always trigger a warning
    # Added to ensure the config validation catches this dangerous configuration   

    #test_debug_mode_production_generates_warning#

    def test_warns_on_production_debug_mode(self):
    
        s = Settings(
            app_env=Environment.PRODUCTION,
            debug=True,
            mt5_login=12345,
            mt5_password="test"
        )
        warnings = s.validate_trading_config()
        assert any("Debug mode is enabled in production" in w for w in warnings)
 
     # EXTRA: Missing notifications in production could lead to missed alerts about critical events
    # Added to ensure operators are warned about potential monitoring gaps
    def test_warns_on_missing_telegram_in_production(self):
        """Test that missing Telegram in production generates warning"""
        s = Settings(
            app_env=Environment.PRODUCTION,
            telegram_enabled=False,
            mt5_login=12345,
            mt5_password="test"
        )
        warnings = s.validate_trading_config()
        assert any("Telegram notifications are disabled in production" in w for w in warnings)

    # EXTRA: Invalid lot size configuration could cause position sizing failures
    # Added to catch misconfigurations where max_lot is less than default_lot
    def test_warns_on_max_lot_less_than_default(self):
        """Test that max_lot < default_lot generates warning"""
        s = Settings(
            max_lot_size=0.02,
            default_lot_size=0.05,
            mt5_login=12345,
            mt5_password="test"
        )
        warnings = s.validate_trading_config()
        assert any("max_lot_size" in w and "default_lot_size" in w for w in warnings)


# ---------------------------------------------------------------------------
# Edge Cases (Added for Comprehensive Coverage)
# ---------------------------------------------------------------------------
# EXTRA: Edge case testing ensures the configuration handles boundary conditions gracefully
# Added to validate behavior at extreme values and malformed inputs

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_min_risk_per_trade_boundary(self):
        """Test minimum acceptable risk per trade (0.1%)"""
        s = Settings(max_risk_per_trade=0.001, mt5_login=0, mt5_password="t")
        assert s.max_risk_per_trade == 0.001

    def test_max_risk_per_trade_boundary(self):
        """Test maximum acceptable risk per trade (5%)"""
        s = Settings(max_risk_per_trade=0.05, mt5_login=0, mt5_password="t")
        assert s.max_risk_per_trade == 0.05

    def test_empty_trade_sessions(self):
        """Test that empty trade_sessions returns empty list"""
        s = Settings(trade_sessions="", mt5_login=0, mt5_password="t")
        sessions = s.active_sessions
        assert sessions == []

    def test_invalid_trade_sessions_ignored(self):
        """Test that invalid session names are ignored (graceful degradation)"""
        s = Settings(
            trade_sessions="london,invalid,tokyo,newyork",
            mt5_login=0,
            mt5_password="t"
        )
        sessions = s.active_sessions
        assert TradingSession.LONDON in sessions
        assert TradingSession.NEWYORK in sessions
        assert len(sessions) == 2  # Only valid sessions included

    def test_custom_database_url(self):
        """Test that custom database URL is used when provided"""
        custom_url = "postgresql://custom:password@custom-host:5433/custom_db"
        s = Settings(
            database_url=custom_url,
            mt5_login=0,
            mt5_password="t"
        )
        assert s.db_url == custom_url

    def test_custom_redis_url(self):
        """Test that custom Redis URL is used when provided"""
        custom_url = "redis://custom-host:6380/1"
        s = Settings(
            redis_url=custom_url,
            mt5_login=0,
            mt5_password="t"
        )
        assert s.redis_connection_url == custom_url


# ---------------------------------------------------------------------------
# Environment Detection (Added for Complete Coverage)
# ---------------------------------------------------------------------------
# EXTRA: Environment detection tests ensure the application correctly identifies
# its runtime environment, which affects critical behaviors like safety warnings

class TestEnvironmentDetection:
    """Tests for environment detection and properties."""

    def test_development_environment(self):
        """Test development environment properties"""
        s = Settings(
            app_env=Environment.DEVELOPMENT,
            mt5_login=0,
            mt5_password="t"
        )
        assert s.is_production is False
        assert s.app_env == Environment.DEVELOPMENT

    def test_staging_environment(self):
        """Test staging environment properties"""
        s = Settings(
            app_env=Environment.STAGING,
            mt5_login=0,
            mt5_password="t"
        )
        assert s.is_production is False
        assert s.app_env == Environment.STAGING

    def test_production_environment(self):
        """Test production environment properties"""
        s = Settings(
            app_env=Environment.PRODUCTION,
            mt5_login=0,
            mt5_password="t"
        )
        assert s.is_production is True
        assert s.app_env == Environment.PRODUCTION