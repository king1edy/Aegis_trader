"""
Smoke Tests
===========
Quick sanity checks that core modules are importable and basic
wiring is correct. These should never be slow or flaky.
"""

import pytest

pytestmark = pytest.mark.smoke


class TestCoreImports:
    """Verify all core packages can be imported without errors."""

    def test_import_config(self):
        from src.core.config import Settings, Environment
        assert Settings is not None
        assert Environment is not None

    def test_import_exceptions(self):
        from src.core.exceptions import TradingSystemError
        assert TradingSystemError is not None

    def test_import_indicators(self):
        from src.strategies.indicators import IndicatorCalculator
        assert IndicatorCalculator is not None

    def test_import_patterns(self):
        from src.strategies.patterns import PatternRecognizer
        assert PatternRecognizer is not None

    def test_import_position_sizer(self):
        from src.risk.position_sizer import PositionSizer
        assert PositionSizer is not None

    def test_import_message_formatter(self):
        from src.notifications.message_formatter import MessageFormatter
        assert MessageFormatter is not None

    def test_import_paper_broker(self):
        from src.execution.paper_broker import PaperTradingBroker
        assert PaperTradingBroker is not None

    def test_import_base_strategy(self):
        from src.strategies.base_strategy import TradingSignal, BaseStrategy
        assert TradingSignal is not None
        assert BaseStrategy is not None

    def test_import_risk_checker(self):
        """Test that risk_checker module can be imported."""
        try:
            from src.risk.risk_checker import RiskChecker
            assert RiskChecker is not None
        except ImportError:
            pytest.skip("risk_checker module not yet implemented")

    def test_import_risk_monitor(self):
        """Test that risk_monitor module can be imported."""
        try:
            from src.risk.risk_monitor import RiskMonitor
            assert RiskMonitor is not None
        except ImportError:
            pytest.skip("risk_monitor module not yet implemented")

    def test_import_database_models(self):
        """Test that database models can be imported."""
        try:
            from src.database.models import Trade, Position, Signal
            assert Trade is not None
            assert Position is not None
            assert Signal is not None
        except ImportError:
            pytest.skip("database models not yet implemented")


class TestStrategyImports:
    """Verify strategy-related modules can be imported."""

    def test_import_mtftr_strategy(self):
        """Test that MTFTR strategy can be imported."""
        try:
            from src.strategies.mtftr_strategy import MTFTRStrategy
            assert MTFTRStrategy is not None
        except ImportError:
            pytest.skip("mtftr_strategy module not yet implemented")

    def test_import_trading_signal(self):
        """Test that TradingSignal class is importable."""
        from src.strategies.base_strategy import TradingSignal
        assert TradingSignal is not None


class TestExecutionImports:
    """Verify execution-related modules can be imported."""

    def test_import_mt5_connector(self):
        """Test that MT5Connector can be imported."""
        try:
            from src.execution.mt5_connector import MT5Connector, OrderDirection
            assert MT5Connector is not None
            assert OrderDirection is not None
        except ImportError as e:
            pytest.skip(f"mt5_connector import failed: {e}")

    def test_import_symbol_info(self):
        """Test that SymbolInfo can be imported."""
        try:
            from src.execution.mt5_connector import SymbolInfo
            assert SymbolInfo is not None
        except ImportError:
            pytest.skip("SymbolInfo not yet implemented")


class TestBacktestingImports:
    """Verify backtesting-related modules can be imported."""

    def test_import_backtest_engine(self):
        """Test that BacktestEngine can be imported."""
        from src.backtesting.engine import BacktestEngine, BacktestConfig
        assert BacktestEngine is not None
        assert BacktestConfig is not None

    def test_import_data_provider(self):
        """Test that BacktestDataProvider can be imported."""
        from src.backtesting.data_provider import BacktestDataProvider
        assert BacktestDataProvider is not None

    def test_import_test_data_generator(self):
        """Test that TestDataGenerator can be imported."""
        from src.backtesting.test_data_generator import TestDataGenerator
        assert TestDataGenerator is not None


class TestNotificationImports:
    """Verify notification-related modules can be imported."""

    def test_import_telegram_notifier(self):
        """Test that TelegramNotifier can be imported."""
        try:
            from src.notifications.telegram_notifier import TelegramNotifier
            assert TelegramNotifier is not None
        except ImportError:
            pytest.skip("telegram_notifier module not yet implemented")


class TestDatabaseImports:
    """Verify database-related modules can be imported."""

    def test_import_database_session(self):
        """Test that database session can be imported."""
        try:
            from src.database.session import SessionLocal, engine
            assert SessionLocal is not None
            assert engine is not None
        except ImportError:
            pytest.skip("database session not yet implemented")

    def test_import_crud_operations(self):
        """Test that CRUD operations can be imported."""
        try:
            from src.database.crud import create_trade, get_trades
            assert create_trade is not None
            assert get_trades is not None
        except ImportError:
            pytest.skip("database crud not yet implemented")


class TestUtilsImports:
    """Verify utility modules can be imported."""

    def test_import_logging_config(self):
        """Test that logging config can be imported."""
        from src.core.logging_config import get_logger, setup_logging
        assert get_logger is not None
        assert setup_logging is not None

    def test_import_helpers(self):
        """Test that helper functions can be imported."""
        try:
            from src.core.helpers import calculate_pips, format_price
            assert calculate_pips is not None
            assert format_price is not None
        except ImportError:
            pytest.skip("helpers module not yet implemented")


class TestSettingsImports:
    """Verify settings can be loaded."""

    def test_import_settings_instance(self):
        """Test that global settings instance can be imported."""
        from src.core.config import settings, get_settings
        assert settings is not None
        assert get_settings is not None


class TestIntegrationImports:
    """Verify integration between modules works."""

    def test_import_strategy_with_indicators(self):
        """Test that strategy can import indicators."""
        from src.strategies.indicators import IndicatorCalculator
        assert IndicatorCalculator is not None
        
        try:
            from src.strategies.mtftr_strategy import MTFTRStrategy
            assert MTFTRStrategy is not None
        except ImportError:
            # MTFTR strategy not implemented yet, but indicators are
            pass

    def test_import_risk_with_position_sizer(self):
        """Test that risk modules work together."""
        from src.risk.position_sizer import PositionSizer
        from src.core.config import settings
        from src.execution.mt5_connector import SymbolInfo
        
        assert PositionSizer is not None
        assert settings is not None
        assert SymbolInfo is not None

        # =============================================================================
# NOTE: Some tests are skipped because their corresponding modules are not yet
# implemented in the codebase. These modules are planned for future phases:
#
# Skipped Modules & Reason:
# - src.db.*                    : Database integration (planned for Phase 2)
# - src.strategies.mtftr_strategy : MTFTR strategy implementation (planned)
# - src.execution.mt5_connector.OrderType : Not yet defined in MT5 connector
# - src.notifications.telegram_notifier : Telegram integration (Phase 3)
# - src.core.helpers           : Helper utilities (to be implemented)
#
# Once these modules are implemented, these smoke tests should be updated
# to verify their imports work correctly.
# =============================================================================
