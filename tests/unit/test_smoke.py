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

    # TODO: test_import_risk_checker
    # TODO: test_import_risk_monitor
    # TODO: test_import_database_models
