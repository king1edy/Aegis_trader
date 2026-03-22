"""
Core Module
===========
Core utilities, configuration, and shared components.
"""

from core.config import Settings, get_settings, settings
from core.logging_config import (
    get_logger,
    setup_logging,
    Loggers,
    LogMessages,
    TradingContextLogger
)
from core.exceptions import (
    TradingSystemError,
    ConfigurationError,
    BrokerConnectionError,
    MT5ConnectionError,
    TradingError,
    OrderExecutionError,
    RiskManagementError,
    RiskLimitExceededError,
    MaxDrawdownExceededError,
    StrategyError,
    ManualOverrideAttemptError,
)

__all__ = [
    # Configuration
    "Settings",
    "get_settings",
    "settings",
    # Logging
    "get_logger",
    "setup_logging",
    "Loggers",
    "LogMessages",
    "TradingContextLogger",
    # Exceptions
    "TradingSystemError",
    "ConfigurationError",
    "BrokerConnectionError",
    "MT5ConnectionError",
    "TradingError",
    "OrderExecutionError",
    "RiskManagementError",
    "RiskLimitExceededError",
    "MaxDrawdownExceededError",
    "StrategyError",
    "ManualOverrideAttemptError",
]
