"""
Custom Exceptions
=================
Centralized exception definitions for the trading automation system.
Using specific exceptions helps with error handling and debugging.
"""

from typing import Any, Optional


class TradingSystemError(Exception):
    """Base exception for all trading system errors."""
    
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


# =============================================================================
# Configuration Errors
# =============================================================================

class ConfigurationError(TradingSystemError):
    """Raised when there's a configuration problem."""
    pass


class InvalidSettingsError(ConfigurationError):
    """Raised when settings validation fails."""
    pass


# =============================================================================
# Connection Errors
# =============================================================================

class ConnectionError(TradingSystemError):
    """Base class for connection-related errors."""
    pass


class BrokerConnectionError(ConnectionError):
    """Raised when unable to connect to the broker."""
    pass


class MT5ConnectionError(BrokerConnectionError):
    """Specific to MetaTrader 5 connection issues."""
    pass


class DatabaseConnectionError(ConnectionError):
    """Raised when unable to connect to the database."""
    pass


class RedisConnectionError(ConnectionError):
    """Raised when unable to connect to Redis."""
    pass


# =============================================================================
# Trading Errors
# =============================================================================

class TradingError(TradingSystemError):
    """Base class for trading-related errors."""
    pass


class InsufficientMarginError(TradingError):
    """Raised when account has insufficient margin for a trade."""
    
    def __init__(
        self,
        required_margin: float,
        available_margin: float,
        symbol: str
    ):
        super().__init__(
            f"Insufficient margin for {symbol}",
            {
                "required_margin": required_margin,
                "available_margin": available_margin,
                "symbol": symbol
            }
        )


class OrderExecutionError(TradingError):
    """Raised when an order fails to execute."""
    
    def __init__(
        self,
        order_type: str,
        symbol: str,
        error_code: Optional[int] = None,
        broker_message: Optional[str] = None
    ):
        super().__init__(
            f"Failed to execute {order_type} order for {symbol}",
            {
                "order_type": order_type,
                "symbol": symbol,
                "error_code": error_code,
                "broker_message": broker_message
            }
        )


class InvalidOrderError(TradingError):
    """Raised when order parameters are invalid."""
    pass


class PositionNotFoundError(TradingError):
    """Raised when trying to modify/close a non-existent position."""
    
    def __init__(self, ticket: int):
        super().__init__(
            f"Position with ticket {ticket} not found",
            {"ticket": ticket}
        )


class SymbolNotFoundError(TradingError):
    """Raised when a trading symbol is not available."""
    
    def __init__(self, symbol: str):
        super().__init__(
            f"Symbol {symbol} not found or not available for trading",
            {"symbol": symbol}
        )


# =============================================================================
# Risk Management Errors
# =============================================================================

class RiskManagementError(TradingSystemError):
    """Base class for risk management errors."""
    pass


class RiskLimitExceededError(RiskManagementError):
    """Raised when a trade would exceed risk limits."""
    
    def __init__(
        self,
        limit_type: str,
        current_value: float,
        limit_value: float
    ):
        super().__init__(
            f"Risk limit exceeded: {limit_type}",
            {
                "limit_type": limit_type,
                "current_value": current_value,
                "limit_value": limit_value
            }
        )


class MaxDrawdownExceededError(RiskManagementError):
    """Raised when maximum drawdown is exceeded."""
    
    def __init__(self, current_drawdown: float, max_drawdown: float):
        super().__init__(
            "Maximum drawdown exceeded - trading paused",
            {
                "current_drawdown": current_drawdown,
                "max_drawdown": max_drawdown
            }
        )


class DailyLossLimitError(RiskManagementError):
    """Raised when daily loss limit is reached."""
    
    def __init__(self, daily_loss: float, daily_limit: float):
        super().__init__(
            "Daily loss limit reached - no more trades today",
            {
                "daily_loss": daily_loss,
                "daily_limit": daily_limit
            }
        )


class MaxTradesExceededError(RiskManagementError):
    """Raised when maximum trades per day is reached."""
    
    def __init__(self, trade_count: int, max_trades: int):
        super().__init__(
            "Maximum daily trades reached",
            {
                "trade_count": trade_count,
                "max_trades": max_trades
            }
        )


class ConsecutiveLossesError(RiskManagementError):
    """Raised when max consecutive losses reached."""
    
    def __init__(self, consecutive_losses: int, pause_hours: int):
        super().__init__(
            f"Max consecutive losses reached - trading paused for {pause_hours} hours",
            {
                "consecutive_losses": consecutive_losses,
                "pause_hours": pause_hours
            }
        )


# =============================================================================
# Strategy Errors
# =============================================================================

class StrategyError(TradingSystemError):
    """Base class for strategy-related errors."""
    pass


class InsufficientDataError(StrategyError):
    """Raised when there's not enough data for calculations."""
    pass


class IndicatorCalculationError(StrategyError):
    """Raised when indicator calculation fails."""
    
    def __init__(self, indicator_name: str, reason: str):
        super().__init__(
            f"Failed to calculate {indicator_name}: {reason}",
            {"indicator_name": indicator_name, "reason": reason}
        )


class SignalValidationError(StrategyError):
    """Raised when a trading signal fails validation."""
    pass


# =============================================================================
# Session Errors
# =============================================================================

class SessionError(TradingSystemError):
    """Base class for session-related errors."""
    pass


class OutsideTradingHoursError(SessionError):
    """Raised when trying to trade outside allowed sessions."""
    
    def __init__(self, current_time: str, allowed_sessions: list[str]):
        super().__init__(
            "Trading not allowed at this time",
            {
                "current_time": current_time,
                "allowed_sessions": allowed_sessions
            }
        )


class MarketClosedError(SessionError):
    """Raised when the market is closed."""
    
    def __init__(self, symbol: str, reason: str = "Market is closed"):
        super().__init__(
            f"Market closed for {symbol}: {reason}",
            {"symbol": symbol, "reason": reason}
        )


# =============================================================================
# Data Errors
# =============================================================================

class DataError(TradingSystemError):
    """Base class for data-related errors."""
    pass


class MarketDataError(DataError):
    """Raised when there's an issue with market data."""
    pass


class StaleDataError(DataError):
    """Raised when data is too old to be reliable."""
    
    def __init__(self, symbol: str, data_age_seconds: float, max_age_seconds: float):
        super().__init__(
            f"Stale data for {symbol}",
            {
                "symbol": symbol,
                "data_age_seconds": data_age_seconds,
                "max_age_seconds": max_age_seconds
            }
        )


class HistoricalDataError(DataError):
    """Raised when unable to fetch historical data."""
    pass


# =============================================================================
# Behavioral Safeguard Errors
# =============================================================================

class BehavioralSafeguardError(TradingSystemError):
    """
    Base class for behavioral safeguard violations.
    These exist to protect you from manual intervention!
    """
    pass


class ManualOverrideAttemptError(BehavioralSafeguardError):
    """Raised when manual override is attempted but disabled."""
    
    def __init__(self, action: str):
        super().__init__(
            f"Manual override attempted: {action}. "
            "Manual intervention is disabled to protect your capital. "
            "Trust the system!",
            {"action": action}
        )


class CooldownPeriodError(BehavioralSafeguardError):
    """Raised when trying to trade during cooldown period."""
    
    def __init__(self, remaining_minutes: int, reason: str):
        super().__init__(
            f"Trading paused: {reason}. {remaining_minutes} minutes remaining.",
            {
                "remaining_minutes": remaining_minutes,
                "reason": reason
            }
        )


# =============================================================================
# API Errors
# =============================================================================

class APIError(TradingSystemError):
    """Base class for API-related errors."""
    pass


class WebhookValidationError(APIError):
    """Raised when webhook validation fails."""
    pass


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, retry_after_seconds: int):
        super().__init__(
            "Rate limit exceeded",
            {"retry_after_seconds": retry_after_seconds}
        )


# =============================================================================
# Notification Errors
# =============================================================================

class NotificationError(TradingSystemError):
    """Base class for notification-related errors."""
    pass


class TelegramNotificationError(NotificationError):
    """Raised when Telegram notification fails."""
    
    def __init__(self, message: str, error_details: str = None):
        super().__init__(
            f"Telegram notification failed: {message}",
            {"error_details": error_details}
        )


