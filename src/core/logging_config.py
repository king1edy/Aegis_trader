"""
Logging Configuration
=====================
Structured logging setup using structlog for consistent, parseable logs.
All logs include context like trade IDs, symbols, and timestamps.
"""

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from structlog.types import EventDict, Processor

from src.core.config import settings


def add_timestamp(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add ISO timestamp to every log entry."""
    event_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
    return event_dict


def add_app_context(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add application context to every log entry."""
    event_dict["app"] = settings.app_name
    event_dict["env"] = settings.app_env.value
    return event_dict


def censor_sensitive_data(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Remove or mask sensitive data from logs."""
    sensitive_keys = ["password", "token", "secret", "api_key", "authorization"]
    
    for key in list(event_dict.keys()):
        key_lower = key.lower()
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            event_dict[key] = "***REDACTED***"
    
    return event_dict


class TradingContextLogger:
    """
    Context manager for adding trading context to logs.
    
    Usage:
        with TradingContextLogger(symbol="XAUUSD", trade_id="12345"):
            logger.info("Opening position", lot_size=0.01)
    """
    
    _context: Dict[str, Any] = {}
    
    def __init__(
        self,
        symbol: Optional[str] = None,
        trade_id: Optional[str] = None,
        strategy: Optional[str] = None,
        **kwargs: Any
    ):
        self.new_context = {}
        if symbol:
            self.new_context["symbol"] = symbol
        if trade_id:
            self.new_context["trade_id"] = trade_id
        if strategy:
            self.new_context["strategy"] = strategy
        self.new_context.update(kwargs)
        self.old_context: Dict[str, Any] = {}
    
    def __enter__(self) -> "TradingContextLogger":
        self.old_context = TradingContextLogger._context.copy()
        TradingContextLogger._context.update(self.new_context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        TradingContextLogger._context = self.old_context


def add_trading_context(
    logger: logging.Logger, method_name: str, event_dict: EventDict
) -> EventDict:
    """Add current trading context to log entry."""
    event_dict.update(TradingContextLogger._context)
    return event_dict


def setup_logging(
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    json_format: bool = False
) -> structlog.BoundLogger:
    """
    Configure structured logging for the application.
    
    Args:
        log_level: Override log level from settings
        log_file: Optional file path for logging
        json_format: Use JSON formatting (recommended for production)
    
    Returns:
        Configured structlog logger
    """
    level = log_level or settings.log_level
    
    # Define shared processors
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        add_timestamp,
        add_app_context,
        add_trading_context,
        censor_sensitive_data,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Configure structlog
    if json_format or settings.is_production:
        # JSON format for production - easy to parse with log aggregators
        renderer = structlog.processors.JSONRenderer()
    else:
        # Pretty console output for development
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback
        )
    
    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # File handler (optional)
    handlers = [console_handler]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers = handlers
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Reduce noise from third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return structlog.get_logger()


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """
    Get a logger instance with optional name binding.
    
    Args:
        name: Optional module/component name to bind to logger
    
    Returns:
        Bound structlog logger
    """
    logger = structlog.get_logger()
    if name:
        logger = logger.bind(component=name)
    return logger


# Pre-configured loggers for common components
class Loggers:
    """Pre-configured loggers for different components."""
    
    @staticmethod
    def trading() -> structlog.BoundLogger:
        """Logger for trading operations."""
        return get_logger("trading")
    
    @staticmethod
    def risk() -> structlog.BoundLogger:
        """Logger for risk management."""
        return get_logger("risk")
    
    @staticmethod
    def execution() -> structlog.BoundLogger:
        """Logger for order execution."""
        return get_logger("execution")
    
    @staticmethod
    def data() -> structlog.BoundLogger:
        """Logger for market data."""
        return get_logger("data")
    
    @staticmethod
    def strategy() -> structlog.BoundLogger:
        """Logger for strategy calculations."""
        return get_logger("strategy")
    
    @staticmethod
    def api() -> structlog.BoundLogger:
        """Logger for API operations."""
        return get_logger("api")


# Example usage and log message templates
class LogMessages:
    """
    Standard log message templates for consistency.
    Use these to ensure consistent logging across the application.
    """
    
    # Trading events
    TRADE_SIGNAL = "Trade signal generated"
    TRADE_OPENED = "Trade opened"
    TRADE_CLOSED = "Trade closed"
    TRADE_MODIFIED = "Trade modified"
    TRADE_REJECTED = "Trade rejected"
    
    # Risk events
    RISK_CHECK_PASSED = "Risk check passed"
    RISK_CHECK_FAILED = "Risk check failed"
    DAILY_LIMIT_REACHED = "Daily trade limit reached"
    DRAWDOWN_WARNING = "Drawdown warning threshold reached"
    DRAWDOWN_CRITICAL = "Critical drawdown - trading paused"
    
    # System events
    SYSTEM_STARTED = "Trading system started"
    SYSTEM_STOPPED = "Trading system stopped"
    CONNECTION_ESTABLISHED = "Broker connection established"
    CONNECTION_LOST = "Broker connection lost"
    CONNECTION_RESTORED = "Broker connection restored"
    
    # Data events
    DATA_RECEIVED = "Market data received"
    DATA_ERROR = "Market data error"
    INDICATOR_CALCULATED = "Indicator calculated"


# Initialize logging on module import
logger = setup_logging(
    log_file=Path("logs/trading.log") if not settings.is_production else None
)
