"""
Database Module
===============
Database models, repositories, and connection management.
"""

from src.database.models import (
    Base,
    Trade,
    TradeStatus,
    TradeOutcome,
    OrderType,
    SignalSource,
    PartialClose,
    TradeModification,
    PriceBar,
    AccountSnapshot,
    DailyPerformance,
    SystemEvent,
    TradingPause,
    Signal,
)
from src.database.repository import (
    engine,
    get_session,
    init_database,
    close_database,
    TradeRepository,
    PerformanceRepository,
    SignalRepository,
    SystemRepository,
)

__all__ = [
    # Models
    "Base",
    "Trade",
    "TradeStatus",
    "TradeOutcome",
    "OrderType",
    "SignalSource",
    "PartialClose",
    "TradeModification",
    "PriceBar",
    "AccountSnapshot",
    "DailyPerformance",
    "SystemEvent",
    "TradingPause",
    "Signal",
    # Database
    "engine",
    "get_session",
    "init_database",
    "close_database",
    # Repositories
    "TradeRepository",
    "PerformanceRepository",
    "SignalRepository",
    "SystemRepository",
]
