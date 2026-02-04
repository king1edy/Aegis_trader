"""
Execution Module
================
Order execution and broker connectivity.
"""

from src.execution.mt5_connector import (
    BrokerInterface,
    MT5Connector,
    DemoConnector,
    get_broker_connector,
    OrderDirection,
    SymbolInfo,
    AccountInfo,
    Position,
    PriceData,
    Tick,
    TradeResult,
)
from src.execution.broker_factory import create_broker, get_broker_status_message
from src.execution.mt5_api_client import MT5APIClient
from src.execution.paper_broker import PaperTradingBroker

__all__ = [
    # Core interfaces
    "BrokerInterface",
    # Broker implementations
    "MT5Connector",
    "DemoConnector",
    "MT5APIClient",
    "PaperTradingBroker",
    # Factory
    "create_broker",
    "get_broker_connector",  # Legacy
    "get_broker_status_message",
    # Data types
    "OrderDirection",
    "SymbolInfo",
    "AccountInfo",
    "Position",
    "PriceData",
    "Tick",
    "TradeResult",
]
