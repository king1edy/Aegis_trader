"""
Backtesting Module

Provides comprehensive backtesting capabilities for trading strategies.

Components:
- BacktestEngine: Event-driven simulation engine
- BacktestDataProvider: Multi-timeframe historical data provider
- MTFTRStrategySimulator: MTFTR strategy implementation for backtesting

Usage:
    from backtesting import run_backtest
    results = run_backtest(symbol="XAUUSD", start_date="2020-01-01", end_date="2025-12-31")
"""

from backtesting.engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestTrade,
    BacktestResult,
    PositionState,
    print_results
)

from backtesting.data_provider import (
    BacktestDataProvider,
    MultiTimeframeBar
)

from backtesting.strategy_simulator import (
    MTFTRStrategySimulator,
    MTFTRBacktestConfig,
    TrendState
)

from backtesting.run_backtest import run_backtest

__all__ = [
    # Engine
    'BacktestEngine',
    'BacktestConfig',
    'BacktestTrade',
    'BacktestResult',
    'PositionState',
    'print_results',
    # Data Provider
    'BacktestDataProvider',
    'MultiTimeframeBar',
    # Strategy Simulator
    'MTFTRStrategySimulator',
    'MTFTRBacktestConfig',
    'TrendState',
    # Runner
    'run_backtest'
]
