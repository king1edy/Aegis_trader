"""
Database Models
===============
SQLAlchemy models for the trading automation system.
Using TimescaleDB for time-series data optimization.
"""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum as PyEnum
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


# =============================================================================
# Enums
# =============================================================================

class OrderType(PyEnum):
    """Order type enumeration."""
    BUY = "BUY"
    SELL = "SELL"
    BUY_LIMIT = "BUY_LIMIT"
    SELL_LIMIT = "SELL_LIMIT"
    BUY_STOP = "BUY_STOP"
    SELL_STOP = "SELL_STOP"


class TradeStatus(PyEnum):
    """Trade lifecycle status."""
    PENDING = "PENDING"          # Signal generated, awaiting execution
    OPEN = "OPEN"                # Position is open
    PARTIALLY_CLOSED = "PARTIALLY_CLOSED"  # Partial take profit hit
    CLOSED = "CLOSED"            # Position fully closed
    CANCELLED = "CANCELLED"      # Order cancelled before fill
    REJECTED = "REJECTED"        # Order rejected by broker


class TradeOutcome(PyEnum):
    """Trade result classification."""
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"


class SignalSource(PyEnum):
    """Source of trading signal."""
    HULL_SUITE = "HULL_SUITE"
    ASIAN_BREAKOUT = "ASIAN_BREAKOUT"
    INSTITUTIONAL_FLOW = "INSTITUTIONAL_FLOW"
    MANUAL = "MANUAL"  # Should rarely be used!
    WEBHOOK = "WEBHOOK"


# =============================================================================
# Core Models
# =============================================================================

class Trade(Base):
    """
    Main trade record.
    
    Stores complete trade lifecycle from signal to closure.
    """
    __tablename__ = "trades"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Broker reference
    ticket = Column(Integer, unique=True, index=True, nullable=True)
    
    # Trade identification
    symbol = Column(String(20), nullable=False, index=True)
    order_type = Column(Enum(OrderType), nullable=False)
    status = Column(Enum(TradeStatus), default=TradeStatus.PENDING, nullable=False, index=True)
    
    # Signal information
    signal_source = Column(Enum(SignalSource), nullable=False)
    strategy_name = Column(String(100), nullable=False)
    signal_time = Column(DateTime(timezone=True), nullable=False)
    
    # Entry details
    entry_price = Column(Numeric(12, 5), nullable=True)
    requested_entry_price = Column(Numeric(12, 5), nullable=True)
    entry_time = Column(DateTime(timezone=True), nullable=True)
    
    # Position sizing
    lot_size = Column(Numeric(8, 2), nullable=False)
    initial_lot_size = Column(Numeric(8, 2), nullable=False)  # Before partial closes
    
    # Risk management levels
    stop_loss = Column(Numeric(12, 5), nullable=False)
    take_profit_1 = Column(Numeric(12, 5), nullable=True)  # First target (e.g., 1:1)
    take_profit_2 = Column(Numeric(12, 5), nullable=True)  # Second target (e.g., 1:2)
    take_profit_final = Column(Numeric(12, 5), nullable=True)  # Final target
    trailing_stop = Column(Numeric(12, 5), nullable=True)
    
    # Exit details
    exit_price = Column(Numeric(12, 5), nullable=True)
    exit_time = Column(DateTime(timezone=True), nullable=True)
    exit_reason = Column(String(100), nullable=True)  # SL, TP1, TP2, Manual, etc.
    
    # Results
    profit_loss = Column(Numeric(12, 2), nullable=True)
    profit_loss_pips = Column(Numeric(10, 1), nullable=True)
    profit_loss_percent = Column(Numeric(6, 4), nullable=True)
    outcome = Column(Enum(TradeOutcome), nullable=True)
    
    # Risk metrics at time of trade
    risk_amount = Column(Numeric(12, 2), nullable=True)  # Dollar risk
    risk_percent = Column(Numeric(6, 4), nullable=True)  # % of account risked
    risk_reward_planned = Column(Numeric(6, 2), nullable=True)
    risk_reward_actual = Column(Numeric(6, 2), nullable=True)
    
    # Account state at trade time
    account_balance_before = Column(Numeric(14, 2), nullable=True)
    account_balance_after = Column(Numeric(14, 2), nullable=True)
    
    # Market context (stored as JSON for flexibility)
    market_context = Column(JSONB, nullable=True)
    # Example: {"atr": 15.5, "adx": 25.3, "trend": "bullish", "session": "london"}
    
    # Strategy-specific data
    strategy_data = Column(JSONB, nullable=True)
    # Example: {"hull_color": "green", "entry_trigger": "color_change"}
    
    # Audit fields
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # Behavioral tracking
    was_manual_intervention = Column(Boolean, default=False)
    intervention_notes = Column(Text, nullable=True)
    
    # Relationships
    partial_closes = relationship("PartialClose", back_populates="trade", cascade="all, delete-orphan")
    modifications = relationship("TradeModification", back_populates="trade", cascade="all, delete-orphan")
    
    # Indexes for common queries
    __table_args__ = (
        Index("ix_trades_symbol_status", "symbol", "status"),
        Index("ix_trades_entry_time", "entry_time"),
        Index("ix_trades_strategy", "strategy_name", "signal_time"),
    )
    
    def __repr__(self) -> str:
        return f"<Trade {self.ticket} {self.symbol} {self.order_type.value} {self.status.value}>"


class PartialClose(Base):
    """
    Records partial position closures (e.g., at TP1).
    """
    __tablename__ = "partial_closes"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trades.id"), nullable=False)
    
    close_time = Column(DateTime(timezone=True), nullable=False)
    close_price = Column(Numeric(12, 5), nullable=False)
    lots_closed = Column(Numeric(8, 2), nullable=False)
    profit_loss = Column(Numeric(12, 2), nullable=False)
    reason = Column(String(50), nullable=False)  # TP1, TP2, Manual, etc.
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship
    trade = relationship("Trade", back_populates="partial_closes")


class TradeModification(Base):
    """
    Audit trail for trade modifications (SL/TP changes).
    """
    __tablename__ = "trade_modifications"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trades.id"), nullable=False)
    
    modification_time = Column(DateTime(timezone=True), nullable=False)
    field_modified = Column(String(50), nullable=False)  # stop_loss, take_profit, etc.
    old_value = Column(Numeric(12, 5), nullable=True)
    new_value = Column(Numeric(12, 5), nullable=True)
    reason = Column(String(200), nullable=True)
    was_automatic = Column(Boolean, default=True)  # False if manual intervention
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship
    trade = relationship("Trade", back_populates="modifications")


# =============================================================================
# Market Data Models
# =============================================================================

class PriceBar(Base):
    """
    OHLCV price data.
    
    This table should be converted to a TimescaleDB hypertable for
    efficient time-series queries.
    """
    __tablename__ = "price_bars"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)  # M1, M5, M15, H1, H4, D1
    timestamp = Column(DateTime(timezone=True), nullable=False)
    
    open = Column(Numeric(12, 5), nullable=False)
    high = Column(Numeric(12, 5), nullable=False)
    low = Column(Numeric(12, 5), nullable=False)
    close = Column(Numeric(12, 5), nullable=False)
    volume = Column(Integer, nullable=False)
    tick_volume = Column(Integer, nullable=True)
    spread = Column(Integer, nullable=True)
    
    __table_args__ = (
        UniqueConstraint("symbol", "timeframe", "timestamp", name="uq_price_bar"),
        Index("ix_price_bars_symbol_tf_time", "symbol", "timeframe", "timestamp"),
    )


# =============================================================================
# Account & Performance Models
# =============================================================================

class AccountSnapshot(Base):
    """
    Periodic snapshots of account state.
    
    Taken at regular intervals and after each trade for tracking.
    """
    __tablename__ = "account_snapshots"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    
    # Account values
    balance = Column(Numeric(14, 2), nullable=False)
    equity = Column(Numeric(14, 2), nullable=False)
    margin = Column(Numeric(14, 2), nullable=False)
    free_margin = Column(Numeric(14, 2), nullable=False)
    margin_level = Column(Numeric(10, 2), nullable=True)  # Percentage
    
    # Floating P/L
    floating_pl = Column(Numeric(14, 2), nullable=False)
    
    # Position counts
    open_positions = Column(Integer, default=0)
    open_orders = Column(Integer, default=0)
    
    # Daily metrics
    daily_pl = Column(Numeric(14, 2), nullable=True)
    daily_trades = Column(Integer, default=0)
    daily_wins = Column(Integer, default=0)
    daily_losses = Column(Integer, default=0)
    
    # Drawdown tracking
    peak_equity = Column(Numeric(14, 2), nullable=True)
    current_drawdown = Column(Numeric(14, 2), nullable=True)
    current_drawdown_percent = Column(Numeric(6, 4), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index("ix_account_snapshots_time", "timestamp"),
    )


class DailyPerformance(Base):
    """
    Aggregated daily performance metrics.
    """
    __tablename__ = "daily_performance"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    date = Column(DateTime(timezone=True), nullable=False, unique=True, index=True)
    
    # Trade counts
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    breakeven_trades = Column(Integer, default=0)
    
    # Profit/Loss
    gross_profit = Column(Numeric(14, 2), default=0)
    gross_loss = Column(Numeric(14, 2), default=0)
    net_profit = Column(Numeric(14, 2), default=0)
    
    # Account values (end of day)
    starting_balance = Column(Numeric(14, 2), nullable=True)
    ending_balance = Column(Numeric(14, 2), nullable=True)
    
    # Performance metrics
    win_rate = Column(Numeric(6, 4), nullable=True)
    profit_factor = Column(Numeric(8, 2), nullable=True)
    average_win = Column(Numeric(14, 2), nullable=True)
    average_loss = Column(Numeric(14, 2), nullable=True)
    largest_win = Column(Numeric(14, 2), nullable=True)
    largest_loss = Column(Numeric(14, 2), nullable=True)
    
    # Risk metrics
    max_drawdown = Column(Numeric(14, 2), nullable=True)
    max_drawdown_percent = Column(Numeric(6, 4), nullable=True)
    
    # Strategy breakdown (JSON)
    strategy_breakdown = Column(JSONB, nullable=True)
    # Example: {"HULL_SUITE": {"trades": 5, "pnl": 125.50}, ...}
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# =============================================================================
# System & Audit Models
# =============================================================================

class SystemEvent(Base):
    """
    System-level events for audit and debugging.
    """
    __tablename__ = "system_events"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    severity = Column(String(20), nullable=False)  # INFO, WARNING, ERROR, CRITICAL
    
    message = Column(Text, nullable=False)
    details = Column(JSONB, nullable=True)
    
    # Related entities (optional)
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trades.id"), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index("ix_system_events_type_time", "event_type", "timestamp"),
    )


class TradingPause(Base):
    """
    Records when trading was paused and why.
    
    Critical for behavioral tracking - helps identify patterns.
    """
    __tablename__ = "trading_pauses"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True), nullable=True)
    
    reason = Column(String(100), nullable=False)
    # Examples: "max_drawdown", "consecutive_losses", "daily_limit", "manual_pause"
    
    trigger_value = Column(Numeric(14, 2), nullable=True)  # What triggered it
    threshold_value = Column(Numeric(14, 2), nullable=True)  # What the limit was
    
    was_automatic = Column(Boolean, default=True)
    notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index("ix_trading_pauses_time", "start_time", "end_time"),
    )


class Signal(Base):
    """
    Records all generated signals (whether executed or not).
    
    Useful for analyzing missed opportunities and signal quality.
    """
    __tablename__ = "signals"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    strategy_name = Column(String(100), nullable=False)
    signal_source = Column(Enum(SignalSource), nullable=False)
    
    direction = Column(String(10), nullable=False)  # BUY or SELL
    strength = Column(Numeric(5, 2), nullable=True)  # Signal strength score
    
    # Proposed trade parameters
    proposed_entry = Column(Numeric(12, 5), nullable=True)
    proposed_sl = Column(Numeric(12, 5), nullable=True)
    proposed_tp = Column(Numeric(12, 5), nullable=True)
    proposed_lot_size = Column(Numeric(8, 2), nullable=True)
    
    # Outcome
    was_executed = Column(Boolean, default=False)
    rejection_reason = Column(String(200), nullable=True)
    # Examples: "risk_limit", "session_filter", "already_in_position", etc.
    
    trade_id = Column(UUID(as_uuid=True), ForeignKey("trades.id"), nullable=True)
    
    # Market conditions at signal time
    market_context = Column(JSONB, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index("ix_signals_strategy_time", "strategy_name", "timestamp"),
    )
