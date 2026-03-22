"""
Database Repository
===================
Data access layer for the trading system.
Provides clean interfaces for database operations.

All repositories accept an optional ``tenant_id`` for multi-tenant
isolation.  When supplied, every query is scoped to that tenant and
every new record gets the tenant_id set automatically.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import AsyncGenerator, List, Optional, Sequence
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from core.config import settings
from core.logging_config import get_logger
from database.models import (
    AccountSnapshot,
    Base,
    DailyPerformance,
    PartialClose,
    Signal,
    SystemEvent,
    Trade,
    TradeModification,
    TradeOutcome,
    TradeStatus,
    TradingPause,
)

logger = get_logger("database")


# =============================================================================
# Database Engine & Session
# =============================================================================

# Create async engine
engine = create_async_engine(
    settings.async_db_url,
    echo=settings.debug,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)

# Session factory
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Async context manager for database sessions.

    Usage:
        async with get_session() as session:
            # do database operations
            await session.commit()
    """
    async with async_session_factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_database() -> None:
    """Verify database connectivity.

    Schema creation is handled by Alembic migrations
    (``alembic upgrade head``).  This function only checks
    that the connection pool can reach the server.
    """
    async with engine.begin() as conn:
        await conn.execute(sa.text("SELECT 1"))
    logger.info("Database connection verified")


async def close_database() -> None:
    """Close database connections."""
    await engine.dispose()
    logger.info("Database connections closed")


# =============================================================================
# Tenant-aware base
# =============================================================================

class TenantMixin:
    """Base for repositories that scope queries by tenant_id."""

    def __init__(self, session: AsyncSession, tenant_id: Optional[UUID] = None):
        self.session = session
        self.tenant_id = tenant_id

    def _scope(self, query, model):
        """Append ``WHERE model.tenant_id == self.tenant_id`` when set."""
        if self.tenant_id is not None:
            return query.where(model.tenant_id == self.tenant_id)
        return query

    def _stamp(self, obj):
        """Set tenant_id on an ORM object before persisting."""
        if self.tenant_id is not None:
            obj.tenant_id = self.tenant_id
        return obj


# =============================================================================
# Trade Repository
# =============================================================================

class TradeRepository(TenantMixin):
    """Repository for trade-related database operations."""

    async def create(self, trade: Trade) -> Trade:
        """Create a new trade record."""
        self._stamp(trade)
        self.session.add(trade)
        await self.session.commit()
        await self.session.refresh(trade)
        logger.info("Trade created", trade_id=str(trade.id), ticket=trade.ticket)
        return trade

    async def get_by_id(self, trade_id: UUID) -> Optional[Trade]:
        """Get trade by ID (scoped to tenant)."""
        query = (
            select(Trade)
            .options(selectinload(Trade.partial_closes))
            .options(selectinload(Trade.modifications))
            .where(Trade.id == trade_id)
        )
        result = await self.session.execute(self._scope(query, Trade))
        return result.scalar_one_or_none()

    async def get_by_ticket(self, ticket: int) -> Optional[Trade]:
        """Get trade by broker ticket number."""
        query = (
            select(Trade)
            .options(selectinload(Trade.partial_closes))
            .where(Trade.ticket == ticket)
        )
        result = await self.session.execute(self._scope(query, Trade))
        return result.scalar_one_or_none()

    async def get_open_trades(self, symbol: Optional[str] = None) -> Sequence[Trade]:
        """Get all open trades, optionally filtered by symbol."""
        query = select(Trade).where(Trade.status == TradeStatus.OPEN)
        if symbol:
            query = query.where(Trade.symbol == symbol)
        result = await self.session.execute(
            self._scope(query, Trade).order_by(Trade.entry_time)
        )
        return result.scalars().all()

    async def get_recent_trades(
        self,
        limit: int = 50,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> Sequence[Trade]:
        """Get recent trades with optional filters."""
        query = select(Trade)
        conditions = []
        if symbol:
            conditions.append(Trade.symbol == symbol)
        if strategy:
            conditions.append(Trade.strategy_name == strategy)
        if conditions:
            query = query.where(and_(*conditions))

        result = await self.session.execute(
            self._scope(query, Trade).order_by(desc(Trade.signal_time)).limit(limit)
        )
        return result.scalars().all()

    async def get_trades_in_range(
        self,
        start_date: datetime,
        end_date: datetime,
        symbol: Optional[str] = None,
    ) -> Sequence[Trade]:
        """Get trades within a date range."""
        query = select(Trade).where(
            and_(Trade.entry_time >= start_date, Trade.entry_time <= end_date)
        )
        if symbol:
            query = query.where(Trade.symbol == symbol)
        result = await self.session.execute(
            self._scope(query, Trade).order_by(Trade.entry_time)
        )
        return result.scalars().all()

    async def get_today_trades(self) -> Sequence[Trade]:
        """Get all trades from today."""
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        query = select(Trade).where(Trade.signal_time >= today_start)
        result = await self.session.execute(
            self._scope(query, Trade).order_by(Trade.signal_time)
        )
        return result.scalars().all()

    async def count_today_trades(self) -> int:
        """Count trades opened today."""
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        query = select(func.count(Trade.id)).where(
            and_(
                Trade.entry_time >= today_start,
                Trade.status.in_([TradeStatus.OPEN, TradeStatus.CLOSED]),
            )
        )
        result = await self.session.execute(self._scope(query, Trade))
        return result.scalar() or 0

    async def get_consecutive_losses(self) -> int:
        """Get count of consecutive losses (most recent first)."""
        query = (
            select(Trade)
            .where(Trade.status == TradeStatus.CLOSED)
            .order_by(desc(Trade.exit_time))
            .limit(20)
        )
        result = await self.session.execute(self._scope(query, Trade))
        trades = result.scalars().all()

        consecutive = 0
        for trade in trades:
            if trade.outcome == TradeOutcome.LOSS:
                consecutive += 1
            else:
                break
        return consecutive

    async def update(self, trade: Trade) -> Trade:
        """Update a trade record."""
        await self.session.commit()
        await self.session.refresh(trade)
        return trade

    async def add_partial_close(self, partial: PartialClose) -> PartialClose:
        """Record a partial position close."""
        self._stamp(partial)
        self.session.add(partial)
        await self.session.commit()
        return partial

    async def add_modification(self, modification: TradeModification) -> TradeModification:
        """Record a trade modification."""
        self._stamp(modification)
        self.session.add(modification)
        await self.session.commit()
        return modification


# =============================================================================
# Performance Repository
# =============================================================================

class PerformanceRepository(TenantMixin):
    """Repository for performance and account metrics."""

    async def save_snapshot(self, snapshot: AccountSnapshot) -> AccountSnapshot:
        """Save an account snapshot."""
        self._stamp(snapshot)
        self.session.add(snapshot)
        await self.session.commit()
        return snapshot

    async def get_latest_snapshot(self) -> Optional[AccountSnapshot]:
        """Get the most recent account snapshot."""
        query = (
            select(AccountSnapshot)
            .order_by(desc(AccountSnapshot.timestamp))
            .limit(1)
        )
        result = await self.session.execute(
            self._scope(query, AccountSnapshot)
        )
        return result.scalar_one_or_none()

    async def get_peak_equity(self) -> Optional[Decimal]:
        """Get the peak equity value."""
        query = select(func.max(AccountSnapshot.equity))
        result = await self.session.execute(
            self._scope(query, AccountSnapshot)
        )
        return result.scalar()

    async def get_daily_performance(
        self,
        date: Optional[datetime] = None,
    ) -> Optional[DailyPerformance]:
        """Get performance for a specific date or today."""
        if date is None:
            date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        query = select(DailyPerformance).where(DailyPerformance.date == date)
        result = await self.session.execute(
            self._scope(query, DailyPerformance)
        )
        return result.scalar_one_or_none()

    async def save_daily_performance(
        self,
        performance: DailyPerformance,
    ) -> DailyPerformance:
        """Save or update daily performance record."""
        self._stamp(performance)
        existing = await self.get_daily_performance(performance.date)
        if existing:
            for key, value in performance.__dict__.items():
                if not key.startswith("_") and key != "id":
                    setattr(existing, key, value)
            await self.session.commit()
            return existing
        else:
            self.session.add(performance)
            await self.session.commit()
            return performance

    async def get_performance_range(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Sequence[DailyPerformance]:
        """Get daily performance records for a date range."""
        query = select(DailyPerformance).where(
            and_(
                DailyPerformance.date >= start_date,
                DailyPerformance.date <= end_date,
            )
        )
        result = await self.session.execute(
            self._scope(query, DailyPerformance).order_by(DailyPerformance.date)
        )
        return result.scalars().all()

    async def calculate_current_drawdown(self) -> tuple[Decimal, Decimal]:
        """Calculate current drawdown from peak.

        Returns (drawdown_amount, drawdown_percent).
        """
        peak = await self.get_peak_equity()
        latest = await self.get_latest_snapshot()

        if not peak or not latest:
            return Decimal("0"), Decimal("0")

        drawdown = peak - latest.equity
        drawdown_percent = (drawdown / peak) * 100 if peak > 0 else Decimal("0")
        return drawdown, drawdown_percent


# =============================================================================
# Signal Repository
# =============================================================================

class SignalRepository(TenantMixin):
    """Repository for signal tracking."""

    async def save(self, signal: Signal) -> Signal:
        """Save a trading signal."""
        self._stamp(signal)
        self.session.add(signal)
        await self.session.commit()
        return signal

    async def get_recent_signals(
        self,
        limit: int = 100,
        strategy: Optional[str] = None,
        executed_only: bool = False,
    ) -> Sequence[Signal]:
        """Get recent signals with optional filters."""
        query = select(Signal)
        conditions = []
        if strategy:
            conditions.append(Signal.strategy_name == strategy)
        if executed_only:
            conditions.append(Signal.was_executed == True)  # noqa: E712
        if conditions:
            query = query.where(and_(*conditions))

        result = await self.session.execute(
            self._scope(query, Signal).order_by(desc(Signal.timestamp)).limit(limit)
        )
        return result.scalars().all()

    async def get_signal_execution_rate(
        self,
        strategy: str,
        days: int = 30,
    ) -> float:
        """Calculate signal execution rate for a strategy."""
        start_date = datetime.now(timezone.utc) - timedelta(days=days)

        total_query = select(func.count(Signal.id)).where(
            and_(Signal.strategy_name == strategy, Signal.timestamp >= start_date)
        )
        total_result = await self.session.execute(
            self._scope(total_query, Signal)
        )
        total = total_result.scalar() or 0
        if total == 0:
            return 0.0

        exec_query = select(func.count(Signal.id)).where(
            and_(
                Signal.strategy_name == strategy,
                Signal.timestamp >= start_date,
                Signal.was_executed == True,  # noqa: E712
            )
        )
        exec_result = await self.session.execute(
            self._scope(exec_query, Signal)
        )
        executed = exec_result.scalar() or 0
        return executed / total


# =============================================================================
# System Events Repository
# =============================================================================

class SystemRepository(TenantMixin):
    """Repository for system events and pauses."""

    async def log_event(
        self,
        event_type: str,
        message: str,
        severity: str = "INFO",
        details: Optional[dict] = None,
        trade_id: Optional[UUID] = None,
    ) -> SystemEvent:
        """Log a system event."""
        event = SystemEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            severity=severity,
            message=message,
            details=details,
            trade_id=trade_id,
        )
        self._stamp(event)
        self.session.add(event)
        await self.session.commit()
        return event

    async def start_trading_pause(
        self,
        reason: str,
        trigger_value: Optional[Decimal] = None,
        threshold_value: Optional[Decimal] = None,
        was_automatic: bool = True,
        notes: Optional[str] = None,
    ) -> TradingPause:
        """Record the start of a trading pause."""
        pause = TradingPause(
            start_time=datetime.now(timezone.utc),
            reason=reason,
            trigger_value=trigger_value,
            threshold_value=threshold_value,
            was_automatic=was_automatic,
            notes=notes,
        )
        self._stamp(pause)
        self.session.add(pause)
        await self.session.commit()

        logger.warning(
            "Trading paused",
            reason=reason,
            trigger_value=float(trigger_value) if trigger_value else None,
        )
        return pause

    async def end_trading_pause(self, pause_id: UUID) -> Optional[TradingPause]:
        """Record the end of a trading pause."""
        query = select(TradingPause).where(TradingPause.id == pause_id)
        result = await self.session.execute(self._scope(query, TradingPause))
        pause = result.scalar_one_or_none()

        if pause:
            pause.end_time = datetime.now(timezone.utc)
            await self.session.commit()
            logger.info("Trading pause ended", pause_id=str(pause_id))

        return pause

    async def get_active_pause(self) -> Optional[TradingPause]:
        """Check if trading is currently paused."""
        query = (
            select(TradingPause)
            .where(TradingPause.end_time.is_(None))
            .order_by(desc(TradingPause.start_time))
            .limit(1)
        )
        result = await self.session.execute(self._scope(query, TradingPause))
        return result.scalar_one_or_none()

    async def get_recent_events(
        self,
        limit: int = 100,
        severity: Optional[str] = None,
        event_type: Optional[str] = None,
    ) -> Sequence[SystemEvent]:
        """Get recent system events with optional filters."""
        query = select(SystemEvent)
        conditions = []
        if severity:
            conditions.append(SystemEvent.severity == severity)
        if event_type:
            conditions.append(SystemEvent.event_type == event_type)
        if conditions:
            query = query.where(and_(*conditions))

        result = await self.session.execute(
            self._scope(query, SystemEvent)
            .order_by(desc(SystemEvent.timestamp))
            .limit(limit)
        )
        return result.scalars().all()
