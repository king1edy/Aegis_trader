"""
Database Repository
===================
Data access layer for the trading system.
Provides clean interfaces for database operations.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import AsyncGenerator, List, Optional, Sequence
from uuid import UUID

from sqlalchemy import and_, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import selectinload

from src.core.config import settings
from src.core.logging_config import get_logger
from src.database.models import (
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
    """Initialize database tables."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables initialized")


async def close_database() -> None:
    """Close database connections."""
    await engine.dispose()
    logger.info("Database connections closed")


# =============================================================================
# Trade Repository
# =============================================================================

class TradeRepository:
    """Repository for trade-related database operations."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create(self, trade: Trade) -> Trade:
        """Create a new trade record."""
        self.session.add(trade)
        await self.session.commit()
        await self.session.refresh(trade)
        logger.info("Trade created", trade_id=str(trade.id), ticket=trade.ticket)
        return trade
    
    async def get_by_id(self, trade_id: UUID) -> Optional[Trade]:
        """Get trade by ID."""
        result = await self.session.execute(
            select(Trade)
            .options(selectinload(Trade.partial_closes))
            .options(selectinload(Trade.modifications))
            .where(Trade.id == trade_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_ticket(self, ticket: int) -> Optional[Trade]:
        """Get trade by broker ticket number."""
        result = await self.session.execute(
            select(Trade)
            .options(selectinload(Trade.partial_closes))
            .where(Trade.ticket == ticket)
        )
        return result.scalar_one_or_none()
    
    async def get_open_trades(self, symbol: Optional[str] = None) -> Sequence[Trade]:
        """Get all open trades, optionally filtered by symbol."""
        query = select(Trade).where(Trade.status == TradeStatus.OPEN)
        if symbol:
            query = query.where(Trade.symbol == symbol)
        result = await self.session.execute(query.order_by(Trade.entry_time))
        return result.scalars().all()
    
    async def get_recent_trades(
        self,
        limit: int = 50,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None
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
            query.order_by(desc(Trade.signal_time)).limit(limit)
        )
        return result.scalars().all()
    
    async def get_trades_in_range(
        self,
        start_date: datetime,
        end_date: datetime,
        symbol: Optional[str] = None
    ) -> Sequence[Trade]:
        """Get trades within a date range."""
        query = select(Trade).where(
            and_(
                Trade.entry_time >= start_date,
                Trade.entry_time <= end_date
            )
        )
        if symbol:
            query = query.where(Trade.symbol == symbol)
        
        result = await self.session.execute(query.order_by(Trade.entry_time))
        return result.scalars().all()
    
    async def get_today_trades(self) -> Sequence[Trade]:
        """Get all trades from today."""
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        result = await self.session.execute(
            select(Trade)
            .where(Trade.signal_time >= today_start)
            .order_by(Trade.signal_time)
        )
        return result.scalars().all()
    
    async def count_today_trades(self) -> int:
        """Count trades opened today."""
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        result = await self.session.execute(
            select(func.count(Trade.id)).where(
                and_(
                    Trade.entry_time >= today_start,
                    Trade.status.in_([TradeStatus.OPEN, TradeStatus.CLOSED])
                )
            )
        )
        return result.scalar() or 0
    
    async def get_consecutive_losses(self) -> int:
        """Get count of consecutive losses (most recent first)."""
        result = await self.session.execute(
            select(Trade)
            .where(Trade.status == TradeStatus.CLOSED)
            .order_by(desc(Trade.exit_time))
            .limit(20)
        )
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
        self.session.add(partial)
        await self.session.commit()
        return partial
    
    async def add_modification(self, modification: TradeModification) -> TradeModification:
        """Record a trade modification."""
        self.session.add(modification)
        await self.session.commit()
        return modification


# =============================================================================
# Performance Repository
# =============================================================================

class PerformanceRepository:
    """Repository for performance and account metrics."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save_snapshot(self, snapshot: AccountSnapshot) -> AccountSnapshot:
        """Save an account snapshot."""
        self.session.add(snapshot)
        await self.session.commit()
        return snapshot
    
    async def get_latest_snapshot(self) -> Optional[AccountSnapshot]:
        """Get the most recent account snapshot."""
        result = await self.session.execute(
            select(AccountSnapshot)
            .order_by(desc(AccountSnapshot.timestamp))
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def get_peak_equity(self) -> Optional[Decimal]:
        """Get the peak equity value."""
        result = await self.session.execute(
            select(func.max(AccountSnapshot.equity))
        )
        return result.scalar()
    
    async def get_daily_performance(
        self,
        date: Optional[datetime] = None
    ) -> Optional[DailyPerformance]:
        """Get performance for a specific date or today."""
        if date is None:
            date = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        
        result = await self.session.execute(
            select(DailyPerformance).where(DailyPerformance.date == date)
        )
        return result.scalar_one_or_none()
    
    async def save_daily_performance(
        self,
        performance: DailyPerformance
    ) -> DailyPerformance:
        """Save or update daily performance record."""
        existing = await self.get_daily_performance(performance.date)
        if existing:
            # Update existing record
            for key, value in performance.__dict__.items():
                if not key.startswith('_') and key != 'id':
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
        end_date: datetime
    ) -> Sequence[DailyPerformance]:
        """Get daily performance records for a date range."""
        result = await self.session.execute(
            select(DailyPerformance)
            .where(
                and_(
                    DailyPerformance.date >= start_date,
                    DailyPerformance.date <= end_date
                )
            )
            .order_by(DailyPerformance.date)
        )
        return result.scalars().all()
    
    async def calculate_current_drawdown(self) -> tuple[Decimal, Decimal]:
        """
        Calculate current drawdown from peak.
        
        Returns:
            Tuple of (drawdown_amount, drawdown_percent)
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

class SignalRepository:
    """Repository for signal tracking."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def save(self, signal: Signal) -> Signal:
        """Save a trading signal."""
        self.session.add(signal)
        await self.session.commit()
        return signal
    
    async def get_recent_signals(
        self,
        limit: int = 100,
        strategy: Optional[str] = None,
        executed_only: bool = False
    ) -> Sequence[Signal]:
        """Get recent signals with optional filters."""
        query = select(Signal)
        
        conditions = []
        if strategy:
            conditions.append(Signal.strategy_name == strategy)
        if executed_only:
            conditions.append(Signal.was_executed == True)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        result = await self.session.execute(
            query.order_by(desc(Signal.timestamp)).limit(limit)
        )
        return result.scalars().all()
    
    async def get_signal_execution_rate(
        self,
        strategy: str,
        days: int = 30
    ) -> float:
        """Calculate signal execution rate for a strategy."""
        start_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        total_result = await self.session.execute(
            select(func.count(Signal.id))
            .where(
                and_(
                    Signal.strategy_name == strategy,
                    Signal.timestamp >= start_date
                )
            )
        )
        total = total_result.scalar() or 0
        
        if total == 0:
            return 0.0
        
        executed_result = await self.session.execute(
            select(func.count(Signal.id))
            .where(
                and_(
                    Signal.strategy_name == strategy,
                    Signal.timestamp >= start_date,
                    Signal.was_executed == True
                )
            )
        )
        executed = executed_result.scalar() or 0
        
        return executed / total


# =============================================================================
# System Events Repository
# =============================================================================

class SystemRepository:
    """Repository for system events and pauses."""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def log_event(
        self,
        event_type: str,
        message: str,
        severity: str = "INFO",
        details: Optional[dict] = None,
        trade_id: Optional[UUID] = None
    ) -> SystemEvent:
        """Log a system event."""
        event = SystemEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            severity=severity,
            message=message,
            details=details,
            trade_id=trade_id
        )
        self.session.add(event)
        await self.session.commit()
        return event
    
    async def start_trading_pause(
        self,
        reason: str,
        trigger_value: Optional[Decimal] = None,
        threshold_value: Optional[Decimal] = None,
        was_automatic: bool = True,
        notes: Optional[str] = None
    ) -> TradingPause:
        """Record the start of a trading pause."""
        pause = TradingPause(
            start_time=datetime.now(timezone.utc),
            reason=reason,
            trigger_value=trigger_value,
            threshold_value=threshold_value,
            was_automatic=was_automatic,
            notes=notes
        )
        self.session.add(pause)
        await self.session.commit()
        
        logger.warning(
            "Trading paused",
            reason=reason,
            trigger_value=float(trigger_value) if trigger_value else None
        )
        return pause
    
    async def end_trading_pause(self, pause_id: UUID) -> Optional[TradingPause]:
        """Record the end of a trading pause."""
        result = await self.session.execute(
            select(TradingPause).where(TradingPause.id == pause_id)
        )
        pause = result.scalar_one_or_none()
        
        if pause:
            pause.end_time = datetime.now(timezone.utc)
            await self.session.commit()
            logger.info("Trading pause ended", pause_id=str(pause_id))
        
        return pause
    
    async def get_active_pause(self) -> Optional[TradingPause]:
        """Check if trading is currently paused."""
        result = await self.session.execute(
            select(TradingPause)
            .where(TradingPause.end_time.is_(None))
            .order_by(desc(TradingPause.start_time))
            .limit(1)
        )
        return result.scalar_one_or_none()
    
    async def get_recent_events(
        self,
        limit: int = 100,
        severity: Optional[str] = None,
        event_type: Optional[str] = None
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
            query.order_by(desc(SystemEvent.timestamp)).limit(limit)
        )
        return result.scalars().all()
