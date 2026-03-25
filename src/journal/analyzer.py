"""
Journal Analyzer
================
Async query functions for trade pattern analysis.

All queries work across both trade_source="ea" (EA webhook events) and
trade_source="manual" (MT5 poller-detected trades) for a unified view.

Every function returns plain Python dicts/lists — no ORM objects — so the
router can serialize them directly to JSON without extra mapping.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Optional

from sqlalchemy import case, cast, desc, func, select, Float

from src.database.models import (
    AccountSnapshot,
    JournalDeal,
    SetupTag,
    Trade,
    TradeOutcome,
    TradeStatus,
)
from src.database.repository import get_session


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _win_rate(wins: int, total: int) -> float:
    return round(wins / total, 4) if total else 0.0


def _profit_factor(gross_profit: float, gross_loss: float) -> Optional[float]:
    if gross_loss == 0:
        return None
    return round(gross_profit / gross_loss, 2)


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

async def summary_stats() -> dict:
    """Overall stats across all closed trades."""
    async with get_session() as session:
        result = await session.execute(
            select(
                func.count(Trade.id).label("total"),
                func.sum(
                    case((Trade.outcome == TradeOutcome.WIN, 1), else_=0)
                ).label("wins"),
                func.sum(
                    case((Trade.outcome == TradeOutcome.LOSS, 1), else_=0)
                ).label("losses"),
                func.sum(
                    case((Trade.outcome == TradeOutcome.BREAKEVEN, 1), else_=0)
                ).label("breakevens"),
                func.sum(Trade.profit_loss).label("net_pnl"),
                func.sum(
                    case((Trade.profit_loss > 0, Trade.profit_loss), else_=0)
                ).label("gross_profit"),
                func.sum(
                    case((Trade.profit_loss < 0, func.abs(Trade.profit_loss)), else_=0)
                ).label("gross_loss"),
                func.avg(Trade.risk_reward_actual).label("avg_rr"),
                func.avg(Trade.profit_loss).label("avg_pnl"),
            ).where(Trade.status == TradeStatus.CLOSED)
        )
        row = result.one()

        total      = row.total or 0
        wins       = row.wins or 0
        losses     = row.losses or 0
        breakevens = row.breakevens or 0
        net_pnl    = _safe_float(row.net_pnl) or 0.0
        g_profit   = _safe_float(row.gross_profit) or 0.0
        g_loss     = _safe_float(row.gross_loss) or 0.0

        # Count open positions separately
        open_result = await session.execute(
            select(func.count(Trade.id)).where(
                Trade.status.in_([TradeStatus.OPEN, TradeStatus.PARTIALLY_CLOSED])
            )
        )
        open_count = open_result.scalar() or 0

        # Source breakdown
        ea_result = await session.execute(
            select(func.count(Trade.id)).where(
                Trade.status == TradeStatus.CLOSED,
                Trade.trade_source == "ea",
            )
        )
        manual_result = await session.execute(
            select(func.count(Trade.id)).where(
                Trade.status == TradeStatus.CLOSED,
                Trade.trade_source == "manual",
            )
        )

        return {
            "total_trades":   total,
            "open_trades":    open_count,
            "wins":           wins,
            "losses":         losses,
            "breakevens":     breakevens,
            "win_rate":       _win_rate(wins, total),
            "net_pnl":        round(net_pnl, 2),
            "gross_profit":   round(g_profit, 2),
            "gross_loss":     round(g_loss, 2),
            "profit_factor":  _profit_factor(g_profit, g_loss),
            "avg_rr":         round(_safe_float(row.avg_rr) or 0.0, 2),
            "avg_pnl":        round(_safe_float(row.avg_pnl) or 0.0, 2),
            "ea_trades":      ea_result.scalar() or 0,
            "manual_trades":  manual_result.scalar() or 0,
        }


# ---------------------------------------------------------------------------
# Breakdowns
# ---------------------------------------------------------------------------

async def by_session() -> list[dict]:
    """Win rate and net P&L per trading session."""
    async with get_session() as session:
        result = await session.execute(
            select(
                Trade.trading_session,
                func.count(Trade.id).label("total"),
                func.sum(
                    case((Trade.outcome == TradeOutcome.WIN, 1), else_=0)
                ).label("wins"),
                func.sum(Trade.profit_loss).label("net_pnl"),
                func.avg(Trade.risk_reward_actual).label("avg_rr"),
            )
            .where(Trade.status == TradeStatus.CLOSED)
            .group_by(Trade.trading_session)
            .order_by(desc("net_pnl"))
        )
        rows = result.all()

    return [
        {
            "session":   row.trading_session or "Unknown",
            "total":     row.total,
            "wins":      row.wins or 0,
            "win_rate":  _win_rate(row.wins or 0, row.total),
            "net_pnl":   round(_safe_float(row.net_pnl) or 0.0, 2),
            "avg_rr":    round(_safe_float(row.avg_rr) or 0.0, 2),
        }
        for row in rows
    ]


async def by_hour() -> list[dict]:
    """Win rate and trade count for each hour of day (0–23 UTC)."""
    async with get_session() as session:
        result = await session.execute(
            select(
                Trade.hour_of_day,
                func.count(Trade.id).label("total"),
                func.sum(
                    case((Trade.outcome == TradeOutcome.WIN, 1), else_=0)
                ).label("wins"),
                func.sum(Trade.profit_loss).label("net_pnl"),
            )
            .where(
                Trade.status == TradeStatus.CLOSED,
                Trade.hour_of_day.isnot(None),
            )
            .group_by(Trade.hour_of_day)
            .order_by(Trade.hour_of_day)
        )
        rows = result.all()

    return [
        {
            "hour":      row.hour_of_day,
            "total":     row.total,
            "wins":      row.wins or 0,
            "win_rate":  _win_rate(row.wins or 0, row.total),
            "net_pnl":   round(_safe_float(row.net_pnl) or 0.0, 2),
        }
        for row in rows
    ]


async def by_day_of_week() -> list[dict]:
    """Win rate per day of week (0=Mon … 6=Sun)."""
    _day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    async with get_session() as session:
        result = await session.execute(
            select(
                Trade.day_of_week,
                func.count(Trade.id).label("total"),
                func.sum(
                    case((Trade.outcome == TradeOutcome.WIN, 1), else_=0)
                ).label("wins"),
                func.sum(Trade.profit_loss).label("net_pnl"),
            )
            .where(
                Trade.status == TradeStatus.CLOSED,
                Trade.day_of_week.isnot(None),
            )
            .group_by(Trade.day_of_week)
            .order_by(Trade.day_of_week)
        )
        rows = result.all()

    return [
        {
            "day":       row.day_of_week,
            "day_name":  _day_names[row.day_of_week] if 0 <= row.day_of_week <= 6 else "?",
            "total":     row.total,
            "wins":      row.wins or 0,
            "win_rate":  _win_rate(row.wins or 0, row.total),
            "net_pnl":   round(_safe_float(row.net_pnl) or 0.0, 2),
        }
        for row in rows
    ]


async def by_setup_tag() -> list[dict]:
    """Win rate and net P&L per setup tag."""
    async with get_session() as session:
        result = await session.execute(
            select(
                Trade.setup_tag,
                func.count(Trade.id).label("total"),
                func.sum(
                    case((Trade.outcome == TradeOutcome.WIN, 1), else_=0)
                ).label("wins"),
                func.sum(Trade.profit_loss).label("net_pnl"),
                func.avg(Trade.risk_reward_actual).label("avg_rr"),
            )
            .where(
                Trade.status == TradeStatus.CLOSED,
                Trade.setup_tag.isnot(None),
            )
            .group_by(Trade.setup_tag)
            .order_by(desc("net_pnl"))
        )
        rows = result.all()

    return [
        {
            "setup":     row.setup_tag,
            "total":     row.total,
            "wins":      row.wins or 0,
            "win_rate":  _win_rate(row.wins or 0, row.total),
            "net_pnl":   round(_safe_float(row.net_pnl) or 0.0, 2),
            "avg_rr":    round(_safe_float(row.avg_rr) or 0.0, 2),
        }
        for row in rows
    ]


async def by_symbol() -> list[dict]:
    """Win rate per symbol."""
    async with get_session() as session:
        result = await session.execute(
            select(
                Trade.symbol,
                func.count(Trade.id).label("total"),
                func.sum(
                    case((Trade.outcome == TradeOutcome.WIN, 1), else_=0)
                ).label("wins"),
                func.sum(Trade.profit_loss).label("net_pnl"),
            )
            .where(Trade.status == TradeStatus.CLOSED)
            .group_by(Trade.symbol)
            .order_by(desc("net_pnl"))
        )
        rows = result.all()

    return [
        {
            "symbol":   row.symbol,
            "total":    row.total,
            "wins":     row.wins or 0,
            "win_rate": _win_rate(row.wins or 0, row.total),
            "net_pnl":  round(_safe_float(row.net_pnl) or 0.0, 2),
        }
        for row in rows
    ]


async def by_direction() -> list[dict]:
    """Win rate split by BUY vs SELL."""
    from src.database.models import OrderType

    async with get_session() as session:
        result = await session.execute(
            select(
                Trade.order_type,
                func.count(Trade.id).label("total"),
                func.sum(
                    case((Trade.outcome == TradeOutcome.WIN, 1), else_=0)
                ).label("wins"),
                func.sum(Trade.profit_loss).label("net_pnl"),
            )
            .where(Trade.status == TradeStatus.CLOSED)
            .group_by(Trade.order_type)
        )
        rows = result.all()

    return [
        {
            "direction": row.order_type.value if row.order_type else "Unknown",
            "total":     row.total,
            "wins":      row.wins or 0,
            "win_rate":  _win_rate(row.wins or 0, row.total),
            "net_pnl":   round(_safe_float(row.net_pnl) or 0.0, 2),
        }
        for row in rows
    ]


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

async def equity_curve(limit: int = 500) -> list[dict]:
    """
    Return recent AccountSnapshot rows for balance/equity time series.
    Limited to `limit` most recent points (downsampled for large histories).
    """
    async with get_session() as session:
        result = await session.execute(
            select(
                AccountSnapshot.timestamp,
                AccountSnapshot.balance,
                AccountSnapshot.equity,
                AccountSnapshot.floating_pl,
            )
            .order_by(desc(AccountSnapshot.timestamp))
            .limit(limit)
        )
        rows = result.all()

    # Return in ascending order for chart rendering
    return [
        {
            "time":       row.timestamp.isoformat(),
            "balance":    _safe_float(row.balance),
            "equity":     _safe_float(row.equity),
            "floating_pl": _safe_float(row.floating_pl),
        }
        for row in reversed(rows)
    ]


# ---------------------------------------------------------------------------
# Trade list queries
# ---------------------------------------------------------------------------

async def list_trades(
    page:       int = 1,
    per_page:   int = 50,
    symbol:     Optional[str] = None,
    direction:  Optional[str] = None,
    session_:   Optional[str] = None,
    setup_tag:  Optional[str] = None,
    source:     Optional[str] = None,
    status:     Optional[str] = None,
) -> dict:
    """Paginated trade list with optional filters."""
    from sqlalchemy import and_

    offset = (page - 1) * per_page
    conditions = []

    if symbol:
        conditions.append(Trade.symbol == symbol.upper())
    if direction:
        from src.database.models import OrderType
        try:
            conditions.append(Trade.order_type == OrderType[direction.upper()])
        except KeyError:
            pass
    if session_:
        conditions.append(Trade.trading_session == session_)
    if setup_tag:
        conditions.append(Trade.setup_tag == setup_tag)
    if source:
        conditions.append(Trade.trade_source == source)
    if status:
        try:
            conditions.append(Trade.status == TradeStatus[status.upper()])
        except KeyError:
            pass

    where_clause = and_(*conditions) if conditions else True

    async with get_session() as db:
        count_result = await db.execute(
            select(func.count(Trade.id)).where(where_clause)
        )
        total = count_result.scalar() or 0

        result = await db.execute(
            select(Trade)
            .where(where_clause)
            .order_by(desc(Trade.entry_time))
            .offset(offset)
            .limit(per_page)
        )
        trades = result.scalars().all()

    return {
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "pages":    (total + per_page - 1) // per_page if total else 0,
        "items":    [_trade_to_dict(t) for t in trades],
    }


async def open_trades() -> list[dict]:
    """Return all currently open / partially-closed positions."""
    async with get_session() as db:
        result = await db.execute(
            select(Trade)
            .where(
                Trade.status.in_([TradeStatus.OPEN, TradeStatus.PARTIALLY_CLOSED])
            )
            .order_by(Trade.entry_time)
        )
        trades = result.scalars().all()
    return [_trade_to_dict(t) for t in trades]


async def list_deals(page: int = 1, per_page: int = 100) -> dict:
    """Paginated raw deal audit log."""
    offset = (page - 1) * per_page

    async with get_session() as db:
        count_result = await db.execute(select(func.count(JournalDeal.id)))
        total = count_result.scalar() or 0

        result = await db.execute(
            select(JournalDeal)
            .order_by(desc(JournalDeal.deal_time))
            .offset(offset)
            .limit(per_page)
        )
        deals = result.scalars().all()

    return {
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "pages":    (total + per_page - 1) // per_page if total else 0,
        "items":    [_deal_to_dict(d) for d in deals],
    }


async def get_tags() -> list[dict]:
    """Return all setup tags."""
    async with get_session() as db:
        result = await db.execute(select(SetupTag).order_by(SetupTag.name))
        tags = result.scalars().all()
    return [{"id": t.id, "name": t.name, "color": t.color, "description": t.description} for t in tags]


# ---------------------------------------------------------------------------
# Serialisers (ORM → dict)
# ---------------------------------------------------------------------------

def _trade_to_dict(t: Trade) -> dict:
    return {
        "id":              str(t.id),
        "ticket":          t.ticket,
        "symbol":          t.symbol,
        "direction":       t.order_type.value if t.order_type else None,
        "status":          t.status.value if t.status else None,
        "trade_source":    t.trade_source,
        "entry_price":     _safe_float(t.entry_price),
        "exit_price":      _safe_float(t.exit_price),
        "entry_time":      t.entry_time.isoformat() if t.entry_time else None,
        "exit_time":       t.exit_time.isoformat() if t.exit_time else None,
        "lot_size":        _safe_float(t.lot_size),
        "stop_loss":       _safe_float(t.stop_loss),
        "take_profit_1":   _safe_float(t.take_profit_1),
        "profit_loss":     _safe_float(t.profit_loss),
        "outcome":         t.outcome.value if t.outcome else None,
        "exit_reason":     t.exit_reason,
        "risk_reward_actual": _safe_float(t.risk_reward_actual),
        "trading_session": t.trading_session,
        "hour_of_day":     t.hour_of_day,
        "day_of_week":     t.day_of_week,
        "setup_tag":       t.setup_tag,
        "journal_notes":   t.journal_notes,
        "mt5_position_id": t.mt5_position_id,
        "strategy_name":   t.strategy_name,
    }


def _deal_to_dict(d: JournalDeal) -> dict:
    return {
        "id":             str(d.id),
        "deal_id":        d.deal_id,
        "position_id":    d.position_id,
        "symbol":         d.symbol,
        "deal_time":      d.deal_time.isoformat() if d.deal_time else None,
        "deal_type":      d.deal_type,
        "entry_type":     d.entry_type,
        "volume":         _safe_float(d.volume),
        "price":          _safe_float(d.price),
        "commission":     _safe_float(d.commission),
        "swap":           _safe_float(d.swap),
        "profit":         _safe_float(d.profit),
        "exit_reason":    d.exit_reason,
        "comment":        d.comment,
    }
