"""
TradingView Webhook Handler
=============================
Core processing logic for TradingView alert webhooks.

Handles BUY/SELL (open) and CLOSE events:
- Persists trades to PostgreSQL via TradeRepository
- Writes crash-safe CSV rows
- Fires Telegram notifications

The handler mirrors the pattern in trade_logging.trade_event_server but
accepts TradingView alert payloads instead of EA BuildJSON payloads.
"""

import csv
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List
from uuid import UUID

from core.config import settings
from core.logging_config import get_logger
from database.models import (
    OrderType,
    SignalSource,
    Trade,
    TradeOutcome,
    TradeStatus,
)
from database.repository import (
    TradeRepository,
    SystemRepository,
    get_session,
)
from notifications import NotificationService
from webhooks.tv_schema import TradingViewAlert

logger = get_logger("tv_handler")

# ---------------------------------------------------------------------------
# Session detection
# ---------------------------------------------------------------------------

_SESSION_RANGES = {
    "London":     (7, 12),
    "NY Overlap": (13, 16),
}


def _detect_session(hour: int) -> str:
    for name, (start, end) in _SESSION_RANGES.items():
        if start <= hour < end:
            return name
    return "Off-Hours"


# ---------------------------------------------------------------------------
# Direction mapping
# ---------------------------------------------------------------------------

_ACTION_TO_ORDER: Dict[str, OrderType] = {
    "BUY":  OrderType.BUY,
    "SELL": OrderType.SELL,
}

# ---------------------------------------------------------------------------
# CSV helpers — same crash-safe pattern as trade_event_server
# ---------------------------------------------------------------------------

TV_CSV_FILE = Path(settings.trade_log_csv_path).parent / "tradingview_events.csv"

TV_CSV_COLUMNS = [
    "timestamp", "event", "symbol", "direction", "strategy_name",
    "timeframe", "entry", "sl", "tp1", "tp2", "lots",
    "exit_price", "pnl", "outcome", "filters", "note",
    "trade_id", "tenant_id",
]


def _ensure_tv_csv_header() -> None:
    TV_CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not TV_CSV_FILE.exists():
        with open(TV_CSV_FILE, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=TV_CSV_COLUMNS).writeheader()


def _append_tv_csv(row: dict) -> None:
    _ensure_tv_csv_header()
    with open(TV_CSV_FILE, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=TV_CSV_COLUMNS, extrasaction="ignore").writerow(row)


# ---------------------------------------------------------------------------
# Telegram helper
# ---------------------------------------------------------------------------

async def _notify_open(alert: TradingViewAlert) -> None:
    try:
        notifier = NotificationService.get_instance()
        if not notifier.is_enabled:
            return
        await notifier.notify_trade_opened(
            symbol=alert.symbol,
            direction=alert.action,
            entry_price=alert.price or 0.0,
            stop_loss=alert.stop_loss or 0.0,
            take_profit_1=alert.take_profit or 0.0,
            take_profit_2=alert.take_profit_2,
            lot_size=alert.quantity or 0.0,
            reason=f"TradingView | {alert.strategy_name}",
        )
    except Exception as exc:
        logger.warning("Telegram notification failed (open)", error=str(exc))


async def _notify_close(alert: TradingViewAlert, entry_price: float, direction: str) -> None:
    try:
        notifier = NotificationService.get_instance()
        if not notifier.is_enabled:
            return
        await notifier.notify_trade_closed(
            symbol=alert.symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=alert.exit_price or 0.0,
            profit_loss=alert.pnl or 0.0,
            lot_size=alert.quantity or 0.0,
            duration_minutes=0,
            exit_reason="TradingView CLOSE",
        )
    except Exception as exc:
        logger.warning("Telegram notification failed (close)", error=str(exc))


# ---------------------------------------------------------------------------
# OPEN handler
# ---------------------------------------------------------------------------

async def handle_open(alert: TradingViewAlert, tenant_id: UUID) -> dict:
    """
    Process a BUY or SELL alert from TradingView.

    Creates a Trade row, writes a CSV backup, and fires a Telegram
    notification.  Returns the trade_id so the CLOSE alert can
    reference it.
    """
    now = datetime.now(timezone.utc)

    trade = Trade(
        symbol=alert.symbol,
        order_type=_ACTION_TO_ORDER[alert.action],
        status=TradeStatus.OPEN,
        signal_source=SignalSource.TRADINGVIEW,
        strategy_name=alert.strategy_name,
        signal_time=now,
        entry_price=Decimal(str(alert.price)) if alert.price else None,
        entry_time=now,
        lot_size=Decimal(str(alert.quantity)) if alert.quantity else Decimal("0.01"),
        initial_lot_size=Decimal(str(alert.quantity)) if alert.quantity else Decimal("0.01"),
        stop_loss=Decimal(str(alert.stop_loss)) if alert.stop_loss else Decimal("0"),
        take_profit_1=Decimal(str(alert.take_profit)) if alert.take_profit else None,
        take_profit_2=Decimal(str(alert.take_profit_2)) if alert.take_profit_2 else None,
        position_state="initial",
        market_context=alert.filters,
        strategy_data={"timeframe": alert.timeframe} if alert.timeframe else None,
        trade_source="tradingview",
        trading_session=_detect_session(now.hour),
        hour_of_day=now.hour,
        day_of_week=now.weekday(),
    )

    # 1. CSV — always first, never raises
    csv_row = {
        "timestamp":     now.isoformat(),
        "event":         "OPEN",
        "symbol":        alert.symbol,
        "direction":     alert.action,
        "strategy_name": alert.strategy_name,
        "timeframe":     alert.timeframe,
        "entry":         alert.price,
        "sl":            alert.stop_loss,
        "tp1":           alert.take_profit,
        "tp2":           alert.take_profit_2,
        "lots":          alert.quantity,
        "exit_price":    None,
        "pnl":           None,
        "outcome":       "OPEN",
        "filters":       str(alert.filters) if alert.filters else "",
        "note":          alert.note,
        "trade_id":      "",  # filled after DB persist
        "tenant_id":     str(tenant_id),
    }
    _append_tv_csv(csv_row)

    # 2. PostgreSQL
    try:
        async with get_session() as session:
            trade_repo = TradeRepository(session, tenant_id)
            system_repo = SystemRepository(session, tenant_id)
            trade = await trade_repo.create(trade)

            await system_repo.log_event(
                event_type="TV_OPEN",
                message=f"TradingView {alert.action} {alert.symbol}",
                severity="INFO",
                details={
                    "strategy": alert.strategy_name,
                    "timeframe": alert.timeframe,
                    "filters": alert.filters,
                },
            )

        logger.info(
            "TradingView OPEN persisted",
            trade_id=str(trade.id),
            symbol=alert.symbol,
            direction=alert.action,
        )
    except Exception as exc:
        logger.error(
            "DB persist failed — CSV record still written",
            error=str(exc),
            symbol=alert.symbol,
        )
        return {
            "status": "logged_csv_only",
            "symbol": alert.symbol,
            "error": "Database unavailable, trade logged to CSV",
        }

    # 3. Telegram
    await _notify_open(alert)

    return {
        "status": "opened",
        "trade_id": str(trade.id),
        "symbol": trade.symbol,
        "direction": alert.action,
        "entry_price": alert.price,
        "session": trade.trading_session,
    }


# ---------------------------------------------------------------------------
# CLOSE handler
# ---------------------------------------------------------------------------

async def handle_close(alert: TradingViewAlert, tenant_id: UUID) -> dict:
    """
    Process a CLOSE alert from TradingView.

    Looks up the open trade by trade_id + tenant_id, sets exit fields,
    determines outcome, writes CSV, and fires Telegram notification.
    """
    now = datetime.now(timezone.utc)
    trade_uuid = UUID(alert.trade_id)

    try:
        async with get_session() as session:
            trade_repo = TradeRepository(session, tenant_id)
            system_repo = SystemRepository(session, tenant_id)

            trade = await trade_repo.get_by_id(trade_uuid)
            if trade is None:
                logger.warning(
                    "Trade not found for CLOSE",
                    trade_id=alert.trade_id,
                    symbol=alert.symbol,
                )
                return {"status": "error", "detail": f"Trade {alert.trade_id} not found"}

            if trade.status not in (TradeStatus.OPEN, TradeStatus.PARTIALLY_CLOSED):
                return {
                    "status": "error",
                    "detail": f"Trade {alert.trade_id} is not open (status={trade.status.value})",
                }

            # Determine P&L and outcome
            pnl = alert.pnl
            if pnl is None and alert.exit_price and trade.entry_price:
                # Calculate from price difference
                exit_dec = Decimal(str(alert.exit_price))
                diff = exit_dec - trade.entry_price
                if trade.order_type == OrderType.SELL:
                    diff = -diff
                pnl = float(diff * trade.lot_size)

            if pnl is not None:
                if pnl > 0:
                    outcome = TradeOutcome.WIN
                elif pnl < 0:
                    outcome = TradeOutcome.LOSS
                else:
                    outcome = TradeOutcome.BREAKEVEN
            else:
                outcome = None

            # Update trade
            trade.exit_price = Decimal(str(alert.exit_price)) if alert.exit_price else None
            trade.exit_time = now
            trade.exit_reason = "TradingView CLOSE"
            trade.profit_loss = Decimal(str(pnl)) if pnl is not None else None
            trade.outcome = outcome
            trade.status = TradeStatus.CLOSED

            # Compute actual R:R if we have the data
            if pnl is not None and trade.stop_loss and trade.entry_price:
                sl_distance = abs(float(trade.entry_price - trade.stop_loss))
                if sl_distance > 0:
                    trade.risk_reward_actual = Decimal(
                        str(round(abs(pnl) / (sl_distance * float(trade.lot_size)), 2))
                    )
                    if pnl < 0:
                        trade.risk_reward_actual = -trade.risk_reward_actual

            await trade_repo.update(trade)

            await system_repo.log_event(
                event_type="TV_CLOSE",
                message=f"TradingView CLOSE {alert.symbol} pnl={pnl}",
                severity="INFO",
                details={
                    "trade_id": str(trade.id),
                    "pnl": pnl,
                    "outcome": outcome.value if outcome else None,
                },
            )

        logger.info(
            "TradingView CLOSE persisted",
            trade_id=str(trade.id),
            pnl=pnl,
            outcome=outcome.value if outcome else None,
        )
    except ValueError:
        return {"status": "error", "detail": f"Invalid trade_id format: {alert.trade_id}"}
    except Exception as exc:
        logger.error("DB close failed", error=str(exc), trade_id=alert.trade_id)
        return {"status": "error", "detail": "Database error during close"}

    # CSV
    csv_row = {
        "timestamp":     now.isoformat(),
        "event":         "CLOSE",
        "symbol":        alert.symbol,
        "direction":     trade.order_type.value if trade.order_type else "",
        "strategy_name": trade.strategy_name,
        "timeframe":     (trade.strategy_data or {}).get("timeframe"),
        "entry":         float(trade.entry_price) if trade.entry_price else None,
        "sl":            float(trade.stop_loss) if trade.stop_loss else None,
        "tp1":           float(trade.take_profit_1) if trade.take_profit_1 else None,
        "tp2":           float(trade.take_profit_2) if trade.take_profit_2 else None,
        "lots":          float(trade.lot_size),
        "exit_price":    alert.exit_price,
        "pnl":           pnl,
        "outcome":       outcome.value if outcome else "",
        "filters":       str(trade.market_context) if trade.market_context else "",
        "note":          alert.note,
        "trade_id":      str(trade.id),
        "tenant_id":     str(tenant_id),
    }
    _append_tv_csv(csv_row)

    # Telegram
    await _notify_close(
        alert,
        entry_price=float(trade.entry_price) if trade.entry_price else 0.0,
        direction=trade.order_type.value if trade.order_type else "UNKNOWN",
    )

    return {
        "status": "closed",
        "trade_id": str(trade.id),
        "symbol": alert.symbol,
        "pnl": pnl,
        "outcome": outcome.value if outcome else None,
    }
