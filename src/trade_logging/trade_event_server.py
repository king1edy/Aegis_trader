"""
Trade Event Logging Server
===========================
Receives trade events from the MT5 EA via HTTP POST.
Writes every event to CSV (crash-safe) then persists to PostgreSQL.
Forwards OPEN and CLOSE events to Telegram via NotificationService.

EA MQL5 settings
----------------
  InpEnableFastAPI  = true
  InpFastAPIURL     = http://127.0.0.1:8000/trade
  InpFastAPITimeout = 2000

Endpoints (registered on ea_router — included in main app)
---------
  POST /trade           — receive event from EA
  GET  /trades          — all events, newest first, optional filters
  GET  /trades/summary  — win rate, P&L, breakdowns by method/session/direction
  GET  /health          — liveness check

Usage in main.py
----------------
  from src.trade_logging.trade_event_server import (
      ea_router, _load_csv_to_memory, _ensure_csv_header
  )
  app.include_router(ea_router)

For standalone testing / backward compat use create_logging_app().
"""

import csv
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.core.config import settings
from src.core.logging_config import get_logger
from src.database.repository import (
    get_session,
    init_database,
    TradeRepository,
    SystemRepository,
)
from src.database.models import (
    Trade,
    TradeStatus,
    TradeOutcome,
    OrderType,
    SignalSource,
    PartialClose,
)
from src.notifications import NotificationService

logger = get_logger("trade_event_server")

VERSION  = "3.0.0"
CSV_FILE = Path(settings.trade_log_csv_path)

CSV_COLUMNS = [
    "timestamp", "event", "ticket", "symbol", "direction", "method",
    "session", "entry", "sl", "tp1", "tp2", "lots", "risk_pct",
    "sl_dist", "exit_price", "pnl", "rr", "outcome",
    "d1_bias", "h4_bias", "pos_state", "balance", "equity", "note",
]

_FULL_CLOSE_EVENTS    = {"SL_HIT", "TIME_EXIT", "TRAIL_EXIT"}
_PARTIAL_CLOSE_EVENTS = {"TP1_HIT", "TP2_HIT"}
_ALL_CLOSE_EVENTS     = _FULL_CLOSE_EVENTS | _PARTIAL_CLOSE_EVENTS

_DIRECTION_MAP: dict[str, OrderType] = {
    "LONG":  OrderType.BUY,
    "SHORT": OrderType.SELL,
}

_OUTCOME_MAP: dict[str, TradeOutcome] = {
    "WIN":         TradeOutcome.WIN,
    "WIN_PARTIAL": TradeOutcome.WIN,
    "LOSS":        TradeOutcome.LOSS,
    "TIME_EXIT":   TradeOutcome.LOSS,
    "BREAKEVEN":   TradeOutcome.BREAKEVEN,
}


# =============================================================================
# Pydantic model — mirrors EA BuildJSON payload exactly
# =============================================================================

class TradeEvent(BaseModel):
    timestamp:  str
    event:      str          # OPEN | TP1_HIT | TP2_HIT | SL_HIT | TIME_EXIT | TRAIL_EXIT
    ticket:     int
    symbol:     str
    direction:  str          # LONG | SHORT
    method:     str          # EMA Bounce | Structure Break | Fibonacci 50/61.8%
    session:    str          # London | NY Overlap | Lunch | Off-Hours
    entry:      float
    sl:         float
    tp1:        float
    tp2:        float
    lots:       float
    risk_pct:   float
    sl_dist:    float
    exit_price: Optional[float] = None
    pnl:        Optional[float] = None
    rr:         Optional[float] = None
    outcome:    str          # OPEN | WIN | WIN_PARTIAL | LOSS | TIME_EXIT
    d1_bias:    str          # LONG | SHORT | NEUTRAL
    h4_bias:    str          # LONG | SHORT | NEUTRAL
    pos_state:  int          # 0=initial  1=tp1_hit  2=tp2_hit
    balance:    float
    equity:     float
    note:       Optional[str] = ""


# =============================================================================
# In-memory store — seeded from CSV on startup, reset on server restart
# =============================================================================

_events_store: List[Dict] = []


# =============================================================================
# CSV helpers
# =============================================================================

def _ensure_csv_header() -> None:
    """Create the CSV file with a header row if it does not exist yet."""
    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not CSV_FILE.exists():
        with open(CSV_FILE, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()


def _append_to_csv(row: dict) -> None:
    _ensure_csv_header()
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore").writerow(row)


def _load_csv_to_memory() -> None:
    """Re-seed the in-memory store from the CSV file on server startup."""
    if not CSV_FILE.exists():
        return
    with open(CSV_FILE, "r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            _events_store.append(row)
    logger.info(
        "Event store seeded from CSV",
        count=len(_events_store),
        path=str(CSV_FILE),
    )


# =============================================================================
# Database persistence
# =============================================================================

async def _persist_event(event: TradeEvent) -> None:
    """Write the EA event to PostgreSQL. Errors never block the CSV write."""
    try:
        async with get_session() as session:
            trade_repo  = TradeRepository(session)
            system_repo = SystemRepository(session)
            now = datetime.now(timezone.utc)

            if event.event == "OPEN":
                trade = Trade(
                    ticket                 = event.ticket,
                    symbol                 = event.symbol,
                    order_type             = _DIRECTION_MAP.get(event.direction.upper(), OrderType.BUY),
                    status                 = TradeStatus.OPEN,
                    signal_source          = SignalSource.WEBHOOK,
                    strategy_name          = "MTFTR_EA",
                    signal_time            = now,
                    entry_price            = Decimal(str(event.entry)),
                    entry_time             = now,
                    lot_size               = Decimal(str(event.lots)),
                    initial_lot_size       = Decimal(str(event.lots)),
                    stop_loss              = Decimal(str(event.sl)),
                    take_profit_1          = Decimal(str(event.tp1)) if event.tp1 else None,
                    take_profit_2          = Decimal(str(event.tp2)) if event.tp2 else None,
                    position_state         = "initial",
                    risk_percent           = Decimal(str(round(event.risk_pct, 6))),
                    account_balance_before = Decimal(str(event.balance)),
                    market_context         = {
                        "d1_bias": event.d1_bias,
                        "h4_bias": event.h4_bias,
                        "session": event.session,
                        "method":  event.method,
                        "sl_dist": event.sl_dist,
                    },
                    strategy_data          = {"pos_state": event.pos_state},
                )
                await trade_repo.create(trade)
                logger.info("OPEN persisted", ticket=event.ticket, symbol=event.symbol)

            elif event.event in _PARTIAL_CLOSE_EVENTS:
                trade = await trade_repo.get_by_ticket(event.ticket)
                if trade:
                    partial = PartialClose(
                        trade_id    = trade.id,
                        close_time  = now,
                        close_price = Decimal(str(event.exit_price)),
                        lots_closed = Decimal(str(event.lots)),
                        profit_loss = Decimal(str(event.pnl)),
                        reason      = event.event,
                    )
                    await trade_repo.add_partial_close(partial)
                    trade.position_state = "tp1_hit" if event.event == "TP1_HIT" else "tp2_hit"
                    trade.status         = TradeStatus.PARTIALLY_CLOSED
                    trade.strategy_data  = {
                        **(trade.strategy_data or {}),
                        "pos_state": event.pos_state,
                    }
                    await trade_repo.update(trade)
                    logger.info(
                        "Partial close persisted",
                        ticket=event.ticket,
                        event=event.event,
                    )

            elif event.event in _FULL_CLOSE_EVENTS:
                trade = await trade_repo.get_by_ticket(event.ticket)
                if trade:
                    trade.exit_price            = Decimal(str(event.exit_price))
                    trade.exit_time             = now
                    trade.exit_reason           = event.event
                    trade.profit_loss           = Decimal(str(event.pnl))
                    trade.outcome               = _OUTCOME_MAP.get(event.outcome)
                    trade.status                = TradeStatus.CLOSED
                    trade.account_balance_after = Decimal(str(event.balance))
                    if event.rr:
                        trade.risk_reward_actual = Decimal(str(round(event.rr, 2)))
                    await trade_repo.update(trade)
                    logger.info("Full close persisted", ticket=event.ticket, pnl=event.pnl)

            # Always write a system event for a full audit trail
            await system_repo.log_event(
                event_type = f"EA_{event.event}",
                message    = f"EA {event.event} #{event.ticket} {event.symbol}",
                severity   = "INFO",
                details    = {
                    "ticket":  event.ticket,
                    "pnl":     event.pnl,
                    "outcome": event.outcome,
                    "session": event.session,
                },
            )

    except Exception as exc:
        logger.error(
            "DB persist failed — CSV record still written",
            error=str(exc),
            ticket=event.ticket,
            trade_event=event.event,
        )


# =============================================================================
# Telegram notifications
# =============================================================================

async def _notify_event(event: TradeEvent) -> None:
    """Forward OPEN and close events to Telegram (if enabled)."""
    try:
        notifier = NotificationService.get_instance()
        if not notifier.is_enabled:
            return

        if event.event == "OPEN":
            await notifier.notify_trade_opened(
                symbol        = event.symbol,
                direction     = event.direction,
                entry_price   = event.entry,
                stop_loss     = event.sl,
                take_profit_1 = event.tp1,
                take_profit_2 = event.tp2 if event.tp2 else None,
                lot_size      = event.lots,
                reason        = f"{event.method} | {event.session}",
            )
        elif event.event in _ALL_CLOSE_EVENTS:
            await notifier.notify_trade_closed(
                symbol           = event.symbol,
                direction        = event.direction,
                entry_price      = event.entry,
                exit_price       = event.exit_price,
                profit_loss      = event.pnl,
                lot_size         = event.lots,
                duration_minutes = 0,
                exit_reason      = event.event,
                is_partial       = event.event in _PARTIAL_CLOSE_EVENTS,
            )
    except Exception as exc:
        logger.warning("Telegram notification failed", error=str(exc))


# =============================================================================
# APIRouter — all EA routes registered here so they can be composed into
# any FastAPI app via app.include_router(ea_router).
# =============================================================================

ea_router = APIRouter(tags=["EA Events"])


@ea_router.get("/health")
async def health():
    return {
        "status":           "ok",
        "version":          VERSION,
        "events_in_memory": len(_events_store),
        "csv_file":         str(CSV_FILE),
        "server_time":      datetime.now(timezone.utc).isoformat(),
    }


@ea_router.post("/trade", status_code=201)
async def receive_trade_event(event: TradeEvent):
    """Receive a trade event from the EA (InpFastAPIURL endpoint)."""
    row = event.model_dump()
    row["received_at"] = datetime.now(timezone.utc).isoformat()

    _append_to_csv(row)           # 1. CSV — always first, never raises
    _events_store.append(row)     # 2. In-memory store
    await _persist_event(event)   # 3. PostgreSQL
    await _notify_event(event)    # 4. Telegram

    exit_str = f"{event.exit_price:.2f}" if event.exit_price else "0.00"
    logger.info(
        "Event received",
        trade_event = event.event,
        ticket      = event.ticket,
        direction   = event.direction,
        method      = event.method,
        entry       = event.entry,
        exit        = exit_str,
        pnl         = event.pnl,
        outcome     = event.outcome,
    )
    return {"status": "logged", "ticket": event.ticket, "event": event.event}


@ea_router.get("/trades")
def get_trades(
    ticket:    Optional[int] = None,
    event:     Optional[str] = None,
    direction: Optional[str] = None,
    outcome:   Optional[str] = None,
    limit:     int           = 200,
):
    """Return trade events newest-first with optional filters."""
    result = list(reversed(_events_store))
    if ticket    is not None:
        result = [e for e in result if str(e.get("ticket")) == str(ticket)]
    if event     is not None:
        result = [e for e in result if e.get("event")     == event.upper()]
    if direction is not None:
        result = [e for e in result if e.get("direction") == direction.upper()]
    if outcome   is not None:
        result = [e for e in result if e.get("outcome")   == outcome.upper()]
    return result[:limit]


@ea_router.get("/trades/summary")
def get_summary():
    """Win rate, P&L, and breakdowns by method / session / direction."""
    closed = [
        e for e in _events_store
        if e.get("event") in (
            "SL_HIT", "TP1_HIT", "TP2_HIT", "TIME_EXIT", "TRAIL_EXIT"
        )
    ]
    if not closed:
        return {"message": "No closed events yet.", "total_events": len(_events_store)}

    wins   = [e for e in closed if e.get("outcome") in ("WIN", "WIN_PARTIAL")]
    losses = [e for e in closed if e.get("outcome") == "LOSS"]

    def _sf(v, d: float = 0.0) -> float:
        try:    return float(v)
        except: return d

    total_pnl = sum(_sf(e.get("pnl")) for e in closed)
    win_pnl   = sum(_sf(e.get("pnl")) for e in wins)
    loss_pnl  = sum(_sf(e.get("pnl")) for e in losses)
    pf        = abs(win_pnl / loss_pnl) if loss_pnl else float("inf")

    def _tally(field: str) -> dict:
        acc: dict = defaultdict(lambda: {"wins": 0, "losses": 0})
        for e in closed:
            k = e.get(field, "Unknown")
            if   e.get("outcome") in ("WIN", "WIN_PARTIAL"): acc[k]["wins"]   += 1
            elif e.get("outcome") == "LOSS":                 acc[k]["losses"] += 1
        return {
            k: {
                **v,
                "win_rate": (
                    round(v["wins"] / (v["wins"] + v["losses"]), 4)
                    if (v["wins"] + v["losses"]) else 0.0
                ),
            }
            for k, v in acc.items()
        }

    return {
        "total_events":  len(_events_store),
        "closed_events": len(closed),
        "wins":          len(wins),
        "losses":        len(losses),
        "win_rate":      round(len(wins) / len(closed), 4) if closed else 0,
        "total_pnl":     round(total_pnl, 2),
        "avg_win":       round(win_pnl   / len(wins),   2) if wins   else 0.0,
        "avg_loss":      round(loss_pnl  / len(losses), 2) if losses else 0.0,
        "profit_factor": round(pf, 2) if pf != float("inf") else "∞",
        "by_method":    _tally("method"),
        "by_session":   _tally("session"),
        "by_direction": _tally("direction"),
    }


# =============================================================================
# Standalone application factory (backward-compat / testing)
# =============================================================================

def create_logging_app() -> FastAPI:
    """
    Build a standalone FastAPI app that wraps ea_router.
    Preserved for backward compatibility and standalone testing.
    For production, main.py composes the unified app directly.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await init_database()
        _load_csv_to_memory()
        _ensure_csv_header()
        logger.info(
            "Trade event logging server ready",
            version = VERSION,
            csv     = str(CSV_FILE),
            loaded  = len(_events_store),
        )
        yield

    app = FastAPI(
        title       = "MTFTR Trade Event Logger",
        description = "Receives EA trade events and persists to DB + CSV",
        version     = VERSION,
        lifespan    = lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins = ["*"],
        allow_methods = ["*"],
        allow_headers = ["*"],
    )
    app.include_router(ea_router)
    return app

