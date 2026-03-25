"""
Deal Mapper
===========
Translates MT5DealRecord / MT5PositionRecord structs into SQLAlchemy model
instances (Trade, PartialClose, TradeModification, JournalDeal, AccountSnapshot).

This module contains *only* mapping logic — it never touches the database
directly.  The poller calls these helpers then persists the results.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional
from uuid import uuid4

from database.models import (
    JournalDeal,
    PartialClose,
    Trade,
    TradeModification,
    TradeOutcome,
    TradeStatus,
    OrderType,
    SignalSource,
    AccountSnapshot,
)
from journal.mt5_reader import MT5DealRecord, MT5PositionRecord
from journal.session_tagger import (
    compute_rr,
    get_day_of_week,
    get_hour_of_day,
    get_session,
)


# ---------------------------------------------------------------------------
# MT5 deal → JournalDeal (raw audit record)
# ---------------------------------------------------------------------------

def deal_to_journal_deal(deal: MT5DealRecord) -> JournalDeal:
    """Create a JournalDeal from a raw MT5DealRecord."""
    return JournalDeal(
        deal_id         = deal.deal_id,
        position_id     = deal.position_id,
        order_id        = deal.order_id,
        symbol          = deal.symbol,
        deal_time       = deal.time,
        deal_type       = deal.deal_type,
        entry_type      = deal.entry_type,
        volume          = Decimal(str(deal.volume)),
        price           = Decimal(str(deal.price)),
        commission      = Decimal(str(deal.commission)) if deal.commission else None,
        swap            = Decimal(str(deal.swap)) if deal.swap else None,
        profit          = Decimal(str(deal.profit)) if deal.profit is not None else None,
        exit_reason     = deal.exit_reason,
        comment         = deal.comment,
        raw_reason_code = deal.raw_reason_code,
    )


# ---------------------------------------------------------------------------
# MT5 position (IN deal) → new Trade
# ---------------------------------------------------------------------------

def position_to_trade(pos: MT5PositionRecord) -> Trade:
    """
    Create a new Trade from an open MT5 position.

    Called when the poller detects a position not yet in the database.
    The SL/TP stored on the position are used; they may later be updated
    by TradeModification records if the trader moves them.
    """
    now = pos.time_open
    return Trade(
        ticket                 = pos.ticket,
        symbol                 = pos.symbol,
        order_type             = OrderType.BUY if pos.direction == "BUY" else OrderType.SELL,
        status                 = TradeStatus.OPEN,
        signal_source          = SignalSource.MANUAL,
        strategy_name          = "Manual",
        signal_time            = now,
        entry_price            = Decimal(str(pos.price_open)),
        requested_entry_price  = Decimal(str(pos.price_open)),
        entry_time             = now,
        lot_size               = Decimal(str(pos.volume)),
        initial_lot_size       = Decimal(str(pos.volume)),
        stop_loss              = Decimal(str(pos.sl)) if pos.sl else Decimal("0"),
        take_profit_1          = Decimal(str(pos.tp)) if pos.tp else None,
        position_state         = "initial",
        # Journal context
        mt5_position_id        = pos.position_id,
        trade_source           = "manual",
        trading_session        = get_session(now),
        hour_of_day            = get_hour_of_day(now),
        day_of_week            = get_day_of_week(now),
    )


def in_deal_to_trade(deal: MT5DealRecord, sl: Optional[float] = None, tp: Optional[float] = None) -> Trade:
    """
    Create a new Trade from an IN deal (used during historical backfill).

    The SL/TP aren't available on the deal itself, so they are passed
    separately when available (e.g. from the matching position record).
    """
    return Trade(
        ticket                 = deal.order_id or deal.deal_id,
        symbol                 = deal.symbol,
        order_type             = OrderType.BUY if deal.deal_type == "BUY" else OrderType.SELL,
        status                 = TradeStatus.OPEN,
        signal_source          = SignalSource.MANUAL,
        strategy_name          = "Manual",
        signal_time            = deal.time,
        entry_price            = Decimal(str(deal.price)),
        requested_entry_price  = Decimal(str(deal.price)),
        entry_time             = deal.time,
        lot_size               = Decimal(str(deal.volume)),
        initial_lot_size       = Decimal(str(deal.volume)),
        stop_loss              = Decimal(str(sl)) if sl else Decimal("0"),
        take_profit_1          = Decimal(str(tp)) if tp else None,
        position_state         = "initial",
        mt5_position_id        = deal.position_id,
        trade_source           = "manual",
        trading_session        = get_session(deal.time),
        hour_of_day            = get_hour_of_day(deal.time),
        day_of_week            = get_day_of_week(deal.time),
    )


# ---------------------------------------------------------------------------
# OUT deal → Trade closure fields
# ---------------------------------------------------------------------------

def apply_out_deal_to_trade(trade: Trade, deal: MT5DealRecord) -> None:
    """
    Update a Trade in-place with exit information from an OUT deal.

    Modifies: exit_price, exit_time, exit_reason, profit_loss, outcome,
              status, account_balance_after, risk_reward_actual.
    """
    trade.exit_price  = Decimal(str(deal.price))
    trade.exit_time   = deal.time
    trade.exit_reason = deal.exit_reason
    trade.profit_loss = Decimal(str(deal.profit + deal.swap + deal.commission))
    trade.status      = TradeStatus.CLOSED

    # Outcome classification
    net = float(trade.profit_loss)
    if net > 0.5:
        trade.outcome = TradeOutcome.WIN
    elif net < -0.5:
        trade.outcome = TradeOutcome.LOSS
    else:
        trade.outcome = TradeOutcome.BREAKEVEN

    # Risk-reward
    if trade.stop_loss and float(trade.stop_loss) != 0 and trade.entry_price:
        rr = compute_rr(
            direction  = "BUY" if trade.order_type == OrderType.BUY else "SELL",
            entry      = float(trade.entry_price),
            exit_price = float(trade.exit_price),
            stop_loss  = float(trade.stop_loss),
        )
        if rr is not None:
            from decimal import Decimal as D
            trade.risk_reward_actual = D(str(rr))


# ---------------------------------------------------------------------------
# SL/TP change → TradeModification
# ---------------------------------------------------------------------------

def make_sl_modification(
    trade_id,
    old_sl: float,
    new_sl: float,
    timestamp: datetime,
) -> TradeModification:
    return TradeModification(
        trade_id          = trade_id,
        modification_time = timestamp,
        field_modified    = "stop_loss",
        old_value         = Decimal(str(old_sl)),
        new_value         = Decimal(str(new_sl)),
        reason            = "Manual SL adjustment detected by MT5 poller",
        was_automatic     = False,
    )


def make_tp_modification(
    trade_id,
    old_tp: float,
    new_tp: float,
    timestamp: datetime,
) -> TradeModification:
    return TradeModification(
        trade_id          = trade_id,
        modification_time = timestamp,
        field_modified    = "take_profit_1",
        old_value         = Decimal(str(old_tp)),
        new_value         = Decimal(str(new_tp)),
        reason            = "Manual TP adjustment detected by MT5 poller",
        was_automatic     = False,
    )


# ---------------------------------------------------------------------------
# Account info → AccountSnapshot
# ---------------------------------------------------------------------------

def account_info_to_snapshot(info: dict) -> AccountSnapshot:
    """Create an AccountSnapshot from MT5Reader.get_account_info() output."""
    now = datetime.now(tz=timezone.utc)
    return AccountSnapshot(
        timestamp    = now,
        balance      = Decimal(str(info.get("balance", 0))),
        equity       = Decimal(str(info.get("equity", 0))),
        margin       = Decimal(str(info.get("margin", 0))),
        free_margin  = Decimal(str(info.get("free_margin", 0))),
        margin_level = Decimal(str(info.get("margin_level", 0))) if info.get("margin_level") else None,
        floating_pl  = Decimal(str(info.get("profit", 0))),
        open_positions = 0,   # filled separately by the poller
        open_orders    = 0,
    )
