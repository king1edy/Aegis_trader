"""
Journal Poller
==============
Asyncio background task that continuously polls EXNESS MT5 for:

  1. New open positions      → create Trade records (manual trades)
  2. SL/TP changes           → create TradeModification records
  3. New closed deals        → update Trade records (exit details)
  4. All raw deals           → append to JournalDeal audit table
  5. Account state           → update AccountSnapshot every 5 minutes

Designed to run as ``asyncio.create_task(poller.run())`` inside the
FastAPI app lifespan.  On non-Windows machines the MT5Reader returns empty
lists immediately (graceful degradation — the EA webhook still works).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from sqlalchemy import select

from src.core.config import get_settings
from src.database.models import (
    AccountSnapshot,
    JournalDeal,
    PartialClose,
    SetupTag,
    Trade,
    TradeModification,
    TradeStatus,
    DEFAULT_SETUP_TAGS,
)
from src.database.repository import get_session
from src.journal.deal_mapper import (
    account_info_to_snapshot,
    apply_out_deal_to_trade,
    deal_to_journal_deal,
    in_deal_to_trade,
    make_sl_modification,
    make_tp_modification,
    position_to_trade,
)
from src.journal.mt5_reader import MT5DealRecord, MT5PositionRecord, MT5Reader

logger = logging.getLogger("journal.poller")
settings = get_settings()


class JournalPoller:
    """
    Polls EXNESS MT5 on a background asyncio task.

    State is held in memory; the database is the durable store.
    Safe to restart — backfill is idempotent (skips already-known deals).
    """

    _SNAPSHOT_INTERVAL = timedelta(minutes=5)

    def __init__(self) -> None:
        self._reader = MT5Reader()
        self._known_positions: dict[int, MT5PositionRecord] = {}
        self._last_deal_time: datetime = datetime.now(tz=timezone.utc)
        self._last_snapshot_time: datetime = datetime.min.replace(tzinfo=timezone.utc)
        self._backfilled = False

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main loop.  Run as an asyncio background task."""
        if not self._reader.connect():
            logger.warning(
                "MT5 connection failed — manual trade polling disabled. "
                "EA webhook journaling continues normally."
            )
            return

        try:
            await self._seed_setup_tags()
            await self._backfill()
            self._backfilled = True
        except Exception as exc:
            logger.error(
                "Database unavailable during poller startup — skipping backfill",
                extra={"error": str(exc)},
            )
        logger.info(
            "MT5 poller running",
            extra={"interval_s": settings.journal_poll_interval_seconds},
        )

        try:
            while True:
                await asyncio.sleep(settings.journal_poll_interval_seconds)
                try:
                    await self._poll_positions()
                    await self._poll_deals()
                    await self._maybe_snapshot()
                except Exception as exc:
                    logger.error("Poller cycle error", extra={"error": str(exc)}, exc_info=True)
        except asyncio.CancelledError:
            logger.info("MT5 poller shutting down")
            self._reader.disconnect()

    # ------------------------------------------------------------------
    # Setup tag seeding
    # ------------------------------------------------------------------

    async def _seed_setup_tags(self) -> None:
        """Insert DEFAULT_SETUP_TAGS if the setup_tags table is empty."""
        async with get_session() as session:
            result = await session.execute(select(SetupTag).limit(1))
            if result.scalar_one_or_none() is not None:
                return  # already seeded
            for tag_data in DEFAULT_SETUP_TAGS:
                session.add(SetupTag(**tag_data))
            await session.commit()
            logger.info("Setup tags seeded", extra={"count": len(DEFAULT_SETUP_TAGS)})

    # ------------------------------------------------------------------
    # Historical backfill
    # ------------------------------------------------------------------

    async def _backfill(self) -> None:
        """
        Load up to journal_history_days of deal history on first run.
        Idempotent: skips deals already in journal_deals table.
        """
        days = settings.journal_history_days
        date_from = datetime.now(tz=timezone.utc) - timedelta(days=days)
        date_to   = datetime.now(tz=timezone.utc)

        logger.info(
            "Backfilling trade history",
            extra={"days": days, "from": date_from.strftime("%Y-%m-%d")},
        )

        deals = await asyncio.to_thread(
            self._reader.get_history_deals, date_from, date_to
        )
        if not deals:
            logger.info("No historical deals found")
            return

        # Group by position_id
        by_position: dict[int, list[MT5DealRecord]] = {}
        for d in deals:
            by_position.setdefault(d.position_id, []).append(d)

        created = 0
        async with get_session() as session:
            # Get already-known deal_ids in one query
            existing_result = await session.execute(select(JournalDeal.deal_id))
            existing_ids = {row[0] for row in existing_result.fetchall()}

            for position_id, pos_deals in by_position.items():
                # Persist JournalDeal rows (skip duplicates)
                for d in pos_deals:
                    if d.deal_id not in existing_ids:
                        session.add(deal_to_journal_deal(d))
                        existing_ids.add(d.deal_id)

                # Check if Trade already exists for this position
                trade_result = await session.execute(
                    select(Trade).where(Trade.mt5_position_id == position_id)
                )
                if trade_result.scalar_one_or_none() is not None:
                    continue  # already journaled

                in_deals  = [d for d in pos_deals if d.entry_type == "IN"]
                out_deals = [d for d in pos_deals if d.entry_type in ("OUT", "OUT_BY")]

                if not in_deals:
                    continue

                entry_deal = in_deals[0]
                trade = in_deal_to_trade(entry_deal)

                if out_deals:
                    exit_deal = out_deals[-1]   # last exit (fully closed)
                    apply_out_deal_to_trade(trade, exit_deal)

                session.add(trade)
                created += 1

            await session.commit()

        # Move the deal cursor forward so the incremental poller doesn't re-process
        if deals:
            self._last_deal_time = max(d.time for d in deals)

        logger.info("Backfill complete", extra={"positions_created": created, "deals_total": len(deals)})

    # ------------------------------------------------------------------
    # Incremental position polling
    # ------------------------------------------------------------------

    async def _poll_positions(self) -> None:
        """
        Detect new open positions and SL/TP modifications.
        """
        current_positions_raw = await asyncio.to_thread(self._reader.get_open_positions)
        current: dict[int, MT5PositionRecord] = {p.position_id: p for p in current_positions_raw}
        now = datetime.now(tz=timezone.utc)

        async with get_session() as session:
            for pos_id, pos in current.items():
                if pos_id not in self._known_positions:
                    # Brand new position — check DB first (may have been opened by EA)
                    result = await session.execute(
                        select(Trade).where(Trade.mt5_position_id == pos_id)
                    )
                    existing = result.scalar_one_or_none()
                    if existing is None:
                        # Truly new manual position
                        trade = position_to_trade(pos)
                        session.add(trade)
                        logger.info(
                            "New manual position detected",
                            extra={"symbol": pos.symbol, "direction": pos.direction, "position_id": pos_id},
                        )
                else:
                    # Known position — check for SL/TP changes
                    prev = self._known_positions[pos_id]
                    if prev.sl != pos.sl and pos.sl != 0:
                        result = await session.execute(
                            select(Trade).where(Trade.mt5_position_id == pos_id)
                        )
                        trade = result.scalar_one_or_none()
                        if trade:
                            mod = make_sl_modification(trade.id, prev.sl, pos.sl, now)
                            session.add(mod)
                            trade.stop_loss = Decimal(str(pos.sl))
                            logger.info(
                                "SL change detected",
                                extra={"position_id": pos_id, "old": prev.sl, "new": pos.sl},
                            )
                    if prev.tp != pos.tp and pos.tp != 0:
                        result = await session.execute(
                            select(Trade).where(Trade.mt5_position_id == pos_id)
                        )
                        trade = result.scalar_one_or_none()
                        if trade:
                            mod = make_tp_modification(trade.id, prev.tp, pos.tp, now)
                            session.add(mod)
                            trade.take_profit_1 = Decimal(str(pos.tp))
                            logger.info(
                                "TP change detected",
                                extra={"position_id": pos_id, "old": prev.tp, "new": pos.tp},
                            )

            await session.commit()

        self._known_positions = current

    # ------------------------------------------------------------------
    # Incremental deal polling
    # ------------------------------------------------------------------

    async def _poll_deals(self) -> None:
        """
        Fetch new deals since last poll, persist JournalDeal rows,
        and update Trade records for closed positions.
        """
        now = datetime.now(tz=timezone.utc)
        deals = await asyncio.to_thread(
            self._reader.get_history_deals, self._last_deal_time, now
        )
        if not deals:
            return

        async with get_session() as session:
            # Get existing deal IDs to avoid duplicates
            existing_result = await session.execute(select(JournalDeal.deal_id))
            existing_ids = {row[0] for row in existing_result.fetchall()}

            for deal in deals:
                # Always persist the raw deal for audit trail
                if deal.deal_id not in existing_ids:
                    session.add(deal_to_journal_deal(deal))
                    existing_ids.add(deal.deal_id)

                # For OUT deals, update the matching Trade record
                if deal.entry_type in ("OUT", "OUT_BY"):
                    result = await session.execute(
                        select(Trade).where(
                            Trade.mt5_position_id == deal.position_id,
                            Trade.status.in_([TradeStatus.OPEN, TradeStatus.PARTIALLY_CLOSED]),
                        )
                    )
                    trade = result.scalar_one_or_none()
                    if trade:
                        apply_out_deal_to_trade(trade, deal)
                        logger.info(
                            "Trade closed via MT5 poller",
                            extra={
                                "position_id": deal.position_id,
                                "reason":      deal.exit_reason,
                                "profit":      deal.profit,
                            },
                        )

            await session.commit()

        self._last_deal_time = max(d.time for d in deals)

    # ------------------------------------------------------------------
    # Account snapshots
    # ------------------------------------------------------------------

    async def _maybe_snapshot(self) -> None:
        """Save an AccountSnapshot every 5 minutes."""
        now = datetime.now(tz=timezone.utc)
        if (now - self._last_snapshot_time) < self._SNAPSHOT_INTERVAL:
            return

        info = await asyncio.to_thread(self._reader.get_account_info)
        if not info:
            return

        snapshot = account_info_to_snapshot(info)
        snapshot.open_positions = len(self._known_positions)

        async with get_session() as session:
            session.add(snapshot)
            await session.commit()

        self._last_snapshot_time = now
