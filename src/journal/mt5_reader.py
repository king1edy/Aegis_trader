"""
MT5 Reader
==========
Read-only interface to MetaTrader 5 via the MetaTrader5 Python library.
Runs on Windows (same machine as the MT5 terminal).

Responsibilities
----------------
- Connect / disconnect to MT5 using credentials from settings
- Fetch currently open positions
- Fetch historical deals (closed trades) for a date range
- Fetch account info (balance, equity, margin)

On non-Windows platforms (e.g. Docker Linux container) the MetaTrader5
library is unavailable; MT5Reader.connect() returns False gracefully and
all fetch methods return empty lists.  The dashboard and EA webhook
continue to work without the poller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger("journal.mt5_reader")


# ---------------------------------------------------------------------------
# Data classes — mirror MT5 deal / position structs
# ---------------------------------------------------------------------------

@dataclass
class MT5DealRecord:
    """One row from mt5.history_deals_get()."""
    deal_id:         int
    position_id:     int
    order_id:        int
    symbol:          str
    time:            datetime          # UTC
    deal_type:       str               # "BUY" | "SELL"
    entry_type:      str               # "IN" | "OUT" | "OUT_BY"
    volume:          float
    price:           float
    commission:      float
    swap:            float
    profit:          float
    exit_reason:     str               # "SL" | "TP" | "MANUAL" | "STOP_OUT" | "MOBILE" | "WEB"
    comment:         str
    raw_reason_code: int


@dataclass
class MT5PositionRecord:
    """One row from mt5.positions_get()."""
    position_id: int
    ticket:      int
    symbol:      str
    direction:   str               # "BUY" | "SELL"
    volume:      float
    price_open:  float
    sl:          float             # 0 if not set
    tp:          float             # 0 if not set
    profit:      float
    swap:        float
    commission:  float
    time_open:   datetime          # UTC
    magic:       int
    comment:     str


# ---------------------------------------------------------------------------
# MT5 reason code → string mapping
# ---------------------------------------------------------------------------
_REASON_MAP: dict[int, str] = {
    0: "MANUAL",
    1: "MOBILE",
    2: "WEB",
    3: "SL",
    4: "TP",
    5: "STOP_OUT",
    6: "ROLLOVER",
    7: "VMARGIN",
    8: "SPLIT",
}

_ENTRY_MAP: dict[int, str] = {
    0: "IN",
    1: "OUT",
    2: "INOUT",
    3: "OUT_BY",
}

_TYPE_MAP: dict[int, str] = {
    0: "BUY",
    1: "SELL",
    2: "BALANCE",
    3: "CREDIT",
    4: "CHARGE",
    5: "CORRECTION",
    6: "BONUS",
    7: "COMMISSION",
    8: "COMMISSION_DAILY",
    9: "COMMISSION_MONTHLY",
    10: "COMMISSION_AGENT_DAILY",
    11: "COMMISSION_AGENT_MONTHLY",
    12: "INTEREST",
    13: "BUY_CANCELED",
    14: "SELL_CANCELED",
    15: "DIVIDEND",
    16: "DIVIDEND_FRANKED",
    17: "TAX",
}


# ---------------------------------------------------------------------------
# MT5Reader
# ---------------------------------------------------------------------------

class MT5Reader:
    """
    Read-only wrapper around the MetaTrader5 Python library.

    Usage::

        reader = MT5Reader()
        if reader.connect():
            positions = reader.get_open_positions()
            deals     = reader.get_history_deals(from_dt, to_dt)
            reader.disconnect()
    """

    def __init__(self) -> None:
        self._mt5 = None
        self._connected = False

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Initialise MT5 connection using credentials from settings.
        Returns True on success, False if MT5 library is unavailable or
        credentials are wrong.
        """
        try:
            import MetaTrader5 as mt5  # Windows-only library
            self._mt5 = mt5
        except ImportError:
            logger.warning(
                "MetaTrader5 library not available (non-Windows environment). "
                "MT5 polling disabled — EA webhook still works."
            )
            return False

        from core.config import get_settings
        s = get_settings()

        kwargs: dict = {}
        if s.mt5_path:
            kwargs["path"] = s.mt5_path
        if s.mt5_login:
            kwargs["login"]    = s.mt5_login
            kwargs["password"] = s.mt5_password
            kwargs["server"]   = s.mt5_server

        if not mt5.initialize(**kwargs):
            err = mt5.last_error()
            logger.error("MT5 initialize failed", extra={"error": err})
            return False

        info = mt5.account_info()
        if info is None:
            logger.error("MT5 connected but account_info() returned None")
            mt5.shutdown()
            return False

        self._connected = True
        logger.info(
            "MT5 connected",
            extra={
                "login":   info.login,
                "server":  info.server,
                "balance": info.balance,
            },
        )
        return True

    def disconnect(self) -> None:
        if self._mt5 and self._connected:
            self._mt5.shutdown()
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    def get_account_info(self) -> dict:
        """Return account balance, equity, margin as a plain dict."""
        if not self._connected or self._mt5 is None:
            return {}
        info = self._mt5.account_info()
        if info is None:
            return {}
        return {
            "login":        info.login,
            "server":       info.server,
            "currency":     info.currency,
            "balance":      float(info.balance),
            "equity":       float(info.equity),
            "margin":       float(info.margin),
            "free_margin":  float(info.margin_free),
            "margin_level": float(info.margin_level) if info.margin_level else None,
            "profit":       float(info.profit),
            "leverage":     info.leverage,
        }

    # ------------------------------------------------------------------
    # Open positions
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[MT5PositionRecord]:
        """Return all currently open positions."""
        if not self._connected or self._mt5 is None:
            return []
        raw = self._mt5.positions_get()
        if raw is None:
            return []
        return [self._map_position(p) for p in raw]

    def _map_position(self, p) -> MT5PositionRecord:
        return MT5PositionRecord(
            position_id = p.identifier,
            ticket      = p.ticket,
            symbol      = p.symbol,
            direction   = "BUY" if p.type == 0 else "SELL",
            volume      = float(p.volume),
            price_open  = float(p.price_open),
            sl          = float(p.sl),
            tp          = float(p.tp),
            profit      = float(p.profit),
            swap        = float(p.swap),
            commission  = float(getattr(p, "commission", 0)),
            time_open   = datetime.fromtimestamp(p.time, tz=timezone.utc),
            magic       = p.magic,
            comment     = p.comment or "",
        )

    # ------------------------------------------------------------------
    # Historical deals
    # ------------------------------------------------------------------

    def get_history_deals(
        self,
        date_from: datetime,
        date_to: datetime,
    ) -> list[MT5DealRecord]:
        """
        Return all deals in [date_from, date_to].

        Only returns trade-type deals (BUY/SELL) — balance adjustments,
        commissions recorded as separate deal types are excluded to keep
        the journal focused on actual position activity.
        """
        if not self._connected or self._mt5 is None:
            return []

        # Ensure timezone-aware
        if date_from.tzinfo is None:
            date_from = date_from.replace(tzinfo=timezone.utc)
        if date_to.tzinfo is None:
            date_to = date_to.replace(tzinfo=timezone.utc)

        raw = self._mt5.history_deals_get(date_from, date_to)
        if raw is None:
            return []

        results = []
        for d in raw:
            deal_type = _TYPE_MAP.get(d.type, "UNKNOWN")
            # Only include actual position deals (BUY/SELL)
            if deal_type not in ("BUY", "SELL"):
                continue
            results.append(self._map_deal(d, deal_type))
        return results

    def _map_deal(self, d, deal_type: str) -> MT5DealRecord:
        entry_type = _ENTRY_MAP.get(d.entry, "UNKNOWN")
        reason_code = int(d.reason) if hasattr(d, "reason") else 0
        exit_reason = _REASON_MAP.get(reason_code, d.comment or "UNKNOWN")
        return MT5DealRecord(
            deal_id         = int(d.ticket),
            position_id     = int(d.position_id),
            order_id        = int(d.order),
            symbol          = d.symbol,
            time            = datetime.fromtimestamp(d.time, tz=timezone.utc),
            deal_type       = deal_type,
            entry_type      = entry_type,
            volume          = float(d.volume),
            price           = float(d.price),
            commission      = float(d.commission),
            swap            = float(d.swap),
            profit          = float(d.profit),
            exit_reason     = exit_reason,
            comment         = d.comment or "",
            raw_reason_code = reason_code,
        )
