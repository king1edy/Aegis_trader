"""
Session Tagger
==============
Pure-function utilities for classifying a UTC datetime into a trading session
and computing convenience fields (hour_of_day, day_of_week, pip_size, R:R).

No imports from the rest of src/ — keeps this module free of circular deps
and fast to import in any context.
"""

from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Session windows (UTC hours, inclusive start, exclusive end)
# ---------------------------------------------------------------------------
_SESSIONS: list[tuple[str, int, int]] = [
    ("Asian",    0,  7),   # 00:00–07:00 UTC
    ("London",   7, 12),   # 07:00–12:00 UTC
    ("New_York", 13, 17),  # 13:00–17:00 UTC
    # London/NY overlap 12:00–13:00 falls in "Off-Hours" by design;
    # adjust here if you want an explicit overlap label.
]

# Pip sizes for common symbols.  Extend as needed.
_PIP_SIZES: dict[str, float] = {
    "XAUUSD": 0.1,
    "XAGUSD": 0.001,
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "AUDUSD": 0.0001,
    "USDCAD": 0.0001,
    "USDCHF": 0.0001,
    "NZDUSD": 0.0001,
    "EURJPY": 0.01,
    "GBPJPY": 0.01,
}

# Day-of-week names (index = Monday = 0)
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def get_session(dt: datetime) -> str:
    """
    Return the trading session name for a UTC datetime.

    Returns one of: "Asian", "London", "New_York", "Off-Hours".
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    hour = dt.hour
    for name, start, end in _SESSIONS:
        if start <= hour < end:
            return name
    return "Off-Hours"


def get_hour_of_day(dt: datetime) -> int:
    """Return 0–23 UTC hour from a datetime."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.hour


def get_day_of_week(dt: datetime) -> int:
    """Return 0 (Monday) … 6 (Sunday) for a datetime."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.weekday()


def get_pip_size(symbol: str) -> float:
    """Return pip size for a symbol.  Defaults to 0.0001 for unknown pairs."""
    return _PIP_SIZES.get(symbol.upper(), 0.0001)


def compute_pips(symbol: str, direction: str, entry: float, exit_price: float) -> float:
    """
    Compute P&L in pips.

    Args:
        direction: "BUY" or "SELL"
        entry, exit_price: prices in quote currency units
    """
    pip = get_pip_size(symbol)
    if pip == 0:
        return 0.0
    raw = (exit_price - entry) if direction.upper() == "BUY" else (entry - exit_price)
    return raw / pip


def compute_rr(
    direction: str,
    entry: float,
    exit_price: float,
    stop_loss: float,
) -> Optional[float]:
    """
    Compute the actual risk-reward ratio achieved on a trade.

    Returns None if stop_loss is missing or the SL distance is zero.
    """
    if stop_loss is None or stop_loss == 0:
        return None
    sl_dist = abs(entry - stop_loss)
    if sl_dist == 0:
        return None
    profit_dist = (exit_price - entry) if direction.upper() == "BUY" else (entry - exit_price)
    return round(profit_dist / sl_dist, 2)
