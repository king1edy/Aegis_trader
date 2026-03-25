"""
Trade Journal Package
=====================
Captures, stores, and analyses every manual and EA trade on EXNESS MT5.

Components
----------
- mt5_reader    : Read-only MT5 interface (positions, deals, account info)
- poller        : Asyncio background task that polls MT5 every N seconds
- deal_mapper   : Maps MT5 deal structs to database models
- session_tagger: Computes trading session, hour, day from UTC timestamps
- analyzer      : Async pattern-analysis queries on the trade history
- router        : FastAPI APIRouter — dashboard HTML + /api/journal/* endpoints
"""

from journal.session_tagger import get_session, get_hour_of_day, get_day_of_week
from journal.mt5_reader import MT5Reader, MT5DealRecord, MT5PositionRecord

__all__ = [
    "MT5Reader",
    "MT5DealRecord",
    "MT5PositionRecord",
    "get_session",
    "get_hour_of_day",
    "get_day_of_week",
]
