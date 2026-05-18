"""
Aegis Trading — FastAPI Entry Point
=====================================

Start the server:

    Development:  cd src && uvicorn main:app --reload
    Production:   cd src && uvicorn main:app --host 0.0.0.0 --port 8000
    Direct:       cd src && python main.py

The MODE is determined by the EA_MODE env variable:

    EA_MODE=true  → passive logging server (EA executes the strategy in MT5)
    EA_MODE=false → full trading system (Python executes the strategy via MT5 bridge)
"""

import asyncio
import signal
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.logging_config import setup_logging
from database.repository import init_database
from trade_logging.trade_event_server import (
    ea_router,
    _load_csv_to_memory,
    _ensure_csv_header,
)
from auth.router import auth_router
from journal.router import journal_router
from journal.poller import JournalPoller
from settings.router import settings_router
from core.rate_limiter import RateLimitMiddleware
from webhooks.tv_router import tv_router

logger = setup_logging()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hook for both EA and trading modes."""

    # ── Shared startup ─────────────────────────────────────────────────────
    try:
        await init_database()
    except Exception as exc:
        logger.warning(
            "Database unavailable at startup — running in CSV-only mode",
            error=str(exc),
        )

    _load_csv_to_memory()
    _ensure_csv_header()

    # Start MT5 journal poller (background task)
    poller = JournalPoller()
    poller_task = asyncio.create_task(poller.run())

    # ── Mode-specific startup ──────────────────────────────────────────────
    trading_task = None

    if settings.ea_mode:
        logger.info(
            "Webhook mode — receiving events from MT5 EA and TradingView",
            host=settings.ea_log_server_host,
            port=settings.ea_log_server_port,
            poll_s=settings.journal_poll_interval_seconds,
        )
    else:
        # DEPRECATED: Direct strategy execution via Python + MT5 bridge.
        # The production ingestion path is POST /webhook/tradingview (or
        # POST /trade for the EA).  This branch is retained for local
        # development and backtesting only.
        logger.warning(
            "EA_MODE=false — starting deprecated direct-execution mode. "
            "For production, use EA_MODE=true with TradingView webhooks."
        )
        from trading_system import TradingSystem

        system = TradingSystem()

        async def _run_trading_system():
            try:
                await system.initialize()
                await system.run()
            except Exception as e:
                logger.exception("Trading system error", error=str(e))

        trading_task = asyncio.create_task(_run_trading_system())
        logger.info("Trading mode — strategy running as background task")

    yield

    # ── Shutdown ───────────────────────────────────────────────────────────
    if trading_task and not trading_task.done():
        trading_task.cancel()
        try:
            await trading_task
        except asyncio.CancelledError:
            pass

    poller_task.cancel()
    try:
        await poller_task
    except asyncio.CancelledError:
        pass


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Aegis Trading",
    description=(
        "TradingView webhook receiver (POST /webhook/tradingview) · "
        "EA event receiver (POST /trade) · "
        "trade journal + dashboard (GET /) · "
        "strategy engine [deprecated] (EA_MODE=false)"
    ),
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RateLimitMiddleware)

# Auth routes:        POST /api/auth/register  POST /api/auth/login  GET /api/auth/me  API key CRUD
app.include_router(auth_router)
# EA webhook routes:  POST /trade  GET /trades  GET /trades/summary  GET /health
app.include_router(ea_router)
# Journal routes:     GET /  GET /api/journal/*
app.include_router(journal_router)
# Settings routes:    GET/PATCH /api/settings  GET /api/settings/subscription  GET /api/settings/rate-limits
app.include_router(settings_router)
# TradingView webhook: POST /webhook/tradingview
app.include_router(tv_router)


# ---------------------------------------------------------------------------
# CLI entry point:  python main.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.ea_log_server_host,
        port=settings.ea_log_server_port,
        reload=True,
    )
