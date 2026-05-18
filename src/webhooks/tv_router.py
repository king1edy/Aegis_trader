"""
TradingView Webhook Router
============================
FastAPI router for TradingView alert ingestion.

Endpoint:
    POST /webhook/tradingview   (auth: X-API-Key header)

Routes BUY/SELL to handle_open() and CLOSE to handle_close().
"""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from auth.dependencies import get_tenant_id_from_api_key
from core.logging_config import get_logger
from webhooks.tv_handler import handle_close, handle_open
from webhooks.tv_schema import TradingViewAlert

logger = get_logger("tv_router")

tv_router = APIRouter(tags=["TradingView Webhooks"])


@tv_router.post("/webhook/tradingview", status_code=201)
async def receive_tradingview_alert(
    alert: TradingViewAlert,
    tenant_id: UUID = Depends(get_tenant_id_from_api_key),
):
    """
    Receive a trade alert from TradingView.

    - BUY/SELL: opens a new trade, returns the ``trade_id`` for use in CLOSE.
    - CLOSE: closes an existing trade by ``trade_id``.

    Authentication: ``X-API-Key`` header (same key used for the EA webhook).
    """
    logger.info(
        "TradingView alert received",
        action=alert.action,
        symbol=alert.symbol,
        strategy=alert.strategy_name,
    )

    if alert.action in ("BUY", "SELL"):
        result = await handle_open(alert, tenant_id)
    else:
        result = await handle_close(alert, tenant_id)

    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("detail", "Unknown error"))

    return result
