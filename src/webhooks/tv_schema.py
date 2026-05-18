"""
TradingView Webhook Schema
===========================
Pydantic models for TradingView alert payloads.

TradingView sends a user-defined JSON body as the alert message.
Users configure their Pine script to populate these fields.
"""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, model_validator


class TradingViewAlert(BaseModel):
    """
    Inbound TradingView webhook alert payload.

    The ``filters`` dict is the key differentiator — it captures the state
    of every confluence filter at signal time, enabling per-filter
    attribution analytics in the journal.
    """

    # Required
    action: Literal["BUY", "SELL", "CLOSE"]
    symbol: str

    # Trade parameters (BUY/SELL)
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    take_profit_2: Optional[float] = None
    quantity: Optional[float] = None

    # Strategy metadata
    strategy_name: str = "TradingView"
    timeframe: Optional[str] = None

    # Filter snapshot — Pine script populates from indicator values at signal time
    # Example: {"ema200_trend": "bullish", "rsi": 47.2, "session": "london"}
    filters: Optional[Dict[str, Any]] = None

    # Close-specific fields
    trade_id: Optional[str] = None          # returned by OPEN, required for CLOSE
    exit_price: Optional[float] = None
    pnl: Optional[float] = None

    # Optional note
    note: Optional[str] = ""

    @model_validator(mode="after")
    def close_requires_trade_id(self) -> "TradingViewAlert":
        if self.action == "CLOSE" and not self.trade_id:
            raise ValueError("trade_id is required for CLOSE action")
        return self
