"""
Settings Schemas
=================
Pydantic request/response models for the settings API.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field


class UserSettingsResponse(BaseModel):
    """Full user settings read."""

    # MT5 Connection
    mt5_login: Optional[int] = None
    mt5_server: Optional[str] = None
    mt5_mode: str

    # Risk Rules
    max_daily_drawdown_pct: float
    max_consecutive_losses: int
    max_lot_size: float
    max_open_positions: int
    max_daily_trades: int
    allowed_sessions: list[str]
    allowed_symbols: list[str]
    pause_on_rule_breach: bool

    # Strategy
    active_strategy_id: Optional[str] = None
    strategy_params: dict[str, Any]

    # Notifications
    telegram_chat_id: Optional[str] = None
    telegram_enabled: bool
    notify_on_trade_open: bool
    notify_on_trade_close: bool
    notify_on_daily_summary: bool
    notify_on_risk_breach: bool

    # UI Preferences
    preferences: dict[str, Any]


class UserSettingsUpdate(BaseModel):
    """Partial update — only non-None fields are applied."""

    # MT5 Connection
    mt5_login: Optional[int] = None
    mt5_server: Optional[str] = None
    mt5_mode: Optional[str] = Field(None, pattern="^(ea|bridge)$")

    # Risk Rules
    max_daily_drawdown_pct: Optional[float] = Field(None, ge=0.01, le=100.0)
    max_consecutive_losses: Optional[int] = Field(None, ge=1, le=100)
    max_lot_size: Optional[float] = Field(None, ge=0.01, le=100.0)
    max_open_positions: Optional[int] = Field(None, ge=1, le=100)
    max_daily_trades: Optional[int] = Field(None, ge=1, le=1000)
    allowed_sessions: Optional[list[str]] = None
    allowed_symbols: Optional[list[str]] = None
    pause_on_rule_breach: Optional[bool] = None

    # Strategy
    active_strategy_id: Optional[str] = None
    strategy_params: Optional[dict[str, Any]] = None

    # Notifications
    telegram_chat_id: Optional[str] = None
    telegram_enabled: Optional[bool] = None
    notify_on_trade_open: Optional[bool] = None
    notify_on_trade_close: Optional[bool] = None
    notify_on_daily_summary: Optional[bool] = None
    notify_on_risk_breach: Optional[bool] = None

    # UI Preferences
    preferences: Optional[dict[str, Any]] = None
