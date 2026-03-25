"""
Settings Router
================
CRUD endpoints for user settings, subscription info, and rate limits.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from auth.dependencies import get_current_user
from auth.models import User
from auth.subscription_models import RateLimits, Subscription, UserSettings
from database.repository import get_session
from settings.schemas import UserSettingsResponse, UserSettingsUpdate

logger = logging.getLogger("settings.router")

settings_router = APIRouter(prefix="/api/settings", tags=["Settings"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _settings_to_response(s: UserSettings) -> UserSettingsResponse:
    """Map ORM object to Pydantic response."""
    return UserSettingsResponse(
        mt5_login=s.mt5_login,
        mt5_server=s.mt5_server,
        mt5_mode=s.mt5_mode,
        max_daily_drawdown_pct=float(s.max_daily_drawdown_pct),
        max_consecutive_losses=s.max_consecutive_losses,
        max_lot_size=float(s.max_lot_size),
        max_open_positions=s.max_open_positions,
        max_daily_trades=s.max_daily_trades,
        allowed_sessions=list(s.allowed_sessions or []),
        allowed_symbols=list(s.allowed_symbols or []),
        pause_on_rule_breach=s.pause_on_rule_breach,
        active_strategy_id=str(s.active_strategy_id) if s.active_strategy_id else None,
        strategy_params=s.strategy_params or {},
        telegram_chat_id=s.telegram_chat_id,
        telegram_enabled=s.telegram_enabled,
        notify_on_trade_open=s.notify_on_trade_open,
        notify_on_trade_close=s.notify_on_trade_close,
        notify_on_daily_summary=s.notify_on_daily_summary,
        notify_on_risk_breach=s.notify_on_risk_breach,
        preferences=s.preferences or {},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@settings_router.get("", response_model=UserSettingsResponse)
async def get_settings(user: User = Depends(get_current_user)):
    """Return the full user settings row."""
    async with get_session() as session:
        result = await session.execute(
            select(UserSettings).where(UserSettings.user_id == user.id)
        )
        settings = result.scalar_one_or_none()
        if settings is None:
            raise HTTPException(status_code=404, detail="Settings not found")
        return _settings_to_response(settings)


@settings_router.patch("", response_model=UserSettingsResponse)
async def update_settings(
    body: UserSettingsUpdate,
    user: User = Depends(get_current_user),
):
    """Partial update — only provided (non-None) fields are applied."""
    async with get_session() as session:
        result = await session.execute(
            select(UserSettings).where(UserSettings.user_id == user.id)
        )
        settings = result.scalar_one_or_none()
        if settings is None:
            raise HTTPException(status_code=404, detail="Settings not found")

        updates = body.model_dump(exclude_unset=True)
        for field, value in updates.items():
            setattr(settings, field, value)

        await session.commit()
        await session.refresh(settings)
        return _settings_to_response(settings)


@settings_router.get("/subscription")
async def get_subscription(user: User = Depends(get_current_user)):
    """Return the user's subscription details."""
    async with get_session() as session:
        result = await session.execute(
            select(Subscription).where(Subscription.user_id == user.id)
        )
        sub = result.scalar_one_or_none()
        if sub is None:
            raise HTTPException(status_code=404, detail="Subscription not found")
        return {
            "tier": sub.tier,
            "status": sub.status,
            "trial_ends_at": sub.trial_ends_at.isoformat() if sub.trial_ends_at else None,
            "current_period_start": sub.current_period_start.isoformat() if sub.current_period_start else None,
            "current_period_end": sub.current_period_end.isoformat() if sub.current_period_end else None,
        }


@settings_router.get("/rate-limits")
async def get_rate_limits(user: User = Depends(get_current_user)):
    """Return the rate limits for the user's current subscription tier."""
    async with get_session() as session:
        result = await session.execute(
            select(RateLimits)
            .join(Subscription, Subscription.tier == RateLimits.tier)
            .where(Subscription.user_id == user.id)
        )
        rl = result.scalar_one_or_none()
        if rl is None:
            raise HTTPException(status_code=404, detail="Rate limits not found")
        return {
            "tier": rl.tier,
            "api_requests_per_minute": rl.api_requests_per_minute,
            "api_requests_per_day": rl.api_requests_per_day,
            "webhook_events_per_minute": rl.webhook_events_per_minute,
            "max_backtests_per_day": rl.max_backtests_per_day,
            "max_strategies": rl.max_strategies,
            "max_connected_accounts": rl.max_connected_accounts,
        }
