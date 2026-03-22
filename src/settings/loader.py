"""
Trading Configuration Loader
==============================
Loads per-user trading configuration from the ``user_settings`` DB table,
falling back to environment defaults when a value is missing or the DB
is unavailable.

Usage::

    from settings.loader import get_trading_config

    # In an async context with a known user:
    config = await get_trading_config(user_id)

    # Fallback (no DB, e.g. single-tenant / local dev):
    config = TradingConfig.from_env()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger("settings.loader")


# ---------------------------------------------------------------------------
# TradingConfig — duck-type-compatible with core.config.Settings for the
# attributes consumed by TradingSystem, RiskChecker, PositionSizer,
# TelegramNotifier, and SessionFilter.
# ---------------------------------------------------------------------------

@dataclass
class TradingConfig:
    """
    Per-user trading configuration.

    Loaded from ``user_settings`` + ``strategy_params`` JSONB, with
    env-based fallbacks from ``core.config.Settings``.
    """

    # ── Symbol & Sessions ────────────────────────────────────────────────
    default_symbol: str = "XAUUSD"
    allowed_symbols: List[str] = field(default_factory=lambda: ["XAUUSD"])
    allowed_sessions: List[str] = field(default_factory=lambda: ["london", "new_york"])
    london_session_start: str = "07:00"
    london_session_end: str = "12:00"
    ny_session_start: str = "13:00"
    ny_session_end: str = "16:00"
    trade_sessions: str = "london,newyork"

    # ── Risk Management ──────────────────────────────────────────────────
    max_risk_per_trade: float = 0.01
    max_daily_risk: float = 0.03
    max_drawdown_percent: float = 0.10
    max_trades_per_day: int = 3
    min_trade_interval_minutes: int = 60
    max_daily_loss_percent: float = 0.03
    max_consecutive_losses: int = 3
    min_margin_level: float = 200.0
    pause_on_rule_breach: bool = True

    # ── Position Sizing ──────────────────────────────────────────────────
    default_lot_size: float = 0.01
    max_lot_size: float = 0.05
    use_atr_sizing: bool = True
    atr_multiplier: float = 1.5
    max_open_positions: int = 2
    max_daily_trades: int = 3

    # ── MTFTR Strategy ───────────────────────────────────────────────────
    mtftr_enabled: bool = True
    mtftr_ema_200: int = 200
    mtftr_ema_50: int = 50
    mtftr_ema_21: int = 21
    mtftr_hull_55: int = 55
    mtftr_hull_34: int = 34
    mtftr_rsi_period: int = 14
    mtftr_atr_period: int = 14
    mtftr_swing_lookback: int = 5
    mtftr_tp1_rr: float = 1.0
    mtftr_tp2_rr: float = 2.0
    mtftr_tp1_close_percent: float = 0.50
    mtftr_tp2_close_percent: float = 0.30
    mtftr_trail_percent: float = 0.20
    mtftr_min_rsi_long: float = 40.0
    mtftr_max_rsi_long: float = 55.0
    mtftr_min_rsi_short: float = 45.0
    mtftr_max_rsi_short: float = 60.0
    mtftr_min_sl_atr: float = 1.0
    mtftr_max_sl_atr: float = 3.0
    mtftr_sl_buffer_atr: float = 0.5
    mtftr_max_trade_hours: int = 8

    # ── MT5 Connection ───────────────────────────────────────────────────
    mt5_login: int = 0
    mt5_server: str = ""
    mt5_mode: str = "ea"

    # ── Notifications ────────────────────────────────────────────────────
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    notify_on_trade_open: bool = True
    notify_on_trade_close: bool = True
    notify_on_signal_generated: bool = True
    notify_on_daily_summary: bool = True
    notify_on_drawdown_warning: bool = True
    notify_on_error: bool = True
    # Alias used by risk_checker via TelegramNotifier
    notify_on_risk_breach: bool = True

    # ── Behavioural Safeguards ───────────────────────────────────────────
    cooldown_after_loss_minutes: int = 30
    pause_duration_hours: int = 4

    # ── Source tracking ──────────────────────────────────────────────────
    _source: str = "env"  # "env" or "db"

    # ------------------------------------------------------------------
    # Factory: build from env-only Settings
    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, env=None) -> "TradingConfig":
        """Create config from environment-based Settings (no DB)."""
        if env is None:
            from core.config import settings as env
        return cls(
            # Symbol & sessions
            default_symbol=env.default_symbol,
            allowed_symbols=[env.default_symbol],
            allowed_sessions=env.trade_sessions.split(","),
            london_session_start=env.london_session_start,
            london_session_end=env.london_session_end,
            ny_session_start=env.ny_session_start,
            ny_session_end=env.ny_session_end,
            trade_sessions=env.trade_sessions,
            # Risk
            max_risk_per_trade=env.max_risk_per_trade,
            max_daily_risk=env.max_daily_risk,
            max_drawdown_percent=env.max_drawdown_percent,
            max_trades_per_day=env.max_trades_per_day,
            min_trade_interval_minutes=env.min_trade_interval_minutes,
            max_daily_loss_percent=env.max_daily_loss_percent,
            max_consecutive_losses=env.max_consecutive_losses,
            min_margin_level=env.min_margin_level,
            # Position sizing
            default_lot_size=env.default_lot_size,
            max_lot_size=env.max_lot_size,
            use_atr_sizing=env.use_atr_sizing,
            atr_multiplier=env.atr_multiplier,
            max_open_positions=env.max_open_positions,
            max_daily_trades=env.max_daily_trades,
            # MTFTR
            mtftr_enabled=env.mtftr_enabled,
            mtftr_ema_200=env.mtftr_ema_200,
            mtftr_ema_50=env.mtftr_ema_50,
            mtftr_ema_21=env.mtftr_ema_21,
            mtftr_hull_55=env.mtftr_hull_55,
            mtftr_hull_34=env.mtftr_hull_34,
            mtftr_rsi_period=env.mtftr_rsi_period,
            mtftr_atr_period=env.mtftr_atr_period,
            mtftr_swing_lookback=env.mtftr_swing_lookback,
            mtftr_tp1_rr=env.mtftr_tp1_rr,
            mtftr_tp2_rr=env.mtftr_tp2_rr,
            mtftr_tp1_close_percent=env.mtftr_tp1_close_percent,
            mtftr_tp2_close_percent=env.mtftr_tp2_close_percent,
            mtftr_trail_percent=env.mtftr_trail_percent,
            mtftr_min_rsi_long=env.mtftr_min_rsi_long,
            mtftr_max_rsi_long=env.mtftr_max_rsi_long,
            mtftr_min_rsi_short=env.mtftr_min_rsi_short,
            mtftr_max_rsi_short=env.mtftr_max_rsi_short,
            mtftr_min_sl_atr=env.mtftr_min_sl_atr,
            mtftr_max_sl_atr=env.mtftr_max_sl_atr,
            mtftr_sl_buffer_atr=env.mtftr_sl_buffer_atr,
            mtftr_max_trade_hours=env.mtftr_max_trade_hours,
            # MT5
            mt5_login=env.mt5_login,
            mt5_server=env.mt5_server,
            # Notifications
            telegram_enabled=env.telegram_enabled,
            telegram_bot_token=env.telegram_bot_token,
            telegram_chat_id=env.telegram_chat_id,
            notify_on_trade_open=env.notify_on_trade_open,
            notify_on_trade_close=env.notify_on_trade_close,
            notify_on_signal_generated=env.notify_on_signal_generated,
            notify_on_daily_summary=env.notify_on_daily_summary,
            notify_on_drawdown_warning=env.notify_on_drawdown_warning,
            notify_on_error=env.notify_on_error,
            # Safeguards
            cooldown_after_loss_minutes=env.cooldown_after_loss_minutes,
            pause_duration_hours=env.pause_duration_hours,
            _source="env",
        )

    # ------------------------------------------------------------------
    # Factory: build from DB UserSettings row + env fallbacks
    # ------------------------------------------------------------------
    @classmethod
    def from_db(cls, user_settings, env=None) -> "TradingConfig":
        """
        Build config by merging a ``UserSettings`` ORM row with env
        defaults.  Strategy-specific params (mtftr_*) are stored in the
        ``strategy_params`` JSONB column.
        """
        if env is None:
            from core.config import settings as env

        sp: Dict[str, Any] = user_settings.strategy_params or {}

        return cls(
            # ── From typed DB columns ────────────────────────────────
            default_symbol=(user_settings.allowed_symbols or [env.default_symbol])[0],
            allowed_symbols=list(user_settings.allowed_symbols or [env.default_symbol]),
            allowed_sessions=list(user_settings.allowed_sessions or env.trade_sessions.split(",")),
            london_session_start=sp.get("london_session_start", env.london_session_start),
            london_session_end=sp.get("london_session_end", env.london_session_end),
            ny_session_start=sp.get("ny_session_start", env.ny_session_start),
            ny_session_end=sp.get("ny_session_end", env.ny_session_end),
            trade_sessions=",".join(user_settings.allowed_sessions or env.trade_sessions.split(",")),

            # Risk — DB columns map to config attrs
            max_risk_per_trade=sp.get("max_risk_per_trade", env.max_risk_per_trade),
            max_daily_risk=sp.get("max_daily_risk", env.max_daily_risk),
            max_drawdown_percent=float(user_settings.max_daily_drawdown_pct) / 100.0,  # DB stores 5.00 → 0.05
            max_trades_per_day=user_settings.max_daily_trades,
            min_trade_interval_minutes=sp.get("min_trade_interval_minutes", env.min_trade_interval_minutes),
            max_daily_loss_percent=float(user_settings.max_daily_drawdown_pct) / 100.0,
            max_consecutive_losses=user_settings.max_consecutive_losses,
            min_margin_level=sp.get("min_margin_level", env.min_margin_level),
            pause_on_rule_breach=user_settings.pause_on_rule_breach,

            # Position sizing
            default_lot_size=sp.get("default_lot_size", env.default_lot_size),
            max_lot_size=float(user_settings.max_lot_size),
            use_atr_sizing=sp.get("use_atr_sizing", env.use_atr_sizing),
            atr_multiplier=sp.get("atr_multiplier", env.atr_multiplier),
            max_open_positions=user_settings.max_open_positions,
            max_daily_trades=user_settings.max_daily_trades,

            # MTFTR — from strategy_params JSONB
            mtftr_enabled=sp.get("mtftr_enabled", env.mtftr_enabled),
            mtftr_ema_200=sp.get("mtftr_ema_200", env.mtftr_ema_200),
            mtftr_ema_50=sp.get("mtftr_ema_50", env.mtftr_ema_50),
            mtftr_ema_21=sp.get("mtftr_ema_21", env.mtftr_ema_21),
            mtftr_hull_55=sp.get("mtftr_hull_55", env.mtftr_hull_55),
            mtftr_hull_34=sp.get("mtftr_hull_34", env.mtftr_hull_34),
            mtftr_rsi_period=sp.get("mtftr_rsi_period", env.mtftr_rsi_period),
            mtftr_atr_period=sp.get("mtftr_atr_period", env.mtftr_atr_period),
            mtftr_swing_lookback=sp.get("mtftr_swing_lookback", env.mtftr_swing_lookback),
            mtftr_tp1_rr=sp.get("mtftr_tp1_rr", env.mtftr_tp1_rr),
            mtftr_tp2_rr=sp.get("mtftr_tp2_rr", env.mtftr_tp2_rr),
            mtftr_tp1_close_percent=sp.get("mtftr_tp1_close_percent", env.mtftr_tp1_close_percent),
            mtftr_tp2_close_percent=sp.get("mtftr_tp2_close_percent", env.mtftr_tp2_close_percent),
            mtftr_trail_percent=sp.get("mtftr_trail_percent", env.mtftr_trail_percent),
            mtftr_min_rsi_long=sp.get("mtftr_min_rsi_long", env.mtftr_min_rsi_long),
            mtftr_max_rsi_long=sp.get("mtftr_max_rsi_long", env.mtftr_max_rsi_long),
            mtftr_min_rsi_short=sp.get("mtftr_min_rsi_short", env.mtftr_min_rsi_short),
            mtftr_max_rsi_short=sp.get("mtftr_max_rsi_short", env.mtftr_max_rsi_short),
            mtftr_min_sl_atr=sp.get("mtftr_min_sl_atr", env.mtftr_min_sl_atr),
            mtftr_max_sl_atr=sp.get("mtftr_max_sl_atr", env.mtftr_max_sl_atr),
            mtftr_sl_buffer_atr=sp.get("mtftr_sl_buffer_atr", env.mtftr_sl_buffer_atr),
            mtftr_max_trade_hours=sp.get("mtftr_max_trade_hours", env.mtftr_max_trade_hours),

            # MT5 connection
            mt5_login=user_settings.mt5_login or env.mt5_login,
            mt5_server=user_settings.mt5_server or env.mt5_server,
            mt5_mode=user_settings.mt5_mode or "ea",

            # Notifications — DB columns + env fallback for bot token
            telegram_enabled=user_settings.telegram_enabled,
            telegram_bot_token=env.telegram_bot_token,  # token stays in env (secret)
            telegram_chat_id=user_settings.telegram_chat_id or env.telegram_chat_id,
            notify_on_trade_open=user_settings.notify_on_trade_open,
            notify_on_trade_close=user_settings.notify_on_trade_close,
            notify_on_signal_generated=sp.get("notify_on_signal_generated", env.notify_on_signal_generated),
            notify_on_daily_summary=user_settings.notify_on_daily_summary,
            notify_on_drawdown_warning=user_settings.notify_on_risk_breach,
            notify_on_error=sp.get("notify_on_error", env.notify_on_error),
            notify_on_risk_breach=user_settings.notify_on_risk_breach,

            # Safeguards
            cooldown_after_loss_minutes=sp.get("cooldown_after_loss_minutes", env.cooldown_after_loss_minutes),
            pause_duration_hours=sp.get("pause_duration_hours", env.pause_duration_hours),
            _source="db",
        )


# ---------------------------------------------------------------------------
# Async loader — the main public API
# ---------------------------------------------------------------------------

async def get_trading_config(user_id: Optional[UUID] = None) -> TradingConfig:
    """
    Load trading configuration for a user.

    Resolution order:
    1. If ``user_id`` is provided → load from ``user_settings`` table
    2. If ``user_id`` is None → try ``settings.default_tenant_id``
    3. If DB is unavailable → fall back to env-only config

    Returns:
        A ``TradingConfig`` instance ready for use by TradingSystem,
        RiskChecker, PositionSizer, etc.
    """
    from core.config import settings as env

    # Determine the user to load for
    target_id = user_id
    if target_id is None and env.default_tenant_id:
        try:
            target_id = UUID(env.default_tenant_id)
        except (ValueError, AttributeError):
            pass

    if target_id is None:
        logger.info("No user_id — using env-only trading config")
        return TradingConfig.from_env(env)

    # Try loading from DB
    try:
        from sqlalchemy import select

        from auth.subscription_models import UserSettings
        from database.repository import get_session

        async with get_session() as session:
            result = await session.execute(
                select(UserSettings).where(UserSettings.user_id == target_id)
            )
            row = result.scalar_one_or_none()

        if row is None:
            logger.warning(
                "No user_settings row for user — falling back to env",
                extra={"user_id": str(target_id)},
            )
            return TradingConfig.from_env(env)

        config = TradingConfig.from_db(row, env)
        logger.info(
            "Trading config loaded from DB",
            extra={"user_id": str(target_id), "source": "db"},
        )
        return config

    except Exception as exc:
        logger.warning(
            "Failed to load trading config from DB — using env fallback",
            extra={"error": str(exc)},
        )
        return TradingConfig.from_env(env)
