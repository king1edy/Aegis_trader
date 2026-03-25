"""
Subscription & Settings Models
===============================
SQLAlchemy models for user subscriptions, per-user settings, and
tier-based rate limits.

These are auth-adjacent (all FK → users) and live in the auth package.
"""

from uuid import uuid4

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    LargeBinary,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import relationship

from database.models import Base


class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    tier = Column(
        String(20),
        CheckConstraint("tier IN ('journal', 'pro', 'autopilot')", name="ck_subscriptions_tier"),
        nullable=False,
        default="journal",
    )
    stripe_customer_id = Column(String(255), nullable=True, index=True)
    stripe_subscription_id = Column(String(255), nullable=True)
    status = Column(
        String(20),
        CheckConstraint(
            "status IN ('trialing', 'active', 'past_due', 'canceled', 'paused')",
            name="ck_subscriptions_status",
        ),
        nullable=False,
        default="trialing",
    )
    trial_ends_at = Column(DateTime(timezone=True), nullable=True)
    current_period_start = Column(DateTime(timezone=True), nullable=True)
    current_period_end = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    user = relationship("User", back_populates="subscription")


class UserSettings(Base):
    __tablename__ = "user_settings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )

    # ── MT5 Connection ────────────────────────────────────────────────────
    mt5_login = Column(Integer, nullable=True)
    mt5_server = Column(String(100), nullable=True)
    mt5_password_enc = Column(LargeBinary, nullable=True)  # encrypted at app level
    mt5_mode = Column(
        String(10),
        CheckConstraint("mt5_mode IN ('ea', 'bridge')", name="ck_user_settings_mt5_mode"),
        nullable=False,
        default="ea",
    )

    # ── Trading Risk Rules ────────────────────────────────────────────────
    max_daily_drawdown_pct = Column(Numeric(5, 2), nullable=False, default=5.00)
    max_consecutive_losses = Column(Integer, nullable=False, default=3)
    max_lot_size = Column(Numeric(10, 2), nullable=False, default=1.00)
    max_open_positions = Column(Integer, nullable=False, default=5)
    max_daily_trades = Column(Integer, nullable=False, default=20)
    allowed_sessions = Column(ARRAY(Text), nullable=False, default=["london", "new_york"])
    allowed_symbols = Column(ARRAY(Text), nullable=False, default=["XAUUSD"])
    pause_on_rule_breach = Column(Boolean, nullable=False, default=True)

    # ── Strategy Configuration ────────────────────────────────────────────
    active_strategy_id = Column(UUID(as_uuid=True), nullable=True)
    strategy_params = Column(JSONB, nullable=False, default={})

    # ── Notification Preferences ──────────────────────────────────────────
    telegram_chat_id = Column(String(100), nullable=True)
    telegram_enabled = Column(Boolean, nullable=False, default=False)
    notify_on_trade_open = Column(Boolean, nullable=False, default=True)
    notify_on_trade_close = Column(Boolean, nullable=False, default=True)
    notify_on_daily_summary = Column(Boolean, nullable=False, default=True)
    notify_on_risk_breach = Column(Boolean, nullable=False, default=True)

    # ── UI / Display Preferences (flexible JSONB bucket) ──────────────────
    preferences = Column(
        JSONB,
        nullable=False,
        default={
            "timezone": "UTC",
            "currency_display": "USD",
            "chart_theme": "dark",
            "dashboard_layout": "default",
        },
    )

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )

    user = relationship("User", back_populates="settings")


class RateLimits(Base):
    """Reference table — one row per subscription tier, NOT per user."""

    __tablename__ = "rate_limits"

    tier = Column(
        String(20),
        CheckConstraint("tier IN ('journal', 'pro', 'autopilot')", name="ck_rate_limits_tier"),
        primary_key=True,
    )
    api_requests_per_minute = Column(Integer, nullable=False)
    api_requests_per_day = Column(Integer, nullable=False)
    webhook_events_per_minute = Column(Integer, nullable=False)
    max_backtests_per_day = Column(Integer, nullable=False)
    max_strategies = Column(Integer, nullable=False)
    max_connected_accounts = Column(Integer, nullable=False)
