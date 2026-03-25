"""Create subscriptions, user_settings, and rate_limits tables.

Revision ID: 006_sub_settings_rl
Revises: 005_tenant_not_null
Create Date: 2026-03-22

Adds the settings system: per-user subscription tracking, a wide
user_settings table, and a reference rate_limits table (one row per tier).
Also backfills default rows for any existing users.
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "006_sub_settings_rl"
down_revision = "005_tenant_not_null"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── subscriptions ─────────────────────────────────────────────────────
    op.create_table(
        "subscriptions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            unique=True,
            nullable=False,
        ),
        sa.Column(
            "tier",
            sa.String(20),
            sa.CheckConstraint("tier IN ('journal', 'pro', 'autopilot')", name="ck_subscriptions_tier"),
            nullable=False,
            server_default="journal",
        ),
        sa.Column("stripe_customer_id", sa.String(255), nullable=True),
        sa.Column("stripe_subscription_id", sa.String(255), nullable=True),
        sa.Column(
            "status",
            sa.String(20),
            sa.CheckConstraint(
                "status IN ('trialing', 'active', 'past_due', 'canceled', 'paused')",
                name="ck_subscriptions_status",
            ),
            nullable=False,
            server_default="trialing",
        ),
        sa.Column("trial_ends_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("current_period_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("current_period_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_subscriptions_user_id", "subscriptions", ["user_id"], unique=True)
    op.create_index("ix_subscriptions_stripe", "subscriptions", ["stripe_customer_id"])

    # ── user_settings ─────────────────────────────────────────────────────
    op.create_table(
        "user_settings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            unique=True,
            nullable=False,
        ),
        # MT5 Connection
        sa.Column("mt5_login", sa.Integer(), nullable=True),
        sa.Column("mt5_server", sa.String(100), nullable=True),
        sa.Column("mt5_password_enc", sa.LargeBinary(), nullable=True),
        sa.Column(
            "mt5_mode",
            sa.String(10),
            sa.CheckConstraint("mt5_mode IN ('ea', 'bridge')", name="ck_user_settings_mt5_mode"),
            nullable=False,
            server_default="ea",
        ),
        # Risk Rules
        sa.Column("max_daily_drawdown_pct", sa.Numeric(5, 2), nullable=False, server_default="5.00"),
        sa.Column("max_consecutive_losses", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("max_lot_size", sa.Numeric(10, 2), nullable=False, server_default="1.00"),
        sa.Column("max_open_positions", sa.Integer(), nullable=False, server_default="5"),
        sa.Column("max_daily_trades", sa.Integer(), nullable=False, server_default="20"),
        sa.Column("allowed_sessions", postgresql.ARRAY(sa.Text()), nullable=False, server_default="{london,new_york}"),
        sa.Column("allowed_symbols", postgresql.ARRAY(sa.Text()), nullable=False, server_default="{XAUUSD}"),
        sa.Column("pause_on_rule_breach", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        # Strategy
        sa.Column("active_strategy_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("strategy_params", postgresql.JSONB(), nullable=False, server_default="{}"),
        # Notifications
        sa.Column("telegram_chat_id", sa.String(100), nullable=True),
        sa.Column("telegram_enabled", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("notify_on_trade_open", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("notify_on_trade_close", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("notify_on_daily_summary", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("notify_on_risk_breach", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        # UI Preferences
        sa.Column(
            "preferences",
            postgresql.JSONB(),
            nullable=False,
            server_default='{"timezone":"UTC","currency_display":"USD","chart_theme":"dark","dashboard_layout":"default"}',
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    op.create_index("ix_user_settings_user_id", "user_settings", ["user_id"], unique=True)

    # ── rate_limits (reference table) ─────────────────────────────────────
    op.create_table(
        "rate_limits",
        sa.Column(
            "tier",
            sa.String(20),
            sa.CheckConstraint("tier IN ('journal', 'pro', 'autopilot')", name="ck_rate_limits_tier"),
            primary_key=True,
        ),
        sa.Column("api_requests_per_minute", sa.Integer(), nullable=False),
        sa.Column("api_requests_per_day", sa.Integer(), nullable=False),
        sa.Column("webhook_events_per_minute", sa.Integer(), nullable=False),
        sa.Column("max_backtests_per_day", sa.Integer(), nullable=False),
        sa.Column("max_strategies", sa.Integer(), nullable=False),
        sa.Column("max_connected_accounts", sa.Integer(), nullable=False),
    )

    # Seed rate limits
    op.execute(
        sa.text(
            "INSERT INTO rate_limits (tier, api_requests_per_minute, api_requests_per_day, "
            "webhook_events_per_minute, max_backtests_per_day, max_strategies, max_connected_accounts) VALUES "
            "('journal',   30,  5000,  60,  2,   1, 1), "
            "('pro',       120, 20000, 300, 20,  10, 3), "
            "('autopilot', 300, 50000, 600, 100, 25, 5)"
        )
    )

    # ── Backfill existing users with default subscription + settings ──────
    op.execute(
        sa.text(
            "INSERT INTO subscriptions (id, user_id, tier, status, created_at, updated_at) "
            "SELECT gen_random_uuid(), u.id, 'journal', 'trialing', now(), now() "
            "FROM users u "
            "WHERE NOT EXISTS (SELECT 1 FROM subscriptions s WHERE s.user_id = u.id)"
        )
    )
    op.execute(
        sa.text(
            "INSERT INTO user_settings (id, user_id, mt5_mode, max_daily_drawdown_pct, "
            "max_consecutive_losses, max_lot_size, max_open_positions, max_daily_trades, "
            "allowed_sessions, allowed_symbols, pause_on_rule_breach, strategy_params, "
            "telegram_enabled, notify_on_trade_open, notify_on_trade_close, "
            "notify_on_daily_summary, notify_on_risk_breach, preferences, created_at, updated_at) "
            "SELECT gen_random_uuid(), u.id, 'ea', 5.00, 3, 1.00, 5, 20, "
            "'{london,new_york}', '{XAUUSD}', true, '{}', false, true, true, true, true, "
            "'{\"timezone\":\"UTC\",\"currency_display\":\"USD\",\"chart_theme\":\"dark\",\"dashboard_layout\":\"default\"}'::jsonb, "
            "now(), now() "
            "FROM users u "
            "WHERE NOT EXISTS (SELECT 1 FROM user_settings us WHERE us.user_id = u.id)"
        )
    )


def downgrade() -> None:
    op.drop_table("rate_limits")
    op.drop_table("user_settings")
    op.drop_table("subscriptions")
