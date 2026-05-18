"""Add TRADINGVIEW to SignalSource enum and GIN index on trades.market_context.

Revision ID: 007_tradingview
Revises: 006_sub_settings_rl
Create Date: 2026-03-23

Enables TradingView webhook ingestion: adds the new enum value so
Trade rows can reference signal_source='TRADINGVIEW', and adds a GIN
index on the JSONB market_context column to speed up filter attribution
queries.
"""

from alembic import op

# revision identifiers
revision = "007_tradingview"
down_revision = "006_sub_settings_rl"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add TRADINGVIEW to the signalsource PostgreSQL enum
    op.execute("ALTER TYPE signalsource ADD VALUE IF NOT EXISTS 'TRADINGVIEW'")

    # GIN index for JSONB filter-attribution queries on market_context
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_trades_market_context "
        "ON trades USING gin (market_context jsonb_path_ops)"
    )


def downgrade() -> None:
    op.drop_index("ix_trades_market_context", table_name="trades")
    # Note: PostgreSQL does not support removing a value from an enum type.
    # To fully reverse, you would need to recreate the enum without TRADINGVIEW.
