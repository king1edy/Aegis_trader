"""Add nullable tenant_id to 10 tables.

Revision ID: 003_add_tenant_id
Revises: 002_users_api_keys
Create Date: 2026-03-22

Adds ``tenant_id`` (FK → users.id, nullable) to all tenant-scoped tables.
Also converts single-column unique constraints to composite (column + tenant_id).
"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "003_add_tenant_id"
down_revision = "002_users_api_keys"
branch_labels = None
depends_on = None

# Tables that get a plain tenant_id column (no unique constraint changes)
_SIMPLE_TABLES = [
    "partial_closes",
    "trade_modifications",
    "account_snapshots",
    "system_events",
    "trading_pauses",
    "signals",
]

# Tables that need a unique constraint swapped to include tenant_id
# (old_constraint_name, table, column, new_constraint_name)
_COMPOSITE_UNIQUE = [
    ("trades",           "ticket",  "uq_trades_ticket_tenant"),
    ("daily_performance", "date",   "uq_daily_perf_date_tenant"),
    ("journal_deals",    "deal_id", "uq_journal_deals_deal_tenant"),
    ("setup_tags",       "name",    "uq_setup_tags_name_tenant"),
]


def upgrade() -> None:
    # --- Simple tables: just add the column + index ---
    for table in _SIMPLE_TABLES:
        op.add_column(
            table,
            sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
        )
        op.create_index(f"ix_{table}_tenant", table, ["tenant_id"])

    # --- Trades: drop old unique on ticket, add tenant_id, add composite unique ---
    op.add_column(
        "trades",
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
    )
    op.create_index("ix_trades_tenant", "trades", ["tenant_id"])
    # Drop the old single-column unique (may be named differently in existing DBs)
    try:
        op.drop_constraint("trades_ticket_key", "trades", type_="unique")
    except Exception:
        pass  # constraint may not exist by this exact name
    op.create_unique_constraint("uq_trades_ticket_tenant", "trades", ["ticket", "tenant_id"])

    # --- daily_performance ---
    op.add_column(
        "daily_performance",
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
    )
    op.create_index("ix_daily_performance_tenant", "daily_performance", ["tenant_id"])
    try:
        op.drop_constraint("daily_performance_date_key", "daily_performance", type_="unique")
    except Exception:
        pass
    op.create_unique_constraint("uq_daily_perf_date_tenant", "daily_performance", ["date", "tenant_id"])

    # --- journal_deals ---
    op.add_column(
        "journal_deals",
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
    )
    op.create_index("ix_journal_deals_tenant", "journal_deals", ["tenant_id"])
    try:
        op.drop_constraint("journal_deals_deal_id_key", "journal_deals", type_="unique")
    except Exception:
        pass
    op.create_unique_constraint("uq_journal_deals_deal_tenant", "journal_deals", ["deal_id", "tenant_id"])

    # --- setup_tags ---
    op.add_column(
        "setup_tags",
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("users.id"), nullable=True),
    )
    op.create_index("ix_setup_tags_tenant", "setup_tags", ["tenant_id"])
    try:
        op.drop_constraint("setup_tags_name_key", "setup_tags", type_="unique")
    except Exception:
        pass
    op.create_unique_constraint("uq_setup_tags_name_tenant", "setup_tags", ["name", "tenant_id"])


def downgrade() -> None:
    all_tables = _SIMPLE_TABLES + ["trades", "daily_performance", "journal_deals", "setup_tags"]
    for table in all_tables:
        op.drop_index(f"ix_{table}_tenant", table_name=table)
        op.drop_column(table, "tenant_id")

    # Restore single-column unique constraints
    op.create_unique_constraint("trades_ticket_key", "trades", ["ticket"])
    op.create_unique_constraint("daily_performance_date_key", "daily_performance", ["date"])
    op.create_unique_constraint("journal_deals_deal_id_key", "journal_deals", ["deal_id"])
    op.create_unique_constraint("setup_tags_name_key", "setup_tags", ["name"])
