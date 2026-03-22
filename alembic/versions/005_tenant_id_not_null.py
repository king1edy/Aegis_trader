"""Make tenant_id NOT NULL on all tenant-scoped tables.

Revision ID: 005_tenant_not_null
Revises: 004_seed_user
Create Date: 2026-03-22

After the data migration assigned every row to a tenant, enforce the
NOT NULL constraint so no unscoped data can be inserted going forward.
"""

import sqlalchemy as sa

from alembic import op

revision = "005_tenant_not_null"
down_revision = "004_seed_user"
branch_labels = None
depends_on = None

_TABLES = [
    "trades",
    "partial_closes",
    "trade_modifications",
    "account_snapshots",
    "daily_performance",
    "system_events",
    "trading_pauses",
    "signals",
    "journal_deals",
    "setup_tags",
]


def upgrade() -> None:
    for table in _TABLES:
        op.alter_column(table, "tenant_id", nullable=False)


def downgrade() -> None:
    for table in _TABLES:
        op.alter_column(table, "tenant_id", nullable=True)
