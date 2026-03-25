"""Seed admin user and assign existing data.

Revision ID: 004_seed_user
Revises: 003_add_tenant_id
Create Date: 2026-03-22

Creates a default admin user and sets tenant_id on all existing rows
so they belong to a known tenant.  The password is read from the
SEED_ADMIN_PASSWORD env var (default: ``changeme123!``).
"""

import os
from uuid import uuid4

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "004_seed_user"
down_revision = "003_add_tenant_id"
branch_labels = None
depends_on = None

# Fixed UUID so it's deterministic & can be referenced by DEFAULT_TENANT_ID
SEED_USER_ID = "00000000-0000-4000-a000-000000000001"

_TENANT_TABLES = [
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
    # Hash password at migration time using passlib (already installed)
    from passlib.context import CryptContext

    pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
    raw_password = os.environ.get("SEED_ADMIN_PASSWORD", "changeme123!")
    hashed = pwd_ctx.hash(raw_password)

    # Insert seed user
    op.execute(
        sa.text(
            "INSERT INTO users (id, email, username, hashed_password, is_active, is_admin, created_at, updated_at) "
            "VALUES (:id, :email, :username, :hashed, true, true, now(), now()) "
            "ON CONFLICT (email) DO NOTHING"
        ).bindparams(
            id=SEED_USER_ID,
            email="admin@aegis.local",
            username="admin",
            hashed=hashed,
        )
    )

    # Assign all orphaned rows to the seed user
    for table in _TENANT_TABLES:
        op.execute(
            sa.text(f"UPDATE {table} SET tenant_id = :uid WHERE tenant_id IS NULL").bindparams(
                uid=SEED_USER_ID
            )
        )


def downgrade() -> None:
    # Set tenant_id back to NULL
    for table in _TENANT_TABLES:
        op.execute(
            sa.text(f"UPDATE {table} SET tenant_id = NULL WHERE tenant_id = :uid").bindparams(
                uid=SEED_USER_ID
            )
        )
    # Remove seed user
    op.execute(
        sa.text("DELETE FROM users WHERE id = :uid").bindparams(uid=SEED_USER_ID)
    )
