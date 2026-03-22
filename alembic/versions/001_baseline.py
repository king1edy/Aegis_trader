"""Baseline — stamp existing schema.

Revision ID: 001_baseline
Revises:
Create Date: 2026-03-22

This is an empty migration that represents the pre-Alembic schema.
If the database already has tables, run ``alembic stamp 001_baseline``
to mark it as current without executing anything.
"""

from alembic import op

revision = "001_baseline"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
