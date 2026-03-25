"""
Journal Migration Script
========================
Idempotent ALTER TABLE migration that adds journal columns to the
existing `trades` table, and creates the `journal_deals` and
`setup_tags` tables if they do not already exist.

Safe to run on a populated database — uses IF NOT EXISTS / IF NOT EXISTS
to avoid errors when re-run.

Usage
-----
    python scripts/migrate_journal.py

Environment
-----------
Reads connection details from .env (via src.core.config.settings).
Requires a running PostgreSQL instance with the `trades` table.
"""

import asyncio
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from src.core.config import get_settings
from src.database.models import Base


ALTER_STATEMENTS = [
    # ── Journal annotation columns ────────────────────────────────────────────
    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS setup_tag VARCHAR(100)",
    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS journal_notes TEXT",

    # ── Context columns (set by MT5 poller for manual trades) ─────────────────
    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS trading_session VARCHAR(20)",
    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS hour_of_day INTEGER",
    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS day_of_week INTEGER",

    # ── Manual trade tracking ─────────────────────────────────────────────────
    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS mt5_position_id BIGINT",
    "ALTER TABLE trades ADD COLUMN IF NOT EXISTS trade_source VARCHAR(20) DEFAULT 'ea'",

    # ── Indexes on the new columns ────────────────────────────────────────────
    "CREATE INDEX IF NOT EXISTS ix_trades_session   ON trades(trading_session)",
    "CREATE INDEX IF NOT EXISTS ix_trades_setup_tag ON trades(setup_tag)",
    "CREATE INDEX IF NOT EXISTS ix_trades_mt5_pos   ON trades(mt5_position_id)",
]


async def run_migration() -> None:
    settings = get_settings()
    engine = create_async_engine(settings.async_db_url, echo=True)

    print(f"\n{'='*60}")
    print("Aegis Trade Journal — database migration")
    print(f"Target: {settings.postgres_host}:{settings.postgres_port}/{settings.postgres_db}")
    print(f"{'='*60}\n")

    async with engine.begin() as conn:
        # 1. Apply ALTER TABLE statements
        print("── Step 1: Adding journal columns to `trades` ──────────────")
        for stmt in ALTER_STATEMENTS:
            print(f"  {stmt[:80]}{'…' if len(stmt)>80 else ''}")
            await conn.execute(text(stmt))

        # 2. Create new tables (journal_deals, setup_tags)
        print("\n── Step 2: Creating journal_deals and setup_tags tables ────")
        await conn.run_sync(Base.metadata.create_all)

    await engine.dispose()
    print("\n✓ Migration complete.\n")


if __name__ == "__main__":
    asyncio.run(run_migration())
